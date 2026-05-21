"""Exchange client adapter for NonKYC REST APIs."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from engine.exchange_client import ExchangeClient, OpenOrder, OrderStatusView
from nonkyc_client.models import OrderRequest
from nonkyc_client.rest import RestClient, RestError, RestRequest


class NonkycRestExchangeClient(ExchangeClient):
    def __init__(self, rest_client: RestClient) -> None:
        self._rest = rest_client

    def get_mid_price(self, symbol: str) -> Decimal:
        ticker = self._rest.get_market_data(symbol)
        if ticker.bid is not None and ticker.ask is not None:
            return (Decimal(ticker.bid) + Decimal(ticker.ask)) / Decimal("2")
        if ticker.last_price is None:
            raise RestError(f"No last price available for {symbol}")
        return Decimal(ticker.last_price)

    def get_orderbook_top(self, symbol: str) -> tuple[Decimal, Decimal]:
        bid, ask, _, _ = self.get_orderbook_top_with_qty(symbol)
        return bid, ask

    def get_orderbook_top_with_qty(
        self, symbol: str
    ) -> tuple[Decimal, Decimal, Decimal, Decimal]:
        response = self._rest.send(
            RestRequest(
                method="GET",
                path="/orderbook",
                params={"ticker_id": symbol, "depth": "1"},
            )
        )
        payload = response.get("data", response.get("result", response))
        if not isinstance(payload, dict):
            raise RestError(f"Unexpected orderbook payload for {symbol}: {payload}")
        bids = self._extract_orderbook_levels(payload.get("bids", []))
        asks = self._extract_orderbook_levels(payload.get("asks", []))
        if not bids or not asks:
            raise RestError(f"Orderbook data missing for {symbol}")
        best_bid = max(bids, key=lambda x: x[0])
        best_ask = min(asks, key=lambda x: x[0])
        return best_bid[0], best_ask[0], best_bid[1], best_ask[1]

    def get_recent_trades(self, symbol: str, limit: int = 200) -> list[dict]:
        # GET /trades?market={symbol} — official endpoint, paginated by trade ID via `since`
        try:
            response = self._rest.send(
                RestRequest(method="GET", path="/trades", params={"market": symbol})
            )
            payload = response.get("data", response.get("result", response))
            if isinstance(payload, list):
                return self._parse_trades(payload)
        except Exception:
            pass
        return []

    @staticmethod
    def _parse_trades(raw: list) -> list[dict]:
        trades = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            ts_raw = entry.get(
                "time", entry.get("timestamp", entry.get("ts", entry.get("created_at")))
            )
            price_raw = entry.get("price", entry.get("rate"))
            if ts_raw is None or price_raw is None:
                continue
            try:
                ts = float(ts_raw)
                # NonKYC timestamps may be in milliseconds; normalise to seconds
                if ts > 1e12:
                    ts /= 1000.0
                trades.append({"ts": ts, "price": str(price_raw)})
            except (ValueError, TypeError):
                continue
        return trades

    def place_limit(
        self,
        symbol: str,
        side: str,
        price: Decimal,
        quantity: Decimal,
        client_id: str | None = None,
        strict_validate: bool | None = None,
    ) -> str:
        order = OrderRequest(
            symbol=symbol,
            side=side,
            order_type="limit",
            price=str(price),
            quantity=str(quantity),
            user_provided_id=client_id,
            strict_validate=strict_validate,
        )
        response = self._rest.place_order(order)
        return response.order_id

    def place_market(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        client_id: str | None = None,
    ) -> str:
        order = OrderRequest(
            symbol=symbol,
            side=side,
            order_type="market",
            price=None,
            quantity=str(quantity),
            user_provided_id=client_id,
        )
        try:
            response = self._rest.place_order(order)
        except RestError as exc:
            message = str(exc).lower()
            if "market" in message or "unsupported" in message:
                raise NotImplementedError(
                    "Market orders are not supported by the NonKYC REST API."
                ) from exc
            raise
        return response.order_id

    def cancel_order(self, order_id: str) -> bool:
        result = self._rest.cancel_order(order_id=order_id)
        return result.success

    def cancel_all(self, market_id: str, order_type: str = "all") -> bool:
        return self._rest.cancel_all_orders_v1(market_id, order_type)

    def get_order(self, order_id: str) -> OrderStatusView:
        response = self._rest.get_order_status(order_id)
        raw = response.raw_payload
        avg_price = self._extract_decimal(
            raw, ("avgPrice", "avg_price", "average", "price")
        )
        filled = self._extract_decimal(raw, ("filled", "filledQty", "filled_qty"))
        updated_at = self._extract_float(
            raw, ("updated", "updatedAt", "timestamp", "time")
        )
        return OrderStatusView(
            status=response.status,
            filled_qty=filled,
            avg_price=avg_price,
            updated_at=updated_at,
        )

    def list_open_orders(self, symbol: str) -> list[OpenOrder]:
        # API uses slash format (XNV/USDT) and requires status=active for open orders
        slash_symbol = symbol.replace("_", "/")
        try:
            response = self._rest.send(
                RestRequest(
                    method="GET",
                    path="/account/orders",
                    params={"symbol": slash_symbol, "status": "active", "limit": "500"},
                )
            )
        except RestError as exc:
            if self._is_not_found_error(exc):
                return []
            raise
        return self._parse_open_orders(response, symbol)

    def _parse_open_orders(
        self, response: dict[str, Any] | list, symbol: str
    ) -> list[OpenOrder]:
        if isinstance(response, list):
            payload = response
        else:
            payload = response.get("data", response.get("result", response))
        if isinstance(payload, dict):
            for key in ("orders", "openOrders", "open_orders", "data", "result"):
                if key in payload:
                    payload = payload[key]
                    break
        if payload in (None, {}):
            return []
        if not isinstance(payload, list):
            raise RestError(f"Unexpected open orders payload for {symbol}: {payload}")
        orders: list[OpenOrder] = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            # Skip orders that are not active (filled/cancelled)
            if entry.get("isActive") is False:
                continue
            order_id = str(
                entry.get(
                    "id",
                    entry.get(
                        "orderId", entry.get("order_id", entry.get("userProvidedId"))
                    ),
                )
            )
            side = entry.get("side", entry.get("type"))
            price_value = entry.get("price", entry.get("rate", entry.get("orderPrice")))
            quantity_value = entry.get(
                "quantity",
                entry.get("remaining", entry.get("amount", entry.get("origQty"))),
            )
            if (
                not order_id
                or side is None
                or price_value is None
                or quantity_value is None
            ):
                continue
            try:
                price = Decimal(str(price_value))
                quantity = Decimal(str(quantity_value))
            except (ValueError, TypeError, InvalidOperation):
                continue
            orders.append(
                OpenOrder(
                    order_id=order_id,
                    symbol=str(entry.get("symbol", symbol)),
                    side=str(side).lower(),
                    price=price,
                    quantity=quantity,
                )
            )
        return orders

    def get_balances(self) -> dict[str, tuple[Decimal, Decimal]]:
        balances = {}
        for balance in self._rest.get_balances():
            balances[balance.asset] = (
                Decimal(balance.available),
                Decimal(balance.held),
            )
        return balances

    @staticmethod
    def _extract_orderbook_levels(levels: Any) -> list[tuple[Decimal, Decimal]]:
        result: list[tuple[Decimal, Decimal]] = []
        if not isinstance(levels, list):
            return result
        for level in levels:
            price_raw = None
            qty_raw = None
            if isinstance(level, dict):
                price_raw = level.get("price")
                qty_raw = level.get("quantity", level.get("amount", level.get("qty", "0")))
            elif isinstance(level, (list, tuple)) and len(level) >= 2:
                price_raw, qty_raw = level[0], level[1]
            elif isinstance(level, (list, tuple)) and len(level) == 1:
                price_raw, qty_raw = level[0], "0"
            if price_raw is None:
                continue
            try:
                result.append((Decimal(str(price_raw)), Decimal(str(qty_raw or "0"))))
            except (ValueError, TypeError, InvalidOperation):
                continue
        return result

    @staticmethod
    def _extract_decimal(payload: Any, keys: tuple[str, ...]) -> Decimal | None:
        if isinstance(payload, dict):
            for key in keys:
                if key in payload and payload[key] is not None:
                    try:
                        return Decimal(str(payload[key]))
                    except Exception:
                        return None
        return None

    @staticmethod
    def _is_not_found_error(exc: Exception) -> bool:
        return isinstance(exc, RestError) and "HTTP error 404" in str(exc)

    @staticmethod
    def _extract_float(payload: Any, keys: tuple[str, ...]) -> float | None:
        if isinstance(payload, dict):
            for key in keys:
                if key in payload and payload[key] is not None:
                    try:
                        return float(payload[key])
                    except Exception:
                        return None
        return None
