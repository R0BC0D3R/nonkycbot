"""XNV advanced market maker: dual ladder MM + trend overlay + 2-leg cross-pair arb."""

from __future__ import annotations

import json
import logging
import random
import statistics
import time
import uuid
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from engine.exchange_client import ExchangeClient, OrderStatusView
from nonkyc_client.rest import RestError

LOGGER = logging.getLogger("nonkyc_bot.strategy.xnv_market_maker")

LOOKBACK_3M = 180.0
LOOKBACK_15M = 900.0
LOOKBACK_1H = 3600.0
HISTORY_MAX_AGE = 7200.0  # 2h rolling window

_W3M = Decimal("0.50")
_W15M = Decimal("0.30")
_W1H = Decimal("0.20")


# ── Config ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PairConfig:
    tick_size: Decimal
    step_size: Decimal
    fee_rate: Decimal
    min_notional: Decimal


@dataclass(frozen=True)
class XnvMarketMakerConfig:
    symbol_usdt: str
    symbol_xmr: str
    # Ladder
    base_order_size: Decimal
    num_levels: int
    level_0_offset_pct: Decimal  # Distance of L0 from mid as fraction of mid
    level_spacing_pct: Decimal   # Additional distance per level as fraction of mid
    size_step_factor: Decimal
    max_order_age_sec: float
    level_reprice_threshold_pct: Decimal  # Min price drift to trigger a level reprice
    # Per-pair precision
    pairs: dict[str, PairConfig]
    # 3-asset inventory
    inventory_target: dict[str, Decimal]  # {"XNV": 0.60, "USDT": 0.30, "XMR": 0.10}
    inventory_tolerance_pct: Decimal
    inventory_skew_pct: Decimal
    # Trend
    trend_threshold: Decimal
    trend_deadzone: Decimal
    ladder_shift_pct: Decimal
    taker_bite_size_factor: Decimal
    taker_bite_interval_sec: float
    taker_max_exposure: Decimal  # Multiples of base_order_size
    xmr_qty_fraction: Decimal  # XNV_XMR order size as fraction of USDT size
    order_size_jitter_pct: Decimal  # ±fraction applied to each order qty
    # Arb
    arb_z_threshold: Decimal
    arb_min_profit_pct: Decimal
    arb_max_order_size: Decimal
    arb_depth_pct: Decimal
    arb_cooldown_sec: float
    arb_ratio_window: int
    arb_retry_ticks: int
    # Operational
    poll_interval_sec: float
    balance_refresh_sec: float
    mode: str


# ── State ──────────────────────────────────────────────────────────────────


@dataclass
class LevelOrder:
    order_id: str
    price: Decimal
    quantity: Decimal
    client_id: str
    created_at: float


class TrendState(Enum):
    NEUTRAL = "neutral"
    UP = "up"
    DOWN = "down"


@dataclass
class PricePoint:
    ts: float
    value: Decimal


@dataclass
class StuckArb:
    pair: str
    side: str
    qty: Decimal
    retry_count: int = 0


@dataclass
class XnvMarketMakerState:
    # open_orders[pair][side][level_index] = LevelOrder
    open_orders: dict[str, dict[str, dict[int, LevelOrder]]] = field(
        default_factory=dict
    )
    trend_state: TrendState = TrendState.NEUTRAL
    trend_exposure_xnv: Decimal = field(default_factory=lambda: Decimal("0"))
    last_bite_ts: float = 0.0
    last_bite_order_id: str | None = None
    last_bite_side: str | None = None
    price_history: list[PricePoint] = field(default_factory=list)
    ratio_history: list[PricePoint] = field(default_factory=list)
    arb_last_executed_ts: float = 0.0
    arb_stuck: StuckArb | None = None


# ── Strategy ───────────────────────────────────────────────────────────────


class XnvMarketMakerStrategy:
    def __init__(
        self,
        client: ExchangeClient,
        config: XnvMarketMakerConfig,
        *,
        state_path: Path | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.state_path = state_path
        self.state = XnvMarketMakerState()
        self._last_balance_refresh = 0.0
        self._last_sync_ts = 0.0
        self._balances: dict[str, tuple[Decimal, Decimal]] = {}
        # (pair, side, level) -> timestamp of last insufficient-funds failure
        self._insuf_funds_ts: dict[tuple[str, str, int], float] = {}
        self._current_mids: dict[str, Decimal] = {}

    # ── Persistence ────────────────────────────────────────────────────────

    def load_state(self) -> None:
        if self.state_path is None or not self.state_path.exists():
            return
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))

        open_orders: dict[str, dict[str, dict[int, LevelOrder]]] = {}
        for pair, sides in raw.get("open_orders", {}).items():
            open_orders[pair] = {}
            for side, levels in sides.items():
                open_orders[pair][side] = {}
                for k_str, o in levels.items():
                    open_orders[pair][side][int(k_str)] = LevelOrder(
                        order_id=o["order_id"],
                        price=Decimal(o["price"]),
                        quantity=Decimal(o["quantity"]),
                        client_id=o["client_id"],
                        created_at=float(o["created_at"]),
                    )

        stuck_raw = raw.get("arb_stuck")
        arb_stuck = (
            StuckArb(
                pair=stuck_raw["pair"],
                side=stuck_raw["side"],
                qty=Decimal(stuck_raw["qty"]),
                retry_count=int(stuck_raw.get("retry_count", 0)),
            )
            if stuck_raw
            else None
        )

        self.state = XnvMarketMakerState(
            open_orders=open_orders,
            trend_state=TrendState(raw.get("trend_state", "neutral")),
            trend_exposure_xnv=Decimal(str(raw.get("trend_exposure_xnv", "0"))),
            last_bite_ts=float(raw.get("last_bite_ts", 0.0)),
            last_bite_order_id=raw.get("last_bite_order_id"),
            last_bite_side=raw.get("last_bite_side"),
            price_history=[
                PricePoint(ts=float(p["ts"]), value=Decimal(p["value"]))
                for p in raw.get("price_history", [])
            ],
            ratio_history=[
                PricePoint(ts=float(p["ts"]), value=Decimal(p["value"]))
                for p in raw.get("ratio_history", [])
            ],
            arb_last_executed_ts=float(raw.get("arb_last_executed_ts", 0.0)),
            arb_stuck=arb_stuck,
        )

    def save_state(self) -> None:
        if self.state_path is None:
            return

        open_orders_raw: dict[str, Any] = {}
        for pair, sides in self.state.open_orders.items():
            open_orders_raw[pair] = {}
            for side, levels in sides.items():
                open_orders_raw[pair][side] = {
                    str(k): {
                        "order_id": o.order_id,
                        "price": str(o.price),
                        "quantity": str(o.quantity),
                        "client_id": o.client_id,
                        "created_at": o.created_at,
                    }
                    for k, o in levels.items()
                }

        payload: dict[str, Any] = {
            "open_orders": open_orders_raw,
            "trend_state": self.state.trend_state.value,
            "trend_exposure_xnv": str(self.state.trend_exposure_xnv),
            "last_bite_ts": self.state.last_bite_ts,
            "last_bite_order_id": self.state.last_bite_order_id,
            "last_bite_side": self.state.last_bite_side,
            "price_history": [
                {"ts": p.ts, "value": str(p.value)} for p in self.state.price_history
            ],
            "ratio_history": [
                {"ts": p.ts, "value": str(p.value)}
                for p in self.state.ratio_history[-self.config.arb_ratio_window :]
            ],
            "arb_last_executed_ts": self.state.arb_last_executed_ts,
            "arb_stuck": (
                {
                    "pair": self.state.arb_stuck.pair,
                    "side": self.state.arb_stuck.side,
                    "qty": str(self.state.arb_stuck.qty),
                    "retry_count": self.state.arb_stuck.retry_count,
                }
                if self.state.arb_stuck
                else None
            ),
        }
        self.state_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def seed_price_history(self) -> None:
        """Pre-populate price history from exchange trade history on startup.

        Fetches the last 2h of trades for XNV_USDT so the 3m/15m/1h trend
        signal is ready immediately. If no trade history is available (common
        for low-volume markets), seeds a single anchor point from the live mid
        price so the 3m signal becomes usable after ~3 min of live polling.
        """
        now = time.time()
        trades = self.client.get_recent_trades(self.config.symbol_usdt, limit=500)

        added = 0
        if trades:
            for t in trades:
                ts = t.get("ts", 0.0)
                price_str = t.get("price")
                if not ts or not price_str or now - ts > HISTORY_MAX_AGE:
                    continue
                try:
                    self.state.price_history.append(
                        PricePoint(ts=ts, value=Decimal(price_str))
                    )
                    added += 1
                except Exception:
                    continue
            self.state.price_history.sort(key=lambda p: p.ts)
            LOGGER.info("Seeded price history with %d trade points", added)
        else:
            # No trade history available — anchor to current mid price so the
            # 3m signal warms up after ~3 min rather than ~60 min.
            try:
                mid = self.client.get_mid_price(self.config.symbol_usdt)
                self.state.price_history.append(PricePoint(ts=now, value=mid))
                LOGGER.info(
                    "No trade history available; anchored to current mid %.6s — "
                    "3m trend signal ready in ~3 min",
                    mid,
                )
            except Exception as exc:
                LOGGER.warning("Could not anchor price history: %s", exc)

    # ── Startup ────────────────────────────────────────────────────────────

    def cancel_all_on_startup(self) -> None:
        """Cancel all open orders on both pairs and clear tracked state."""
        if self.config.mode in {"monitor", "dry-run"}:
            LOGGER.info("DRY-RUN: skipping startup cancel")
            return
        total = 0
        for symbol in (self.config.symbol_usdt, self.config.symbol_xmr):
            try:
                orders = self.client.list_open_orders(symbol)
            except Exception as exc:
                LOGGER.warning("Startup list_open_orders %s failed: %s", symbol, exc)
                continue
            for order in orders:
                try:
                    self.client.cancel_order(order.order_id)
                    total += 1
                except Exception as exc:
                    LOGGER.warning("Startup cancel %s failed: %s", order.order_id, exc)
        LOGGER.info("Startup: cancelled %d open orders", total)
        self.state.open_orders.clear()

    # ── Main loop ──────────────────────────────────────────────────────────

    def poll_once(self) -> None:
        now = time.time()
        self._refresh_balances(now)
        # Settle previous bite order unconditionally (fill/cancel check)
        self._settle_previous_bite(now)
        # Sync ladder order statuses every 30s using list_open_orders comparison
        if now - self._last_sync_ts >= 30.0:
            self._sync_all_order_statuses()
            self._last_sync_ts = now

        try:
            usdt_bid, usdt_ask, usdt_bid_qty, usdt_ask_qty = (
                self.client.get_orderbook_top_with_qty(self.config.symbol_usdt)
            )
            xmr_bid, xmr_ask, xmr_bid_qty, xmr_ask_qty = (
                self.client.get_orderbook_top_with_qty(self.config.symbol_xmr)
            )
        except RestError as exc:
            LOGGER.warning("Orderbook fetch failed: %s", exc)
            return

        if not _valid_book(usdt_bid, usdt_ask) or not _valid_book(xmr_bid, xmr_ask):
            LOGGER.warning(
                "Invalid orderbook: USDT bid=%s ask=%s  XMR bid=%s ask=%s",
                usdt_bid, usdt_ask, xmr_bid, xmr_ask,
            )
            return

        usdt_mid = (usdt_bid + usdt_ask) / Decimal("2")
        xmr_mid = (xmr_bid + xmr_ask) / Decimal("2")
        self._current_mids[self.config.symbol_usdt] = usdt_mid
        self._current_mids[self.config.symbol_xmr] = xmr_mid

        self._record_price_snapshot(usdt_mid, now)
        self._record_ratio_snapshot(usdt_mid, xmr_mid, now)

        # Priority 1: arb
        arb_fired = False
        if now - self.state.arb_last_executed_ts >= self.config.arb_cooldown_sec:
            arb_fired = self._check_and_execute_arb(
                usdt_bid, usdt_ask, usdt_bid_qty, usdt_ask_qty,
                xmr_bid, xmr_ask, xmr_bid_qty, xmr_ask_qty,
                now,
            )
        if not arb_fired and self.state.arb_stuck is not None:
            self._retry_stuck_arb()

        # Priority 2: trend signal + taker bites on XNV_USDT
        score = self._compute_trend_score(now)
        self._update_trend_state(score)

        if not arb_fired and self.state.trend_state != TrendState.NEUTRAL:
            self._maybe_place_taker_bite(usdt_bid, usdt_ask, now)

        # Priority 3: ladder maintenance
        usdt_offset = (
            self._inventory_skew(self.config.symbol_usdt, usdt_mid)
            + self._trend_offset(usdt_mid)
        )
        xmr_offset = (
            self._inventory_skew(self.config.symbol_xmr, xmr_mid)
            + self._trend_offset(xmr_mid)
        )

        self._run_ladder(
            self.config.symbol_usdt, usdt_bid, usdt_ask, usdt_offset, now
        )
        self._run_ladder(
            self.config.symbol_xmr, xmr_bid, xmr_ask, xmr_offset, now
        )

        self.save_state()

    # ── Component 1: Ladder ────────────────────────────────────────────────

    def _run_ladder(
        self,
        pair: str,
        best_bid: Decimal,
        best_ask: Decimal,
        center_offset: Decimal,
        now: float,
    ) -> None:
        mid = (best_bid + best_ask) / Decimal("2")

        # Clamp offset so it can never push L0 prices outside the valid maker range.
        # Without this, a large trend + inventory skew combination can push desired
        # sell L0 below best_bid (or buy L0 above best_ask), causing endless cancel/replace.
        max_offset = mid * self.config.level_0_offset_pct * Decimal("0.9")
        center_offset = max(min(center_offset, max_offset), -max_offset)

        desired = self._compute_desired_levels(pair, best_bid, best_ask, center_offset)
        existing = self.state.open_orders.get(pair, {})

        # Place or replace desired levels
        for (side, k), (desired_price, desired_qty) in desired.items():
            fail_ts = self._insuf_funds_ts.get((pair, side, k), 0.0)
            if now - fail_ts < self.config.taker_bite_interval_sec:
                continue  # Still in cooldown after insufficient-funds failure
            current = existing.get(side, {}).get(k)
            if current is None:
                self._place_level(pair, side, k, desired_price, desired_qty)
            elif self._level_needs_replace(
                current, desired_price, pair, now, side, best_bid, best_ask
            ):
                self._cancel_level(pair, side, k, current.order_id)
                self._place_level(pair, side, k, desired_price, desired_qty)

        # Cancel levels no longer in desired set
        for side in ("buy", "sell"):
            for k in list(existing.get(side, {}).keys()):
                if (side, k) not in desired:
                    self._cancel_level(pair, side, k, existing[side][k].order_id)


    def _compute_desired_levels(
        self,
        pair: str,
        best_bid: Decimal,
        best_ask: Decimal,
        center_offset: Decimal,
    ) -> dict[tuple[str, int], tuple[Decimal, Decimal]]:
        spread = best_ask - best_bid
        if spread <= 0:
            return {}

        pair_cfg = self.config.pairs[pair]
        mid = (best_bid + best_ask) / Decimal("2")

        result: dict[tuple[str, int], tuple[Decimal, Decimal]] = {}

        for k in range(self.config.num_levels):
            # Fixed % of mid per level — independent of current spread width.
            # offset_k is the total distance from mid as a fraction of mid.
            #   buy_k  = mid * (1 - offset_k) + center_offset
            #   sell_k = mid * (1 + offset_k) + center_offset
            offset_k = self.config.level_0_offset_pct + k * self.config.level_spacing_pct
            buy_raw = mid * (Decimal("1") - offset_k) + center_offset
            sell_raw = mid * (Decimal("1") + offset_k) + center_offset

            qty = self.config.base_order_size * (
                Decimal("1") + k * self.config.size_step_factor
            )
            if pair == self.config.symbol_xmr:
                qty *= self.config.xmr_qty_fraction
            jitter = Decimal(str(random.uniform(
                float(1 - self.config.order_size_jitter_pct),
                float(1 + self.config.order_size_jitter_pct),
            )))
            qty = _quantize_qty(qty * jitter, pair_cfg.step_size)
            if qty <= 0:
                continue

            buy_price = _quantize_price(buy_raw, pair_cfg.tick_size, side="buy")
            sell_price = _quantize_price(sell_raw, pair_cfg.tick_size, side="sell")

            if (
                buy_price > 0
                and buy_price < best_ask
                and buy_price * qty >= pair_cfg.min_notional
            ):
                result[("buy", k)] = (buy_price, qty)

            if (
                sell_price > 0
                and sell_price >= best_bid
                and sell_price * qty >= pair_cfg.min_notional
            ):
                result[("sell", k)] = (sell_price, qty)

        return result

    def _level_needs_replace(
        self,
        order: LevelOrder,
        desired_price: Decimal,
        pair: str,
        now: float,
        side: str,
        best_bid: Decimal,
        best_ask: Decimal,
    ) -> bool:
        if now - order.created_at >= self.config.max_order_age_sec:
            return True
        # Reprice immediately if order has crossed strictly inside the spread
        if side == "buy" and order.price > best_ask:
            LOGGER.info("[%s] buy L? @ %s crossed ask %s — repricing", pair, order.price, best_ask)
            return True
        if side == "sell" and order.price < best_bid:
            LOGGER.info("[%s] sell L? @ %s crossed bid %s — repricing", pair, order.price, best_bid)
            return True
        # Large-drift safety net (price moved significantly since placement)
        if order.price > 0:
            drift = abs(order.price - desired_price) / order.price
            if drift >= self.config.level_reprice_threshold_pct:
                return True
        return False

    def _cancel_all_ladder(self, pair: str) -> None:
        existing = self.state.open_orders.get(pair, {})
        for side in ("buy", "sell"):
            for k, order in list(existing.get(side, {}).items()):
                self._cancel_level(pair, side, k, order.order_id)

    def _place_level(
        self,
        pair: str,
        side: str,
        level: int,
        price: Decimal,
        qty: Decimal,
    ) -> None:
        if self.config.mode in {"monitor", "dry-run"}:
            LOGGER.info(
                "[%s] DRY-RUN place %s L%d qty=%s @ %s", pair, side, level, qty, price
            )
            return
        client_id = f"xnv-mm-{uuid.uuid4().hex[:12]}"
        try:
            order_id = self.client.place_limit(
                pair, side, price, qty, client_id=client_id, strict_validate=True
            )
        except RestError as exc:
            if "insufficient funds" in str(exc).lower():
                LOGGER.warning(
                    "[%s] Insufficient funds: %s L%d qty=%s @ %s — skipping for %ds",
                    pair, side, level, qty, price, self.config.taker_bite_interval_sec,
                )
                self._insuf_funds_ts[(pair, side, level)] = time.time()
            else:
                LOGGER.error("[%s] Place %s L%d failed: %s", pair, side, level, exc)
            return

        # Subtract a small random age so not all orders placed at the same time
        # expire simultaneously, spreading out age-based reprices.
        stagger = random.uniform(0, 30.0)
        self.state.open_orders.setdefault(pair, {}).setdefault(side, {})[
            level
        ] = LevelOrder(
            order_id=order_id,
            price=price,
            quantity=qty,
            client_id=client_id,
            created_at=time.time() - stagger,
        )
        LOGGER.info(
            "[%s] Placed %s L%d %s qty=%s @ %s", pair, side, level, order_id, qty, price
        )

    def _cancel_level(
        self, pair: str, side: str, level: int, order_id: str
    ) -> None:
        if self.config.mode in {"monitor", "dry-run"}:
            LOGGER.info(
                "[%s] DRY-RUN cancel %s L%d %s", pair, side, level, order_id
            )
            self.state.open_orders.get(pair, {}).get(side, {}).pop(level, None)
            return
        try:
            self.client.cancel_order(order_id)
            LOGGER.info("[%s] Cancelled %s L%d %s", pair, side, level, order_id)
        except RestError as exc:
            if "not found" in str(exc).lower():
                # Order already gone — likely filled before we could cancel
                self._log_fill_status(pair, side, level, order_id, "cancel-race")
            else:
                LOGGER.error("[%s] Cancel %s failed: %s", pair, order_id, exc)
        self.state.open_orders.get(pair, {}).get(side, {}).pop(level, None)

    # ── Component 2: Trend overlay ─────────────────────────────────────────

    def _record_price_snapshot(self, mid: Decimal, now: float) -> None:
        self.state.price_history.append(PricePoint(ts=now, value=mid))
        cutoff = now - HISTORY_MAX_AGE
        self.state.price_history = [
            p for p in self.state.price_history if p.ts >= cutoff
        ]

    def _price_at(self, lookback: float, now: float) -> Decimal | None:
        """Most recent price snapshot recorded at or before (now - lookback)."""
        target = now - lookback
        candidates = [p for p in self.state.price_history if p.ts <= target]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.ts).value

    def _compute_trend_score(self, now: float) -> Decimal | None:
        if not self.state.price_history:
            return None
        current = self.state.price_history[-1].value

        p3m = self._price_at(LOOKBACK_3M, now)
        if p3m is None or p3m == 0:
            return None  # Need at least 3m to emit any signal

        p15m = self._price_at(LOOKBACK_15M, now)
        p1h = self._price_at(LOOKBACK_1H, now)

        c3m = (current - p3m) / p3m

        if p15m is None or p15m == 0:
            return c3m  # Only 3m available

        c15m = (current - p15m) / p15m

        if p1h is None or p1h == 0:
            # 3m + 15m: renormalise weights
            total_w = _W3M + _W15M
            return (c3m * _W3M + c15m * _W15M) / total_w

        c1h = (current - p1h) / p1h
        return c3m * _W3M + c15m * _W15M + c1h * _W1H

    def _update_trend_state(self, score: Decimal | None) -> None:
        if score is None:
            return
        current = self.state.trend_state

        if current == TrendState.NEUTRAL:
            if score > self.config.trend_threshold:
                self.state.trend_state = TrendState.UP
                LOGGER.info("Trend NEUTRAL→UP  score=%s", score)
            elif score < -self.config.trend_threshold:
                self.state.trend_state = TrendState.DOWN
                LOGGER.info("Trend NEUTRAL→DOWN  score=%s", score)
        elif current == TrendState.UP:
            if score < self.config.trend_deadzone:
                self.state.trend_state = TrendState.NEUTRAL
                LOGGER.info("Trend UP→NEUTRAL  score=%s  exposure=%s", score, self.state.trend_exposure_xnv)
        elif current == TrendState.DOWN:
            if score > -self.config.trend_deadzone:
                self.state.trend_state = TrendState.NEUTRAL
                LOGGER.info("Trend DOWN→NEUTRAL  score=%s  exposure=%s", score, self.state.trend_exposure_xnv)

    def _trend_offset(self, mid: Decimal) -> Decimal:
        if self.state.trend_state == TrendState.NEUTRAL:
            return Decimal("0")
        sign = Decimal("1") if self.state.trend_state == TrendState.UP else Decimal("-1")
        return sign * self.config.ladder_shift_pct * mid

    def _settle_previous_bite(self, now: float) -> None:
        """Check the last bite order; credit exposure if filled, cancel if still open."""
        if self.state.last_bite_order_id is None:
            return
        prev_id = self.state.last_bite_order_id
        LOGGER.info("Bite settle: checking order %s", prev_id)
        try:
            prev_status = self.client.get_order(prev_id)
            LOGGER.info(
                "Bite settle: id=%s status=%r filled_qty=%s avg_price=%s",
                prev_id, prev_status.status, prev_status.filled_qty, prev_status.avg_price,
            )
            if _is_final(prev_status):
                filled = prev_status.filled_qty or Decimal("0")
                if filled > 0:
                    if self.state.last_bite_side == "sell":
                        self.state.trend_exposure_xnv -= filled
                    else:
                        self.state.trend_exposure_xnv += filled
                    LOGGER.info(
                        "Taker bite %s (%s) confirmed filled=%s  exposure=%s",
                        prev_id, self.state.last_bite_side, filled, self.state.trend_exposure_xnv,
                    )
                else:
                    LOGGER.info(
                        "Taker bite %s final (status=%s) but filled_qty=0 — not crediting exposure",
                        prev_id, prev_status.status,
                    )
                self.state.last_bite_order_id = None
                self.state.last_bite_side = None
            else:
                # Still open — cancel and restart cooldown.
                LOGGER.info(
                    "Taker bite %s status=%s (not final) — cancelling, restarting cooldown",
                    prev_id, prev_status.status,
                )
                try:
                    self.client.cancel_order(prev_id)
                except RestError as cancel_exc:
                    LOGGER.warning("Bite cancel %s failed: %s", prev_id, cancel_exc)
                self.state.last_bite_order_id = None
                self.state.last_bite_side = None
                self.state.last_bite_ts = now
        except RestError as exc:
            LOGGER.warning("Bite settle get_order %s failed: %s — clearing id", prev_id, exc)
            self.state.last_bite_order_id = None
            self.state.last_bite_side = None

    def _maybe_place_taker_bite(
        self,
        best_bid: Decimal,
        best_ask: Decimal,
        now: float,
    ) -> None:
        if now - self.state.last_bite_ts < self.config.taker_bite_interval_sec:
            return

        max_exp = self.config.taker_max_exposure * self.config.base_order_size
        # Signed exposure: negative = net short, positive = net long.
        # Block deepening in the current trend direction; unwinding is always allowed.
        exp = self.state.trend_exposure_xnv
        if self.state.trend_state == TrendState.DOWN and exp <= -max_exp:
            LOGGER.info("Taker bite skipped: short exposure %s >= max %s", -exp, max_exp)
            return
        if self.state.trend_state == TrendState.UP and exp >= max_exp:
            LOGGER.info("Taker bite skipped: long exposure %s >= max %s", exp, max_exp)
            return

        pair_cfg = self.config.pairs[self.config.symbol_usdt]
        jitter = Decimal(str(random.uniform(
            float(1 - self.config.order_size_jitter_pct),
            float(1 + self.config.order_size_jitter_pct),
        )))
        bite_qty = _quantize_qty(
            self.config.taker_bite_size_factor * self.config.base_order_size * jitter,
            pair_cfg.step_size,
        )
        if bite_qty <= 0:
            return

        if self.state.trend_state == TrendState.UP:
            side, price = "buy", _quantize_price(best_ask, pair_cfg.tick_size, side="buy")
        else:
            side, price = "sell", _quantize_price(best_bid, pair_cfg.tick_size, side="sell")

        # Balance gate: ensure actual inventory backs the bite.
        if side == "sell":
            avail_xnv = self._balances.get("XNV", (Decimal("0"), Decimal("0")))[0]
            if avail_xnv < bite_qty:
                LOGGER.info(
                    "Taker bite skipped: insufficient XNV — available=%s needed=%s",
                    avail_xnv, bite_qty,
                )
                return
        else:
            avail_usdt = self._balances.get("USDT", (Decimal("0"), Decimal("0")))[0]
            needed = bite_qty * price
            if avail_usdt < needed:
                LOGGER.info(
                    "Taker bite skipped: insufficient USDT — available=%s needed=%s",
                    avail_usdt, needed,
                )
                return

        if self.config.mode in {"monitor", "dry-run"}:
            LOGGER.info("DRY-RUN taker bite %s %s @ %s", side, bite_qty, price)
            self.state.last_bite_ts = now
            return

        client_id = f"xnv-bite-{uuid.uuid4().hex[:12]}"
        try:
            order_id = self.client.place_limit(
                self.config.symbol_usdt,
                side,
                price,
                bite_qty,
                client_id=client_id,
                strict_validate=False,
            )
            self.state.last_bite_order_id = order_id
            self.state.last_bite_side = side
            self.state.last_bite_ts = now
            LOGGER.info(
                "Taker bite %s qty=%s @ %s  id=%s  exposure=%s (pending)",
                side, bite_qty, price, order_id, self.state.trend_exposure_xnv,
            )
        except RestError as exc:
            LOGGER.warning("Taker bite failed: %s", exc)

    # ── Component 3: 2-leg arb ─────────────────────────────────────────────

    def _record_ratio_snapshot(
        self, usdt_mid: Decimal, xmr_mid: Decimal, now: float
    ) -> None:
        if xmr_mid <= 0:
            return
        self.state.ratio_history.append(PricePoint(ts=now, value=usdt_mid / xmr_mid))
        cap = self.config.arb_ratio_window * 2
        if len(self.state.ratio_history) > cap:
            self.state.ratio_history = self.state.ratio_history[-self.config.arb_ratio_window :]

    def _arb_stats(self) -> tuple[Decimal, Decimal] | None:
        window = self.state.ratio_history[-self.config.arb_ratio_window :]
        min_samples = max(10, self.config.arb_ratio_window // 4)
        if len(window) < min_samples:
            return None
        vals = [float(p.value) for p in window]
        mean = statistics.mean(vals)
        stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
        if stdev == 0:
            return None
        return Decimal(str(mean)), Decimal(str(stdev))

    def _check_and_execute_arb(
        self,
        usdt_bid: Decimal,
        usdt_ask: Decimal,
        usdt_bid_qty: Decimal,
        usdt_ask_qty: Decimal,
        xmr_bid: Decimal,
        xmr_ask: Decimal,
        xmr_bid_qty: Decimal,
        xmr_ask_qty: Decimal,
        now: float,
    ) -> bool:
        stats = self._arb_stats()
        if stats is None:
            return False

        mean, stdev = stats
        usdt_mid = (usdt_bid + usdt_ask) / Decimal("2")
        xmr_mid = (xmr_bid + xmr_ask) / Decimal("2")
        z = (usdt_mid / xmr_mid - mean) / stdev

        total_fees = (
            self.config.pairs[self.config.symbol_usdt].fee_rate
            + self.config.pairs[self.config.symbol_xmr].fee_rate
        )

        if z > self.config.arb_z_threshold:
            # XNV_USDT is high relative to XNV_XMR.
            # Sell XNV_USDT (take usdt_bid) + buy XNV_XMR (pay xmr_ask).
            # Profit: usdt received minus implied USDT cost of XMR spent.
            # implied cost = xmr_ask * mean  (converting XMR to USDT via rolling mean)
            profit_pct = usdt_bid / (xmr_ask * mean) - Decimal("1") - total_fees
            if profit_pct < self.config.arb_min_profit_pct:
                return False
            size = self._arb_size(usdt_bid_qty, xmr_ask_qty)
            if size <= 0:
                return False
            LOGGER.info(
                "Arb A triggered z=%.3f profit_pct=%.4f size=%s",
                float(z), float(profit_pct), size,
            )
            return self._execute_arb("A", size, usdt_bid, xmr_ask, now)

        if z < -self.config.arb_z_threshold:
            # XNV_XMR is high relative to XNV_USDT.
            # Buy XNV_USDT (pay usdt_ask) + sell XNV_XMR (take xmr_bid).
            # Profit: XMR received (valued at mean) minus USDT paid.
            profit_pct = xmr_bid * mean / usdt_ask - Decimal("1") - total_fees
            if profit_pct < self.config.arb_min_profit_pct:
                return False
            size = self._arb_size(usdt_ask_qty, xmr_bid_qty)
            if size <= 0:
                return False
            LOGGER.info(
                "Arb B triggered z=%.3f profit_pct=%.4f size=%s",
                float(z), float(profit_pct), size,
            )
            return self._execute_arb("B", size, usdt_ask, xmr_bid, now)

        return False

    def _arb_size(self, qty1: Decimal, qty2: Decimal) -> Decimal:
        step = self.config.pairs[self.config.symbol_usdt].step_size
        size = min(
            self.config.arb_max_order_size,
            qty1 * self.config.arb_depth_pct,
            qty2 * self.config.arb_depth_pct,
        )
        return max(Decimal("0"), _quantize_qty(size, step))

    def _execute_arb(
        self,
        direction: str,
        size: Decimal,
        usdt_price: Decimal,
        xmr_price: Decimal,
        now: float,
    ) -> bool:
        # Direction A: sell XNV_USDT + buy  XNV_XMR
        # Direction B: buy  XNV_USDT + sell XNV_XMR
        usdt_side = "sell" if direction == "A" else "buy"
        xmr_side = "buy" if direction == "A" else "sell"

        if self.config.mode in {"monitor", "dry-run"}:
            LOGGER.info(
                "DRY-RUN arb %s  usdt_%s @ %s  xmr_%s @ %s  size=%s",
                direction, usdt_side, usdt_price, xmr_side, xmr_price, size,
            )
            self.state.arb_last_executed_ts = now
            return True

        # Pre-flight notional checks
        usdt_pair = self.config.pairs[self.config.symbol_usdt]
        xmr_pair = self.config.pairs[self.config.symbol_xmr]
        if size * usdt_price < usdt_pair.min_notional:
            LOGGER.warning(
                "Arb %s: USDT notional %s below minimum %s — skipping",
                direction, size * usdt_price, usdt_pair.min_notional,
            )
            return False
        if size * xmr_price < xmr_pair.min_notional:
            LOGGER.warning(
                "Arb %s: XMR notional %s below minimum %s — skipping",
                direction, size * xmr_price, xmr_pair.min_notional,
            )
            return False

        # Leg 1: XNV_XMR (less liquid — fail fast here)
        xmr_cid = f"xnv-arb-xmr-{uuid.uuid4().hex[:12]}"
        try:
            xmr_oid = self.client.place_limit(
                self.config.symbol_xmr, xmr_side, xmr_price, size,
                client_id=xmr_cid, strict_validate=False,
            )
            LOGGER.info(
                "Arb %s leg1 XNV_XMR %s qty=%s @ %s → %s",
                direction, xmr_side, size, xmr_price, xmr_oid,
            )
        except RestError as exc:
            LOGGER.warning("Arb %s leg1 XNV_XMR failed (abort): %s", direction, exc)
            return False

        # Leg 2: XNV_USDT (more liquid)
        usdt_cid = f"xnv-arb-usdt-{uuid.uuid4().hex[:12]}"
        try:
            usdt_oid = self.client.place_limit(
                self.config.symbol_usdt, usdt_side, usdt_price, size,
                client_id=usdt_cid, strict_validate=False,
            )
            LOGGER.info(
                "Arb %s leg2 XNV_USDT %s qty=%s @ %s → %s",
                direction, usdt_side, size, usdt_price, usdt_oid,
            )
        except RestError as exc:
            LOGGER.warning(
                "Arb %s leg2 XNV_USDT failed — leg1 already placed, stuck: %s",
                direction, exc,
            )
            self.state.arb_stuck = StuckArb(
                pair=self.config.symbol_usdt, side=usdt_side, qty=size, retry_count=0
            )

        self.state.arb_last_executed_ts = now
        return True

    def _retry_stuck_arb(self) -> None:
        stuck = self.state.arb_stuck
        if stuck is None:
            return
        if stuck.retry_count >= self.config.arb_retry_ticks:
            LOGGER.error(
                "Stuck arb unresolved after %d retries: %s %s qty=%s — manual action required",
                stuck.retry_count, stuck.pair, stuck.side, stuck.qty,
            )
            self.state.arb_stuck = None
            return

        LOGGER.warning(
            "Retrying stuck arb leg (attempt %d/%d): %s %s qty=%s",
            stuck.retry_count + 1, self.config.arb_retry_ticks,
            stuck.pair, stuck.side, stuck.qty,
        )
        pair_cfg = self.config.pairs[stuck.pair]
        try:
            bid, ask = self.client.get_orderbook_top(stuck.pair)
            price = ask if stuck.side == "buy" else bid
            price = _quantize_price(price, pair_cfg.tick_size, side=stuck.side)
            cid = f"xnv-arb-retry-{uuid.uuid4().hex[:12]}"
            oid = self.client.place_limit(
                stuck.pair, stuck.side, price, stuck.qty,
                client_id=cid, strict_validate=False,
            )
            LOGGER.info("Stuck arb resolved: %s %s @ %s → %s", stuck.side, stuck.qty, price, oid)
            self.state.arb_stuck = None
        except RestError as exc:
            LOGGER.warning("Stuck arb retry failed: %s", exc)
            stuck.retry_count += 1

    # ── Inventory skew ─────────────────────────────────────────────────────

    def _inventory_skew(self, pair: str, mid: Decimal) -> Decimal:
        if not self._balances or mid <= 0:
            return Decimal("0")

        xnv_bal = self._balances.get("XNV", (Decimal("0"), Decimal("0")))[0]
        usdt_bal = self._balances.get("USDT", (Decimal("0"), Decimal("0")))[0]
        xmr_bal = self._balances.get("XMR", (Decimal("0"), Decimal("0")))[0]

        usdt_mid = self._current_mids.get(self.config.symbol_usdt, Decimal("0"))
        xmr_mid_c = self._current_mids.get(self.config.symbol_xmr, Decimal("0"))

        if usdt_mid <= 0:
            return Decimal("0")

        implied_xmr_usdt = (usdt_mid / xmr_mid_c) if xmr_mid_c > 0 else Decimal("0")

        total = xnv_bal * usdt_mid + usdt_bal + xmr_bal * implied_xmr_usdt
        if total <= 0:
            return Decimal("0")

        xnv_ratio = xnv_bal * usdt_mid / total
        target = self.config.inventory_target.get("XNV", Decimal("0.5"))
        diff = xnv_ratio - target

        tol = max(self.config.inventory_tolerance_pct, Decimal("0.0001"))
        if abs(diff) <= tol:
            return Decimal("0")

        factor = max(min(diff / tol, Decimal("1")), Decimal("-1"))
        return mid * self.config.inventory_skew_pct * factor

    # ── Helpers ────────────────────────────────────────────────────────────

    def _refresh_balances(self, now: float) -> None:
        if now - self._last_balance_refresh < self.config.balance_refresh_sec:
            return
        try:
            self._balances = self.client.get_balances()
            self._last_balance_refresh = now
        except RestError as exc:
            LOGGER.warning("Balance refresh failed: %s", exc)

    def _log_fill_status(
        self, pair: str, side: str, level: int, order_id: str, context: str = ""
    ) -> Decimal:
        """Fetch and log the fill status of an order that has left the book.

        Returns the confirmed filled quantity (0 if unknown or unfilled).
        """
        try:
            status = self.client.get_order(order_id)
            filled = status.filled_qty
            avg = status.avg_price
            if filled is not None and filled > Decimal("0"):
                LOGGER.info(
                    "[%s] %s L%d FILLED%s: filled=%s avg=%s id=%s",
                    pair, side.upper(), level,
                    f" ({context})" if context else "",
                    filled, avg, order_id,
                )
                return filled
            else:
                LOGGER.info(
                    "[%s] %s L%d %s%s: id=%s",
                    pair, side.upper(), level,
                    status.status.upper() if status.status else "GONE",
                    f" ({context})" if context else "",
                    order_id,
                )
        except RestError as exc:
            LOGGER.debug("Fill check %s: %s", order_id, exc)
        return Decimal("0")

    def _sync_all_order_statuses(self) -> None:
        """Compare tracked orders against live open orders; log and remove any that filled or disappeared."""
        for pair in (self.config.symbol_usdt, self.config.symbol_xmr):
            try:
                live_ids = {o.order_id for o in self.client.list_open_orders(pair)}
            except RestError as exc:
                LOGGER.warning("list_open_orders %s: %s", pair, exc)
                continue
            to_remove: list[tuple[str, int]] = []
            for side, levels in self.state.open_orders.get(pair, {}).items():
                for k, order in list(levels.items()):
                    if order.order_id not in live_ids:
                        to_remove.append((side, k))
                        filled_qty = self._log_fill_status(pair, side, k, order.order_id)
                        self._adjust_exposure_for_ladder_fill(side, filled_qty)
            for side, k in to_remove:
                self.state.open_orders.get(pair, {}).get(side, {}).pop(k, None)

    def _adjust_exposure_for_ladder_fill(self, side: str, filled_qty: Decimal) -> None:
        """Adjust signed exposure when a ladder order fills."""
        if filled_qty <= 0:
            return
        before = self.state.trend_exposure_xnv
        # Buy fills increase exposure (less short / more long).
        # Sell fills decrease exposure (more short / less long).
        if side == "buy":
            self.state.trend_exposure_xnv += filled_qty
        else:
            self.state.trend_exposure_xnv -= filled_qty
        LOGGER.info(
            "Exposure adjusted by ladder %s fill %s: %s → %s",
            side, filled_qty, before, self.state.trend_exposure_xnv,
        )


# ── Module-level helpers ────────────────────────────────────────────────────


def _valid_book(bid: Decimal, ask: Decimal) -> bool:
    return bid > 0 and ask > 0 and ask > bid


def _is_final(status: OrderStatusView) -> bool:
    return status.status.lower() in {
        "filled", "closed", "done", "cancelled", "canceled", "rejected", "expired",
    }


def _quantize_price(price: Decimal, tick: Decimal, *, side: str) -> Decimal:
    if tick <= 0:
        return price
    rounding = ROUND_DOWN if side == "buy" else ROUND_UP
    return (price / tick).to_integral_value(rounding=rounding) * tick


def _quantize_qty(qty: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return qty
    return (qty / step).to_integral_value(rounding=ROUND_DOWN) * step


def describe() -> str:
    return (
        "XNV advanced market maker: dual 5-level ladder on XNV_USDT + XNV_XMR, "
        "trend-following overlay (3m/15m/1h), 2-leg cross-pair arb."
    )
