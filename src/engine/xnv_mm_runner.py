"""Runner for the XNV advanced market maker."""

from __future__ import annotations

import logging
import time
from decimal import Decimal
from pathlib import Path

from engine.rest_client_factory import build_exchange_client
from strategies.xnv_market_maker import (
    PairConfig,
    XnvMarketMakerConfig,
    XnvMarketMakerStrategy,
)

LOGGER = logging.getLogger(__name__)


def build_strategy(config: dict, state_path: Path) -> XnvMarketMakerStrategy:
    symbol_usdt = config.get("symbol_usdt", "XNV_USDT")
    symbol_xmr = config.get("symbol_xmr", "XNV_XMR")

    pairs_raw = config.get("pairs", {})

    def _pair_cfg(symbol: str) -> PairConfig:
        raw = pairs_raw.get(symbol, {})
        return PairConfig(
            tick_size=Decimal(str(raw.get("tick_size", "0"))),
            step_size=Decimal(str(raw.get("step_size", "0"))),
            fee_rate=Decimal(str(raw.get("fee_rate", "0.001"))),
            min_notional=Decimal(str(raw.get("min_notional", "0.001"))),
        )

    inv_raw = config.get("inventory_target", {})
    inventory_target = {
        k: Decimal(str(v)) for k, v in inv_raw.items()
    }
    if not inventory_target:
        inventory_target = {
            "XNV": Decimal("0.60"),
            "USDT": Decimal("0.30"),
            "XMR": Decimal("0.10"),
        }

    strategy_config = XnvMarketMakerConfig(
        symbol_usdt=symbol_usdt,
        symbol_xmr=symbol_xmr,
        # Ladder
        base_order_size=Decimal(str(config.get("base_order_size", "10"))),
        num_levels=int(config.get("num_levels", 5)),
        level_0_offset_pct=Decimal(str(config.get("level_0_offset_pct", "0.003"))),
        level_spacing_pct=Decimal(str(config.get("level_spacing_pct", "0.50"))),
        size_step_factor=Decimal(str(config.get("size_step_factor", "0.5"))),
        max_order_age_sec=float(config.get("max_order_age_sec", 120)),
        level_reprice_threshold_pct=Decimal(
            str(config.get("level_reprice_threshold_pct", "0.004"))
        ),
        # Per-pair precision
        pairs={symbol_usdt: _pair_cfg(symbol_usdt), symbol_xmr: _pair_cfg(symbol_xmr)},
        # Inventory
        inventory_target=inventory_target,
        inventory_tolerance_pct=Decimal(
            str(config.get("inventory_tolerance_pct", "0.05"))
        ),
        inventory_skew_pct=Decimal(str(config.get("inventory_skew_pct", "0.20"))),
        # Trend
        trend_threshold=Decimal(str(config.get("trend_threshold", "0.003"))),
        trend_deadzone=Decimal(str(config.get("trend_deadzone", "0.001"))),
        ladder_shift_pct=Decimal(str(config.get("ladder_shift_pct", "0.30"))),
        taker_bite_size_factor=Decimal(
            str(config.get("taker_bite_size_factor", "0.30"))
        ),
        taker_bite_interval_sec=float(config.get("taker_bite_interval_sec", 300)),
        taker_max_exposure=Decimal(str(config.get("taker_max_exposure", "5.0"))),
        xmr_qty_fraction=Decimal(str(config.get("xmr_qty_fraction", "0.60"))),
        order_size_jitter_pct=Decimal(str(config.get("order_size_jitter_pct", "0.07"))),
        # Arb
        arb_z_threshold=Decimal(str(config.get("arb_z_threshold", "2.0"))),
        arb_min_profit_pct=Decimal(str(config.get("arb_min_profit_pct", "0.002"))),
        arb_max_order_size=Decimal(str(config.get("arb_max_order_size", "50"))),
        arb_depth_pct=Decimal(str(config.get("arb_depth_pct", "0.30"))),
        arb_cooldown_sec=float(config.get("arb_cooldown_sec", 60)),
        arb_ratio_window=int(config.get("arb_ratio_window", 100)),
        arb_retry_ticks=int(config.get("arb_retry_ticks", 5)),
        # Operational
        poll_interval_sec=float(config.get("poll_interval_sec", 15)),
        balance_refresh_sec=float(config.get("balance_refresh_sec", 30)),
        mode=config.get("mode", "live"),
        fills_csv=config.get("fills_csv"),
        cost_basis_scale_range_pct=Decimal(str(config.get("cost_basis_scale_range_pct", "0.20"))),
        cost_basis_min_sell_factor=Decimal(str(config.get("cost_basis_min_sell_factor", "0.20"))),
        cost_basis_max_sell_factor=Decimal(str(config.get("cost_basis_max_sell_factor", "2.0"))),
    )

    exchange = build_exchange_client(config)
    return XnvMarketMakerStrategy(exchange, strategy_config, state_path=state_path)


def run_xnv_mm(config: dict, state_path: Path) -> None:
    strategy = build_strategy(config, state_path)
    strategy.load_state()
    strategy.cancel_all_on_startup()
    strategy.seed_price_history()

    LOGGER.info(
        "XNV market maker running. pairs=%s+%s levels=%d poll=%ss mode=%s  "
        "exposure=%s  avg_cost=%s  held=%s",
        strategy.config.symbol_usdt, strategy.config.symbol_xmr,
        strategy.config.num_levels, strategy.config.poll_interval_sec,
        strategy.config.mode, strategy.state.trend_exposure_xnv,
        strategy.state.avg_cost_xnv or "unset",
        strategy.state.held_xnv_qty,
    )

    try:
        while True:
            strategy.poll_once()
            time.sleep(strategy.config.poll_interval_sec)
    except KeyboardInterrupt:
        pass
    finally:
        LOGGER.info("Shutting down — cancelling all open orders...")
        strategy.cancel_all_on_startup()
        strategy.close()
