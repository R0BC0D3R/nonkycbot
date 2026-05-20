"""Cancel all open orders for a symbol using the bot's state file."""

import sys
import json
import argparse
from pathlib import Path

import yaml

sys.path.insert(0, '/root/nonkycbot/src')
from engine.rest_client_factory import build_exchange_client


def cancel_all(symbol: str, config_file: str) -> None:
    with open(config_file) as f:
        config = yaml.safe_load(f)

    state_path = config.get('state_path', f'state/{symbol.lower()}_grid_state.json')
    state_file = Path(state_path)

    if not state_file.exists():
        print(f"State file not found: {state_path}")
        print("Bot may not have run yet, or state was already cleared.")
        return

    state = json.loads(state_file.read_text(encoding='utf-8'))
    order_ids = list(state.get('open_orders', {}).keys())

    if not order_ids:
        print(f"No tracked orders in state file: {state_path}")
        return

    print(f"Found {len(order_ids)} tracked orders. Cancelling...")
    client = build_exchange_client(config)

    cancelled = 0
    for order_id in order_ids:
        try:
            ok = client.cancel_order(order_id)
            print(f"  {order_id}: {'OK' if ok else 'FAILED (no confirmation)'}")
            if ok:
                cancelled += 1
        except Exception as exc:
            print(f"  {order_id}: ERROR - {exc}")

    print(f"\nCancelled {cancelled}/{len(order_ids)} orders.")

    # Remove cancelled orders from state so the bot starts clean
    for order_id in order_ids:
        state['open_orders'].pop(order_id, None)
    state_file.write_text(json.dumps(state, indent=2), encoding='utf-8')
    print(f"State file updated: {state_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cancel all tracked grid orders')
    parser.add_argument('symbol', help='Trading pair e.g. XNV_XMR')
    parser.add_argument('--config', default='conf_xnv_xmr_grid.yml', help='Config file')
    args = parser.parse_args()
    cancel_all(args.symbol, args.config)
