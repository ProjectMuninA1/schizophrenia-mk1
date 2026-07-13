"""Command-line interface.

Examples:
    python -m stockpipe train   --config config.yaml
    python -m stockpipe predict --config config.yaml
    python -m stockpipe retrain --config config.yaml
    python -m stockpipe train   --ticker MSFT --top-n 15
"""

from __future__ import annotations

import argparse

from .config import Config
from .pipeline import predict, retrain, train


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if args.ticker:
        cfg.data.ticker = args.ticker
    if args.start:
        cfg.data.start = args.start
    if args.end:
        cfg.data.end = args.end
    if args.top_n is not None:
        cfg.feature_selection.top_n = args.top_n
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stockpipe", description=__doc__)
    parser.add_argument("command", choices=["train", "predict", "retrain"])
    parser.add_argument("--config", default="config.yaml", help="path to config YAML")
    parser.add_argument("--ticker", help="override ticker symbol")
    parser.add_argument("--start", help="override start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="override end date (YYYY-MM-DD)")
    parser.add_argument("--top-n", type=int, dest="top_n", help="override number of features")
    parser.add_argument("--quiet", action="store_true", help="reduce logging")
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    cfg = _apply_overrides(cfg, args)
    verbose = not args.quiet

    if args.command == "train":
        train(cfg, verbose=verbose)
    elif args.command == "predict":
        predict(cfg, verbose=verbose)
    elif args.command == "retrain":
        retrain(cfg, verbose=verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
