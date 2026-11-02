"""Public training entry point. Defaults are revision examples, not recovered paper settings."""
import argparse
import json
from regsignhg.config import load_config
from regsignhg.engine import run_training

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, help="New output directory; existing directories are never overwritten")
    parser.add_argument("--variant", default="full", choices=["full", "no_hyper", "no_nodecl", "no_graphcl", "no_intent", "no_aug"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.variant)
    if args.device: cfg["train"]["device"] = args.device
    summary = run_training(cfg, args.output, args.allow_synthetic)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

