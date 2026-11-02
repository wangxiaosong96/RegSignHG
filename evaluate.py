import argparse
import json
from regsignhg.engine import evaluate_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Reload a frozen validation-selected checkpoint and evaluate its saved test split")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()
    metrics, _, _ = evaluate_checkpoint(args.checkpoint, args.data_dir, args.device, args.allow_synthetic)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()

