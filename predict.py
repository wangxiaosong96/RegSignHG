import argparse
import csv
import torch
from regsignhg.engine import restore, predict

def main():
    parser = argparse.ArgumentParser(description="Predict CONDITIONAL signs for pairs of entities present in a checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pairs", required=True, help="CSV with drug,gene; query signs are not required")
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-dir")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()
    ck, data, model, (dx, gx, pos, neg, _) = restore(args.checkpoint, args.data_dir, args.device, args.allow_synthetic)
    dmap = {x: i for i, x in enumerate(data.drugs)}; gmap = {x: i for i, x in enumerate(data.genes)}
    with open(args.pairs, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"drug", "gene"}.issubset(reader.fieldnames or []): raise ValueError("Queries require drug,gene")
        rows = [(r["drug"].strip(), r["gene"].strip()) for r in reader]
    if not rows: raise ValueError("Query file is empty")
    for d, g in rows:
        if d not in dmap or g not in gmap: raise ValueError(f"Unknown entity: {d}/{g}; this release does not claim cold-start support")
    pairs = torch.tensor([(dmap[d], gmap[g]) for d, g in rows], dtype=torch.long, device=dx.device)
    probabilities = predict(model, dx, gx, pos, neg, pairs)
    with open(args.output, "x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["drug", "gene", "p_activation", "p_inhibition"])
        for (d, g), p in zip(rows, probabilities): writer.writerow([d, g, float(p), 1 - float(p)])

if __name__ == "__main__":
    main()
