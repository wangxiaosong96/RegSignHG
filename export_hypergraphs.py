from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import torch
from regsignhg.engine import restore

@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Export actual learned incidence memberships without assigning biological intent names")
    p.add_argument("--checkpoint", required=True); p.add_argument("--output", required=True)
    p.add_argument("--data-dir"); p.add_argument("--allow-synthetic", action="store_true")
    a = p.parse_args()
    ck, data, model, (dx, gx, pos, neg, pairs) = restore(a.checkpoint, a.data_dir, "cpu", a.allow_synthetic)
    if model.mode != "hypergraph": raise ValueError("Pairwise ablation has no hypergraphs")
    model.eval(); encoded = model.encode(dx, gx, pos, neg)
    ids, types = data.drugs + data.genes, ["drug"] * len(data.drugs) + ["gene"] * len(data.genes)
    with open(a.output, "x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["sign", "intent", "prototype", "node_type", "entity_id", "incidence_weight"])
        for s, channels in zip(["activation", "inhibition"], encoded["incidences"]):
            for channel, incidence in enumerate(channels, 1):
                for node, edge in (incidence > 0).nonzero().cpu().tolist():
                    writer.writerow([s, channel, edge, types[node], ids[node], float(incidence[node, edge])])

if __name__ == "__main__": main()

