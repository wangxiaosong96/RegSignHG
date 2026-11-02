from pathlib import Path
from importlib import metadata
import csv
import json
import platform
import random
import subprocess
import sys
import numpy as np
import torch
from torch.nn import functional as F
from .data import load_dataset, make_splits, digest
from .model import RegSignHG, build_graph
from .losses import node_contrastive, graph_contrastive, decorrelation, positive_view, corrupted_view
from .metrics import compute_metrics

def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

def select_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; explicitly select cpu")
    return torch.device(name)

def environment():
    packages = {d.metadata["Name"]: d.version for d in metadata.distributions() if d.metadata.get("Name")}
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.DEVNULL)
    except (subprocess.SubprocessError, OSError):
        freeze = "unavailable"
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip())
    except (subprocess.SubprocessError, OSError):
        commit, dirty = None, None
    return {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
            "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "packages": packages, "pip_freeze": freeze, "git_commit": commit, "git_dirty": dirty,
            "source_sha256": {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*.py")) if "__pycache__" not in p.parts}}

def tensors(data, device):
    return (torch.as_tensor(data.drug_features, device=device), torch.as_tensor(data.gene_features, device=device),
            torch.as_tensor(data.pairs, device=device), torch.as_tensor(data.labels, device=device))

@torch.no_grad()
def predict(model, dx, gx, pos, neg, pairs):
    model.eval()
    logits, _ = model(dx, gx, pos, neg, pairs)
    # Eq. 15 class order is [+1, -1], despite binary metric labels being 1/0.
    return logits.softmax(-1)[:, 0].cpu().numpy()

def save_predictions(path, data, indices, probabilities):
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row_index", "drug", "gene", "sign", "p_activation", "p_inhibition"])
        for i, probability in zip(indices, probabilities):
            d, g = data.pairs[i]
            writer.writerow([i, data.drugs[d], data.genes[g], 1 if data.labels[i] else -1,
                             format(float(probability), ".10g"), format(1 - float(probability), ".10g")])

def train_one(data, split, cfg, training_seed, folder, device):
    folder.mkdir(parents=True, exist_ok=False)
    # The split is fixed independently of the initialization seed.
    effective_seed = training_seed * 1000 + split["fold"]
    set_seed(effective_seed)
    dx, gx, pairs, labels = tensors(data, device)
    tr = torch.tensor(split["train"], dtype=torch.long, device=device)
    va = torch.tensor(split["validation"], dtype=torch.long, device=device)
    te = torch.tensor(split["test"], dtype=torch.long, device=device)
    pos, neg = build_graph(pairs[tr], labels[tr])
    graph_node_ids = torch.unique(torch.cat([pairs[tr, 0], pairs[tr, 1] + len(data.drugs)]))
    model = RegSignHG(len(data.drugs), len(data.genes), dx.shape[1], gx.shape[1], cfg["model"]).to(device)
    t, weights, aug = cfg["train"], cfg["loss"], cfg["augmentation"]
    optimizer = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    best_auc, best_epoch, waiting = -float("inf"), None, 0
    best_state = None
    with open(folder / "history.csv", "w", newline="", encoding="utf-8") as stream:
        logger = csv.DictWriter(stream, fieldnames=["epoch", "loss", "supervised", "node_cl", "graph_cl", "decorrelation", "validation_AUC"])
        logger.writeheader()
        for epoch in range(1, t["epochs"] + 1):
            model.train()
            logits, original = model(dx, gx, pos, neg, pairs[tr])
            supervised = F.cross_entropy(logits, 1 - labels[tr])
            zero = supervised.new_zeros(())
            node, graph, decor = zero, zero, zero
            if weights["node_weight"] or weights["graph_weight"]:
                view1, view2 = positive_view(model, dx, gx, pos, neg, aug), positive_view(model, dx, gx, pos, neg, aug)
                if weights["node_weight"]:
                    node = node_contrastive(view1["all"], view2["all"], weights["node_tau"], weights["node_batch"], graph_node_ids)
                if weights["graph_weight"]:
                    negative = [corrupted_view(model, dx, gx, pos, neg, aug)["all"] for _ in range(aug["negative_graphs"])]
                    graph = graph_contrastive(view1["all"], view2["all"], negative, weights["graph_tau"])
            if weights["decor_weight"] and cfg["model"]["mode"] == "hypergraph":
                decor = decorrelation(original["intents"], original["all"])
            loss = supervised + weights["node_weight"] * node + weights["graph_weight"] * graph + weights["decor_weight"] * decor
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at seed {training_seed}, fold {split['fold']}, epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            # Test labels and test predictions are never used in this loop.
            val_prob = predict(model, dx, gx, pos, neg, pairs[va])
            validation = compute_metrics(data.labels[split["validation"]], val_prob, t["threshold"])
            logger.writerow({"epoch": epoch, "loss": float(loss.detach()), "supervised": float(supervised.detach()),
                             "node_cl": float(node.detach()), "graph_cl": float(graph.detach()),
                             "decorrelation": float(decor.detach()), "validation_AUC": validation["AUC"]})
            stream.flush()
            if validation["AUC"] > best_auc:
                best_auc, best_epoch, waiting = validation["AUC"], epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                waiting += 1
            if waiting >= t["patience"]:
                break
    if best_state is None:
        raise RuntimeError("No valid validation checkpoint")
    model.load_state_dict(best_state)
    checkpoint = {"implementation": "RegSignHG revision 0.1.0", "config": cfg,
                  "state_dict": best_state, "drugs": data.drugs, "genes": data.genes,
                  "drug_dim": dx.shape[1], "gene_dim": gx.shape[1], "split": split,
                  "training_seed": training_seed, "effective_seed": effective_seed,
                  "best_epoch": best_epoch, "best_validation_auc": best_auc,
                  "input_files": data.manifest["files"]}
    torch.save(checkpoint, folder / "best.pt")
    write_json(folder / "split.json", split)
    test_prob = predict(model, dx, gx, pos, neg, pairs[te])
    val_prob = predict(model, dx, gx, pos, neg, pairs[va])
    metrics = compute_metrics(data.labels[split["test"]], test_prob, t["threshold"])
    save_predictions(folder / "test_predictions.csv", data, split["test"], test_prob)
    save_predictions(folder / "validation_predictions.csv", data, split["validation"], val_prob)
    record = {"seed": training_seed, "fold": split["fold"], "best_epoch": best_epoch,
              "validation_AUC": best_auc, **metrics}
    write_json(folder / "metrics.json", record)
    print(f"seed={training_seed} fold={split['fold']} epoch={best_epoch} AUC={metrics['AUC']:.6f}", flush=True)
    return record

def summarize(records):
    metrics = ["AUC", "AUPR", "Binary-F1"]
    groups = {seed: [r for r in records if r["seed"] == seed] for seed in sorted({r["seed"] for r in records})}
    per_seed = []
    for seed, rows in groups.items():
        per_seed.append({"seed": seed, **{k: {"mean": float(np.mean([r[k] for r in rows])),
                                                   "fold_sample_std": float(np.std([r[k] for r in rows], ddof=1))} for k in metrics}})
    across = {}
    for k in metrics:
        values = [row[k]["mean"] for row in per_seed]
        across[k] = {"mean": float(np.mean(values)), "sample_std_of_seed_means": float(np.std(values, ddof=1)) if len(values) > 1 else None}
    return {"note": "Descriptive statistics only. Folds and seeds are not independent datasets; no significance claim is computed.",
            "run_count": len(records), "per_seed": per_seed, "across_seed_means": across}

def run_training(cfg, output, allow_synthetic=False):
    torch.set_num_threads(cfg["train"]["threads"])
    device = select_device(cfg["train"]["device"])
    data = load_dataset(cfg["dataset"]["data_dir"], allow_synthetic)
    if cfg["dataset"]["name"] != data.manifest["provenance"]["dataset_name"]:
        raise ValueError("Config dataset.name does not match provenance.json")
    splits = make_splits(data.labels, cfg["train"]["folds"], cfg["train"]["split_seed"], cfg["train"]["val_ratio"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "resolved_config.json", cfg)
    write_json(output / "environment.json", environment())
    write_json(output / "data_audit.json", data.manifest)
    write_json(output / "all_splits.json", splits)
    records = []
    if data.manifest["provenance"]["synthetic"]:
        print("SYNTHETIC SMOKE TEST ONLY: no paper or biomedical performance claims", flush=True)
    for seed in cfg["train"]["seeds"]:
        for split in splits:
            folder = output / f"seed_{seed}" / f"fold_{split['fold']}"
            records.append(train_one(data, split, cfg, seed, folder, device))
    with open(output / "runs.csv", "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    summary = summarize(records)
    write_json(output / "summary.json", summary)
    return summary

def restore(checkpoint_path, data_dir=None, device_name="cpu", allow_synthetic=False):
    device = select_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    data = load_dataset(data_dir or cfg["dataset"]["data_dir"], allow_synthetic)
    for name, original in checkpoint["input_files"].items():
        if data.manifest["files"][name]["sha256"] != original["sha256"]:
            raise ValueError(f"Input changed since training: {name}")
    if data.drugs != checkpoint["drugs"] or data.genes != checkpoint["genes"]:
        raise ValueError("Entity mapping differs from checkpoint")
    torch.set_num_threads(cfg["train"]["threads"])
    model = RegSignHG(len(data.drugs), len(data.genes), checkpoint["drug_dim"], checkpoint["gene_dim"], cfg["model"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    dx, gx, pairs, labels = tensors(data, device)
    tr = torch.tensor(checkpoint["split"]["train"], dtype=torch.long, device=device)
    pos, neg = build_graph(pairs[tr], labels[tr])
    return checkpoint, data, model, (dx, gx, pos, neg, pairs)

def evaluate_checkpoint(checkpoint_path, data_dir=None, device="cpu", allow_synthetic=False):
    ck, data, model, (dx, gx, pos, neg, pairs) = restore(checkpoint_path, data_dir, device, allow_synthetic)
    ids = ck["split"]["test"]
    probabilities = predict(model, dx, gx, pos, neg, pairs[ids])
    return compute_metrics(data.labels[ids], probabilities, ck["config"]["train"]["threshold"]), probabilities, ids

