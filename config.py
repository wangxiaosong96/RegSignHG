from copy import deepcopy
from pathlib import Path
import yaml

DEFAULTS = {
    "dataset": {"name": "", "data_dir": ""},
    "model": {"hidden_dim": 128, "identity_dim": 32, "layers": 1,
              "intents": 4, "intent_dim": 32, "prototypes": 64,
              "top_r": 8, "dropout": 0.2, "mode": "hypergraph"},
    "train": {"folds": 5, "split_seed": 2026, "seeds": [1, 2, 3],
              "val_ratio": 0.1, "epochs": 200, "patience": 20,
              "lr": 0.001, "weight_decay": 0.00001, "grad_clip": 5.0,
              "device": "auto", "threads": 1, "threshold": 0.5},
    "loss": {"node_weight": 0.1, "graph_weight": 0.1, "decor_weight": 0.001,
             "node_tau": 0.2, "graph_tau": 0.2, "node_batch": 512},
    "augmentation": {"edge_drop": 0.1, "feature_mask": 0.1,
                     "negative_sign_flip": 0.5, "negative_graphs": 2},
}

def load_config(path, variant="full"):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping")
    cfg = deepcopy(DEFAULTS)
    for section, values in raw.items():
        if section not in cfg or not isinstance(values, dict):
            raise ValueError(f"Unknown or invalid section: {section}")
        for key, value in values.items():
            if key not in cfg[section]:
                raise ValueError(f"Unknown option: {section}.{key}")
            cfg[section][key] = value
    if not all(cfg["dataset"].values()):
        raise ValueError("dataset.name and dataset.data_dir are required")
    if variant == "no_hyper":
        cfg["model"]["mode"] = "pairwise"
    elif variant == "no_nodecl":
        cfg["loss"]["node_weight"] = 0.0
    elif variant == "no_graphcl":
        cfg["loss"]["graph_weight"] = 0.0
    elif variant == "no_intent":
        cfg["model"]["intents"] = 1
    elif variant == "no_aug":
        cfg["augmentation"]["edge_drop"] = 0.0
        cfg["augmentation"]["feature_mask"] = 0.0
    elif variant != "full":
        raise ValueError(f"Unknown variant: {variant}")
    m, t, a, loss = (cfg[k] for k in ["model", "train", "augmentation", "loss"])
    for key in ["hidden_dim", "identity_dim", "layers", "intents", "intent_dim", "prototypes", "top_r"]:
        if not isinstance(m[key], int) or isinstance(m[key], bool) or m[key] < 1:
            raise ValueError(f"model.{key} must be a positive integer")
    if m["top_r"] > m["prototypes"] or m["mode"] not in ["hypergraph", "pairwise"]:
        raise ValueError("Invalid top_r or model mode")
    for name, value in [("dropout", m["dropout"]), ("edge_drop", a["edge_drop"]),
                        ("feature_mask", a["feature_mask"]), ("negative_sign_flip", a["negative_sign_flip"])]:
        if not 0 <= value < 1:
            raise ValueError(f"{name} must be in [0,1)")
    for key in ["folds", "epochs", "patience", "threads"]:
        if not isinstance(t[key], int) or t[key] < 1:
            raise ValueError(f"train.{key} must be a positive integer")
    if t["folds"] < 2 or not 0 < t["val_ratio"] < 0.5 or not 0 < t["threshold"] < 1:
        raise ValueError("Invalid folds, validation ratio or threshold")
    if not t["seeds"] or len(set(t["seeds"])) != len(t["seeds"]) or not all(isinstance(x, int) for x in t["seeds"]):
        raise ValueError("train.seeds must contain distinct integers")
    if t["lr"] <= 0 or t["weight_decay"] < 0 or t["grad_clip"] <= 0:
        raise ValueError("Invalid optimizer settings")
    if loss["node_tau"] <= 0 or loss["graph_tau"] <= 0 or loss["node_batch"] < 2:
        raise ValueError("Invalid contrastive temperatures or node_batch")
    if any(loss[k] < 0 for k in ["node_weight", "graph_weight", "decor_weight"]):
        raise ValueError("Loss weights must be nonnegative")
    if not isinstance(a["negative_graphs"], int) or a["negative_graphs"] < 1:
        raise ValueError("At least one corrupted negative graph is required")
    cfg["dataset"]["data_dir"] = str(Path(cfg["dataset"]["data_dir"]).resolve())
    cfg["variant"] = variant
    return cfg

