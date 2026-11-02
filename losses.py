import torch
from torch.nn import functional as F

def node_contrastive(z1, z2, temperature, max_nodes, node_ids):
    if len(node_ids) > max_nodes:
        node_ids = node_ids[torch.randperm(len(node_ids), device=node_ids.device)[:max_nodes]]
    a, b = F.normalize(z1[node_ids], dim=-1), F.normalize(z2[node_ids], dim=-1)
    scores = a @ b.T / temperature
    target = torch.arange(len(a), device=a.device)
    return (F.cross_entropy(scores, target) + F.cross_entropy(scores.T, target)) / 2

def graph_readout(z):
    """Explicit revision choice: arithmetic mean across encoded nodes."""
    return F.normalize(z.mean(0, keepdim=True), dim=-1)

def graph_contrastive(z1, z2, corrupted_graph_embeddings, temperature):
    """Corrected Eq. 14; corrupted graphs must be RE-ENCODED."""
    anchor, positive = graph_readout(z1), graph_readout(z2)
    negatives = torch.cat([graph_readout(x) for x in corrupted_graph_embeddings], 0)
    pos = (anchor * positive).sum(-1, keepdim=True)
    neg = anchor @ negatives.T
    scores = torch.cat([pos, neg], 1) / temperature
    return F.cross_entropy(scores, torch.zeros(1, dtype=torch.long, device=z1.device))

def decorrelation(sign_intents, reference):
    """Column-normalized channels; sum ordered cross-channel terms per sign."""
    total = reference.sum() * 0
    for channels in sign_intents:
        normalized = [F.normalize(x, dim=0) for x in channels]
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                total = total + 2 * (normalized[i].T @ normalized[j]).square().sum()
    return total / 2

def dropout_edges(edges, probability):
    if probability == 0 or edges.shape[1] == 0:
        return edges
    keep = torch.rand(edges.shape[1], device=edges.device) >= probability
    return edges[:, keep]

def mask_features(x, probability):
    return x if probability == 0 else x * (torch.rand_like(x) >= probability)

def positive_view(model, dx, gx, pos, neg, aug):
    return model.encode(mask_features(dx, aug["feature_mask"]), mask_features(gx, aug["feature_mask"]),
                        dropout_edges(pos, aug["edge_drop"]), dropout_edges(neg, aug["edge_drop"]))

def corrupted_view(model, dx, gx, pos, neg, aug):
    """Shuffle INPUT features within node type; flip signs only in negative views."""
    dp = torch.randperm(len(dx), device=dx.device)
    gp = torch.randperm(len(gx), device=gx.device)
    ps = torch.rand(pos.shape[1], device=pos.device) < aug["negative_sign_flip"]
    ns = torch.rand(neg.shape[1], device=neg.device) < aug["negative_sign_flip"]
    bad_pos = torch.cat([pos[:, ~ps], neg[:, ns]], 1)
    bad_neg = torch.cat([neg[:, ~ns], pos[:, ps]], 1)
    return model.encode(dx[dp], gx[gp], bad_pos, bad_neg)

