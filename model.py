import math
import torch
from torch import nn
from torch.nn import functional as F

def build_graph(pairs, labels):
    """Only call with training pairs and their observed sign labels."""
    return pairs[labels == 1].T.contiguous(), pairs[labels == 0].T.contiguous()

class FeatureEncoder(nn.Module):
    """Eq. 2: frozen supplied features plus trainable entity embeddings."""
    def __init__(self, nd, ng, dd, gd, cfg):
        super().__init__()
        h, identity = cfg["hidden_dim"], cfg["identity_dim"]
        self.drug_identity, self.gene_identity = nn.Embedding(nd, identity), nn.Embedding(ng, identity)
        self.drug_project, self.gene_project = nn.Linear(dd + identity, h), nn.Linear(gd + identity, h)
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, dx, gx):
        d = self.drug_project(torch.cat([dx, self.drug_identity.weight], -1))
        g = self.gene_project(torch.cat([gx, self.gene_identity.weight], -1))
        return self.dropout(F.relu(d)), self.dropout(F.relu(g))

class RelationLayer(nn.Module):
    """Eqs. 3-8: per-sign symmetric degrees and a gate BEFORE hypergraphs."""
    def __init__(self, h, dropout):
        super().__init__()
        self.gene_messages = nn.ModuleList([nn.Linear(h, h, bias=False) for _ in range(2)])
        self.drug_messages = nn.ModuleList([nn.Linear(h, h, bias=False) for _ in range(2)])
        self.gate = nn.Linear(3 * h, 2)
        self.fuse = nn.Linear(3 * h, h)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def aggregate(dx, gx, edges, d_transform, g_transform):
        drug_ids, gene_ids = edges
        dm, gm = torch.zeros_like(dx), torch.zeros_like(gx)
        if drug_ids.numel() == 0:
            return dm, gm
        dd = torch.bincount(drug_ids, minlength=len(dx)).to(dx.dtype).clamp_min(1)
        gd = torch.bincount(gene_ids, minlength=len(gx)).to(gx.dtype).clamp_min(1)
        weight = (dd[drug_ids] * gd[gene_ids]).rsqrt().unsqueeze(1)
        dm.index_add_(0, drug_ids, g_transform(gx[gene_ids]) * weight)
        gm.index_add_(0, gene_ids, d_transform(dx[drug_ids]) * weight)
        return dm, gm

    def forward(self, dx, gx, pos, neg):
        messages = [self.aggregate(dx, gx, edge, self.drug_messages[s], self.gene_messages[s])
                    for s, edge in enumerate([pos, neg])]
        x = torch.cat([dx, gx])
        positive = torch.cat(messages[0]); negative = torch.cat(messages[1])
        alpha = self.gate(torch.cat([positive, negative, x], -1)).softmax(-1)
        r = self.fuse(torch.cat([x, positive * alpha[:, :1], negative * alpha[:, 1:]], -1))
        return self.dropout(F.relu(r)), alpha

class IntentHypergraph(nn.Module):
    """Eqs. 9-11. Top-r masked full softmax is NOT renormalized."""
    def __init__(self, cfg):
        super().__init__()
        self.project = nn.Linear(cfg["hidden_dim"], cfg["intent_dim"])
        self.prototypes = nn.Parameter(torch.randn(cfg["prototypes"], cfg["intent_dim"]) * 0.02)
        self.raw_weights = nn.Parameter(torch.full((cfg["prototypes"],), math.log(math.expm1(1.0))))
        self.theta = nn.Linear(cfg["intent_dim"], cfg["intent_dim"], bias=False)
        self.top_r = cfg["top_r"]
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, r):
        z = F.relu(self.project(r))
        full = (z @ self.prototypes.T / math.sqrt(z.shape[1])).softmax(-1)
        values, indices = full.topk(self.top_r, dim=1)
        incidence = torch.zeros_like(full).scatter(1, indices, values)
        dv = incidence.sum(1).clamp_min(1e-8).rsqrt().unsqueeze(1)
        de = incidence.sum(0).clamp_min(1e-8).unsqueeze(1)
        hyperedges = (incidence.T @ (dv * z)) / de
        hyperedges = hyperedges * F.softplus(self.raw_weights).unsqueeze(1)
        out = self.theta(dv * (incidence @ hyperedges))
        return self.dropout(F.relu(out)), incidence

class RegSignHG(nn.Module):
    def __init__(self, num_drugs, num_genes, drug_dim, gene_dim, cfg):
        super().__init__()
        self.num_drugs = num_drugs
        self.mode = cfg["mode"]
        self.features = FeatureEncoder(num_drugs, num_genes, drug_dim, gene_dim, cfg)
        self.relations = nn.ModuleList([RelationLayer(cfg["hidden_dim"], cfg["dropout"]) for _ in range(cfg["layers"])])
        if self.mode == "hypergraph":
            self.hypergraphs = nn.ModuleList([nn.ModuleList([IntentHypergraph(cfg) for _ in range(cfg["intents"])]) for _ in range(2)])
            self.fusion = nn.Sequential(nn.Linear(2 * cfg["intents"] * cfg["intent_dim"], cfg["hidden_dim"]), nn.ReLU(), nn.Dropout(cfg["dropout"]))
        self.decoder = nn.Sequential(nn.Linear(4 * cfg["hidden_dim"], cfg["hidden_dim"]), nn.ReLU(),
                                     nn.Dropout(cfg["dropout"]), nn.Linear(cfg["hidden_dim"], 2))

    def encode(self, dx, gx, pos, neg):
        d, g = self.features(dx, gx)
        for layer in self.relations:
            r, gate = layer(d, g, pos, neg)
            d, g = r[:self.num_drugs], r[self.num_drugs:]
        representations, incidences = [[], []], [[], []]
        if self.mode == "hypergraph":
            for sign, channels in enumerate(self.hypergraphs):
                for channel in channels:
                    z, incidence = channel(r)
                    representations[sign].append(z)
                    incidences[sign].append(incidence)
            fused = self.fusion(torch.cat(representations[0] + representations[1], -1))
        else:
            fused = r
        return {"all": fused, "intents": representations, "incidences": incidences, "gate": gate}

    def decode(self, encoded, pairs):
        z = encoded["all"]
        d, g = z[pairs[:, 0]], z[self.num_drugs + pairs[:, 1]]
        return self.decoder(torch.cat([d, g, d * g, (d - g).abs()], -1))

    def forward(self, dx, gx, pos, neg, pairs):
        encoded = self.encode(dx, gx, pos, neg)
        return self.decode(encoded, pairs), encoded

