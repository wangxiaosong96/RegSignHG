# Manuscript modules and implementation choices

This table maps the supplied Bioinformatics proof to the NEW revision code.
It is not evidence that this implementation produced the submitted results.

| Manuscript component | Formula | Code |
|---|---|---|
| Frozen molecular/protein features plus learned IDs | Eq. 2 | `regsignhg/model.py::FeatureEncoder` |
| Positive/negative observed training neighborhoods | Eqs. 3-4 | `regsignhg/model.py::build_graph` |
| Symmetric degree-normalized signed messages | Eqs. 5-6 | `regsignhg/model.py::RelationLayer.aggregate` |
| Early two-sign gate and concatenation | Eqs. 7-8 | `regsignhg/model.py::RelationLayer.forward` |
| Per-sign, per-intent projections and prototypes | Eq. 9 | `regsignhg/model.py::IntentHypergraph.forward` |
| Top-r retained assignment values | Text below Eq. 9 | `IntentHypergraph.forward` |
| Unweighted incidence node/hyperedge degrees | Eq. 10 | `IntentHypergraph.forward` |
| Learnable weighted hypergraph propagation | Eq. 11 | `IntentHypergraph.forward` |
| Concatenation of sign and intent outputs | Eq. 12 | `regsignhg/model.py::RegSignHG.encode` |
| Node view contrast | Eq. 13 | `regsignhg/losses.py::node_contrastive` |
| Explicit graph readout | Section 2.4.3 | `regsignhg/losses.py::graph_readout` |
| Corrected negative graph contrast | Eq. 14 correction | `regsignhg/losses.py::graph_contrastive` |
| Re-encoded corrupted graph negatives | Section 2.4.3 | `regsignhg/losses.py::corrupted_view` |
| Four-part pair representation and two-class output | Eq. 15 | `regsignhg/model.py::RegSignHG.decode` |
| Activation/inhibition supervised CE | Eq. 16 | `regsignhg/engine.py::train_one` |
| Channel decorrelation | Eq. 17 | `regsignhg/losses.py::decorrelation` |
| Combined losses and L2 weight decay | Section 2.5 | `train_one` and Adam configuration |
| Train-only graph with inner validation | Section 2.6 | `make_splits`, `train_one` |
| AUC, AUPR, positive-class Binary-F1 | Section 2.6 | `regsignhg/metrics.py::compute_metrics` |

## Choices that must be reported

1. `RegSignHG` uses an early gate, matching the submitted equations. Earlier
   CSIH bundles used late adaptive fusion and are not interchangeable.
2. The default relation encoder has one layer; extra layers are configurable.
   Hidden size, channels, prototypes, temperatures and loss weights are revision
   examples, not the recovered original hyperparameters.
3. Eq. 9 is implemented as softmax across all prototypes, followed by zeroing
   values outside top-r, with no re-normalization. Incidences are stored densely.
4. Hyperedge weights are positive via softplus; incidence degrees are unweighted
   as written in Eq. 10. Unused hyperedges are safely clamped before division.
5. Graph readout is arithmetic mean over all encoded nodes followed by L2
   normalization. No biological annotation is used in readout.
6. Node contrast uses a sampled batch of training-incident nodes, symmetric
   cross-entropy in both view directions and cosine similarity. It does not
   materialize an all-node contrastive matrix on genuine datasets.
7. Graph contrast is one-anchor cross-entropy against the other positive view
   and the configured number of explicitly re-encoded corrupted graphs.
   Negative graphs shuffle input features within each node type and independently
   flip selected training-edge signs. Only negative graphs permit sign changes.
8. Node features are masked by fixed probabilities; training edges are dropped
   at fixed probabilities without flipping positive-view signs. There is no
   unimplemented adaptive augmentation claim.
9. Decorrelation uses column-normalized propagated intent representations and
   the ordered cross-channel squared Frobenius sum, averaged across two signs.
   This resolves an underspecified normalization in the original text and
   must be documented alongside the selected coefficient.
10. Decoder classes are ordered `[activation, inhibition]`. Metric label 1
    means activation and uses softmax column zero. The pair vector is
    `[drug, gene, drug*gene, abs(drug-gene)]`, with input width `4*hidden_dim`.
11. Adam weight decay is the explicitly configured regularization; loss scalars
    are means for supervised and contrastive CE. The manuscript's displayed
    sums and regularization coefficient should be reconciled with these scales.
12. Validation AUC is the only early-stopping criterion. Test pairs do not
    enter any message graph, augmentation, loss or checkpoint selection.
13. All entity features are available in edge CV; this is a transductive edge
    experiment, not an inductive entity experiment. Cold start requires a
    separately implemented protocol and ID-embedding handling.
14. Training queries are observed training edges, consistent with the submitted
    train-graph protocol. This release does not claim leave-one-query-out
    masking of training edges; validation and test edges are strictly excluded.

