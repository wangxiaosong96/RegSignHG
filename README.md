# Processed data contract

Raw data is not included. The scripts audit processed inputs; they do not
silently invent DGIdb, DrugBank or LINCS evidence.

For each dataset directory supply:

```csv
drug,gene,sign
CANONICAL_DRUG_A,HGNC_SYMBOL_A,-1
CANONICAL_DRUG_B,HGNC_SYMBOL_B,1
```

The example names above are schema examples, not biological evidence.
Each canonical pair must occur once with exactly one resolved sign.
Remove or resolve conflicts using a documented preprocessing policy before
splitting. Missing pairs are unlabeled, never confirmed inhibitory interactions.

`provenance.json` must record real values using the following fields:

| Field | Required meaning |
|---|---|
| dataset_name | Matches config: DGIdb, DrugBank or LINCS_L1000 |
| source_version | Actual release, accession/version and access date |
| source_files | Original source filenames or stable source identifiers |
| processing_description | ID mapping, sign mapping, duplicate/conflict filtering; include LINCS aggregation rules where applicable |
| synthetic | false for real data |

Include additional reproducibility fields such as raw file SHA256 values,
mapping dictionaries, raw/filtered counts and executable preprocessing scripts.
Writing metadata alone does not establish genuine source provenance.

Each NPZ contains:

- `ids`: one-dimensional NumPy Unicode array, matching actual feature rows;
- `features`: finite float matrix, one row per ID;
- `metadata`: a scalar JSON string, never a pickle object.

The metadata needs `encoder`, `checkpoint`, and `label_independent: true`.
The expected encoder names are `Uni-Mol` for drugs and `ESM-2` for genes.
Add checkpoint SHA256, pretrained package version, pooling and extraction
settings. A declared encoder name does not make an unrelated DGI embedding
a pretrained representation.

If genuine NPY files already exist, prepare an ID CSV with `id` in exactly
their original row order and an actual feature provenance JSON, then run:

```bash
python scripts/import_features.py --array drug_features.npy --ids drug_feature_rows.csv --metadata drug_feature_metadata.json --output data/DGIdb/drug_features.npz
```

Do not generate the row-ID file by guessing a sorted list if original alignment
is unavailable. Recover it from the feature-generation records.

Frozen pretrained features may be shared across edge folds because they are
computed without interaction labels. Graph-derived DGI features cannot be
relabelled in this format as label-independent; they need a separate fold-wise
pipeline and a matching manuscript description.

LINCS labels describe perturbation-induced transcriptional responses; database
target-action labels have different semantics. Keep dataset evaluations
separate. Record measured landmark versus inferred genes, treatment context,
QC and exclusion rules. This package does not supply missing LINCS raw data.

