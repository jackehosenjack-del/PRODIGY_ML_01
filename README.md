# PRODIGY_ML_01

Task 01 — Linear Regression model to predict house prices.

## Model history

### v1 — Initial baseline

The original model used three features:

- `GrLivArea`
- `BedroomAbvGr`
- `FullBath`

### v2 — Reproducible evaluation

Added baseline comparison, holdout metrics, and 5-fold cross-validation.

Reference metrics:

- CV RMSE: **52,340.54**
- CV R²: **0.5548**

### v3 — Quality_Size_Basics

Controlled feature selection identified `Quality_Size_Basics` as the strongest candidate:

- `GrLivArea`
- `BedroomAbvGr`
- `FullBath`
- `OverallQual`
- `TotalBsmtSF`
- `GarageCars`

Validation results:

| Metric | v2 3-feature baseline | Model v3 |
|---|---:|---:|
| CV RMSE | 52,340.54 | **40,525.05** |
| CV R² | 0.5548 | **0.7305** |
| Locked holdout RMSE | 52,975.72 | **39,240.84** |
| Locked holdout MAE | 35,788.06 | **25,290.76** |
| Locked holdout R² | 0.6341 | **0.7992** |

The locked holdout RMSE is **25.93% lower** than the v2 baseline.

> Methodological note: feature selection was performed before the locked holdout confirmation using the available dataset. The locked holdout is therefore a reproducible confirmation check, not a completely untouched external test set.

## Reproducibility

- Python: **3.11**
- Random state: **42**
- Holdout test size: **20%**
- Cross-validation: **5-fold KFold, shuffled, random state 42**
- Dependencies are pinned in `requirements.txt`.
- `model_v3.py` deterministically trains the Model v3 feature set and creates a `joblib` model artifact plus metadata.
- CI stores the generated Model v3 artifact for every successful run.

The metadata records the feature set, target, evaluation metrics, Python/scikit-learn versions, source commit, row count, and training timestamp.

## CI quality gate

Every push to `main` and every pull request runs:

`Tests → Evaluation v2 → Evaluation v3 → Locked Validation → Model v3 Artifact → Training Pipeline`

A failure in any stage fails CI.
