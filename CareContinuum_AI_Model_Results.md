# CareContinuum AI — Model Results Record

## Project objective
Predict whether a patient will be readmitted after discharge using the UCI Diabetes 130-US Hospitals dataset (1999–2008).

Binary target: `NO` → 0 (Not Readmitted); `>30` and `<30` → 1 (Readmitted).

This is an educational prediction prototype, not a clinically validated decision system.

## Dataset preparation
- Original dataset after adding binary target: 101,766 rows × 51 columns.
- Excluded 2,423 encounters with discharge disposition IDs `[11, 13, 14, 19, 20]`.
- Dropped `weight`, `citoglipton`, and `examide`.
- Replaced `?` with missing values; categorical missing values were filled with `Unknown`.
- Final modelling dataset: 99,343 rows × 48 columns.
- Features: 42; target: `readmission_binary`.
- Target proportions: 52.9% not readmitted, 47.1% readmitted.
- Excluded from model features: `encounter_id`, `patient_nbr`, `discharge_disposition_id`, `payer_code`, `readmitted`, and `readmission_binary`.
- Kept `patient_nbr` separately for patient-grouped splitting.
- Patient-grouped 80:20 split (`random_state=42`): training 79,541 rows; test 19,802 rows; patient overlap 0.
- A patient-grouped validation split held out 15% of the training partition (`random_state=42`).
- Full model feature types: 34 categorical and 8 numerical. `admission_type_id` and `admission_source_id` were cast to strings.

## CatBoost configuration
`iterations=700`, `learning_rate=0.05`, `depth=6`, `loss_function="Logloss"`, `eval_metric="F1"`, `random_seed=42`, `early_stopping_rounds=50`.

## Feature-set comparison on the same validation split

| Model | Features | Validation accuracy | Validation F1 |
|---|---:|---:|---:|
| Full CatBoost | 42 | 0.6409 | 0.5775 |
| Reduced CatBoost | 22 | 0.6420 | 0.5760 |

The reduced feature set removed 20 individual medication columns while retaining `insulin`, `diabetesMed`, and `change`. It was selected as a simpler model with nearly identical validation F1 and slightly higher validation accuracy. This is a practical feature-efficiency choice, not evidence of statistical superiority.

Best validation iteration: full model 453; reduced model 316.

## Final held-out test evaluation — selected reduced model
- Test accuracy: **0.6369 (63.69%)**
- Test F1 for positive class (`Readmitted`): **0.5846**

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Not Readmitted | 0.64 | 0.72 | 0.68 | 10,496 |
| Readmitted | 0.63 | 0.54 | 0.58 | 9,306 |
| Macro average | 0.64 | 0.63 | 0.63 | 19,802 |
| Weighted average | 0.64 | 0.64 | 0.63 | 19,802 |

Confusion matrix (rows = actual, columns = predicted; class order 0, 1):

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 7,552 | 2,944 |
| Actual 1 | 4,247 | 5,059 |

The model correctly identified 5,059 readmitted patients and missed 4,247. Positive-class recall is approximately 54.4%, so the model misses a substantial share of readmissions.

## Current conclusion
The reduced 22-feature CatBoost model achieved 63.69% test accuracy and 0.5846 test F1. It offers a simpler feature set while maintaining validation performance close to the 42-feature model. Readmission recall remains a limitation and must be disclosed.

## Reproducibility and next steps
- Notebook: `notebooks/02_model_training.ipynb`
- Preserve the untouched test set; do not tune model settings based on its results.
- Next: save the fitted reduced CatBoost model and its feature/categorical-column metadata under `models/`, then build the Streamlit application under `app/`.
- Verify that the application uses the exact 22 input features and categorical handling used during training.

## Important limitation
This is an educational prototype. It is not clinically validated, must not be used to make patient-care decisions, and does not establish causal relationships between features and readmission.
