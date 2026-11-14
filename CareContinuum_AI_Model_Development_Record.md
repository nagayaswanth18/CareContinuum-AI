# CareContinuum AI — Model Development Decision Record

**Purpose:** Document why the project moved from a 42-feature model to a 22-feature model and then investigated a 10-feature compact model.

**Status:** The 22-feature model has completed held-out test evaluation. The tuned 10-feature model has validation results; its final held-out test evaluation is still pending. Therefore, this record does not claim that the 10-feature model is better.

## 1. Project and data preparation

The project uses the UCI Diabetes 130-US Hospitals dataset (1999–2008) to build an educational prototype that estimates whether a hospital encounter will be followed by readmission.

Binary target:
- `NO` → 0 (Not Readmitted)
- `>30` and `<30` → 1 (Readmitted)

This target represents any recorded readmission category, not specifically readmission within 30 days.

Data preparation:
- Excluded discharge disposition IDs `[11, 13, 14, 19, 20]`.
- Dropped `weight`, `citoglipton`, and `examide`.
- Replaced missing categorical values with `Unknown`.
- Cleaned modelling dataset: 99,343 encounters and 48 columns.
- Model feature matrix: 42 features.
- Kept patient identifiers separately for grouped splitting, not as model inputs.
- Patient-grouped train/test split: 79,541 training encounters and 19,802 held-out test encounters.
- No patient appeared in both train and test sets.
- A patient-grouped validation split was made from the training partition for model selection.

Grouping by patient helps reduce leakage from repeated encounters belonging to the same patient.

## 2. First model: 42 features

CatBoost configuration:
- Iterations: 700
- Learning rate: 0.05
- Depth: 6
- Loss: Logloss
- Evaluation metric: F1
- Random seed: 42
- Early stopping: 50 rounds

Validation results:
- Accuracy: **0.6409**
- Readmitted-class F1: **0.5775**
- Best iteration: 453

Feature importance highlighted variables such as `number_inpatient`, diagnosis codes, admission source, number of diagnoses, medical specialty, admission type, number of medications, emergency visits, and age.

**Interpretation:** This model established a performance baseline, but using all 42 features would make the user-facing prediction form lengthy.

## 3. Second model: 22 features

The next step removed the 20 individual medication columns while retaining the broader medication indicators `insulin`, `diabetesMed`, and `change`. This reduced the input count from 42 to 22.

Validation comparison:

| Model | Features | Accuracy | Readmitted-class F1 |
|---|---:|---:|---:|
| Full model | 42 | 0.6409 | 0.5775 |
| Reduced model | 22 | 0.6420 | 0.5760 |

The 22-feature model had marginally lower F1 but slightly higher accuracy, while requiring 20 fewer inputs. It was selected as a simpler candidate with nearly unchanged validation performance.

### Held-out test results for the 22-feature model

- Accuracy: **0.6369** (63.69%)
- Readmitted-class F1: **0.5846**
- Readmitted-class precision: approximately **0.63**
- Readmitted-class recall: approximately **0.54**

Confusion matrix (rows = actual class; columns = predicted class):

| | Predicted Not Readmitted | Predicted Readmitted |
|---|---:|---:|
| Actual Not Readmitted | 7,552 | 2,944 |
| Actual Readmitted | 4,247 | 5,059 |

It correctly identified 5,059 readmitted encounters and missed 4,247. This is why recall matters alongside accuracy and F1.

## 4. Why investigate a 10-feature model?

Although 22 features were fewer than 42, the Streamlit application still required users to enter 22 values. The goal was to investigate whether a compact model could make the application easier to use while retaining performance close to the 22-feature model.

The 10 candidate features were selected from the leading feature-importance results of the 42-feature model. They were candidates to test, not features proven to be optimal:

1. `number_inpatient`
2. `diag_1`
3. `admission_source_id`
4. `diag_2`
5. `number_diagnoses`
6. `medical_specialty`
7. `admission_type_id`
8. `num_medications`
9. `number_emergency`
10. `age`

## 5. Initial 10-feature model

The initial 10-feature model used CatBoost with 700 maximum iterations, learning rate 0.05, depth 6, F1 evaluation, random seed 42, and early stopping after 50 rounds.

| Model | Features | Accuracy | Readmitted-class F1 |
|---|---:|---:|---:|
| 22-feature model | 22 | 0.6420 | 0.5760 |
| Initial 10-feature model | 10 | 0.6380 | 0.5695 |

The initial 10-feature model lost 0.0040 in accuracy and 0.0065 in F1 compared with the 22-feature model. The decrease was small, but it motivated further controlled tuning.

## 6. Eight hyperparameter configurations for the 10-feature model

The same training/validation split and the same 10 features were used. The following parameters were varied:

- Depth: 4 or 6
- Learning rate: 0.03 or 0.05
- L2 regularization: 3 or 5

All runs used maximum iterations 700, Logloss, F1 evaluation, random seed 42, and early stopping after 50 rounds.

| Model | Depth | Learning rate | L2 | Best iteration | Accuracy | F1 | Precision | Recall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 0.03 | 3 | 334 | 0.6368 | 0.5602 | 0.6264 | 0.5066 |
| 2 | 4 | 0.03 | 5 | 4 | 0.6221 | 0.5209 | 0.6182 | 0.4501 |
| 3 | 4 | 0.05 | 3 | 425 | 0.6389 | 0.5678 | 0.6258 | 0.5197 |
| 4 | 4 | 0.05 | 5 | 295 | 0.6352 | 0.5619 | 0.6219 | 0.5124 |
| 5 | 6 | 0.03 | 3 | 2 | 0.6256 | 0.5305 | 0.6203 | 0.4633 |
| 6 | 6 | 0.03 | 5 | 2 | 0.6256 | 0.5305 | 0.6203 | 0.4633 |
| 7 | 6 | 0.05 | 3 | 321 | 0.6380 | 0.5695 | 0.6229 | 0.5245 |
| 8 | 6 | 0.05 | 5 | 484 | 0.6380 | **0.5727** | 0.6209 | 0.5315 |

The eight experiments took approximately **400 seconds** in total.

### Best tuned configuration: Model 8

- Depth: **6**
- Learning rate: **0.05**
- L2 regularization: **5**
- Best iteration: **484**
- Validation accuracy: **0.6380**
- Validation F1: **0.5727**
- Validation precision: **0.6209**
- Validation recall: **0.5315**

Tuning improved the 10-feature model's F1 from 0.5695 to 0.5727, an increase of 0.0032. Accuracy remained 0.6380.

## 7. Overall validation comparison

| Model | Features | Validation accuracy | Validation F1 |
|---|---:|---:|---:|
| Full model | 42 | 0.6409 | 0.5775 |
| Reduced model | 22 | 0.6420 | 0.5760 |
| Initial compact model | 10 | 0.6380 | 0.5695 |
| Tuned compact model | 10 | 0.6380 | 0.5727 |

## 8. Reason for changing the approach

The progression was based on a practical trade-off between predictive performance and application usability:

1. The 42-feature model established the initial performance baseline.
2. The 22-feature model removed 20 inputs while retaining very similar validation performance.
3. The Streamlit application still required 22 user inputs, so a more compact 10-feature alternative was investigated.
4. The 10 features were selected as candidates based on the earlier feature-importance analysis.
5. The initial 10-feature model was slightly weaker, so eight hyperparameter configurations were tested.
6. Tuning improved the 10-feature model's F1, but its validation F1 (0.5727) remains slightly below the 22-feature model (0.5760).

**Current conclusion:** The 10-feature model is a promising usability-oriented alternative, but it has not yet demonstrated better performance than the 22-feature model. The final choice should be made after evaluating the tuned 10-feature model on the same untouched test set and comparing it fairly with the already evaluated 22-feature model.

## 9. Evaluation limitations and next step

- Do not use the held-out test set for further tuning or feature selection.
- The final held-out test evaluation of the tuned 10-feature model is pending.
- The model is an educational prototype, not a clinically validated decision-support system.
- The target combines `<30` and `>30`, so it predicts recorded readmission versus no recorded readmission, not specifically 30-day readmission.
- Report precision and recall, especially for the readmitted class, alongside accuracy and F1.

**Update this record with the final 10-feature test metrics and final model-selection decision after the held-out test evaluation.**


## 10. Final held-out test evaluation of the tuned 10-feature model

After selecting the best 10-feature hyperparameter configuration using the validation set, the final model was refitted on the full training partition (training + validation; 79,541 encounters) for 484 iterations. The held-out test set (19,802 encounters) was then used for final evaluation.

Final configuration:
- Features: the 10 listed in Section 4
- Iterations: 484
- Depth: 6
- Learning rate: 0.05
- L2 regularization: 5
- Random seed: 42

Final test metrics:
- Accuracy: **0.6346** (63.46%)
- Readmitted-class F1: **0.5836**
- Readmitted-class precision: **0.6282**
- Readmitted-class recall: **0.5449**

Classification report:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Not Readmitted | 0.64 | 0.71 | 0.67 | 10,496 |
| Readmitted | 0.63 | 0.54 | 0.58 | 9,306 |
| Macro average | 0.63 | 0.63 | 0.63 | 19,802 |
| Weighted average | 0.63 | 0.63 | 0.63 | 19,802 |

Confusion matrix (rows = actual class; columns = predicted class):

| | Predicted Not Readmitted | Predicted Readmitted |
|---|---:|---:|
| Actual Not Readmitted | 7,495 | 3,001 |
| Actual Readmitted | 4,235 | 5,071 |

## 11. Final comparison on the held-out test set

| Metric | 22-feature model | Tuned 10-feature model |
|---|---:|---:|
| Accuracy | 0.6369 | 0.6346 |
| Readmitted-class F1 | 0.5846 | 0.5836 |
| Readmitted-class precision | approximately 0.63 | 0.6282 |
| Readmitted-class recall | approximately 0.54 | 0.5449 |
| Number of input features | 22 | 10 |

The 10-feature model's accuracy is 0.23 percentage points lower and its readmitted-class F1 is 0.10 percentage points lower. Its readmitted-class recall is 0.12 percentage points higher. The compact model correctly identified 5,071 readmitted encounters compared with 5,059 for the 22-feature model, while producing 3,001 false positives compared with 2,944.

These differences are small. No statistical significance test or confidence interval has been calculated, so the results should not be described as proving that either model is statistically superior.

## 12. Final model-selection rationale

The project selected the tuned 10-feature model as the preferred application model because it reduces the number of user-provided inputs from 22 to 10 while retaining very similar held-out test performance.

This is a usability-versus-performance engineering decision:
- The 22-feature model has marginally higher test accuracy and F1.
- The 10-feature model has marginally higher readmitted-class recall and requires 12 fewer inputs.
- The 10-feature model therefore offers a simpler user experience with only a very small observed reduction in accuracy and F1.

The 22-feature model should be retained as a comparison/backup model. The selection does **not** establish that the 10-feature model is statistically better or clinically suitable.

**Final project decision:** Use the tuned 10-feature model as the preferred model for the CareContinuum AI demonstration application, while retaining the 22-feature model and this experiment record as evidence of the model-selection process.

## 13. Evidence trail and reproducibility notes

The following outputs were recorded during the development process:
1. Validation metrics for the 42-feature model.
2. Validation metrics for the 22-feature model.
3. Held-out test metrics and confusion matrix for the 22-feature model.
4. The 10 selected candidate features and the feature-importance rationale.
5. Validation metrics for the initial 10-feature model.
6. The eight tested hyperparameter configurations, their best iterations, and validation metrics.
7. The selected 10-feature configuration: depth 6, learning rate 0.05, L2 regularization 5, 484 iterations.
8. Final 10-feature held-out test metrics and confusion matrix.
9. The final comparison and rationale for preferring the compact model in the application.

For a reproducible submission, retain the notebook outputs, trained model files, metadata files, and this decision record together. The written record summarizes the results supplied during the experiment; the notebook remains the primary executable evidence.

