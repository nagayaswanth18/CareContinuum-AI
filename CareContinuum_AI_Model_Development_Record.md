# CareContinuum AI — Model Development Record

## 1. Purpose

This document records the experimentation and modelling decisions made during the development of CareContinuum AI. It covers the initial three-class experiments, the transition to a binary readmission target, patient-grouped data splitting, CatBoost feature reduction, hyperparameter tuning, and final model selection.

The project is an educational prototype and is not clinically validated.

## 2. Dataset and initial preparation

Dataset: *Diabetes 130-US Hospitals for Years 1999–2008* from the UCI Machine Learning Repository.

- Original dataset: 101,766 encounters and 50 columns.
- Excluded discharge disposition IDs: `[11, 13, 14, 19, 20]`.
- Removed `weight`, `citoglipton`, and `examide`.
- Replaced `?` with missing values; missing categorical values were filled with `Unknown`.
- Cleaned modelling dataset: 99,343 encounters and 48 columns.

The first notebook investigated the original three-category target (`NO`, `>30`, `<30`). The later CatBoost workflow combined `>30` and `<30` into a single readmitted class and treated `NO` as not readmitted.

**Target interpretation:** the final binary target represents any recorded readmission versus no recorded readmission. It does not specifically predict readmission within 30 days.

## 3. Initial three-class experiments

### 3.1 Cross-validation and evaluation approach

The first notebook used five-fold patient-grouped cross-validation for the Logistic Regression comparisons. Accuracy and macro F1 were recorded. Macro F1 was important because it gives each class equal weight and helps expose poor minority-class performance.

### 3.2 Dummy Classifier baseline

The majority-class Dummy Classifier established a baseline:

| Metric | Result |
|---|---:|
| Accuracy | 52.84% |
| Macro F1 | 0.2305 |

This provided a reference point for the trainable models.

### 3.3 Logistic Regression

Configuration visible in the notebook:

```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

Five-fold cross-validation results:

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Accuracy | 57.35% | 0.0029 |
| Macro F1 | 0.3914 | 0.0045 |

The model improved over the majority baseline, but minority-class performance remained weak, particularly for the `<30` class.

### 3.4 Class-weighted Logistic Regression

Configuration:

```python
LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)
```

Results:

| Metric | Standard Logistic Regression | Class-weighted Logistic Regression |
|---|---:|---:|
| Accuracy | 57.35% | 48.17% |
| Macro F1 | 0.3914 | 0.4262 |

Class weighting reduced accuracy but improved macro F1. The notebook's displayed class-level results showed the `<30` class recall increasing from approximately 0.02 to 0.40. This illustrated the trade-off between overall accuracy and minority-class detection.

### 3.5 Random Forest exploration

The initial Random Forest configuration shown in the notebook was:

```python
RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=1
)
```

The full 200-tree run took a very long time and did not complete within the experiment. To explore the algorithm without waiting for the full run, a smaller experiment was conducted using a 10% subset and a reduced configuration.

The exploratory validation result was:

| Metric | Result |
|---|---:|
| Accuracy | 54.43% |
| Macro F1 | 0.3591 |

This was a reduced exploratory experiment, not a completed five-fold evaluation of the original 200-tree configuration. It should not be compared directly with the five-fold Logistic Regression averages.

### 3.6 Lessons from the first notebook

- A majority-class baseline is necessary to contextualize model performance.
- Accuracy alone does not adequately represent performance across imbalanced classes.
- Class weighting improved macro F1 and minority-class recall, at the cost of lower accuracy.
- Random Forest required more computation than was practical for the initial full configuration.
- These experiments motivated further modelling and evaluation in the subsequent binary CatBoost workflow.

## 4. Binary target and modelling data

The later workflow mapped the target:

| Original `readmitted` value | `readmission_binary` |
|---|---:|
| `NO` | 0 |
| `>30` | 1 |
| `<30` | 1 |

The final cleaned binary modelling dataset had 99,343 rows. The feature matrix had 42 features. `patient_nbr` was kept separately as the group label and not used as a model feature. Encounter identifiers and target-related columns were removed from the feature matrix.

The binary target distribution shown in the notebook was approximately:

- Class 0 (not readmitted): 52.9%
- Class 1 (readmitted): 47.1%

## 5. Patient-grouped splitting

The workflow used patient identifiers for grouped splitting to prevent encounters from the same patient appearing in both training and test partitions.

The final split used:
- Training partition: 79,541 encounters
- Held-out test partition: 19,802 encounters
- No shared patients between train and test.

A validation partition was created from the training data for model selection. The final held-out test set was reserved for final evaluation.

## 6. CatBoost with 42 features

The initial CatBoost model used the 42-feature matrix, including categorical features handled by CatBoost.

Main configuration:

```python
CatBoostClassifier(
    iterations=700,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="F1",
    random_seed=42,
    early_stopping_rounds=50,
    allow_writing_files=False
)
```

Validation results:

| Metric | Result |
|---|---:|
| Accuracy | 0.6409 |
| Readmitted-class F1 | 0.5775 |
| Best iteration | 453 |

Feature importance from this model was used to help identify a smaller feature set for subsequent experiments.

## 7. CatBoost with 22 features

The project then removed 20 individual medication columns while retaining `insulin`, `diabetesMed`, and `change`.

Validation results:

| Metric | Result |
|---|---:|
| Accuracy | 0.6420 |
| Readmitted-class F1 | 0.5760 |

Held-out test results:

| Metric | Result |
|---|---:|
| Accuracy | 0.6369 |
| Readmitted-class precision | approximately 0.6321 |
| Readmitted-class recall | approximately 0.5437 |
| Readmitted-class F1 | 0.5846 |

Confusion matrix (rows = actual class; columns = predicted class):

```text
[[7552, 2944],
 [4247, 5059]]
```

This model was saved as `carecontinuum_catboost.cbm`, with its associated metadata and category options.

## 8. Compact 10-feature model

To simplify the application, a compact model was investigated using the following features:

```python
compact_features = [
    "number_inpatient",
    "diag_1",
    "admission_source_id",
    "diag_2",
    "number_diagnoses",
    "medical_specialty",
    "admission_type_id",
    "num_medications",
    "number_emergency",
    "age"
]
```

The 10-feature model contains six categorical and four numerical features.

Initial validation results:

| Metric | Result |
|---|---:|
| Accuracy | 0.6380 |
| Readmitted-class F1 | 0.5695 |

## 9. Hyperparameter tuning of the 10-feature model

Eight configurations were compared. The search varied:
- Depth: 4 or 6
- Learning rate: 0.03 or 0.05
- L2 regularization: 3 or 5

Maximum iterations were 700, with Logloss, F1 evaluation, random seed 42, and early stopping after 50 rounds.

| Model | Depth | Learning rate | L2 | Best iteration | Validation accuracy | Validation F1 | Precision | Recall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 0.03 | 3 | 334 | 0.6368 | 0.5602 | 0.6264 | 0.5066 |
| 2 | 4 | 0.03 | 5 | 4 | 0.6221 | 0.5209 | 0.6182 | 0.4501 |
| 3 | 4 | 0.05 | 3 | 425 | 0.6389 | 0.5678 | 0.6258 | 0.5197 |
| 4 | 4 | 0.05 | 5 | 295 | 0.6352 | 0.5619 | 0.6219 | 0.5124 |
| 5 | 6 | 0.03 | 3 | 2 | 0.6256 | 0.5305 | 0.6203 | 0.4633 |
| 6 | 6 | 0.03 | 5 | 2 | 0.6256 | 0.5305 | 0.6203 | 0.4633 |
| 7 | 6 | 0.05 | 3 | 321 | 0.6380 | 0.5695 | 0.6229 | 0.5245 |
| 8 | 6 | 0.05 | 5 | 484 | 0.6380 | **0.5727** | 0.6209 | 0.5315 |

Configuration 8 was selected based on validation F1.

Selected configuration:
- Depth: 6
- Learning rate: 0.05
- L2 regularization: 5
- Best iteration: 484

Tuning increased validation F1 from 0.5695 to 0.5727.

## 10. Final held-out test evaluation of the tuned 10-feature model

After selecting the hyperparameters using validation data, the final model was refitted on the full training partition (79,541 encounters) for 484 iterations. It was then evaluated on the held-out test partition (19,802 encounters).

| Metric | Test result |
|---|---:|
| Accuracy | 0.6346 |
| Readmitted-class precision | 0.6282 |
| Readmitted-class recall | 0.5449 |
| Readmitted-class F1 | 0.5836 |

Classification report:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Not readmitted (0) | 0.64 | 0.71 | 0.67 | 10,496 |
| Readmitted (1) | 0.63 | 0.54 | 0.58 | 9,306 |
| Macro average | 0.63 | 0.63 | 0.63 | 19,802 |
| Weighted average | 0.63 | 0.63 | 0.63 | 19,802 |

Confusion matrix (rows = actual class; columns = predicted class):

```text
[[7495, 3001],
 [4235, 5071]]
```

The saved model was reloaded and its predictions were checked against the original predictions.

Saved artefacts:
- `carecontinuum_catboost_10features.cbm`
- `model_metadata_10features.json`
- `category_options_10features.json`
- `numeric_stats.json`

## 11. Comparison and final model-selection decision

| Metric | 22-feature model | Tuned 10-feature model |
|---|---:|---:|
| Test accuracy | 0.6369 | 0.6346 |
| Readmitted-class F1 | 0.5846 | 0.5836 |
| Readmitted-class recall | approximately 0.5437 | 0.5449 |
| Number of input features | 22 | 10 |

The 22-feature model has marginally higher test accuracy and F1. The 10-feature model has marginally higher readmitted-class recall and requires 12 fewer inputs.

The tuned 10-feature model was selected as the preferred application model because it provides a simpler user experience while retaining very similar held-out test performance. The 22-feature model is retained as a comparison and backup model.

This decision does not establish statistical superiority, and neither model has been clinically validated.

## 12. Final 10-feature model feature importance

| Feature | Importance |
|---|---:|
| `number_inpatient` | 24.913 |
| `diag_1` | 12.616 |
| `admission_source_id` | 9.974 |
| `diag_2` | 9.382 |
| `age` | 8.119 |
| `admission_type_id` | 7.914 |
| `number_diagnoses` | 7.560 |
| `medical_specialty` | 6.734 |
| `num_medications` | 6.525 |
| `number_emergency` | 6.264 |

These are global model feature-importance scores, not causal effects or clinical risk estimates.

## 13. Application implementation

The Streamlit application uses the compact 10-feature CatBoost model. It loads the saved model, metadata, category options, and numerical statistics. The interface accepts 10 inputs and displays estimated class probabilities and a threshold-based classification using a 0.50 threshold.

The app includes the model's held-out test metrics and an educational-use disclaimer. An individual prediction probability is a model output, not a diagnosis or a clinically validated risk estimate.

## 14. Limitations

- The source data is historical (1999–2008).
- The final binary target combines readmissions within 30 days and after 30 days.
- The model has not undergone external or prospective validation.
- The reported results represent one patient-grouped held-out test evaluation.
- The Random Forest full 200-tree run did not complete; only a smaller exploratory experiment was recorded.
- The three-class first-notebook metrics and binary CatBoost metrics concern different target definitions and should not be compared as if they were the same task.
- Feature importance does not demonstrate causality.
- This is an academic project, not a clinical decision-support system.

## 15. References

1. Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). *Diabetes 130-US Hospitals for Years 1999–2008*. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J
2. Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records*. BioMed Research International, 2014, 781670. https://doi.org/10.1155/2014/781670
3. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). *CatBoost: Unbiased Boosting with Categorical Features*. NeurIPS 31. https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html
