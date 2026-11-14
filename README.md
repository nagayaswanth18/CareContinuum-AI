# CareContinuum AI
### Machine Learning-Based Hospital Readmission Prediction

CareContinuum AI is a machine learning project that predicts whether a patient will have a recorded hospital readmission based on hospital encounter information. The project uses the Diabetes 130-US Hospitals dataset and CatBoost classification.

The goal is to explore how machine learning can support the analysis of hospital readmission patterns through a simple, interactive web application.

> **Disclaimer:** This project is developed for educational and research purposes only. It is not a clinically validated system and must not be used for medical diagnosis or treatment decisions.

## Project Overview

Hospital readmissions are an important aspect of healthcare analysis. This project applies machine learning techniques to historical hospital records to classify patient encounters into two categories:

- **Not Readmitted (0)**
- **Readmitted (1)**

The target combines the original `>30` and `<30` categories into one readmitted class, while `NO` represents no recorded readmission.

**Important:** This project predicts any recorded readmission category in the dataset. It does not specifically predict 30-day readmission.

## Dataset

**Dataset:** Diabetes 130-US Hospitals for Years 1999–2008

- Source: UCI Machine Learning Repository
- Original records: 101,766
- Original attributes: 50
- Data represents hospital encounters involving patients with diabetes.

The dataset was cleaned and prepared before model development. After preprocessing, the modeling dataset contained 99,343 records.

Dataset citation:

Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). *Diabetes 130-US Hospitals for Years 1999–2008*. UCI Machine Learning Repository.

DOI: https://doi.org/10.24432/C5230J

## Data Preprocessing

The preprocessing steps included:

1. Converted missing-value markers (`?`) into missing values.
2. Replaced missing categorical values with `Unknown`.
3. Removed columns with excessive missing values or limited modeling usefulness.
4. Excluded selected discharge disposition categories.
5. Converted the original readmission labels into a binary target.
6. Removed identifiers and target-related columns from model inputs.

The patient identifier was retained separately to support patient-grouped data splitting and reduce data leakage.

## Model Development

CatBoost Classifier was selected because it supports categorical features and is suitable for structured/tabular datasets.

Three feature configurations were explored:

- **42-feature model:** Initial model using a broad set of available predictors.
- **22-feature model:** Reduced model after removing individual medication columns.
- **10-feature model:** Compact model selected for the final web application.

The final model uses 10 features:

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

Patient-grouped splitting was used so that records from the same patient did not appear in both training and testing sets.

## Final Model Performance

The selected 10-feature CatBoost model was evaluated on a held-out test set.

| Metric | Result |
|---|---:|
| Accuracy | 63.46% |
| Readmitted-class Precision | 62.82% |
| Readmitted-class Recall | 54.49% |
| Readmitted-class F1-score | 58.36% |

**Test set size:** 19,802 encounters.

These metrics describe performance on this dataset and do not establish clinical effectiveness or generalization to other hospitals.

## Web Application

CareContinuum AI includes an interactive Streamlit application.

The application allows users to:
- Enter patient encounter information.
- Generate a readmission prediction.
- View the predicted class.
- View the estimated readmission probability.
- Review model performance information.

## Project Structure

```text
CareContinuum-AI/
├── app/
│   └── app.py
├── data/
├── docs/
├── models/
│   ├── carecontinuum_catboost_10features.cbm
│   ├── model_metadata_10features.json
│   ├── category_options_10features.json
│   └── numeric_stats.json
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   └── 02_model_training.ipynb
├── .gitignore
├── requirements.txt
├── README.md
├── CareContinuum_AI_Model_Results.md
└── CareContinuum_AI_Model_Development_Record.md
```

## Installation and Usage

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd CareContinuum-AI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app/app.py
```

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- CatBoost
- Streamlit
- Jupyter Notebook

## Project Documentation

Additional documentation is available in:

- `CareContinuum_AI_Model_Results.md`
- `CareContinuum_AI_Model_Development_Record.md`

## References

1. Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). *Diabetes 130-US Hospitals for Years 1999–2008*. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J

2. Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records*. BioMed Research International, 2014, 781670. https://doi.org/10.1155/2014/781670

3. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). *CatBoost: Unbiased Boosting with Categorical Features*. NeurIPS 31. https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html

---

**CareContinuum AI — Exploring Machine Learning for Hospital Readmission Analysis.**