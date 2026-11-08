# Online Shopper Purchase Intention — ML Assignment 2

## Problem statement
Build and compare five supervised classification models that predict whether an online shopping session ends in a purchase (`Revenue=True`). This is a binary classification problem. The work includes exploratory analysis, leakage-aware preprocessing, model evaluation, and a Streamlit application that evaluates saved models on an uploaded labelled test CSV without retraining.

## Dataset description
- **Dataset:** Online Shoppers Purchasing Intention Dataset
- **Source:** UCI Machine Learning Repository (Dataset 468)
- **Official page:** https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- **DOI:** https://doi.org/10.24432/C5F88Q
- **Size:** 12,330 sessions; 17 original input attributes and one target attribute (`Revenue`).
- **Target:** `Revenue` (0 = no purchase, 1 = purchase).
- **Class balance:** UCI reports 10,422 non-purchase sessions (84.5%) and 1,908 purchase sessions (15.5%).
- **Feature handling:** `PageValues` is excluded from the model inputs as a leakage-aware modelling decision because it is a conversion-related analytics measure. The remaining input variables are preprocessed in a scikit-learn pipeline.
- **Citation:** Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q.

## GitHub Repository Link
https://github.com/Ameya-2026aj05079/ML-Assignment-2

## Models used and evaluation metrics
The notebook generates `model_metrics.csv` and writes the measured values below into this README after execution. All models use the same stratified holdout test split. Precision, Recall and F1 refer to the positive class (`Revenue=1`). AUC is ROC-AUC calculated from predicted probabilities or decision scores.

| ML Model | Accuracy | ROC-AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| ML Model | Accuracy | ROC-AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.6310 | 0.7352 | 0.2600 | 0.7487 | 0.3860 | 0.2607 |
| Support Vector Machine | 0.6476 | 0.7360 | 0.2596 | 0.6885 | 0.3771 | 0.2417 |
| Decision Tree | 0.6715 | 0.6999 | 0.2643 | 0.6283 | 0.3721 | 0.2309 |
| Gaussian Naive Bayes | 0.2709 | 0.6774 | 0.1719 | 0.9712 | 0.2921 | 0.1245 |
| K-Nearest Neighbors | 0.8382 | 0.7037 | 0.3836 | 0.0733 | 0.1231 | 0.1104 |

### Observations on model performance
- **Logistic Regression:** ROC-AUC 0.7352, F1 0.3860, Recall 0.7487, MCC 0.2607. Accuracy is 0.6310. Interpret these together because the positive purchase class is the minority.
- **Support Vector Machine:** ROC-AUC 0.7360, F1 0.3771, Recall 0.6885, MCC 0.2417. Accuracy is 0.6476. Interpret these together because the positive purchase class is the minority.
- **Decision Tree:** ROC-AUC 0.6999, F1 0.3721, Recall 0.6283, MCC 0.2309. Accuracy is 0.6715. Interpret these together because the positive purchase class is the minority.
- **Gaussian Naive Bayes:** ROC-AUC 0.6774, F1 0.2921, Recall 0.9712, MCC 0.1245. Accuracy is 0.2709. Interpret these together because the positive purchase class is the minority.
- **K-Nearest Neighbors:** ROC-AUC 0.7037, F1 0.1231, Recall 0.0733, MCC 0.1104. Accuracy is 0.8382. Interpret these together because the positive purchase class is the minority.

Because the target is imbalanced, accuracy is interpreted together with ROC-AUC, Recall, F1 and MCC. A model that predicts every session as non-purchase can obtain about 84.5% accuracy while failing to identify purchases.

### Overall winner
**Logistic Regression** was selected using positive-class F1 as the primary ranking metric and ROC-AUC as a secondary indicator. Its F1 was 0.3860 and ROC-AUC was 0.7352 on this holdout split.

## Repository structure
```text
ML-Assignment-2/
├── ML_Assignment_2_Online_Shoppers.ipynb
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_metrics.csv
├── model_metadata.json
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── gaussian_nb.joblib
    └── svm.joblib
```


