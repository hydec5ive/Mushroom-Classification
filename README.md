# Mushroom Edibility Classification — Multimodal ML System

Can machine learning determine if a mushroom is safe to eat? This project compares 9 tabular ML models against CNN-based image classification on a 58,000+ sample dataset, evaluating whether visual or feature-based approaches perform better for edibility prediction.

---

## Key Results

- **Best model:** Random Forest — 99.7% test accuracy
- **Models compared:** 9 tabular ML models + CNN architectures (ResNet-50, VGG-16)
- **Finding:** Tabular feature-based models significantly outperformed image-based deep learning for this classification task

---

## Tech Stack

- **Language:** Python
- **ML:** scikit-learn (Random Forest, XGBoost, SVM, Logistic Regression, AdaBoost, KNN), TensorFlow/Keras (CNN)
- **Data:** pandas, numpy
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook

---

## Project Structure

```
mushroom-classification-ml/
├── README.md
├── requirements.txt
├── data/
│   └── mushroom_dataset.csv
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_tabular_models.ipynb
│   └── 03_cnn_image_models.ipynb
└── results/
    ├── model_accuracy_comparison.png
    └── confusion_matrices.png
```

---

## Models Compared

| Model | Test Accuracy | Type |
|---|---|---|
| Random Forest | 99.7% | Tabular |
| XGBoost | ~99% | Tabular |
| SVM | ~98% | Tabular |
| AdaBoost | ~97% | Tabular |
| Logistic Regression | ~95% | Tabular |
| KNN | ~94% | Tabular |
| ResNet-50 | Lower | Image (CNN) |
| VGG-16 | Lower | Image (CNN) |
| Custom CNN | Lower | Image (CNN) |

---


## Dataset

- 58,000+ labeled mushroom samples
- Features include cap shape, color, odor, gill size, habitat, and more
- Binary classification target: edible vs poisonous

---

## Key Takeaway

Structured tabular features (cap shape, odor, gill size) provided far more predictive signal than raw image data for mushroom edibility classification. This highlights an important ML design principle — image-based deep learning is not always superior to well-engineered tabular features, especially when domain-specific attributes are available.

---

## Authors

Haoyuan Deng & Dimitry Pianykh

## Course

DS4400 — Machine Learning & Data Mining  
Northeastern University, D'Amore-McKim School of Business
