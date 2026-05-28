 ❤️ Heart Attack Risk Predictor

A machine-learning web application that predicts the risk of heart disease based on clinical parameters. Built with **Python**, **Scikit-learn**, and **Streamlit**.

---

📌 Project Overview

Heart disease is one of the leading causes of death worldwide. Early detection through clinical data can significantly improve patient outcomes. This project trains multiple classification models on a heart disease dataset and deploys the best-performing model (KNN) as an interactive Streamlit web application.

---

🗂️ Project Structure

```
heart-attack-predictor/
│
├── app.py                  # Streamlit frontend
├── heart_attack.ipynb      # EDA, preprocessing, and model training notebook
├── heart.csv               # Raw dataset (918 patients)
│
├── KNN_heart.pkl           # Trained KNN model
├── scaler.pkl              # Fitted StandardScaler
├── columns.pkl             # Feature column names (post one-hot encoding)
│
└── README.md
```

---

📊 Dataset

**Source:** [Kaggle – Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

| Feature | Description |
|---|---|
| `Age` | Age of the patient (years) |
| `Sex` | Sex (M = Male, F = Female) |
| `ChestPainType` | ATA · NAP · TA · ASY |
| `RestingBP` | Resting blood pressure (mm Hg) |
| `Cholesterol` | Serum cholesterol (mg/dL) |
| `FastingBS` | Fasting blood sugar > 120 mg/dL (1 = True) |
| `RestingECG` | Normal · ST · LVH |
| `MaxHR` | Maximum heart rate achieved |
| `ExerciseAngina` | Exercise-induced angina (Y/N) |
| `Oldpeak` | ST depression induced by exercise |
| `ST_Slope` | Slope of peak exercise ST segment (Up · Flat · Down) |
| `HeartDisease` | **Target** — 1 = Heart Disease, 0 = Normal |

---

⚙️ ML Pipeline

1. Exploratory Data Analysis
- Distribution plots for `Age`, `RestingBP`, `Cholesterol`, `MaxHR`
- Count plots segmented by `HeartDisease` for categorical features

2. Preprocessing
- Zero-value imputation for `Cholesterol` and `RestingBP` (replaced with column mean)
- Label encoding for `Sex` (M→1, F→0)
- One-hot encoding (with `drop_first=True`) for `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`
- `StandardScaler` applied to all features

3. Models Evaluated

| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | ~85% | ~86% |
| **KNN** | **~87%** | **~88%** |
| Naive Bayes | ~84% | ~85% |
| Decision Tree | ~79% | ~79% |
| SVM (RBF Kernel) | ~87% | ~88% |

> KNN was selected for deployment based on accuracy and F1 score.

4. Saved Artifacts
| File | Contents |
|---|---|
| `KNN_heart.pkl` | Trained `KNeighborsClassifier` |
| `scaler.pkl` | Fitted `StandardScaler` |
| `columns.pkl` | Ordered list of 15 model input features |

---

🖥️ Streamlit App

Features
- Clean two-column form for all 11 clinical inputs
- One-hot encoding & scaling handled automatically
- Binary prediction + probability breakdown
- Visual risk progress bar
- Expandable input summary table
- Medical disclaimer

Input Fields
| Field | Type | Options / Range |
|---|---|---|
| Age | Number | 1 – 120 |
| Sex | Select | Male / Female |
| Chest Pain Type | Select | ATA · NAP · TA · ASY |
| Resting Blood Pressure | Number | 0 – 300 mm Hg |
| Cholesterol | Number | 0 – 700 mg/dL |
| Fasting Blood Sugar > 120 | Checkbox | True / False |
| Resting ECG | Select | Normal · ST · LVH |
| Max Heart Rate | Number | 60 – 220 |
| Exercise Angina | Select | Yes / No |
| Oldpeak | Decimal | 0.0 – 10.0 |
| ST Slope | Select | Up · Flat · Down |

---

🚀 Getting Started

Prerequisites

```bash
pip install streamlit scikit-learn pandas numpy joblib
```

Run the App

```bash
# Clone / download the project, then:
streamlit run app.py
```

Make sure the following files are in the **same directory** as `app.py`:

```
KNN_heart.pkl   scaler.pkl   columns.pkl
```

The app will open at `http://localhost:8501` in your browser.

---

📦 Dependencies

| Package | Version |
|---|---|
| Python | ≥ 3.8 |
| streamlit | ≥ 1.30 |
| scikit-learn | ≥ 1.3 |
| pandas | ≥ 2.0 |
| numpy | ≥ 1.24 |
| joblib | ≥ 1.3 |

---

⚠️ Disclaimer

This application is developed **for educational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for any health concerns.

---

🧑‍💻 Author

Built as part of a machine learning classification project on heart disease prediction.
