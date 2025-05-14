# 🧠 Shallow Neural Network – Pima Indians Diabetes

Welcome to a foundational machine learning project where we implement a **shallow neural network from scratch** using only NumPy.

This hands-on exercise is designed to strengthen your understanding of the core concepts behind neural networks — without relying on high-level libraries like TensorFlow or PyTorch.

---

## 📌 Project Highlights

- ✅ Fully manual implementation of a feedforward neural network
- ✅ Trained on the Pima Indians Diabetes dataset
- ✅ Gradient Descent + Backpropagation from scratch
- ✅ ReLU & Sigmoid activations implemented manually
- ✅ Evaluation metrics and prediction logic included
- ✅ No external deep learning libraries used

---

## 📊 Dataset Overview

The [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) contains 768 records of Native American women with 8 clinical features:

| Feature                  | Description                             |
|--------------------------|-----------------------------------------|
| Pregnancies              | Number of times pregnant                |
| Glucose                  | Plasma glucose concentration            |
| BloodPressure            | Diastolic blood pressure (mm Hg)        |
| SkinThickness            | Triceps skinfold thickness (mm)         |
| Insulin                  | 2-Hour serum insulin (mu U/ml)          |
| BMI                     | Body mass index                         |
| DiabetesPedigreeFunction | Diabetes pedigree function              |
| Age                     | Age (years)                             |
| Outcome                 | Class variable (0 = no diabetes, 1 = diabetes) |

---

## ⚙️ How It Works

1. **Data Loading**
   - Train & test CSVs are loaded with Pandas
   - Features are normalized (mean=0, std=1)
   - Bias terms are added

2. **Model Architecture**
   - Input layer: 9 features (8 + bias)
   - Hidden layer: 1000 neurons with ReLU activation
   - Output layer: 1 neuron with Sigmoid activation

3. **Training**
   - Custom class `Model` implements:
     - `predict()`
     - `update_weights_for_one_epoch()`
     - `fit()`
   - Optimized with manual gradient descent and MSE loss

4. **Evaluation**
   - Accuracy is measured using a validation split
   - Threshold at 0.5 for binary classification

5. **Prediction**
   - Predictions are made on normalized test data
   - Output is saved in `prediction.npy`

---

## 🗂️ Project Files

| File                   | Description                                   |
|------------------------|-----------------------------------------------|
| `play_with_shallow.ipynb` | Jupyter notebook with full code & explanation |
| `model.py`             | Extracted model class for submission/export   |
| `processed_test_data.csv` | Normalized test data used for final predictions |
| `prediction.npy`       | Prediction results for test data              |
| `result.zip`           | Zipped package with all necessary files       |

---

## 🚀 Quick Start

```bash
git clone https://github.com/your-username/shallow-nn-diabetes.git
cd shallow-nn-diabetes
jupyter notebook play_with_shallow.ipynb
