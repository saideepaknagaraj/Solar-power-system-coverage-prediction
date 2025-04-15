
# ☀️ Solar Power System Coverage Prediction

## 📘 Overview

As the global demand for renewable energy rises, solar power has emerged as a sustainable and impactful solution. This project explores the effectiveness of various machine learning algorithms in predicting the coverage of solar power systems across U.S. tiles, leveraging features from the DeepSolar dataset.

## 🎯 Objective

To evaluate and compare three supervised learning models — **Logistic Regression**, **Decision Tree**, and **Random Forest** — in classifying solar power system coverage as *high* or *low*, and determine the optimal hyperparameters for best performance.

---

## 📊 Dataset

- 📦 **Source**: DeepSolar database (subset)
- 📈 **Rows**: 10,926
- 🧾 **Features**: 15, including:
  - Average household income
  - Employment rate
  - Population density
  - Housing characteristics
  - Heating fuel ratios
  - Land/water area
  - Solar radiation and temperature metrics
- 🎯 **Target**: Binary variable `solar_system_coverage` (high = 1, low = 0)

---

## 🧰 Prerequisites

Ensure R is installed with the following libraries:

```R
install.packages(c("ROCR", "rpart", "partykit", "rpart.plot", "caret", "randomForest"))
```

Place the dataset file `data_hw3_deepsolar.RData` in the same directory as your script.

---

## ⚙️ Execution Workflow

1. Load and preprocess the dataset
2. Convert target variable to binary
3. Split into training (80%) and test (20%) sets
4. Train models using k-fold cross-validation
5. Evaluate performance via:
   - Accuracy
   - Sensitivity
   - Specificity
   - ROC Curve & AUC
6. Hyperparameter tuning for `cp`, `mtry`, `ntree`

---

## 🧠 Models

### 🔹 Logistic Regression
- Built using `glm()` with `family=binomial`
- Evaluated at both default threshold (0.5) and optimal `tau=0.10`
- ROC-AUC used for threshold selection
- Achieved:
  - Accuracy: **92.64%**
  - Sensitivity: **71.52%**

---

### 🔸 Classification Tree
- Implemented via `rpart()`
- Cross-validated over 101 `cp` values (0–0.1)
- Best model found at `cp = 0.014`
- Achieved:
  - Accuracy: **93.8%**
  - Sensitivity: **82.19%**

---

### 🟩 Random Forest
- Built using `randomForest()` with importance = TRUE
- Evaluated over 20 combinations of `mtry` and `ntree`
- Best model:
  - `mtry = 3`, `ntree = 125`
- Achieved:
  - Accuracy: **95%** (validation), **54%** (test)
  - Sensitivity: **86.7%** (validation)
  - Specificity: **95.5%**

---

## 📈 Results

| Model               | Accuracy | Sensitivity | Specificity |
|--------------------|----------|-------------|-------------|
| Logistic Regression| 92.64%   | 71.52%      | N/A         |
| Decision Tree      | 93.8%    | 82.19%      | N/A         |
| Random Forest      | **95%**  | **86.7%**   | **95.5%**   |

🏆 **Random Forest** achieved the best performance overall.

---

## ✅ Conclusion

This study demonstrates that Random Forests offer superior performance in predicting high solar coverage areas based on socioeconomic, environmental, and geographical features. This model can be useful for solar energy companies, policy makers, and researchers in optimizing solar infrastructure deployment.

---

## 👨‍💻 Author

*Developed as part of a machine learning analysis project using R.*

---

## 📎 License

MIT License
