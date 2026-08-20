# Iris Flower Classification — Multinomial Logistic Regression from Scratch

A two-stage machine learning pipeline for classifying Iris flowers into `Iris-setosa`,
`Iris-versicolor`, and `Iris-virginica`. Data preprocessing is done in **Python**
(NumPy/Pandas), and the classification model itself — multinomial logistic regression
with softmax and gradient descent — is implemented **from scratch in C++**, with no
ML libraries used anywhere in the pipeline.

---

## Project Structure

```
.
├── IRIS.csv                          # raw dataset (input)
├── iris_flower_classification.py     # preprocessing: cleaning, split, scaling
├── iris_train.csv                    # generated: scaled training set (120 rows)
├── iris_test.csv                     # generated: scaled testing set (30 rows)
├── test.cpp                          # core engine: logistic regression model
└── README.md
```

---

## Output

<img width="419" height="839" alt="image" src="https://github.com/user-attachments/assets/031c172d-5fee-469d-a292-e7048866336c" />


---


## 1. Data Preprocessing (Python — `iris_flower_classification.py`)

### 1.1 Exploratory Data Analysis
The raw dataset (`IRIS.csv`) was checked for null values, column data types, and
class balance (3 species, no missing values). Boxplots and histograms were used to
inspect the distribution and spread of each feature before deciding on outlier
treatment.

### 1.2 Label Encoding
The categorical `species` column is mapped to integers so it can be used in a
numeric model:

| Species          | Label |
|-------------------|:-----:|
| Iris-setosa        | 0     |
| Iris-versicolor     | 1     |
| Iris-virginica       | 2     |

### 1.3 Outlier Handling — IQR Capping
Rather than dropping outlier rows (which would shrink an already small dataset),
each feature is **capped** to the IQR bounds:

```
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
```

Any value outside `[lower_bound, upper_bound]` is clipped to the nearest bound
using `np.clip`, preserving all 150 rows while pulling in extreme values.

### 1.4 Train/Test Split — implemented from scratch
No `sklearn.model_selection` is used. `train_test_split_scratch()` shuffles row
indices with a fixed `random_state` (for reproducibility), then slices the first
80% as the training set and the remaining 20% as the test set:

- **Training set:** 120 rows
- **Test set:** 30 rows

### 1.5 Feature Scaling — Min-Max Normalization, implemented from scratch
`min_max_scale_scratch()` scales every feature to a `[0, 1]` range using the
**min and max of the training set only** (the test set is transformed with those
same bounds, never its own — this avoids data leakage):

```
X_scaled = (X - X_train_min) / (X_train_max - X_train_min)
```

The scaled train/test sets are written out as `iris_train.csv` and
`iris_test.csv`, which the C++ engine reads directly.

---

## 2. Core Model (C++ — `test.cpp`)

### 2.1 Why Multinomial Logistic Regression
Since there are 3 classes, this is a multi-class problem. Multinomial (a.k.a.
"softmax") logistic regression generalizes binary logistic regression by
producing a probability distribution over all 3 classes at once, instead of
one-vs-rest binary decisions.

### 2.2 Model Parameters
- **Weight matrix `wt`** — shape `4 × 3` (4 input features → 3 class scores)
- **Bias vector `b`** — shape `3 × 1` (one bias per class)

### 2.3 Algorithm

**Step 1 — Linear scores (forward pass)**
For every row `i` and class `j`, compute a raw score by taking a weighted sum of
the 4 features plus a bias term:

```
z[i][j] = Σ (X[i][k] * wt[k][j])  for k = 0..3     +   b[j]
```

**Step 2 — Softmax activation**
Convert the 3 raw scores per row into probabilities that sum to 1:

```
softmax(z[i][j]) = exp(z[i][j]) / Σ exp(z[i][c])   for c = 0..2
```

(The implementation subtracts the row max before exponentiating — this is a
standard numerical-stability trick and doesn't change the result, it just keeps
`exp()` from overflowing.)

**Step 3 — One-hot encode the true labels**
Each label (`0`, `1`, or `2`) is expanded into a length-3 vector, e.g. label `1`
becomes `[0, 1, 0]`, so it can be compared directly against the predicted
probability distribution.

**Step 4 — Loss: Categorical Cross-Entropy**
Measures how far the predicted probabilities are from the true one-hot labels:

```
loss = -(1/n) * Σ Σ y_true[i][j] * log(z[i][j])
```

**Step 5 — Backpropagation (gradient computation)**
Because softmax + cross-entropy has a clean combined derivative, the gradient
w.r.t. the raw scores simplifies to just the difference between predicted
probabilities and the true one-hot labels:

```
dz[i][j] = z[i][j] - y_true[i][j]

dW[k][j] = (1/n) * Σ X[i][k] * dz[i][j]     over all rows i
db[j]    = (1/n) * Σ dz[i][j]               over all rows i
```

**Step 6 — Gradient Descent Update**
Weights and biases are nudged in the direction that reduces the loss, scaled by
the learning rate `lr`:

```
wt[k][j] = wt[k][j] - lr * dW[k][j]
b[j]     = b[j]     - lr * db[j]
```

Steps 1–6 repeat once per epoch (batch gradient descent — the whole training set
is used for every update, not mini-batches).

**Step 7 — Prediction**
For a given row, the predicted class is whichever of the 3 softmax outputs is
largest (`argmax`):

```
prediction[i] = argmax(z[i][0], z[i][1], z[i][2])
```

**Step 8 — Accuracy**
```
accuracy = (correct predictions / total predictions) * 100
```

### 2.4 Class Design

| Component | Role |
|---|---|
| `operation()` | forward pass — computes raw scores `z` from `X` and current weights |
| `softmax()` | converts raw scores into class probabilities |
| `Y_OHE()` | one-hot encodes the integer labels |
| `crossEntropyLoss()` | computes training loss for a given epoch |
| `backward()` | computes gradients and updates `wt` / `b` |
| `train()` | runs `operation → softmax → backward` for `ep` epochs |
| `predict()` | runs a forward pass + softmax on unseen data (test set) using the trained weights |
| `y_pred()` / `getOutput()` | returns predicted labels via argmax |

---

## 3. How to Run

**Preprocessing:**
```bash
python iris_flower_classification.py
```
Produces `iris_train.csv` and `iris_test.csv` in the same folder.

**Training + Evaluation:**
```bash
g++ -std=c++17 -O2 -o iris_model test.cpp
./iris_model
```
Make sure `iris_train.csv` and `iris_test.csv` are in the same directory as the
executable (or your Code::Blocks project folder).

---

## 4. Results

| Metric | Value |
|---|---|
| Training rows | 120 |
| Testing rows | 30 |
| Features | 4 (sepal length/width, petal length/width) |
| Classes | 3 (setosa, versicolor, virginica) |
| Epochs | ~8,000 |
| Learning rate | 0.5 |
| Final training loss | ≈ 0.077 |
| **Training accuracy** | **98.33%** |
| **Testing accuracy** | **100%** |

The loss curve decreases smoothly and monotonically across epochs with no
divergence, and test accuracy holds at 100% with no sign of overfitting — a
reasonable result given how linearly separable two of the three Iris classes are.

---

## 5. Possible Improvements

- **Early stopping** — stop training automatically once the loss change between
  epochs drops below a small threshold, instead of a fixed epoch count.
- **L2 regularization** — penalize large weights to improve generalization on
  noisier datasets.
- **Mini-batch gradient descent** — process the data in smaller batches instead
  of the full training set every epoch, for scalability to larger datasets.
- **Learning rate decay** — reduce the learning rate over time for finer
  convergence in later epochs.
- **Confusion matrix / per-class precision-recall** — a single accuracy number
  hides which specific classes get confused with each other.

---

## Tech Stack
- **Python:** NumPy, Pandas, Matplotlib (EDA/preprocessing only — no scikit-learn)
- **C++:** Standard library only (`<vector>`, `<fstream>`, `<sstream>`, `<cmath>`) — no ML libraries
