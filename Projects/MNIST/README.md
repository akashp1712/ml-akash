---
layout: default
nav_exclude: true
---

# Classification Problem
### classifying hand written digits images
----
## MNIST - small images of handwritten digits

## Data
**Instances**: 70,000 <br/>
**Features**: 784 features (28 x 28)


## Training (3 cross-fold validation)

### 1. SVM classifier :: OvO strategy(default)
cross_val_score [0.977  0.9738 0.9739]

### 2. SVM classifier :: OvR strategy
cross_val_score [0.97685 0.9738  0.97495]

### 3. SGDClassifier
cross_val_score [0.87365 0.85835 0.8689]

### 4. RandomForestClassifier
cross_val_score [0.9646  0.96255 0.9666]

### 5. KNeighborsClassifier
cross_val_score [0.9676  0.9671  0.96755]

----

#### Note: Know that cross_val_score takes a lot of time as it trains the multiple models underhood and test especially In the case od SVM OvO and SVM OvR.

