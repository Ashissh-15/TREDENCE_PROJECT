# README.md

# Self-Pruning Neural Network for CIFAR-10 Classification

## Case Study Submission

This project implements a **Self-Pruning Neural Network** using learnable gate mechanisms for automatic weight pruning during training. The goal is to reduce unnecessary network connections while maintaining strong classification performance.

Instead of performing pruning after training, this approach allows the network to **learn which connections are important and which can be removed during training itself**.

The project was implemented using **PyTorch** on the **CIFAR-10 dataset**, and focuses on analyzing the trade-off between:

* Model Accuracy
* Network Sparsity
* Pruning Strength (controlled using λ)

---

# Problem Statement

Traditional neural networks are often over-parameterized, containing many unnecessary weights that do not significantly contribute to prediction performance.

These redundant connections:

* increase memory usage
* increase computation cost
* slow down inference
* reduce deployment efficiency

The objective of this case study is to build a neural network that can:

1. Automatically identify unimportant connections
2. Suppress or remove them during training
3. Maintain competitive classification accuracy
4. Improve model efficiency through sparsity

This is achieved using **learnable gates + L1 regularization**.

---

# Core Idea

Each weight in the network is assigned a learnable gate score.

Instead of using the original weight directly:

[
W
]

we use:

[
W' = W \cdot \sigma(S)
]

Where:

* (W) = original weight
* (S) = learnable gate score
* (\sigma(S)) = sigmoid activation producing gate values between 0 and 1
* (W') = effective pruned weight

### Interpretation

* Gate near **1** → important connection → keep
* Gate near **0** → unimportant connection → prune

This allows pruning to happen automatically during training.

---

# Why L1 Regularization?

To encourage pruning, an additional sparsity loss is introduced.

## Total Loss Function

[
L = L_{cls} + \lambda \sum g
]

Where:

* (L_{cls}) = classification loss (CrossEntropyLoss)
* (\sum g) = sum of all gate values
* (\lambda) = sparsity control parameter

### Why L1?

L1 regularization pushes values toward exact zero, making it ideal for pruning.

This creates a balance between:

* maintaining accuracy
* reducing unnecessary connections

---

# Dataset Used

## CIFAR-10

CIFAR-10 is a standard image classification dataset consisting of:

* 60,000 color images
* 10 classes
* Image size: 32 × 32 × 3

### Classes

* airplane
* automobile
* bird
* cat
* deer
* dog
* frog
* horse
* ship
* truck

### Split

* Training Images: 50,000
* Testing Images: 10,000

---

# Model Architecture

A fully connected feed-forward neural network was used.

## Architecture

Input Image (32×32×3)
↓
Flatten
↓
PrunableLinear (3072 → 512)
↓
BatchNorm
↓
ReLU
↓
PrunableLinear (512 → 256)
↓
BatchNorm
↓
ReLU
↓
PrunableLinear (256 → 10)
↓
Output Layer

---

# Important Improvement: Gate Initialization

Initially, gate scores were randomly initialized:

```python
torch.randn(...)
```

This caused most gates to start near:

[
\sigma(0) = 0.5
]

which made pruning difficult.

To improve pruning behavior, gate scores were initialized using:

```python
-1.5
```

or

```python
-2.0
```

because:

[
\sigma(-2) \approx 0.12
]

This allowed the model to begin with weaker connections and made pruning much more effective.

This significantly improved sparsity results.

---

# Hyperparameter Experiments

The following parameters were tested:

## Learning Rate

```python
1e-3
```

## Gate Initialization

```python
-1.5
-2.0
```

## Lambda Values

```python
1e-4
1e-3
1e-2
```

## Epochs

```python
5
10
```

These experiments were used to study the sparsity–accuracy trade-off.

---

# Evaluation Metrics

## 1. Accuracy

Measured using classification accuracy on the CIFAR-10 test dataset.

## 2. Sparsity

A connection is considered pruned if:

[
gate < 0.01
]

Sparsity is calculated as:

[
\text{Sparsity (%)} =
\frac{\text{Pruned Gates}}
{\text{Total Gates}}
\times 100
]

---

# Key Results

## Balanced Model

### Configuration

* Learning Rate = 0.001
* Gate Initialization = -1.5
* Lambda = 1e-4
* Epochs = 10

### Performance

* Accuracy = **56.93%**
* Sparsity = **52.07%**

This provided the best practical trade-off between accuracy and pruning.

---

## Aggressive Pruning Model

### Configuration

* Learning Rate = 0.001
* Gate Initialization = -2.0
* Lambda = 1e-3
* Epochs = 10

### Performance

* Accuracy ≈ **56%**
* Sparsity ≈ **97%**

This proved that the network was highly over-parameterized and could retain strong performance even after extreme pruning.

---

# Gate Distribution Analysis

Histogram analysis showed:

## Early Training (5 epochs)

Most gate values remained above the pruning threshold:

```text
0.015 → 0.03
```

Result:

* Accuracy remained good
* Sparsity ≈ 0%

This indicated that pruning had started, but was not yet strong enough.

---

## Later Training (10 epochs)

A large portion of gates shifted below the threshold:

```text
gate < 0.01
```

Result:

* Sparsity increased significantly
* Accuracy improved further

This demonstrated that pruning develops gradually over training.

---

# Key Observations

## 1. More Epochs = More Pruning

Pruning required sufficient training time.

5 epochs often produced near-zero sparsity, while 10 epochs produced strong pruning.

---

## 2. Lambda Controls Pruning Strength

Higher λ produced:

* higher sparsity
* slightly lower accuracy

This confirmed correct pruning behavior.

---

## 3. Gate Initialization Matters

Negative initialization improved pruning significantly.

This was one of the most important findings.

---

## 4. Accuracy Was Preserved

Even with high sparsity, the model maintained strong classification accuracy.

This proved that many original connections were redundant.

---

# Conclusion

This project successfully implemented a **Self-Pruning Neural Network** using learnable gates and L1 regularization.

The model was able to:

* identify unnecessary connections
* prune them automatically during training
* maintain strong classification performance
* significantly reduce network complexity

The best balanced model achieved:

* **56.93% Accuracy**
* **52.07% Sparsity**

while the aggressive model demonstrated nearly:

* **97% Sparsity**

with minimal accuracy degradation.

This confirms that self-pruning is an effective strategy for improving model efficiency without major performance loss.

The project demonstrates strong practical applicability for deploying efficient neural networks in real-world systems where memory and computation constraints are important.

---

# Technologies Used

* Python
* PyTorch
* NumPy
* Matplotlib
* Kaggle Notebook
* CIFAR-10 Dataset

---

# Author

Case Study Submission for Recruitment Evaluation

Self-Pruning Neural Network Implementation
