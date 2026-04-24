# Self-Pruning Neural Network for CIFAR-10 Classification

## Case Study Submission

This project implements a **Self-Pruning Neural Network** using learnable gate mechanisms for automatic weight pruning during training. The goal is to reduce unnecessary network connections while maintaining strong classification performance.

Instead of performing pruning after training, this approach allows the network to **learn which connections are important and which can be removed during training itself**.

The project was implemented using **PyTorch** on the **CIFAR-10 dataset**, and focuses on analyzing the trade-off between:

- Model Accuracy  
- Network Sparsity  
- Pruning Strength (controlled using λ)

---

## Architecture

```text
Input Image (32 × 32 × 3)
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
```

---

# Why L1 Regularization?

To encourage pruning, an additional sparsity loss is introduced.

## Total Loss Function

\[
L = L_{cls} + \lambda \sum g
\]

Where:

- \(L_{cls}\) = classification loss (CrossEntropyLoss)  
- \(\lambda\) = sparsity control parameter  
- \(\sum g\) = total gate activation values

This regularization encourages gates to move toward zero, helping the model automatically prune unnecessary connections.

## Example

\[
\sigma(-2) \approx 0.12
\]

This means initializing gates with negative values like -1.5 or -2.0 starts the model with weaker connections, making pruning much more effective.

---

# Experimental Results with Gate Distribution Histograms

Multiple experiments were performed using different gate initialization values, lambda values, and training epochs to analyze pruning behavior.

The pruning threshold used was:

\[
\text{gate} < 0.01
\]

Any gate value below this threshold is considered pruned.

---

## Best Balanced Model

### Configuration

- Learning Rate = 0.001
- Gate Initialization = -1.5
- Lambda = 0.0001
- Epochs = 10

### Performance

- Accuracy = **56.93%**
- Sparsity = **52.07%**

This provided the best balance between model accuracy and pruning efficiency.

---

## Most Aggressive Pruning Model

### Configuration

- Learning Rate = 0.001
- Gate Initialization = -2.0
- Lambda = 0.001
- Epochs = 10

### Performance

- Accuracy = **56.15%**
- Sparsity = **97.74%**

This demonstrated that the network could retain strong performance even after extreme pruning.

---

# Conclusion

This project successfully implemented a **Self-Pruning Neural Network** using learnable gates and L1 regularization.

The model was able to:

- identify unnecessary connections  
- prune them automatically during training  
- maintain strong classification performance  
- significantly reduce network complexity

This confirms that self-pruning is an effective strategy for improving model efficiency without major performance loss.

---

# Technologies Used

- Python  
- PyTorch  
- NumPy  
- Matplotlib  
- Kaggle Notebook  
- CIFAR-10 Dataset
