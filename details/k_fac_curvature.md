# Kronecker-Factored Approximate Curvature (K-FAC)

K-FAC is an optimization method that approximates the Fisher Information Matrix (FIM) to perform natural gradient descent efficiently. It breaks down the FIM into Kronecker products of smaller matrices, avoiding the expensive $O(N^3)$ matrix inversion bottleneck.

## Mathematical Formulation

For a layer with input $a_{l-1}$ and pre-activation gradients $g_l$, the Fisher block $F_l$ is:
$$F_l = \mathbb{E} \left[ \text{vec}(\nabla_{W_l} L) \text{vec}(\nabla_{W_l} L)^T \right] = \mathbb{E} \left[ (a_{l-1} a_{l-1}^T) \otimes (g_l g_l^T) \right]$$

K-FAC approximates this expected tensor product as the Kronecker product of expectations:
$$F_l \approx \mathbb{E} \left[ a_{l-1} a_{l-1}^T \right] \otimes \mathbb{E} \left[ g_l g_l^T \right] = A_{l-1} \otimes G_l$$

This allows the inverse to be computed efficiently since:
$$F_l^{-1} \approx A_{l-1}^{-1} \otimes G_l^{-1}$$

This reduces the inversion complexity from $O(d_{in}^3 d_{out}^3)$ to $O(d_{in}^3 + d_{out}^3)$.

## Approximation Pipeline

```mermaid
graph LR
    Grad[Compute Gradients] --> Activations[Extract Activations A_{l-1}]
    Grad --> PreGrads[Extract Pre-activation Gradients G_l]
    Activations --> CovA[Compute Covariance Matrix E[a a^T]]
    PreGrads --> CovG[Compute Covariance Matrix E[g g^T]]
    CovA --> InvA[Invert CovA]
    CovG --> InvG[Invert CovG]
    InvA --> Kronecker[Kronecker Product of Inverses]
    InvG --> Kronecker
    Kronecker --> Step[Apply Curvature-Corrected Step]
```

[Back to README](../README.md)
