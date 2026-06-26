# The Classical Numerical Optimization Era (Pre-Deep Learning)

Trust-region methods originated in classical numerical optimization to solve non-linear least-squares and general non-linear unconstrained optimization problems. The core idea is to approximate the complex objective function locally with a quadratic model and define a region (the "trust region") around the current point where this model is expected to be an accurate representation.

## Mathematical Formulation

At iteration $k$, we solve the subproblem:
$$\min_{p} m_k(p) = f(x_k) + g_k^T p + \frac{1}{2} p^T B_k p \quad \text{s.t.} \quad \|p\| \le \Delta_k$$

Where:
* $g_k = \nabla f(x_k)$ is the gradient.
* $B_k$ is the Hessian $\nabla^2 f(x_k)$ or an approximation.
* $\Delta_k$ is the trust-region radius.

The trust-region radius $\Delta_k$ is updated based on the ratio of actual reduction to predicted reduction:
$$\rho_k = \frac{f(x_k) - f(x_k + p_k)}{m_k(0) - m_k(p_k)}$$

* If $\rho_k$ is close to 1, the model is accurate; we accept the step and may increase $\Delta_k$.
* If $\rho_k$ is close to 0 or negative, the model is inaccurate; we reject the step and shrink $\Delta_k$.

## Algorithmic Workflow

```mermaid
graph TD
    Start([Start Iteration k]) --> BuildModel[Build Quadratic Model m_k]
    BuildModel --> SolveSub[Solve Subproblem for step p_k s.t. ||p_k|| <= Delta_k]
    SolveSub --> CalcRatio[Compute ratio rho_k of Actual vs Predicted reduction]
    CalcRatio --> CheckRatio{Is rho_k > threshold?}
    CheckRatio -- No --> Shrink[Shrink Delta_k] --> SolveSub
    CheckRatio -- Yes --> Accept[Accept Step: x_{k+1} = x_k + p_k]
    Accept --> UpdateRadius{Is rho_k very high?}
    UpdateRadius -- Yes --> Expand[Expand Delta_{k+1}] --> Next([Next Iteration])
    UpdateRadius -- No --> Keep[Keep Delta_{k+1} = Delta_k] --> Next
```

[Back to README](../README.md)
