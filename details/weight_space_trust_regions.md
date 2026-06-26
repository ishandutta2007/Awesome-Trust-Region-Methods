# Weight-Space Trust Regions

Weight-space trust regions constrain updates directly on the parameters (weights) of the model: $\|\Delta \theta\|_2 \le \Delta$. This approach represents the direct application of classical trust-region optimization to neural networks.

## Mathematical Formulation

The optimization step $\Delta \theta$ is obtained by solving:
$$\min_{\Delta \theta} L(\theta) + \nabla L(\theta)^T \Delta \theta + \frac{1}{2} \Delta \theta^T H \Delta \theta \quad \text{s.t.} \quad \|\Delta \theta\|_2 \le \Delta$$

In deep learning:
* **Pros:** Easy to compute geometrically.
* **Cons:** Extremely fragile because neural network parameter space is highly non-linear. A small change in a single critical weight can completely destroy the policy behavior, while large changes in other weights might have zero effect (the parameter-to-behavior mapping is not isometric).

## Geometric Intuition

```mermaid
graph LR
    subgraph Parameter Space
        A((Current Weights theta))
        B((Proposed Step))
        A -- Delta (L2 Norm Boundary) --> Circle(Trust Region Boundary)
    end
    Circle --> Mapping[Non-linear Neural Network Map]
    subgraph Action Space / Behavior
        Mapping --> Collapse([Potential Behavioral Collapse])
    end
```

[Back to README](../README.md)
