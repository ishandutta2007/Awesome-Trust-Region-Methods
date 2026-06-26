# Distribution-Space Trust Regions

Distribution-space trust regions define boundaries in the probability distribution space of the model outputs rather than the weight space. This guarantees that the step size corresponds directly to changes in model behavior, preventing policy collapse in reinforcement learning.

## Mathematical Formulation

Instead of a parameter norm constraint, we constrain the expected KL divergence:
$$\mathbb{E}_{s} \left[ D_{KL}(\pi_{\theta}(\cdot|s) \parallel \pi_{\theta + \Delta \theta}(\cdot|s)) \right] \le \delta$$

This ensures that the update steps are scaled based on their actual behavioral impact. For example, if a weight change of 0.01 changes the output distribution by a KL of 5.0, the step is scaled down. If it changes the distribution by 0.0001, the step is allowed to be larger.

## Contrast: Weight Space vs. Distribution Space

```mermaid
graph TD
    subgraph Weight Space Constraint
        W1[Small Step in Weights] -->|Highly Sensitive Area| B1[Huge Shift in Output Distribution]
        W2[Large Step in Weights] -->|Insensitive Area| B2[Tiny Shift in Output Distribution]
    end
    subgraph Distribution Space Constraint
        D1[Scale weight steps adaptively] -->|Enforces| B3[Bounded Shift in Output Distribution <= delta]
    end
```

[Back to README](../README.md)
