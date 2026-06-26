# The Statistical Distance Revolution (TRPO)

Trust Region Policy Optimization (TRPO) brought the mathematical rigor of classical trust-region methods to Deep Reinforcement Learning. Instead of constraining weight changes, TRPO constrains the change in the policy's action distribution using Kullback-Leibler (KL) divergence, ensuring stable updates even with complex function approximators (neural networks).

## Mathematical Formulation

TRPO solves the following constrained optimization problem at each step:
$$\max_{\theta} \mathbb{E}_{s \sim \rho_{\theta_{old}}, a \sim \pi_{\theta_{old}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A_{\theta_{old}}(s, a) \right]$$
$$\text{subject to } \mathbb{E}_{s \sim \rho_{\theta_{old}}} \left[ D_{KL}(\pi_{\theta_{old}}(\cdot|s) \parallel \pi_\theta(\cdot|s)) \right] \le \delta$$

Where $A(s, a)$ is the advantage function. Because of the complex KL constraint, TRPO uses a second-order approximation:
* The objective is approximated by the gradient $g$.
* The constraint is approximated by the Fisher Information Matrix (FIM) $H$.

This yields a search direction:
$$s \approx H^{-1} g$$

Using the Conjugate Gradient algorithm, TRPO computes $H^{-1} g$ without explicitly forming the massive matrix $H$.

## Algorithmic Workflow

```mermaid
sequenceDiagram
    participant Env as Environment
    participant Agent as TRPO Agent
    participant Opt as Optimizer
    Env->>Agent: Rollout trajectories (s, a, r)
    Agent->>Agent: Estimate Advantage A(s, a)
    Agent->>Opt: Compute policy gradient g
    Opt->>Opt: Conjugate Gradient to find search direction s = H^-1 * g
    Opt->>Opt: Perform Line Search to satisfy KL constraint <= delta
    Opt->>Agent: Update weights theta_new = theta_old + alpha * s
    Agent->>Env: Interact with updated policy
```

[Back to README](../README.md)
