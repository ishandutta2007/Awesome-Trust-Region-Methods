# Autonomous Robotic Locomotion & Control Stacks

In quadruped and humanoid robotics, trust-region constraints are essential to ensure the safety and physical stability of the robot during learning. Large policy updates can output unstable torque commands, causing violent movements that damage the mechanical hardware.

## Safety Constraints

The control stack maps sensor observations (joint positions, IMU readings) to joint torque targets:
$$\tau_t = \pi_\theta(s_t)$$

Applying trust-region constraints (TRPO/PPO) ensures that the joint velocities and torques change smoothly between policy updates:
$$D_{KL}(\pi_{\theta_{old}}(\cdot|s) \parallel \pi_\theta(\cdot|s)) \le \delta$$

This mathematical guarantee prevents joint velocity spikes, ensuring smooth transitions and stable dynamic balance.

## Control Stack Architecture

```mermaid
graph TD
    Sensors[IMU, Joint Encoders] --> State[State Estimation]
    State --> Policy[PPO Control Policy]
    Policy --> |Torque Command tau| Motors[Joint Actuators]
    Motors --> Env[Robot State Transition]
    Env --> Sensors
    Policy -->|Update step constrained by| TR[Trust-Region Boundary]
```

[Back to README](../README.md)
