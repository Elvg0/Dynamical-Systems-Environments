# Dynamical-Systems-Environments
Various dynamical systems environments for designing, training and testing control and reinforcement learning algorithms.

# Cruise Control

Given the first order differential equation descibing the velocity of some vehicle on a flat surface:

$$m\dot{v}+cv=F$$

Where $v$ is the velocity of the vehicle, $m$ the mass, $c$ the momentum loss due to air resistance or surface friction, and $F$ the force generated by the engine. The simplified control system, where the control is given by $u$, is given by the equation:

$$\dot{v}+cv=u$$

The default value of $c$ is 0.05. Being $v_0$ the target velocity (10 by default), the reward function if given by:

$$r(v) = -\||v-v_0\||^2_2 + 100_{v=v_0}$$

The system is simulated using Euler's method with timestep of 0.05 by default. Each episode is 100 timesteps. The observation and action space are $\mathbb{R}$.

<p align="center">
  <img src="cargif.gif" alt="animated" />
</p>
