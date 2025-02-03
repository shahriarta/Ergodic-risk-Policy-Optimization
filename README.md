# Ergodic-Risk Constrained Policy Optimization: The Linear Quadratic Case

## Overview
This repository contains the implementation of **Ergodic-Risk Constrained Policy Optimization** for Linear Quadratic Regulation (LQR). The approach introduces an **ergodic-risk criterion** to capture long-term cumulative risk in stochastic control systems, particularly when process noise exhibits heavy-tailed distributions.

The project extends classical LQR by incorporating **ergodic-risk constraints** and developing a **primal-dual policy optimization method** to balance performance with robustness against rare but large fluctuations in process noise.

## Key Features
- Introduces **ergodic-risk criteria** to capture long-term risk sensitivity.
- Ensures risk-aware control by constraining an **asymptotic variance measure**.
- Provides **primal-dual optimization** to compute an optimal risk-sensitive LQR policy.
- Demonstrates robustness against **heavy-tailed noise distributions**.
- Implements the **Functional Central Limit Theorem (F-CLT)** to analyze long-term risk behavior.

## Problem Formulation
We consider a discrete-time **stochastic linear system** with process noise:
\[
x_{t+1} = A x_t + B u_t + H w_{t+1}, \quad t \geq 0
\]
where \( x_t \) and \( u_t \) are the state and input vectors, and \( w_t \) represents i.i.d. process noise with potentially heavy tails.

The standard LQR cost is given by:
\[
J(K) = \limsup_{T\to\infty} \frac{1}{T} \sum_{t=0}^{T} \mathbb{E}[ x_t^\top Q x_t + u_t^\top R u_t]
\]
where \( Q \geq 0 \) and \( R > 0 \) define the quadratic performance metric.

### Ergodic-Risk Constraints
The **ergodic-risk criterion** is formulated to measure long-term risk using the **cumulative uncertainty variable**:
\[
C_t = g(x_t, u_t) - \mathbb{E}[ g(x_t, u_t) | \mathcal{F}_{t-1}]
\]
where \( g(x_t, u_t) \) is a risk functional. The objective is to optimize performance while ensuring:
\[
\gamma_N^2(K) \leq \bar\beta
\]
where \( \gamma_N^2(K) \) is the **asymptotic conditional variance** of the ergodic-risk criterion.

## Implementation Details
- **Primal-Dual Algorithm**: Solves the constrained policy optimization problem using gradient-based updates.
- **Policy Optimization**: The optimal policy \( K^* \) is found by iteratively updating the controller gain using:
  \[
  K^*(\lambda) = -(R + B^\top P_{(K^*(\lambda),\lambda)} B)^{-1} B^\top P_{(K^*(\lambda),\lambda)} A
  \]
  where \( P_{(K,\lambda)} \) satisfies a modified **Lyapunov equation**.
- **Strong Duality**: Ensures convergence via complementary slackness in the dual problem.

## Usage
### Requirements
- Python 3.x
- NumPy
- SciPy
- Matplotlib

## Running the Simulations
### Simulation 1: It is the simulation of Grumman X29 aircraft in the Normal-Digital Powered-Approach mode.
```bash
python main.py 
```
### Simulation 2: It is the simulation of 100 randomly sampled systems and illustrates the convergence of the proposed Ergodic-risk primal-dual algorithm.


### Example Simulation
The repository includes an **X-29 aircraft control example** demonstrating the resilience of the ergodic-risk optimal policy under heavy-tailed noise.

## Results
- The ergodic-risk optimal policy shows **greater robustness** against process noise compared to classical LQR.
- The algorithm effectively balances **performance** and **long-term risk sensitivity**.

## References
- S. Talebi and N. Li, “Uniform Ergodicity and Ergodic-Risk Constrained Policy Optimization,” Sept. 2024. arXiv:5856500.
- S. Talebi and N. Li, "Ergodic-Risk Constrained Policy Optimization: The Linear Quadratic Case." 2025. American Control Conference.

## Acknowledgments
This work is supported by **NSF AI Institute 2112085**.

## License
This project is licensed under the terms of the GNU General Public License v3.0 (GPL-3.0).

