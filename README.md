# CS-DARTS: Coherent State-Level Quantum Architecture Search

CS-DARTS is a fully differentiable, single-pass meta-optimizer for discovering the optimal topological depth of Variational Quantum Algorithms (VQAs). 

By resolving phase-loss through **Coherent State-Vector Interpolation** and closing the continuous-to-discrete gap via **Stochastic Soft-Rounding (SSR)**, CS-DARTS operates flawlessly within classical $\mathcal{O}(2^N)$ memory limits. It serves as an uncompromised pre-compilation engine to extract maximally efficient, noise-resilient hardware topologies prior to QPU deployment.

## Core Innovations
1. **The Quantum Super-State:** Instead of interpolating final energy scalars (which destroys quantum phase and cross-terms), CS-DARTS interpolates pure wavefunctions using an Annealed Gaussian Soft-Mask. This preserves physical interference in the gradient landscape.
2. **$\mathcal{O}(2^N)$ Memory Scaling:** Bypasses the $\mathcal{O}(4^N)$ exponential memory explosions that crash standard mixed-state density matrix solvers (like $\rho$DARTS).
3. **Stochastic Soft-Rounding (SSR):** Utilizes a Straight-Through Estimator (STE) in the final training epochs. This enforces a strict integer depth dynamically during optimization, yielding a perfectly rounded discrete circuit in a *single pass* without the need for post-search fine-tuning.

## Installation

This framework is built natively on Google JAX and PennyLane for accelerated XLA-compilation.

```bash
pip install pennylane jax jaxlib optax

