
```python
import jax
import jax.numpy as jnp
import pennylane as qml
import optax
from typing import Tuple

# Ensure 64-bit precision for stable quantum gradients
jax.config.update("jax_enable_x64", True)

class CoherentSuperManifold:
    """Manages the State-Vector continuous relaxation and Stochastic Soft-Rounding (SSR)."""
    def __init__(self, max_depth: int, sigma_0: float = 0.5, gamma: float = 0.98):
        self.max_depth = max_depth
        self.depth_array = jnp.arange(1, max_depth + 1)
        self.sigma_0 = sigma_0
        self.gamma = gamma

    def get_weights(self, delta: jnp.ndarray, epoch: int, use_ste: bool) -> jnp.ndarray:
        if use_ste:
            # Straight-Through Estimator for exact discrete rounding in final epochs
            discrete_delta = jnp.round(delta)
            delta_ste = delta + jax.lax.stop_gradient(discrete_delta - delta)
            w_mask = jnp.where(self.depth_array == discrete_delta, 1.0, 0.0)
            # Attach gradient to the hard-mask via STE trick
            return w_mask + (delta_ste * 0.0) 
        else:
            # Annealed Gaussian Soft-Mask
            sigma_t = self.sigma_0 * (self.gamma ** epoch)
            w_mask = jnp.exp(-((self.depth_array - delta) ** 2) / (2 * sigma_t ** 2))
            return w_mask / jnp.sum(w_mask)

class CSDARTSOptimizer:
    def __init__(self, num_wires: int, max_depth: int, complexity_penalty: float = 0.1): 
        self.num_wires = num_wires
        self.max_depth = max_depth
        self.penalty = complexity_penalty
        self.dev = qml.device('default.qubit', wires=num_wires)
        self.manifold = CoherentSuperManifold(max_depth)
        
        # Build Hamiltonian Matrix explicitly for fast classical JAX vector-math
        coeffs, obs = [], []
        for i in range(num_wires - 1): 
            coeffs.extend([1.0, 1.0, 1.0])
            obs.extend([qml.PauliX(i) @ qml.PauliX(i+1), 
                        qml.PauliY(i) @ qml.PauliY(i+1), 
                        qml.PauliZ(i) @ qml.PauliZ(i+1)])
        self.H = qml.Hamiltonian(coeffs, obs)
        self.H_matrix = jnp.array(qml.matrix(self.H))

        self._build_state_qnodes()

    def _build_state_qnodes(self):
        """Builds QNodes that output complex state-vectors rather than scalars."""
        self.circuits = []
        for depth in range(1, self.max_depth + 1):
            @qml.qnode(self.dev, interface="jax")
            def _circuit(w_slice, d=depth):
                for w in range(self.num_wires): qml.Hadamard(wires=w)
                for i in range(d):
                    # Basic Hardware-Efficient Entanglement Block
                    for w in range(self.num_wires):
                        qml.RY(w_slice[i, w], wires=w)
                    for w in range(self.num_wires - 1):
                        qml.CNOT(wires=[w, w+1])
                return qml.state()
            self.circuits.append(_circuit)

    def coherent_loss(self, params: Tuple[jnp.ndarray, jnp.ndarray], epoch: int) -> jnp.ndarray:
        theta, delta = params
        use_ste = epoch > 100 # Activate SSR in the final 50 epochs
        
        # 1. Fetch State Vectors
        states = jnp.stack([circ(theta) for circ in self.circuits])
        
        # 2. Get Manifold Weights
        w = self.manifold.get_weights(delta, epoch, use_ste)
        
        # 3. Superpose & Normalize (The CS-DARTS Core)
        super_state = jnp.sum(w[:, None] * states, axis=0)
        norm = jnp.linalg.norm(super_state)
        super_state = super_state / norm
        
        # 4. Global Expectation & Penalty
        energy = jnp.vdot(super_state, jnp.dot(self.H_matrix, super_state)).real
        return energy + (self.penalty * delta)

    def fit(self, epochs: int = 150):
        key = jax.random.PRNGKey(42)
        theta = jax.random.normal(key, shape=(self.max_depth, self.num_wires)) * 0.1
        delta = jnp.array(float(self.max_depth)) # Start deep, let the penalty prune
        
        optimizer = optax.adam(0.05)
        opt_state = optimizer.init((theta, delta))
        
        @jax.jit
        def update_step(p, o_state, ep):
            loss_val, grads = jax.value_and_grad(self.coherent_loss)(p, ep)
            updates, new_o_state = optimizer.update(grads, o_state, p)
            new_p = optax.apply_updates(p, updates)
            # Clip delta between 1 and max_depth
            return (new_p[0], jnp.clip(new_p[1], 1.0, float(self.max_depth))), new_o_state, loss_val

        params = (theta, delta)
        for epoch in range(1, epochs + 1):
            params, opt_state, loss = update_step(params, opt_state, epoch)
            if epoch % 25 == 0:
                print(f"Epoch {epoch:03} | Loss: {loss:.4f} | Topology Depth Δ: {params[1]:.4f}")
        
        return params

if __name__ == "__main__":
    NUM_WIRES, MAX_DEPTH = 6, 8
    print("--- Starting CS-DARTS Pre-Compilation ---")
    opt = CSDARTSOptimizer(num_wires=NUM_WIRES, max_depth=MAX_DEPTH)
    final_theta, final_delta = opt.fit(epochs=150)
    print(f"\nFinal Discretized Topology Depth: {int(jnp.round(final_delta).item())}")
