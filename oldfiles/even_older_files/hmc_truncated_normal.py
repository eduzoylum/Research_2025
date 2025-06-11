import numpy as np
import multiprocessing as mp

class HMCTruncatedNormal:
    def __init__(self, mean, cov, lower=None, upper=None, x0=None, step_size=None, num_steps=20):
        """
        HMC sampler for truncated multivariate normal.

        Parameters:
        - mean (np.array): Mean vector of the normal distribution.
        - cov (np.array): Covariance matrix.
        - lower (np.array or None): Lower bounds (None means -inf).
        - upper (np.array or None): Upper bounds (None means +inf).
        - x0 (np.array or None): Starting point (None defaults to reflected mean if out of bounds).
        - step_size (float or None): HMC step size (None defaults to automatic scaling).
        - num_steps (int): Number of leapfrog steps per proposal.
        """
        self.mean = np.asarray(mean)
        self.cov = np.asarray(cov)
        self.dim = self.mean.size

        # Ensure at least one bound is provided
        if lower is None and upper is None:
            raise ValueError("At least one of `lower` or `upper` must be provided.")

        # Convert lower and upper bounds to arrays, setting defaults if necessary
        self.lower = np.full(self.dim, -np.inf) if lower is None else np.asarray(lower)
        self.upper = np.full(self.dim, np.inf) if upper is None else np.asarray(upper)

        # Validate dimensions
        if self.lower.shape != (self.dim,) or self.upper.shape != (self.dim,):
            raise ValueError("`lower` and `upper` bounds must have the same shape as `mean`.")

        # Compute precision matrix (inverse covariance) for efficiency
        self.precision = np.linalg.inv(self.cov)

        # Set default step size if not provided (scaled to problem dimension)
        self.step_size = step_size if step_size is not None else 0.1 * (1.0 / np.sqrt(self.dim))
        self.num_steps = num_steps

        # Initialize starting point, reflecting if necessary
        if x0 is None:
            self.x0 = self.mean.copy()
        else:
            self.x0 = np.asarray(x0)

        # Ensure starting point is within bounds
        if np.any(self.x0 < self.lower) or np.any(self.x0 > self.upper):
            self.x0 = self._reflect_to_bounds

    def _reflect_to_bounds(self, x):
        """Reflects x into the valid region if it is outside the bounds."""
        x_reflected = x.copy()
        for i in range(self.dim):
            if x_reflected[i] < self.lower[i]:
                x_reflected[i] = 2 * self.lower[i] - x_reflected[i]  # Reflect back
            elif x_reflected[i] > self.upper[i]:
                x_reflected[i] = 2 * self.upper[i] - x_reflected[i]
        return x_reflected

    def _potential_energy(self, x):
        """Computes potential energy U(x) = -log P(x) (ignoring constants)."""
        diff = x - self.mean
        return 0.5 * diff.dot(self.precision).dot(diff)

    def _gradient_potential(self, x):
        """Computes the gradient of the log-density: ∇ log P(x) = -Σ⁻¹(x - μ)."""
        return -self.precision.dot(x - self.mean)

    def _hmc_step(self, current_x):
        """Performs one HMC step with reflection at boundaries."""
        p = np.random.normal(size=self.dim)  # Resample momentum at every step
        current_p = p.copy()
        current_H = self._potential_energy(current_x) + 0.5 * np.dot(current_p, current_p)

        x_proposed = current_x.copy()
        p_proposed = current_p.copy()

        # Half-step momentum update
        p_proposed += 0.5 * self.step_size * self._gradient_potential(x_proposed)

        for _ in range(self.num_steps):
            x_new = x_proposed + self.step_size * p_proposed  # Position update
            
            # Reflect if out of bounds
            for i in range(self.dim):
                if x_new[i] < self.lower[i]:
                    x_new[i] = 2 * self.lower[i] - x_new[i]  # Reflect back
                    p_proposed[i] *= -1  # Reverse momentum
                elif x_new[i] > self.upper[i]:
                    x_new[i] = 2 * self.upper[i] - x_new[i]
                    p_proposed[i] *= -1

            x_proposed = x_new.copy()
            p_proposed += self.step_size * self._gradient_potential(x_proposed)

        # Final half-step momentum update
        p_proposed += -0.5 * self.step_size * self._gradient_potential(x_proposed)

        proposed_H = self._potential_energy(x_proposed) + 0.5 * np.dot(p_proposed, p_proposed)
        accept_prob = np.exp(current_H - proposed_H)
        accept = (np.random.rand() < accept_prob)

        return x_proposed if accept else current_x

    def sample(self, num_samples, num_chains=1):
        """Draws samples using parallel HMC."""
        if num_chains > 1:
            with mp.Pool(num_chains) as pool:
                chains = pool.starmap(HMCTruncatedNormal._run_chain_wrapper,
                                      [(self, num_samples)] * num_chains)
            return np.vstack(chains)
        else:
            return self._run_chain(num_samples)

    @staticmethod
    def _run_chain_wrapper(instance, num_samples):
        return instance._run_chain(num_samples)

    def _run_chain(self, num_samples):
        """Runs a single Markov chain."""
        samples = []
        current_x = self.x0.copy()
        for _ in range(num_samples):
            current_x = self._hmc_step(current_x)
            samples.append(current_x.copy())
        return np.array(samples)
