import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
from jax import grad, vmap
from functools import partial


def truncated_multivariate_gaussian_logpdf(x, mu, Sigma, a, b):
    """
    Computes the log probability density of a multivariate Gaussian truncated to [a, b].
    """
    dim = x.shape[0]
    normalizer = jnp.prod(jstats.norm.cdf(b, loc=mu, scale=jnp.sqrt(jnp.diag(Sigma))) - 
                           jstats.norm.cdf(a, loc=mu, scale=jnp.sqrt(jnp.diag(Sigma))))
    logpdf = jstats.multivariate_normal.logpdf(x, mean=mu, cov=Sigma)
    return logpdf - jnp.log(normalizer)


@jax.jit
def grad_logpdf(x, mu, Sigma, a, b):
    """
    Computes the gradient of the log-density of the truncated multivariate Gaussian.
    """
    return grad(truncated_multivariate_gaussian_logpdf, argnums=0)(x, mu, Sigma, a, b)


def leapfrog(x, p, mu, Sigma, a, b, step_size, num_steps):
    """
    Leapfrog integration for Hamiltonian dynamics with rejection sampling.
    """
    num_steps = int(num_steps)  # Ensure num_steps is a Python integer
    x, p = jnp.asarray(x), jnp.asarray(p)  # Ensure JAX arrays
    p = p + 0.5 * step_size * grad_logpdf(x, mu, Sigma, a, b)
    for _ in range(num_steps - 1):
        x = x + step_size * p
        p = p + step_size * grad_logpdf(x, mu, Sigma, a, b)
    
    x = x + step_size * p
    p = p + 0.5 * step_size * grad_logpdf(x, mu, Sigma, a, b)
    
    return x, p


@partial(jax.jit, static_argnums=(7,))
def hmc_single_sample(key, x_init, mu, Sigma, a, b, step_size, num_steps):
    """
    Runs HMC for a single sample in parallel, allowing for custom initial points.
    """
    dim = mu.shape[0]
    subkey1, subkey2, subkey3 = jax.random.split(key, 3)
    x = jnp.clip(x_init + 1e-4 * jax.random.normal(subkey1, shape=x_init.shape), a, b)  # Add slight noise to x_init
    p = jax.random.normal(subkey2, shape=(dim,))  # Sample momentum
    x_new, p_new = leapfrog(x, p, mu, Sigma, a, b, step_size, num_steps)
    
    # Use jax.lax.cond() to handle rejection
    valid_sample = jnp.all((x_new >= a) & (x_new <= b))
    x_final = jax.lax.cond(valid_sample, lambda: x_new, lambda: x)
    
    # Metropolis-Hastings acceptance
    log_prob_old = truncated_multivariate_gaussian_logpdf(x, mu, Sigma, a, b) - 0.5 * jnp.dot(p, p)
    log_prob_new = truncated_multivariate_gaussian_logpdf(x_final, mu, Sigma, a, b) - 0.5 * jnp.dot(p_new, p_new)
    
    return jax.lax.cond(jnp.log(jax.random.uniform(subkey3)) < log_prob_new - log_prob_old,
                         lambda: x_final,
                         lambda: x)


def sample(mu, Sigma, a, b, num_samples=1000, step_size=0.05, num_steps=10, x_init=None):
    """
    Generates exactly num_samples from a truncated multivariate Gaussian using HMC with rejection sampling.
    Allows for custom initial points.
    """
    mu, Sigma, a, b = map(jnp.asarray, (mu, Sigma, a, b))  # Ensure inputs are JAX arrays
    key = jax.random.PRNGKey(np.random.randint(0, 1e6))  # Randomized seed for different runs
    keys = jax.random.split(key, num_samples)  # Generate unique keys for parallel sampling
    num_steps = int(num_steps)  # Convert to Python integer before passing to JIT
    
    x_init = jnp.asarray(x_init) if x_init is not None else jnp.tile(mu, (num_samples, 1))
    if x_init.shape == (mu.shape[0],):  # If single initial point, broadcast to all samples
        x_init = jnp.tile(x_init, (num_samples, 1))
    
    samples = vmap(hmc_single_sample, in_axes=(0, 0, None, None, None, None, None, None))(keys, x_init, mu, Sigma, a, b, step_size, num_steps)
    
    return np.array(samples)