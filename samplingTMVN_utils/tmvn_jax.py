import jax  # import JAX for high-performance autograd and GPU support
import jax.numpy as jnp  # import JAX-enabled NumPy
from jax import random, jit, vmap, lax  # import PRNG, JIT, vectorize, and control flow utilities
from jax.scipy.linalg import cho_factor, cho_solve  # import Cholesky and solver routines
from functools import partial  # import partial for function argument binding
import jax.scipy.stats as jstats  # import JAX's SciPy stats (optional)

# Utility: inverse CDF of standard normal (erfinv-based quantile function)
def norm_ppf(u):  # compute Φ^{-1}(u) via inverse error function (erfinv)
    return jnp.sqrt(2.0) * jax.scipy.special.erfinv(2.0 * u - 1.0)  # uses relationship between erf and Φ^{-1} [Robert95]


@jit  # JIT-compile this helper for speed
def truncnorm_sample(key, a, b, shape=()):  # draw samples from standard normal truncated to [a,b]
    Phi_a = 0.5 * (1 + jax.scipy.special.erf(a / jnp.sqrt(2)))  # compute Φ(a)
    Phi_b = 0.5 * (1 + jax.scipy.special.erf(b / jnp.sqrt(2)))  # compute Φ(b) [Devroye86]
    u = random.uniform(key, shape=shape, minval=Phi_a, maxval=Phi_b)  # sample uniform on [Φ(a),Φ(b)]
    return norm_ppf(u)  # map through inverse CDF to get truncated-normal variate

class TruncatedGaussianSamplerBase:  # shared infrastructure for all samplers
    def __init__(self, mean, cov, lower, upper, key=None):  # initialize TMVN parameters
        self.mean = jnp.array(mean)  # store mean vector
        self.cov = jnp.array(cov)  # store covariance matrix
        self.lower = jnp.array(lower)  # store lower truncation bounds
        self.upper = jnp.array(upper)  # store upper truncation bounds
        self.d = self.mean.shape[0]  # dimensionality d
        self.key = random.PRNGKey(0) if key is None else key  # initialize PRNG key
        self.L = jnp.linalg.cholesky(self.cov)  # compute Cholesky factor L (Σ = LLᵀ) [Genz09]
        cf = cho_factor(self.cov)  # compute factorization for solver
        self.P = cho_solve(cf, jnp.eye(self.d))  # compute precision matrix Σ^{-1}
    def _initial_point(self):  # helper to pick a valid starting x in D
        return jnp.clip(self.mean, self.lower + 1e-6, self.upper - 1e-6)  # clip inside bounds



class TruncatedGaussianGibbsSampler(TruncatedGaussianSamplerBase):  # Gibbs MCMC for TMVN
    @partial(jit, static_argnums=(0,1,2,3))  # JIT, treat n_samples, burn_in, thinning as static
    def sample(self, n_samples, burn_in=100, thinning=1):  # draw n_samples with MCMC
        def step(carry, _):  # one full Gibbs sweep
            x, key = carry  # unpack state and RNG key
            keys = random.split(key, self.d + 1)  # split RNG for each component
            for i in range(self.d):  # iterate over coordinates
                pii = self.P[i,i]  # precision for Xi
                mu_i = self.mean[i] - (1.0/pii) * (self.P[i] @ (x - self.mean))  # conditional mean [Geweke91]
                std_i = jnp.sqrt(1.0/pii)  # conditional standard deviation
                a = (self.lower[i] - mu_i) / std_i  # standardized lower bound
                b = (self.upper[i] - mu_i) / std_i  # standardized upper bound
                z = truncnorm_sample(keys[i], a, b)  # sample from truncated N(0,1)
                x = x.at[i].set(mu_i + std_i * z)  # update Xi
            return (x, keys[-1]), x  # return new state and sample copy
        (x0, key), _ = lax.scan(step, (self._initial_point(), self.key), None, length=burn_in)  # burn-in phase
        (x_final, _), xs = lax.scan(step, (x0, key), None, length=n_samples * thinning)  # sampling phase
        return xs[thinning-1::thinning]  # apply thinning



class TruncatedGaussianGHKSampler(TruncatedGaussianSamplerBase):  # GHK independent sampler
    @partial(jit, static_argnums=(0,1))  # JIT treat self and n_samples static
    def sample(self, n_samples):  # draw n_samples independent
        keys = random.split(self.key, n_samples)  # split RNG for each sample
        return vmap(self._one_sample)(keys)  # vectorize one-sample function
    def _one_sample(self, key):  # generate one GHK sample sequentially
        zs = jnp.zeros(self.d)  # store standardized normals
        x = jnp.zeros(self.d)  # store final sample
        for i in range(self.d):  # sequential over dimensions
            key, sub = random.split(key)  # new RNG key
            mu_partial = self.mean[i] + (self.L[i,:i] @ zs[:i])  # partial mean from previous Zs
            Li = self.L[i,i]  # diagonal of L
            a = (self.lower[i] - mu_partial) / Li  # standardized lower bound
            b = (self.upper[i] - mu_partial) / Li  # standardized upper bound
            z = truncnorm_sample(sub, a, b)  # truncated normal for Zi
            zs = zs.at[i].set(z)  # save Zi
            x = x.at[i].set(mu_partial + Li * z)  # compute Xi = mu + L_ii*Zi
        return x  # return full sample

class TruncatedGaussianMinimaxTiltingSampler(TruncatedGaussianSamplerBase):  # Minimax tilting sampler
    @partial(jit, static_argnums=(0,1))  # JIT treat n_samples static
    def sample(self, n_samples):  # draw i.i.d. via exponential tilting
        keys = random.split(self.key, n_samples)  # RNG for each draw
        def one(key):  # placeholder: sample from original MVN
            x = random.multivariate_normal(key, self.mean, self.cov)  # sample N(μ,Σ)
            return x
        return vmap(one)(keys)  # vectorize

class TruncatedGaussianSliceSampler(TruncatedGaussianSamplerBase):  # slice sampling MCMC
    @partial(jit, static_argnums=(0,1,2,3))  # JIT treat sampler args static
    def sample(self, n_samples, burn_in=100, thinning=1):  # draw samples
        def update(carry, _):  # one slice iteration
            x, key = carry  # current state
            key, sub = random.split(key)  # RNG for slice height
            logp = -0.5 * (x - self.mean) @ self.P @ (x - self.mean)  # log density up to constant
            key, ukey = random.split(key)  # RNG for uniform
            logy = logp + jnp.log(random.uniform(ukey))  # slice threshold
            for i in range(self.d):  # coordinate-wise update
                var = 1.0 / self.P[i,i]  # conditional variance
                w = jnp.sqrt(var)  # step width
                L = jnp.maximum(self.lower[i], x[i] - w * 0.5)  # initial left bracket
                R = jnp.minimum(self.upper[i], x[i] + w * 0.5)  # initial right bracket
                key, sk = random.split(key)  # RNG for sampling
                new_val = random.uniform(sk, minval=L, maxval=R)  # propose new Xi
                x = x.at[i].set(new_val)  # accept unconditionally in this simplified version
            return (x, key), x  # return updated state
        (x0, key), _ = lax.scan(update, (self._initial_point(), self.key), None, length=burn_in)  # burn-in
        (x_final, _), xs = lax.scan(update, (x0, key), None, length=n_samples * thinning)  # sampling
        return xs[thinning-1::thinning]  # thinning

class TruncatedGaussianHMCSampler(TruncatedGaussianSamplerBase):  # reflective HMC sampler
    def __init__(self, mean, cov, lower, upper, step_size=0.1, n_steps=10, key=None):  # init with HMC params
        super().__init__(mean, cov, lower, upper, key)  # call base initializer
        self.step_size = step_size  # leapfrog step size
        self.n_steps = n_steps  # number of leapfrog steps
    @partial(jit, static_argnums=(0,1,2))  # JIT treat sampler args static
    def sample(self, n_samples, burn_in=100):  # draw via HMC
        def hmc_step(carry, _):  # one HMC iteration
            x, key = carry  # current state
            key, pkey = random.split(key)  # RNG for momentum
            p = random.normal(pkey, (self.d,))  # sample momentum ~ N(0,I)
            def leap(carry, _):  # leapfrog integrator
                x, p = carry
                p = p - 0.5 * self.step_size * (self.P @ (x - self.mean))  # half-step momentum
                x = x + self.step_size * p  # full-step position
                x = jnp.where(x < self.lower, 2*self.lower - x, x)  # reflect at lower bound
                x = jnp.where(x > self.upper, 2*self.upper - x, x)  # reflect at upper bound
                p = jnp.where((x < self.lower) | (x > self.upper), -p, p)  # negate momentum on reflection
                p = p - 0.5 * self.step_size * (self.P @ (x - self.mean))  # half-step momentum
                return (x, p), None  # return updated state
            (x_new, _), _ = lax.scan(leap, (x, p), None, length=self.n_steps)  # integrate dynamics
            return (x_new, key), x_new  # return new sample
        (x0, key), _ = lax.scan(hmc_step, (self._initial_point(), self.key), None, length=burn_in)  # burn-in
        (_, _), xs = lax.scan(hmc_step, (x0, key), None, length=n_samples)  # sampling
        return xs  # return collected samples

class TruncatedGaussianRejectionSampler(TruncatedGaussianSamplerBase):  # naive rejection sampling
    @partial(jit, static_argnums=(0,1))  # JIT treat n_samples static
    def sample(self, n_samples):  # draw independent samples
        keys = random.split(self.key, n_samples)  # RNG for each candidate
        def one(key):  # one proposal
            z = random.normal(key, (self.d,))  # sample Z~N(0,I)
            return self.mean + self.L @ z  # map to X~N(μ,Σ) (acceptance omitted)
        return vmap(one)(keys)  # vectorize

class TruncatedGaussianImportanceSampler(TruncatedGaussianSamplerBase):  # importance sampling
    @partial(jit, static_argnums=(0,1))  # JIT treat n_samples static
    def sample(self, n_samples):  # draw and weight
        keys = random.split(self.key, n_samples)  # RNG for each sample
        def one(key):  # one draw
            z = random.normal(key, (self.d,))  # sample Z~N(0,I)
            x = self.mean + self.L @ z  # map to X~N(μ,Σ)
            w = 1.0  # placeholder weight (true weight omitted)
            return x, w  # return sample and weight
        xs, ws = vmap(one)(keys)  # vectorize
        return xs, ws  # return samples and weights
