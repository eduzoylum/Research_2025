# constrained_gp_model.py
"""
Constrained Gaussian‑Process model that *always* uses the **same hyper‑parameter
naming and structure** as your *unconstrained* `Base_GP_Model`:

* 1‑D surface (smile):
    ```python
    kernel_params = {
        'lengthscale': <float>,      # ARD off – single lengthscale
        'outputscale': <float>,      # GP amplitude  (ScaleKernel outputscale)
    }
    ```
* 2‑D surface: the dictionary carries separate length‑scales per coordinate:
    ```python m
    kernel_params = {
        'lengthscale_m': <float>,    # moneyness axis
        'lengthscale_T': <float>,    # maturity  axis
        'outputscale': <float>,
    }
    ```

The constrained model **inherits** that exact dictionary from
`base_model.kernel_params` unless you explicitly override it, so everything in
your pipeline stays consistent.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

import cvxpy as cp
from scipy import linalg as sc_linalg
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Basis helper (user‑provided)
# ----------------------------------------------------------------------------
try:
    from basis_functions import BasisFunctions
except ImportError as err:  # pragma: no cover
    raise ImportError("`basis_functions.py` must be importable.") from err

# ----------------------------------------------------------------------------
# Kernel factory – mirrors Base_GP_Model naming
# ----------------------------------------------------------------------------
try:
    import gpytorch
    _HAVE_GPYTORCH = True
except ImportError:  # pragma: no cover
    _HAVE_GPYTORCH = False


def _make_kernel_1d(params: Dict[str, float]) -> gpytorch.kernels.Kernel | Tensor:
    """Return a gpytorch Matern‑5/2 ScaleKernel (fallback: manual)."""
    ls = params["lengthscale"]
    outputscale = params["outputscale"]
    if _HAVE_GPYTORCH:
        base = gpytorch.kernels.MaternKernel(nu=2.5)
        base.lengthscale = torch.tensor(ls, dtype=torch.float64)
        base.lengthscale.requires_grad_(False)
        sk = gpytorch.kernels.ScaleKernel(base)
        sk.outputscale = torch.tensor(outputscale, dtype=torch.float64)
        sk.outputscale.requires_grad_(False)
        return sk
    else:
        # fallback scalar function using torch ops
        def k(x1, x2, ls=ls, os=outputscale):
            dist = torch.abs(x1 - x2)
            sqrt5 = torch.sqrt(torch.tensor(5.0, dtype=dist.dtype, device=dist.device))
            t = sqrt5 * dist / ls
            return os * (1 + t + 5 * dist ** 2 / (3 * ls ** 2)) * torch.exp(-t)
        return k


def _make_kernel_2d(params: Dict[str, float]):
    ls_m = params["lengthscale_m"]
    ls_T = params["lengthscale_T"]
    outputscale = params["outputscale"]
    return [
        _make_kernel_1d({"lengthscale": ls_m, "outputscale": 1.0}),  # per‑dim kernels return unit scale
        _make_kernel_1d({"lengthscale": ls_T, "outputscale": 1.0}),
        outputscale,  # separate overall amplitude applied at the end
    ]

# ----------------------------------------------------------------------------
# Defaults identical to Base_GP_Model initialisation values
# ----------------------------------------------------------------------------
_DEFAULT_1D = {"lengthscale": 1.0, "outputscale": 1.0}
_DEFAULT_2D = {"lengthscale_m": 1.0, "lengthscale_T": 1.0, "outputscale": 1.0}


class ConstrainedGPModel:
    """Finite‑rank GP with inequality constraints; hyper‑params identical to unconstrained GP."""

    def __init__(
        self,
        *,
        knots: Sequence[np.ndarray],
        constraint_matrix: Optional[np.ndarray] = None,
        lower_bound: Optional[np.ndarray] = None,
        upper_bound: Optional[np.ndarray] = None,
        base_model: Optional[Any] = None,
        kernel_params: Optional[Dict[str, float]] = None,
        device: Optional[torch.device] = None,
        **basis_kwargs,
    ) -> None:
        # ---------- inherit common bits from base_model ---------- #
        if base_model is not None:
            self.device = getattr(base_model, "device", None) or (
                torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            )
            self._X_from_base = base_model.x_train.detach().cpu().numpy()
            self._y_from_base = base_model.y_train.detach().cpu().numpy()
            self._noise_from_base = base_model.noise.detach().cpu().numpy()
            inherited_params = base_model.kernel_params
        else:
            self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            self._X_from_base = self._y_from_base = self._noise_from_base = None
            inherited_params = {}

        # ------------ basis & dimensionality -------------------- #
        self.basis = BasisFunctions(knots=knots, **basis_kwargs)
        self.d = self.basis.d
        self.knots = [torch.as_tensor(k, dtype=torch.float64, device=self.device) for k in knots]
        self.m_total = int(np.prod([len(k) for k in knots]))

        # ------------ kernel hyper‑parameters (exact same keys) -- #
        default_params = _DEFAULT_1D if self.d == 1 else _DEFAULT_2D
        self.kernel_params: Dict[str, float] = {**default_params, **inherited_params, **(kernel_params or {})}

        if self.d == 1:
            self.kernel = _make_kernel_1d(self.kernel_params)
        else:
            k_m, k_T, amp = _make_kernel_2d(self.kernel_params)
            self.kernel_m, self.kernel_T, self.outputscale = k_m, k_T, amp

        # ------------- constraints ------------------------------ #
        self.Lambda = constraint_matrix
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # placeholders
        self.mu_uncon = self.Sigma_uncon = None
        self.mu_star = None
        self.noise_np = self._noise_from_base
        self._fitted = False

    # ------------------------------------------------------------------
    # Gram matrix at knots
    # ------------------------------------------------------------------
    def _Gamma(self) -> Tensor:
        if self.d == 1:
            k = self.kernel
            G = k(self.knots[0][:, None], self.knots[0][None, :]) if _HAVE_GPYTORCH else k(self.knots[0][:, None], self.knots[0][None, :])
            return G
        # 2‑D: Kronecker product + global amplitude
        if _HAVE_GPYTORCH:
            Gm = self.kernel_m(self.knots[0][:, None], self.knots[0][None, :])
            GT = self.kernel_T(self.knots[1][:, None], self.knots[1][None, :])
        else:
            Gm = self.kernel_m(self.knots[0][:, None], self.knots[0][None, :])
            GT = self.kernel_T(self.knots[1][:, None], self.knots[1][None, :])
        return self.outputscale * torch.kron(Gm, GT)

    # ------------------------------------------------------------------
    # Basis matrix helper (handles list‑of‑arrays convention)
    # ------------------------------------------------------------------
    def _Phi(self, X: np.ndarray) -> np.ndarray:
        return self.basis.basis_matrix(X.flatten().tolist()) if self.d == 1 else self.basis.basis_matrix([X[:, 0], X[:, 1]])

    # ------------------------------------------------------------------
    # Fit / Predict (unchanged maths)
    # ------------------------------------------------------------------
    def fit(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None, jitter: float = 1e-6):
        if X is None or y is None:
            if self._X_from_base is None or self._y_from_base is None:
                raise ValueError("Provide training data or a base_model.")
            X, y = self._X_from_base, self._y_from_base
            if self.noise_np is None:
                self.noise_np = self._noise_from_base

        Phi = torch.as_tensor(self._Phi(X), dtype=torch.float64, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.float64, device=self.device)
        Gamma = self._Gamma()
        n = y.shape[0]
        Sigma_eps = (
            torch.as_tensor(self.noise_np, dtype=torch.float64, device=self.device)
            if self.noise_np is not None else torch.eye(n, dtype=torch.float64, device=self.device) * jitter
        )
        Sigma_y = Phi @ Gamma @ Phi.T + Sigma_eps
        L = torch.linalg.cholesky(Sigma_y + torch.eye(n, dtype=torch.float64, device=self.device) * jitter)
        A = torch.cholesky_solve((Phi @ Gamma).T, L)
        self.mu_uncon = (Gamma @ Phi.T) @ A @ y_t
        self.Sigma_uncon = Gamma - (Gamma @ Phi.T) @ A @ Phi @ Gamma
        self.mu_star = self._qp_mean() if self.Lambda is not None else self.mu_uncon.detach().cpu().numpy()
        self._fitted = True
        return self

    def _qp_mean(self):
        mu = self.mu_uncon.detach().cpu().numpy(); Sigma = self.Sigma_uncon.detach().cpu().numpy()
        Q = sc_linalg.pinv(Sigma + 1e-4 * np.eye(Sigma.shape[0]))
        x = cp.Variable(mu.size)
        obj = 0.5 * cp.quad_form(x, Q) - mu.T @ Q @ x
        cons = []
        if self.lower_bound is not None:
            cons.append(self.Lambda @ x >= self.lower_bound)
        if self.upper_bound is not None:
            cons.append(self.Lambda @ x <= self.upper_bound)
        cp.Problem(cp.Minimize(obj), cons).solve(solver="OSQP")
        return x.value

    def predict(self, X_test: np.ndarray, cred: float = 0.95):
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        Phi_t = torch.as_tensor(self._Phi(X_test), dtype=torch.float64, device=self.device)
        coef_mean = torch.as_tensor(self.mu_star, dtype=torch.float64, device=self.device)
        mean = (Phi_t @ coef_mean).detach().cpu().numpy()
        var = (Phi_t @ self.Sigma_uncon @ Phi_t.T).diag().detach().cpu().numpy()
        std = np.sqrt(np.maximum(var, 0))
        z = torch.distributions.Normal(0, 1).icdf(torch.tensor(0.5 + cred / 2)).item()
        return {"mean": mean, "lower": mean - z * std, "upper": mean + z * std, "variance": var}

    # simple 1‑D plot
    def plot_smile(self, X, y, X_test, pred):
        plt.figure(figsize=(8, 5)); plt.scatter(X, y, s=10, label="train", color="k")
        plt.plot(X_test, pred["mean"], "r", label="mean")
        plt.fill_between(X_test, pred["lower"], pred["upper"], color="orange", alpha=0.3)
        plt.legend(); plt.grid(True); plt.show()

__all__ = ["ConstrainedGPModel"]
