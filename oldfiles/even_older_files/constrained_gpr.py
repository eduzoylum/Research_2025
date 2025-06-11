import torch
import gpytorch
from models.base_gpr import BaseGPR

class ConstrainedGPR(BaseGPR):
    """
    Gaussian Process Regression with Linear Inequality Constraints.
    Implements finite-dimensional Gaussian approximation similar to lineqGPR.
    """
    def __init__(self, train_x, train_y, likelihood, basis_functions, constraints, kernel=None):
        """
        Initializes the Constrained GPR model.
        
        Args:
            train_x (Tensor): Training input data.
            train_y (Tensor): Training target values.
            likelihood (gpytorch.likelihoods.Likelihood): Likelihood function.
            basis_functions (callable): Function to compute basis expansion.
            constraints (dict): Dictionary containing constraint matrices (A, b).
            kernel (gpytorch.kernels.Kernel, optional): Custom kernel. Defaults to RBF.
        """
        super().__init__(train_x, train_y, likelihood, kernel)
        self.basis_functions = basis_functions
        self.constraints = constraints
        self.num_train = train_x.shape[0]  # Store number of training points

    def apply_basis_expansion(self, x):
        """
        Expands the input using predefined basis functions.
        
        Args:
            x (Tensor): Input points.
        
        Returns:
            Tensor: Expanded feature representation.
        """
        return self.basis_functions(x)
    
    def apply_constraints(self, mean, covar):
        """
        Modifies the GP posterior distribution to enforce constraints.
        Uses Gaussian conditioning similar to lineqGPR.
        
        Args:
            mean (Tensor): Unconstrained mean vector of shape [num_train].
            covar (Tensor): Unconstrained covariance matrix of shape [num_train, num_train].
        
        Returns:
            Tuple[Tensor, Tensor]: Constrained mean and covariance.
        """
        A, b = self.constraints['A'], self.constraints['b']
        
        # Ensure covariance matrix is extracted from LazyTensor if needed
        if isinstance(covar, gpytorch.lazy.LazyTensor):
            covar = covar.evaluate()
        
        # Ensure A has the correct shape (A should map from num_train to constraint dimensions)
        if A.shape[1] != self.num_train:
            raise ValueError(f"Incompatible dimensions: A.shape={A.shape}, expected second dimension={self.num_train}")
        
        # Ensure b has correct shape
        b = b.view(-1, 1) if b.dim() == 1 else b
        
        # Compute conditional mean and covariance with linear constraints
        ACAT = A @ covar @ A.T
        ACAT_inv = torch.linalg.inv(ACAT + 1e-4 * torch.eye(ACAT.shape[0], device=ACAT.device))  # Regularized inverse
        
        mean_constrained = (mean - (covar @ A.T) @ ACAT_inv @ (A @ mean - b)).view(mean.shape)
        P = torch.eye(self.num_train, device=covar.device) - A.T @ ACAT_inv @ A
        covar_constrained = P @ covar @ P.T + 1e-4 * torch.eye(self.num_train, device=covar.device)
        covar_constrained = covar_constrained.expand(covar.shape)  # Ensure shape consistency
        
        return mean_constrained.squeeze(), covar_constrained

    def forward(self, x):
        """
        Forward pass applying basis expansion and constraints.
        
        Args:
            x (Tensor): Input test points.
        
        Returns:
            MultivariateNormal: GP posterior with constrained mean.
        """
        expanded_x = self.apply_basis_expansion(x)
        mean_x = self.mean_module(expanded_x).squeeze(-1)  # Ensure correct shape
        
        # Compute covariance using original training data, not expanded features
        covar_x = self.covar_module(self.train_inputs[0])
        
        # Ensure covariance matrix is extracted if needed
        if isinstance(covar_x, gpytorch.lazy.LazyTensor):
            covar_x = covar_x.evaluate()
        
        # Apply constraints to the mean and covariance
        mean_x, covar_x = self.apply_constraints(mean_x, covar_x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
