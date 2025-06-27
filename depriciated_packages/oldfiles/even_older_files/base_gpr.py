import torch
import gpytorch

class BaseGPR(gpytorch.models.ExactGP):
    """
    Standard Gaussian Process Regression (GPR) model using GPyTorch.
    This serves as the base class for constrained GPR models.
    """
    def __init__(self, train_x, train_y, likelihood, kernel=None):
        """
        Initializes the BaseGPR model.
        
        Args:
            train_x (Tensor): Training input data (features).
            train_y (Tensor): Training output data (targets).
            likelihood (gpytorch.likelihoods.Likelihood): GPyTorch likelihood function.
            kernel (gpytorch.kernels.Kernel, optional): Custom kernel function. Defaults to RBF.
        """
        super(BaseGPR, self).__init__(train_x, train_y, likelihood)
        
        # Define the GP mean function (zero mean by default)
        self.mean_module = gpytorch.means.ZeroMean()
        
        # Define the GP kernel (RBF by default, can be overridden)
        self.covar_module = kernel if kernel else gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (Tensor): Input test points.
        
        Returns:
            MultivariateNormal: GP posterior distribution.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)