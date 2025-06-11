import torch
import gpytorch
from matplotlib import pyplot as plt


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()  # Changed from LinearMean
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0))
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def trainGP(X_train, y_train, noise_matrix, test_x = None, test_noise = None):

    # Convert to tensors with proper dtype
    train_x = torch.tensor(X_train, dtype=torch.float64)
    train_y = torch.tensor(y_train, dtype=torch.float64)

    noise = torch.tensor(noise_matrix.diagonal(), dtype=torch.float64)



    # Initialize likelihood with fixed noise
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noise)
    model = GPModel(train_x, train_y, likelihood)

    # Use the Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Increased learning rate

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Training loop with better monitoring
    training_iterations = 500
    losses = []

    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Iter {i+1}/{training_iterations} - Loss: {loss.item():.3f}')
            print(f'  lengthscale: {model.covar_module.base_kernel.lengthscale.item():.3f}')
            print(f'  outputscale: {model.covar_module.outputscale.item():.3f}')

    # Plot training curve
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.title("Negative Log Likelihood During Training")
    plt.xlabel("Iterations")
    plt.ylabel("NLL")
    plt.show()

    # Model evaluation and prediction
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Make predictions on fine grid
        if test_x is None:
            test_x = train_x
            test_noise = noise
            observed_pred = likelihood(model(test_x), noise=test_noise)
        elif test_noise is not None:
            test_x = torch.tensor(test_x, dtype=torch.float64)
            test_noise = torch.tensor(test_noise.diagonal(), dtype=torch.float64)
            observed_pred = likelihood(model(test_x), noise=test_noise)
        else:
            test_x = torch.tensor(test_x, dtype=torch.float64)
            observed_pred = likelihood(model(test_x))
        
        # Plot the results
        f, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot training data
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*', label='Observed Data')
        
        # Plot predictive means
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b', label='Mean')
        
        # Get lower and upper confidence bounds (2 standard deviations)
        lower, upper = observed_pred.confidence_region()
        
        # Plot uncertainty bounds
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.3, color='blue', label='Confidence')
        
        ax.legend()
        ax.set_title('GP Regression')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.tight_layout()
        plt.show()

    # Print final parameters
    print(f'Final lengthscale: {model.covar_module.base_kernel.lengthscale.item():.4f}')
    print(f'Final outputscale: {model.covar_module.outputscale.item():.4f}')

    return model.covar_module.base_kernel.lengthscale.item(), model.covar_module.outputscale.item(), model.covar_module


    def estimate_derivatives(model, likelihood, test_x):
        """
        Estimate function value, first derivative and second derivative at test points.
        
        Args:
            model: Trained GP model
            likelihood: GP likelihood
            test_x: Tensor or numpy array of test point(s)
            
        Returns:
            Dictionary containing means and variances for function value, 
            first derivative, and second derivative
        """
        model.eval()
        likelihood.eval()
        
        # Convert input to appropriate tensor
        if not isinstance(test_x, torch.Tensor):
            test_x = torch.tensor(test_x, dtype=torch.float64)
        
        # Ensure test_x has the right shape
        is_scalar = (test_x.dim() == 0)
        
        if is_scalar:
            test_x = test_x.reshape(1, 1)
        elif test_x.dim() == 1:
            test_x = test_x.unsqueeze(-1)
        
        # Small step size for finite difference
        h = 1e-5
        
        # Function values
        with torch.no_grad():
            f_distribution = likelihood(model(test_x))
            f_mean = f_distribution.mean
            f_var = f_distribution.variance
            
            # Points for finite differences
            x_plus_h = test_x + h
            x_minus_h = test_x - h
            
            # Function values at these points
            f_plus_h = likelihood(model(x_plus_h)).mean
            f_minus_h = likelihood(model(x_minus_h)).mean
            
            # First derivative - central difference
            f_prime_mean = (f_plus_h - f_minus_h) / (2 * h)
            
            # Second derivative - central difference
            f_second_mean = (f_plus_h - 2 * f_mean + f_minus_h) / (h**2)
        
        # Reshape results if input was a scalar
        if is_scalar:
            f_mean = f_mean.item()
            f_var = f_var.item()
            f_prime_mean = f_prime_mean.item()
            f_second_mean = f_second_mean.item()
        
        return {
            'function': (f_mean, f_var),
            'first_derivative': f_prime_mean,
            'second_derivative': f_second_mean
        } 