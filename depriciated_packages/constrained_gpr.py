import numpy as np
import cvxpy as cp  # Import CVXPY for quadratic programming
import matplotlib.pyplot as plt
from scipy import linalg as sc_linalg
from basis_functions import BasisFunctions
from scipy.spatial.distance import cdist
import hamiltonianMonteCarlo as hmc

class ConstrainedGaussianProcess:
    """
    Implements a finite-dimensional Gaussian Process with linear inequality constraints.
    """
    def __init__(self, noise=None, constraint_matrix=None, lower_bound=None, upper_bound=None, kernel=None, kernel_params = None, **kwargs):
        """
        Initialize the constrained GP model.
        
        Parameters:
        - knots (list): Knots (d dimensional list of lists of length (m_1, m_2, ..., m_d))
        - m (int or list): Number of knots
        - delta_m (int or list): Space between knots
        - d (int): Dimension of input space
        - noise (np.ndarray): Observation noise covariance matrix
        - constraint_matrix (np.ndarray): Linear inequality constraint matrix
        - lower_bound (np.ndarray): Lower bound vector
        - upper_bound (np.ndarray): Upper bound vector
        - kernel (callable): Kernel function
        - **kwargs: Additional keyword arguments.
        
        Returns:
        - Phi_j(x): Value of the j-th basis function at x.
        """

        self.basis = BasisFunctions(**kwargs)
        self.m = self.basis.m

        self.constraint_matrix = constraint_matrix
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if kernel is None:
            self.kernel = [self.matern_kernel for dim in range(self.basis.d)]

        self.kernel_params = kernel_params

        self.mu, self.Sigma = None, None  # Placeholder for posterior mean and covariance
        self.mu_star = None  # Placeholder for constrained posterior mean
        self.posterior_samples = None  # Store posterior samples
        
        self.x_train = None  # Store training data
        self.y_train = None
        self.noise = noise

    def rbf_kernel(self, x1, x2):
        """
        Radial Basis Function (RBF) kernel.
        """

        if self.kernel_params is None:
            self.kernel_params = {'variance': 0.2, 'length_scale': 0.5}

        variance = self.kernel_params['variance']
        length_scale = self.kernel_params['length_scale']

        dists = (x1 - x2) ** 2
        return variance * np.exp(-0.5 * (dists / length_scale) ** 2)
    
    def matern_kernel(self, x1, x2):
        """
        Matern kernel function with different smoothness parameters.
        
        Parameters:
        - x1, x2: Input points
        - nu: Smoothness parameter (0.5, 1.5, or 2.5)
        
        Returns:
        - Kernel value
        """
        if self.kernel_params is None:
            self.kernel_params = {'variance': 0.5468, 'length_scale': 0.5754, 'nu': 2.5}

        variance = self.kernel_params['variance']
        length_scale = self.kernel_params['length_scale']
        nu = self.kernel_params.get('nu', 2.5)  # Default to 2.5 (Matern 5/2)
        
        distance = np.abs(x1 - x2)
        
        if nu == 0.5:
            # Matern 1/2 (exponential kernel)
            return variance * np.exp(-distance / length_scale)
        
        elif nu == 1.5:
            # Matern 3/2
            sqrt3_dist = np.sqrt(3) * distance / length_scale
            return variance * (1 + sqrt3_dist) * np.exp(-sqrt3_dist)
        
        else:  # Default to nu = 2.5 (Matern 5/2)
            # Matern 5/2
            sqrt5_dist = np.sqrt(5) * distance / length_scale
            return variance * (1 + sqrt5_dist + 5 * (distance ** 2) / (3 * length_scale ** 2)) * np.exp(-sqrt5_dist)

    def fit(self, X, y):
        """
        Train the finite-dimensional GP model with linear inequality constraints.

         Parameters:
        - X (np.ndarray): Training data (d x n)
        - y (np.ndarray): Training labels (n x 1)
        
        Returns:
        - Unconstrained mean and covariance of the GP.
        - Constrained mean of the GP.
        """
        # Store training data
        self.x_train = X
        self.y_train = y

        # Compute the basis matrix Phi
        Phi = self.basis.basis_matrix(X)

        # Compute Gamma for the given set of knots and kernels for each dimension.
        m_total = np.prod(self.basis.m)
        Gamma = np.zeros([m_total, m_total])
        
        for i in range(m_total):
            for j in range(m_total):

                if self.basis.d == 1:
                    
                    kernel = self.kernel[0]
                    Gamma[i, j] = kernel(self.basis.knots[0][i], self.basis.knots[0][j])

                else:
                    i_knots = self.basis.indices_to_knots(i)
                    j_knots = self.basis.indices_to_knots(j)
                    
                    product = 1
                    for dim in range(self.basis.d):
                        kernel = self.kernel[dim]
                        product *= kernel(i_knots[dim], j_knots[dim])
                    
                    Gamma[i, j] = product

        self.Gamma = Gamma

        if self.noise is None:
            self.noise = np.eye(len(y)) * 1e-6

        Sigma_y = Phi @ Gamma @ Phi.T + self.noise
        common_term =  Gamma @ Phi.T @ np.linalg.inv(Sigma_y)
        self.Sigma_y = Sigma_y
        mu_unconstrained = common_term @ y
        Sigma_unconstrained = Gamma - common_term @ Phi @ Gamma

        self.common_term = common_term
        self.Phi = Phi
        

        self.mu = mu_unconstrained
        
        if self.constraint_matrix is not None and (self.lower_bound is not None or self.upper_bound is not None):
            self.Sigma = Sigma_unconstrained  
            self.mu_star = self.compute_constrained_mean(mu_unconstrained, Sigma_unconstrained)
        else:
            self.mu, self.Sigma = mu_unconstrained, Sigma_unconstrained
    
        return self.mu, self.Sigma, self.mu_star

    def compute_constrained_mean(self, mu, Sigma):
        """
        Solve the quadratic optimization problem using a dedicated QP solver.
        """

        #Q = np.linalg.inv(Sigma + 10e-3  * np.eye(np.prod(self.basis.m))) # FIX HERE # WHY IS JITTER HAVE TO BE 0.05???
        #scale_factor = np.linalg.norm(Sigma, ord =np.inf)
        #Q = np.linalg.inv(Sigma/scale_factor) # FIX HERE # WHY IS JITTER HAVE TO BE 0.05???
        #Q = Q*scale_factor

        #Q = np.linalg.inv(Sigma + 10e-4  * np.eye(np.prod(self.basis.m))) # FIX HERE # WHY IS JITTER HAVE TO BE 0.05???
        Q = sc_linalg.pinv(Sigma + 10e-4  * np.eye(np.prod(self.basis.m))) # FIX HERE # WHY IS JITTER HAVE TO BE 0.05???
        
    
        Q = cp.psd_wrap(Q)
        p = -Q @ mu
        x = cp.Variable(np.prod(self.basis.m))
        constraints = []
        
        if self.lower_bound is not None:
            constraints.append(self.constraint_matrix @ x >= self.lower_bound)
        if self.upper_bound is not None:
            constraints.append(self.constraint_matrix @ x <= self.upper_bound)
        prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, Q) + p.T @ x), constraints)
        
        prob.solve(solver ='OSQP')
        
        if prob.status != cp.OPTIMAL:
            raise RuntimeError("Quadratic programming failed to enforce constraints.")
        return x.value

    def sample_posterior(self, num_samples=500, numerical_stability = 0.75):
        """
        Sample from the constrained posterior using Algorithm 1 from the paper.
        """
        Lambda = self.constraint_matrix

        if Lambda is None:
            # If no constraints are provided, sample from the unconstrained GP
            return np.random.multivariate_normal(self.mu, self.Sigma, num_samples)
        
        # Sample η from a truncated normal distribution
                
        mean_eta = numerical_stability * Lambda @ self.mu
        cov_eta = numerical_stability**2 * Lambda @ self.Sigma @ Lambda.T + np.eye(Lambda.shape[0]) * 1e-6 # Add jitter

        lower_bounds = numerical_stability * (self.lower_bound if self.lower_bound is not None else -np.inf * np.ones_like(mean_eta))
        upper_bounds = numerical_stability * (self.upper_bound if self.upper_bound is not None else np.inf * np.ones_like(mean_eta))
        
        #Calculate v*
        nu_star = Lambda @ self.mu_star

        # Temporary direct sampler

        def trunc_normal(mean, cov, lower, upper, num_samples, max_iter=10e5):
            final_samples = []
            iter = 0
            while len(final_samples) < num_samples and (iter < max_iter):
                samples = np.random.multivariate_normal(mean, cov, num_samples)
                samples = samples[((samples > lower) & (samples < upper)).all(axis=1)]
                reflected_samples = 2*mean - samples
                reflected_samples = reflected_samples[((reflected_samples > lower) & (reflected_samples < upper)).all(axis=1)]
                samples = np.vstack((samples, reflected_samples))
                for sample in samples:
                    if len(final_samples) < num_samples:
                        final_samples.append(sample)
                    else:
                        break
                iter += 1
            return np.array(final_samples)


        # Sample η using HMC
        eta_samples = trunc_normal(mean_eta, cov_eta, lower_bounds, upper_bounds, num_samples)
        #eta_samples = hmc.sample(mu= mean_eta, Sigma=cov_eta, a=lower_bounds, b=upper_bounds, num_samples=num_samples, x_init=nu_star)
        
    
        # Solve for ξ using Λξ = η
        
        eta_samples = eta_samples.T / numerical_stability
        xi_samples = np.linalg.pinv(Lambda) @ eta_samples

        self.posterior_samples = xi_samples.T
        self.eta_samples = eta_samples 

        return xi_samples.T
    
    def plot_posterior_samples(self, X_test=None, y_test=None, num_samples=1000, numerical_stability=0.6, band_quantiles=(0.025, 0.975), true_y = None):
        """
        Plot posterior samples from the constrained GP as a credible band based on quantiles.
        """
        if self.posterior_samples is None:
            self.posterior_samples = self.sample_posterior(num_samples, numerical_stability)
        elif len(self.posterior_samples) < num_samples:
            self.posterior_samples = self.sample_posterior(num_samples, numerical_stability)
        
        plt.figure(figsize=(10, 6))
        
        # Calculate quantiles for the credible band
        lower_q, upper_q = band_quantiles
        lower_band = np.quantile(self.posterior_samples, lower_q, axis=0)
        upper_band = np.quantile(self.posterior_samples, upper_q, axis=0)
        
        if X_test is None:

            # Plot the credible band with transparency
            plt.fill_between(self.basis.knots[0], lower_band, upper_band, 
                            alpha=0.9, color='orange', 
                            label=f'{int((upper_q-lower_q)*100)}% Credible Band')
            
            # Plot training points
            plt.scatter(self.x_train, self.y_train, color='blue', label='Training Points', s=10)
        
        if X_test is not None:
            test_basis = self.basis.basis_matrix(X_test)
            test_samples = test_basis @ self.posterior_samples.T

            lower_q, upper_q = band_quantiles
            lower_band = np.quantile(test_samples, lower_q, axis=1)
            upper_band = np.quantile(test_samples, upper_q, axis=1)
            sample_mean = np.mean(test_samples, axis=1)


            plt.fill_between(X_test, lower_band, upper_band, 
                                    alpha=0.9, color='orange', 
                                    label=f'Centered {int((upper_q-lower_q)*100)}% Quantile')

            plt.plot(X_test, sample_mean, color='purple', label='Sample Mean', linewidth=0.5, linestyle='--')


        # Plot test points if provided
        if X_test is not None and y_test is not None:
            plt.scatter(X_test, y_test, color='green', label='Test Points', s = 0.1)
        
        if true_y is not None:
            plt.plot(self.x_train, true_y, color='gray', label='True Function', linestyle='dashed', alpha = 0.5)

        

        # Plot posterior mean
        if self.mu_star is not None:
            plt.plot(self.basis.knots[0], self.mu_star, color='darkgreen', linestyle='dashdot', label='Posterior Mean')
            plt.plot(self.basis.knots[0], self.mu, color='purple', linestyle='dashdot', label='Unconstrained Mean')
            #plt.vlines(self.basis.knots[0], ymin= 0 , ymax = 1,  linestyle = 'dashed', label = 'Constraints', alpha = 0.2,)
        else:
            plt.plot(self.basis.knots[0], self.mu, color='purple', linestyle='dashdot', label='Mean')
        

        y_min, y_max = plt.gca().get_ylim()
        plt.vlines(self.basis.knots[0], ymin= y_min , ymax = y_max, label = 'Knots', alpha = 0.1, color = 'r')


        plt.xlabel('x')
        plt.ylabel('GP Sample Value')
        plt.title('Constrained GP with Credible Band')
        plt.legend()
        plt.grid()
        plt.show()


# Example Usage
if __name__ == "__main__":
    knots = np.linspace(0, 1, 10)  # Define knot locations
    X_train = np.linspace(0, 1, 20)  # 20 training points
    y_train = X_train ** 2  # Convex function: f(x) = x^2
    noise_matrix = np.diag(0.05 * np.ones_like(X_train))  # Small observation noise
    
    # Define convexity constraints (second derivative must be non-negative at the knots)
    Lambda = np.zeros((len(knots) - 2, len(knots)))
    for i in range(len(knots) - 2):
        Lambda[i, i] = 1
        Lambda[i, i + 1] = -2
        Lambda[i, i + 2] = 1
    lower_bound = np.zeros(len(knots) - 2)  # Enforce convexity constraint
    
    constrained_gp = ConstrainedGaussianProcess(knots, noise=noise_matrix, constraint_matrix=Lambda, lower_bound=lower_bound)
    constrained_gp.fit(X_train, y_train)
    

    # Plot posterior samples
    constrained_gp.plot_posterior_samples()
