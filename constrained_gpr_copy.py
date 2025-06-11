import numpy as np
import torch
import cvxpy as cp
from scipy.linalg import pinvh, pinv
import matplotlib.pyplot as plt
import warnings
import jax
import jax.numpy as jnp
from jax import random, lax
from basis_functions import BasisFunctions
from BS_utils import bs_functions as BS

class ConstrainedGPFromBase:
	def __init__(
		self,
		base_gp,
		basis_kwargs=None,
		iterated = False
	):
		"""
		Constrained GP model using projection onto basis functions, initialized from a trained base GP.

		Parameters:
		- base_gp: Trained Base_GP_Model object (unconstrained GP).
		- basis_kwargs: dict or None, passed to BasisFunctions (e.g. {'m': 30}, {'delta_m': 0.05}, {'knots': [...]})
		
		Example usage of ConstrainedGPFromBase:

		1. Basic usage with default basis configuration (auto m and d):
			>>> constrained_gp = ConstrainedGPFromBase(base_gp)

		2. Specify number of knots (m):
			- For 1D:
			>>> constrained_gp = ConstrainedGPFromBase(base_gp, basis_kwargs={'m': 25})

			- For 2D:
			>>> constrained_gp = ConstrainedGPFromBase(base_gp, basis_kwargs={'m': [20, 15]})

		3. Specify knot spacing instead of number:
			>>> constrained_gp = ConstrainedGPFromBase(base_gp, basis_kwargs={'delta_m': [0.05, 0.1]})

		4. Provide explicit knot locations (must match input dimension):
			>>> knots_1d = [np.linspace(0, 1, 30)]
			>>> constrained_gp = ConstrainedGPFromBase(base_gp, basis_kwargs={'knots': knots_1d})

			>>> knots_2d = [np.linspace(0, 1, 25), np.linspace(0, 1, 30)]
			>>> constrained_gp = ConstrainedGPFromBase(base_gp, basis_kwargs={'knots': knots_2d})

		Note:
		- `d` (dimension) is automatically inferred from the base GP.
		- Constraints (`constraint_matrix`, `lower_bound`, `upper_bound`) can also be passed at initialization.
		- If no `basis_kwargs` is provided, the default is m=20 for 1D, m=[15,15,...] for d>1.
				
		"""
		self.base_gp = base_gp
		self.data = base_gp.data
		self.CallData = base_gp.CallData
		self.PutData = base_gp.PutData
		self.div_type = base_gp.div_type
		self.col_map = base_gp.col_map
		self.option_indices = base_gp.option_indices
		
		if not iterated:
			self.iter_id = 1
		else:
			self.iter_id = base_gp.iter_id + 1


		# Scaling functions for m, T and y.
		self.m_scaler = base_gp.m_scaler
		self.T_scaler = base_gp.T_scaler
		self.y_scaler = base_gp.y_scaler
		self.y_scaling_factor = base_gp.y_scaling_factor

		# Determine input dimension from base GP
		self.input_dim = base_gp.input_dim
		self.y_label = base_gp.y_label
		self.sep_obs = base_gp.sep_obs

		# Create basis functions (user can pass m, delta_m, or knots)
		if basis_kwargs is None:
			# default: 1D → m=20, 2D → m=[15,15]
			default_m = 20 if self.input_dim == 1 else [15] * self.input_dim
			basis_kwargs = {'m': default_m, 'd': self.input_dim}
		else:
			if 'd' not in basis_kwargs:
				basis_kwargs['d'] = self.input_dim

		self.basis = BasisFunctions(**basis_kwargs)
		self.m = self.basis.m  # save for convenience
		self.knots = self.basis.knots[0]
		self.iterated = iterated # To check if the base_GP is constrained or not

		if not self.iterated:
			# Extract training data from base GP
			self.x_train = base_gp.x_train.detach().cpu().numpy()
			self.y_train = base_gp.y_train.detach().cpu().numpy()

			# Noise: Use base GP’s estimated observation noise if available
			self.noise = np.diag(base_gp.likelihood.noise.cpu().detach().numpy()) if base_gp.likelihood.noise is not None else np.eye(self.y_train.shape[1])* 1e-6
		
			# Extract learned kernel and wrap it to a NumPy-compatible callable
			full_kernel = base_gp.model.covar_module
			self.kernel = self._wrap_gpytorch_kernel(full_kernel)
		else:
			self.x_train = base_gp.x_train
			self.y_train = base_gp.y_train

			self.noise = base_gp.noise
			self.kernel = base_gp.kernel

		# Posterior placeholders
		self.mu = None
		self.Sigma = None
		self.mu_star = None
		self.posterior_samples = None

	def fit(self, solver = 'SCS'):
		"""
		Compute the unconstrained (and optionally constrained) posterior
		for the finite-dimensional GP using the given base GP.

		Parameters:
			solver (str): Solver for the optimization problem. Default is 'SCS'.
						Avaliable solvers: ['CLARABEL', 'OSQP', 'SCIPY', 'SCS']   

		Returns:
			mu (np.ndarray): Unconstrained posterior mean
			Sigma (np.ndarray): Unconstrained posterior covariance
			mu_star (np.ndarray or None): Constrained posterior mode, if constraints are provided
		"""

		# Step 1: Basis matrix Φ (n_samples x m_total)
		Phi = self.basis.basis_matrix(self.x_train)  # stays in NumPy for now

		# Step 2: Kernel matrix Γ over knots (m_total x m_total)
		Gamma = self._compute_gamma()  # shape (m_total, m_total)
		Gamma += np.eye(Gamma.shape[0]) * 1e-4 # Add jitter for numerical stability. This makes sure Gamma is positive definite.
		y = self.y_train

		# Step 3: Σ_y = ΦΓΦᵀ + Σ_noise
		Sigma_y = Phi @ Gamma @ Phi.T + self.noise
		common_term = Gamma @ Phi.T @ np.linalg.inv(Sigma_y)
		self.Sigma_y = Sigma_y

		# Try to add later on.!!!!
		# Step 4: Inverse Σ_y
		#try:
		#    L = torch.linalg.cholesky(Sigma_y)
		#    inv_Sigma_y = torch.cholesky_inverse(L)
		#except RuntimeError:
		#    inv_Sigma_y = torch.linalg.pinv(Sigma_y)
		#    # Warning:
		#    print("Cholesky decomposition failed, using pseudo-inverse instead.")

		# Step 5: Posterior mean and covariance

		μ = common_term @ y
		Σ = Gamma - common_term @ Phi @ Gamma

		self.mu = μ
		self.Phi = Phi
		self.Sigma = Σ
		self.Sigma_y = Sigma_y
		self.Gamma = Gamma 		
		Λ = self.constraint_matrix 

		# Step 6: Optionally apply linear constraints
		if self.constraint_matrix is not None and (
			self.lower_bound is not None or self.upper_bound is not None
		):
			self.mu_star = self.compute_constrained_mode(self.mu, self.Sigma, solver=solver)
		else:
			self.mu_star = None

		mean_eta = Λ @ μ
		if self.mu_star is not None:
			mean_star_eta =  (Λ @ self.mu_star)
		cov_eta =   (Λ @ Σ @ Λ.T) #+ 1e-6 * np.eye(Λ.shape[0])
		lower_eta =  (self.lower_bound if self.lower_bound is not None else -np.inf * np.ones_like(mean_eta))
		upper_eta =  (self.upper_bound if self.upper_bound is not None else np.inf * np.ones_like(mean_eta))

		results_eta = {
			'mean_eta': mean_eta,
			'cov_eta': cov_eta,
			'lower_eta': lower_eta,
			'upper_eta': upper_eta,
			'mean_star_eta': mean_star_eta if self.mu_star is not None else None,
		}
		self.eta_parameters = results_eta


		return self

	def compute_constrained_mode(self, mu, Sigma, solver='SCS'):

		m = mu.shape[0]
		Lambda = self.constraint_matrix
		lb = self.lower_bound
		ub = self.upper_bound

		# Scale things to avoid extreme values
		#mu_scale = np.max(np.abs(mu))
		#mu = mu / mu_scale
		#lb = lb / mu_scale if lb is not None else None
		#ub = ub / mu_scale if ub is not None else None

		# Strong regularization to avoid ill-conditioning
		jitter = 1e-6
		Sigma_reg = Sigma + jitter * np.eye(m)
		Q_np = np.linalg.inv(Sigma_reg) # / (mu_scale**2)

		Q = cp.psd_wrap(Q_np)
		p = -Q @ mu
		xi = cp.Variable(np.prod(self.basis.m))

		constraints = []
		if lb is not None:
			constraints.append(Lambda @ xi >= lb)
		if ub is not None:
			constraints.append(Lambda @ xi <= ub)

		problem = cp.Problem(cp.Minimize(0.5 * cp.quad_form(xi, Q) + p.T @ xi), constraints)
		problem.solve(solver=solver, verbose=False)

		if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
			raise RuntimeError(f"QP failed: {problem.status}")
		
		return xi.value  # * mu_scale  # scale back to original units

	def sample_posterior(self, sampler=None, num_samples=1000, numerical_stability=1, seed=0, **kwargs):
		"""
		Sample from the constrained posterior using a custom or default truncated Gaussian sampler.

		Parameters:
		- sampler: an instance of a sampler from TMN_sampler.py (e.g. TruncatedGaussianGHKSampler)
		- num_samples: number of samples to draw
		- numerical_stability: scaling factor for better conditioning
		- seed: random seed for reproducibility
		- kwargs: passed to sampler.sample()

		Returns:
		- posterior_samples: (num_samples, m) samples of ξ
		"""
		if self.constraint_matrix is None:
			return np.random.multivariate_normal(self.mu, self.Sigma, num_samples)

		# TO BE FIXED#
		#mean_eta[0] = self.lower_bound[0]
		#mean_eta[-1] = self.lower_bound[-1]
		# TO BE FIXED

		# JAX input conversion
		#mean_eta_jax = jnp.array(mean_eta)
		#if self.mu_star is not None:
		#    #mean_star_eta_jax = jnp.array(mean_star_eta)
		#cov_eta_jax = jnp.array(cov_eta)
		#lower_eta_jax = jnp.array(lower_eta)
		#upper_eta_jax = jnp.array(upper_eta)

		# def trunc_normal(mean, cov, lower, upper, num_samples, max_iter=10e5):
		# 	final_samples = []
		# 	iter = 0
		# 	while len(final_samples) < num_samples and (iter < max_iter):
		# 		samples = np.random.multivariate_normal(mean, cov, num_samples)
		# 		samples = samples[((samples > lower) & (samples < upper)).all(axis=1)]
		# 		reflected_samples = 2*mean - samples
		# 		reflected_samples = reflected_samples[((reflected_samples > lower) & (reflected_samples < upper)).all(axis=1)]
		# 		samples = np.vstack((samples, reflected_samples))
		# 		for sample in samples:
		# 			if len(final_samples) < num_samples:
		# 				final_samples.append(sample)
		# 			else:
		# 				break
		# 		iter += 1
		# 	return np.array(final_samples)

		# filtered_samples = trunc_normal(self.eta_parameters['mean_eta'], 
		# 								self.eta_parameters['cov_eta'],
		# 								self.eta_parameters['lower_eta'],
		# 								self.eta_parameters['upper_eta'],
		# 								num_samples=num_samples)
		# #filtered_samples = samples[np.all(samples >= lower_eta, axis=1)]
		# filtered_samples = filtered_samples / numerical_stability

		mean, cov = self.eta_parameters['mean_eta'], self.eta_parameters['cov_eta']
		lower, upper = self.eta_parameters['lower_eta'], self.eta_parameters['upper_eta']
		mod = self.eta_parameters['mean_star_eta']

		potential_jitter = [1e-6, 1e-5 ,1e-4]
		has_nan = True; jitter_ind = 0

		while has_nan and jitter_ind < len(potential_jitter)-1:
			cov = cov + np.eye(cov.shape[0]) * potential_jitter[jitter_ind]
			cov = 0.5 * (cov + cov.T)  # Ensure symmetry

			samples = sample_MVTN(mean=mean, cov=cov, lower=lower, upper=upper, mod = mod, n_samples=num_samples)
			# Check if there are any NaN values in the 'samples' array
			has_nan = np.isnan(samples).any()
			jitter_ind += 1

		self.eta_parameters['jitter'] = potential_jitter[jitter_ind]
		self.eta_samples = samples

		Λ = self.constraint_matrix
		Λ_pinv = np.linalg.pinv(Λ)
		xi_samples = Λ_pinv @ samples.T

		self.posterior_samples = xi_samples.T

		return self

	def predict(self, add_x_test=None, add_label=None):
		"""
		Predict the posterior mean and 95% confidence intervals at test inputs.

		Parameters:
		- x_test: normalized test inputs, shape (n, d)
		- label: optional string to tag predictions (for multiple test sets)

		Returns:
		- Dictionary with keys: 'mean', 'lower', 'upper', 'variance'
		"""
		if self.posterior_samples is None or self.eta_samples.shape[0] == 0:
			raise RuntimeError("You must sample from the posterior before predict().")

		dim = self.input_dim

		if dim == 1:
			x_test = self.m_scaler.normalize(self.data['m'].unique())
		elif dim == 2:
			x_test = self.m_scaler.normalize(self.data['m'].unique())
			x_test = np.column_stack((x_test, self.T_scaler.normalize(self.data['T'].unique())))
		else:
			raise ValueError("Prediction only supported for 1D or 2D.")

		self.pred = self._get_pred(x_test)        
		if add_x_test is not None:
			if add_label is None:
				warnings.warn("add_label is None. Setting it to 'extra'")
				add_label = 'extra'
			predictions = self._get_pred(add_x_test, label=add_label)
			setattr(self, f"pred_{add_label}", predictions)
		
		return self

	def _get_pred(self, x_test, label = None):
		"""
		Compute the posterior mean and covariance at test inputs.

		Parameters:
		- x_test: normalized test inputs, shape (n, d)

		Returns:
		- mu_star: posterior mean
		- Sigma_star: posterior covariance
		"""
		Phi_test = self.basis.basis_matrix(x_test)  # shape (n_test, m_total)
		test_samples = Phi_test @ self.posterior_samples.T  # shape (n_test, n_samples)
		test_samples = self.y_scaler.inverse(test_samples)

		pred_mean = test_samples.mean(axis=1)
		pred_lower = np.quantile(test_samples, 0.025, axis=1)
		pred_upper = np.quantile(test_samples, 0.975, axis=1)
		pred_var = test_samples.var(axis=1)

		if self.basis.d == 1:
			setattr(self, f"x_test_{label}", self.m_scaler.inverse(x_test))
			if label is None:
				self.x_test = self.m_scaler.inverse(x_test)
			else:
				setattr(self, f"x_test_{label}", self.m_scaler.inverse(x_test))

		elif self.basis.d == 2:
			m = self.m_scaler.inverse(x_test[:, 0])
			T = self.T_scaler.inverse(x_test[:, 1])
			if label is None:
				self.m_test = m
				self.T_test = T
				self.x_test = np.column_stack([m, T])
			else:
				setattr(self, f"m_test_{label}", m)
				setattr(self, f"T_test_{label}", T)
		else:
			raise ValueError("Prediction only supported for 1D or 2D.")

		results = {
			'mean': pred_mean,
			'lower': pred_lower,
			'upper': pred_upper,
			'variance': pred_var,
			'samples': test_samples.T
		}

		return results

	def get_finite_difference_gradient(self, x_test = None, epsilon_m = None):
		"""
		Compute finite difference estimates of the first and second derivatives of the predicted function
		(samples) at test points, both in the native and normalized (transformed) spaces.

		Parameters
		----------
		self : object
			The model instance containing prediction samples and scalers.
		x_test : np.ndarray, optional
			Test points in normalized space at which to compute derivatives. If None, uses self.x_test_extra.
		epsilon_m : float, optional
			Step size for finite differences in normalized space. If None, computed from x_test spacing.

		Returns
		-------
		self : object
			The model instance with added attributes:
			- self.pred_deriv_fd: dict with finite difference results in native space.
			- self.f_tilde_fd: dict with finite difference results in normalized space.

		Notes
		-----
		- The function computes derivatives using forward/backward differences at boundaries and central differences in the interior.
		- Results include mean, std, quantiles, and a random subset of samples for each derivative.
		- Adjustments are made for normalization and scaling to provide results in both native and normalized spaces.
		"""
		if x_test is None or epsilon_m is None:
			x_test = self.m_scaler.normalize(self.x_test_extra)
			epsilon_m = x_test[1] - x_test[0]

		test_samples = self.pred_extra['samples'].T  # shape (n_samples, len(x_eps))
		first_derivative, second_derivative, function_val = self._finite_difference_derivative_calculator(x_test, test_samples, epsilon_m)

		# Adjust for m normalization:
		m_spread =  (self.m_scaler.inverse(1) - self.m_scaler.inverse(0))
		first_derivative = first_derivative / m_spread
		second_derivative = second_derivative / (m_spread**2)

		def get_results(derivative_samples):
			mean = np.mean(derivative_samples, axis=1)
			std = np.std(derivative_samples, axis=1)
			quantiles = np.quantile(derivative_samples, np.linspace(0,1,11), axis=1)
			rand_indices = np.random.choice(derivative_samples.shape[1], 5, replace=False)
			derivative_samples = derivative_samples[:, rand_indices].T
			return {
				'mean': mean,
				'std': std,
				'quantiles': quantiles,
				'samples': derivative_samples
			}
		
		# Derivative in Native space
		results = {
			'first_derivative': get_results(first_derivative),
			'second_derivative': get_results(second_derivative),
			'function_val': get_results(function_val)
		}

		self.pred_deriv_fd = results

		self.pred_deriv ={
			'first_derivative': first_derivative.mean(axis=1),
			'second_derivative': second_derivative.mean(axis=1),
			'function_val': function_val.mean(axis=1)
		}


		# Adjust for transformations
		y_max, y_min = self.y_scaler.max_, self.y_scaler.min_
		y_spread = y_max - y_min

		first_derivative = first_derivative * m_spread / y_spread * self.y_scaling_factor
		second_derivative = second_derivative * (m_spread**2) / y_spread * self.y_scaling_factor
		function_val = self.y_scaler.normalize(function_val)

		results = {
			'first_derivative': get_results(first_derivative),
			'second_derivative': get_results(second_derivative),
			'function_val': get_results(function_val)
		}
		
		self.f_tilde_fd = results

		self.f_tilde = {
			'first_derivative': first_derivative.mean(axis=1),
			'second_derivative': second_derivative.mean(axis=1),
			'function_val': function_val.mean(axis=1)
		}

		return self

	def get_price_gradient(self, x_test=None, epsilon_m=None):
		if x_test is None or epsilon_m is None:
			x_test = self.m_scaler.normalize(self.x_test_extra)
			epsilon_m = x_test[1] - x_test[0]

		_, test_samples = self._get_price_samples_1d()

		test_samples_Call = test_samples['Call']['samples']
		test_samples_Put = test_samples['Put']['samples']

		call_results, call_strikes = self._price_finite_difference_gradient_calculator(test_samples_Call, x_test=x_test, epsilon_m=epsilon_m)

		Call = {
			'derivative': call_results,
			'strikes': call_strikes
		}

		put_results, put_strikes = self._price_finite_difference_gradient_calculator(test_samples_Put, x_test=x_test, epsilon_m=epsilon_m)
		
		Put = {
			'derivative': put_results,
			'strikes': put_strikes
		}

		results = {
			'Call': Call,
			'Put': Put
		}

		self.price_gradient = results

		return self

	# Plotting functions

	def plot_smile_comparison(self, figsize=(24, 10), zoom=80, 
							x_version = 'log_m',
							y_version = 'TIV'):
			fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
			self._plot_smile(ax=ax2, x_version=x_version, y_version=y_version)
			self._plot_smile_centered(ax=ax1, zoom=zoom, x_version=x_version, y_version=y_version)
			plt.tight_layout()
			plt.show()
			return self

	def _plot_smile(self, figsize=(20, 10), ax=None,
					x_version = 'm',
					y_version = 'TIV'):

		with torch.no_grad():

			if y_version == 'IV':
				old_base, old_extra = self.base_gp._get_price_samples_1d()
				base, extra = self._get_price_samples_1d()
				y_label = 'IV'
			elif y_version == 'TIV':
				y_label = 'TIV'


			if ax is None:
				fig, ax = plt.subplots(figsize=figsize)

			if x_version == 'strike':
				x_centered_loc_call = self.CallData[x_version]
				x_centered_loc_put = self.PutData[x_version]

				T = self.data['T'].iloc[0]
				x_extra_loc = self._strike_calculator(T = T)
				x_train_loc = self.m_scaler.inverse(self.x_train)
				x_train_loc = self._strike_calculator(T = T, m = x_train_loc)
				x_label = 'Strike Price'

				knots = self.m_scaler.inverse(self.basis.knots[0])
				knot_locs = self._strike_calculator(T = T, m = knots)

			elif x_version == 'log_m':
				x_centered_loc_call = self.CallData['m']
				x_centered_loc_put = self.PutData['m']

				x_extra_loc = self.x_test_extra
				x_train_loc = self.m_scaler.inverse(self.x_train)
				x_label = 'Log-Moneyness'

				knot_locs = self.m_scaler.inverse(self.basis.knots[0])

			elif x_version == 'm':
				x_centered_loc_call = np.exp(self.CallData[x_version])
				x_centered_loc_put = np.exp(self.PutData[x_version])

				x_extra_loc = self.x_test_extra
				x_extra_loc = np.exp(x_extra_loc)
				x_train_loc = self.m_scaler.inverse(self.x_train)
				x_train_loc = np.exp(x_train_loc)
				x_label = 'Moneyness'

				knot_locs = np.exp(self.m_scaler.inverse(self.basis.knots[0]))

			else:
				raise ValueError("x_version must be 'strike', 'log_m', or 'm'")

			for ind, knot in enumerate(knot_locs):
				if ind == 0:
					ax.axvline(x=knot, color='gray', linestyle='--', alpha=0.2, label = 'Knot Locations')
				else:
					ax.axvline(x=knot, color='gray', linestyle='--', alpha=0.2)

			ax.plot(x_centered_loc_call, 
					self.CallData['mid_' + y_label], 
					label='Call: Mid-Price ' + y_label, color='firebrick',
					alpha=0.6)		
			ax.fill_between(x_centered_loc_call, 
							self.CallData['ask_' + y_label].values, 
							self.CallData['bid_' + y_label].values,
							alpha=0.8, label='Call: Bid-Ask ' + y_label + ' Spread',
							color = 'orange')

			ax.plot(x_centered_loc_put, 
					self.PutData['mid_' + y_label], 
					label='Put: Mid-Price ' + y_label, color='b',
					alpha=0.6)	

			ax.fill_between(x_centered_loc_put, 
					self.PutData['ask_' + y_label].values, 
					self.PutData['bid_' + y_label].values,
							alpha=0.8, label='Put: Bid-Ask ' + y_label + ' Spread',
							color = 'lightblue')


			if y_version == 'TIV':
				# Base GP
				ax.fill_between(x_extra_loc, 
								self.base_gp.pred_extra['lower'],  
								self.base_gp.pred_extra['upper'], 
								alpha=0.45, label='Unconstrained 95% CI for ' + y_label,
								color = 'indigo')		
				ax.plot(x_extra_loc, 
						self.base_gp.pred_extra['mean'], 
						'indigo',
						alpha = 0.45)
				
				# Constrained GP
				ax.fill_between(x_extra_loc, 
								self.pred_extra['lower'],  
								self.pred_extra['upper'], 
								alpha=0.30, label='Constrained 95% CI for ' + y_label,
								color = 'lime')		
				ax.plot(x_extra_loc, 
						self.pred_extra['mean'], 
						'lime',
						alpha = 0.30)
				
				ax.plot(x_train_loc, 
					self.y_scaler.inverse(self.y_train),
					'k*', label='Training Data', markersize=4)
			
			elif y_version == 'IV':
				# Base GP
				ax.fill_between(x_extra_loc, 
								old_extra['IV']['lb'],  
								old_extra['IV']['ub'], 
								alpha=0.45, label='Unconstrained 95% CI for ' + y_label,
								color = 'indigo')		
				ax.plot(x_extra_loc, 
						old_extra['IV']['mean'], 
						'indigo',
						alpha = 0.45)
				
				# Constrained GP
				ax.fill_between(x_extra_loc, 
								extra['IV']['lb'],  
								extra['IV']['ub'], 
								alpha=0.30, label='Constrained 95% CI for ' + y_label,
								color = 'lime')		
				ax.plot(x_extra_loc, 
						extra['IV']['mean'], 
						'lime',
						alpha = 0.30)
				
				
				T = self.data['T'].iloc[0]
				ax.plot(x_train_loc,
					np.sqrt(self.y_scaler.inverse(self.y_train) / T),
					'k*', label='Training Data', markersize=1)	

			
			ax.set_xlabel(x_label)
			y_lab = 'Implied Volatility' if y_label == 'IV' else 'Total Implied Variance'
			ax.set_ylabel(y_lab)
			ax.set_title(f"{y_lab} vs {x_label}\n Expiry Date: {self.data['expiration'].iloc[0]} -- Quote Date: {self.data['quote_date'].iloc[0]}")
			ax.legend()
			return ax
		
	def _plot_smile_centered(self, figsize=(20, 10),
								zoom=80, ax=None,
								x_version = 'm',
								y_version = 'TIV'):

		with torch.no_grad():

			if y_version == 'IV':
				old_base, old_extra = self.base_gp._get_price_samples_1d()
				base, extra = self._get_price_samples_1d()
				center_around = base['IV']['mean']
				center_around_extra = extra['IV']['mean']
				y_label = 'IV'
			elif y_version == 'TIV':
				center_around = self.pred['mean']
				center_around_extra = self.pred_extra['mean']
				y_label = 'TIV'


			if ax is None:
				fig, ax = plt.subplots(figsize=figsize)

			if x_version == 'strike':
				x_centered_loc_call = self.CallData[x_version]
				x_centered_loc_put = self.PutData[x_version]

				T = self.data['T'].iloc[0]
				x_extra_loc = self._strike_calculator(T = T)
				x_train_loc = self.m_scaler.inverse(self.x_train)
				x_train_loc = self._strike_calculator(T = T, m = x_train_loc)
				x_label = 'Strike Price'

				knots = self.m_scaler.inverse(self.basis.knots[0])
				knot_locs = self._strike_calculator(T = T, m = knots)

			elif x_version == 'log_m':
				x_centered_loc_call = self.CallData['m']
				x_centered_loc_put = self.PutData['m']

				x_extra_loc = self.x_test_extra
				x_train_loc = self.m_scaler.inverse(self.x_train)
				x_label = 'Log-Moneyness'

				knot_locs = self.m_scaler.inverse(self.basis.knots[0])

			elif x_version == 'm':
				x_centered_loc_call = np.exp(self.CallData[x_version])
				x_centered_loc_put = np.exp(self.PutData[x_version])

				x_extra_loc = self.x_test_extra
				x_extra_loc = np.exp(x_extra_loc)
				x_train_loc = self.m_scaler.inverse(self.x_train)
				x_train_loc = np.exp(x_train_loc)
				x_label = 'Moneyness'

				knot_locs = np.exp(self.m_scaler.inverse(self.basis.knots[0]))

			else:
				raise ValueError("x_version must be 'strike', 'log_m', or 'm'")

			for ind, knot in enumerate(knot_locs):
				if ind == 0:
					ax.axvline(x=knot, color='gray', linestyle='--', alpha=0.2, label = 'Knot Locations')
				else:
					ax.axvline(x=knot, color='gray', linestyle='--', alpha=0.2)


			ax.plot(x_centered_loc_call, 
					self.CallData['mid_' + y_label] - center_around[self.option_indices['base']['call']], 
					label='Call: Mid-Price ' + y_label, color='firebrick',
					alpha=0.8)		
			ax.fill_between(x_centered_loc_call, 
							self.CallData['ask_' + y_label].values - center_around[self.option_indices['base']['call']],
							self.CallData['bid_' + y_label].values - center_around[self.option_indices['base']['call']],
							alpha=0.6, label='Call: Bid-Ask ' + y_label + ' Spread',
							color = 'orange')

			ax.plot(x_centered_loc_put, 
					self.PutData['mid_' + y_label] - center_around[self.option_indices['base']['put']], 
					label='Put: Mid-Price ' + y_label, color='b',
					alpha=0.8)	

			ax.fill_between(x_centered_loc_put, 
					self.PutData['ask_' + y_label].values - center_around[self.option_indices['base']['put']], 
					self.PutData['bid_' + y_label].values - center_around[self.option_indices['base']['put']],
							alpha=0.6, label='Put: Bid-Ask ' + y_label + ' Spread',
							color = 'lightblue')


			if y_version == 'TIV':
				# Base GP
				ax.fill_between(x_extra_loc, 
								self.base_gp.pred_extra['lower'] - center_around_extra,  
								self.base_gp.pred_extra['upper'] - center_around_extra, 
								alpha=0.45, label='Unconstrained 95% CI for ' + y_label,
								color = 'indigo')		
				ax.plot(x_extra_loc, 
						self.base_gp.pred_extra['mean'] - center_around_extra, 
						'indigo',
						alpha = 0.45)
				
				# Constrained GP
				ax.fill_between(x_extra_loc, 
								self.pred_extra['lower'] - center_around_extra,  
								self.pred_extra['upper'] - center_around_extra, 
								alpha=0.30, label='Constrained 95% CI for ' + y_label,
								color = 'lime')		
				ax.plot(x_extra_loc, 
						self.pred_extra['mean'] - center_around_extra, 
						'lime',
						alpha = 0.45)
				
				ax.plot(x_train_loc, 
					np.zeros_like(self.x_train),
					'k*', label='Training Data', markersize=3)
			
			elif y_version == 'IV':
				# Base GP
				ax.fill_between(x_extra_loc, 
								old_extra['IV']['lb'] - center_around_extra,  
								old_extra['IV']['ub'] - center_around_extra, 
								alpha=0.45, label='Unconstrained 95% CI for ' + y_label,
								color = 'indigo')		
				ax.plot(x_extra_loc, 
						old_extra['IV']['mean'] - center_around_extra, 
						'indigo',
						alpha = 0.45)
				
				# Constrained GP
				ax.fill_between(x_extra_loc, 
								extra['IV']['lb'] - center_around_extra,  
								extra['IV']['ub'] - center_around_extra, 
								alpha=0.30, label='Constrained 95% CI for ' + y_label,
								color = 'lime')		
				ax.plot(x_extra_loc, 
						extra['IV']['mean'] - center_around_extra, 
						'lime',
						alpha = 0.45)
				
				
				T = self.data['T'].iloc[0]
				ax.plot(x_train_loc,
					np.zeros_like(self.x_train),
					'k*', label='Training Data', markersize=1)	

			
			ax.set_xlabel(x_label)
			y_lab = 'Implied Volatility' if y_label == 'IV' else 'Total Implied Variance'
			ax.set_ylabel(y_lab)
			ax.set_title(f"{y_lab} - Centered around Mean Prediction\n Expiry Date: {self.data['expiration'].iloc[0]} -- Quote Date: {self.data['quote_date'].iloc[0]}")
			ax.legend(loc='lower center')

			y_min, y_max = ax.get_ylim()
			y_abs = np.max([np.abs(y_min), np.abs(y_max)])
			perc = (100-zoom) / 100 
			ax.set_ylim(-y_abs * perc, y_abs * perc)
			return ax

	def plot_price_comparison(self, figsize=(24, 10), zoom=80, 
							x_version = 'log_m',
							x_lim_call = (None, None),
							x_lim_put = (None, None)):
		fig, (ax1, ax2) = plt.subplots(2, 2, figsize=figsize)
		self._plot_price(ax=ax1[1], x_version=x_version, option_type = 'Call',
						x_lim = x_lim_call)
		self._plot_price_centered(ax=ax1[0], zoom=zoom, x_version=x_version, option_type = 'Call')

		self._plot_price(ax=ax2[1], x_version=x_version, option_type = 'Put',
						x_lim = x_lim_put)
		self._plot_price_centered(ax=ax2[0], zoom=zoom, x_version=x_version, option_type = 'Put')
		plt.tight_layout()
		plt.show()
		return self

	def _plot_price(self, figsize=(20, 10),
					ax=None,
					option_type = 'Call',
					x_version = 'm',
					x_lim = (None, None)):
		with torch.no_grad():

			old_base, old_extra = self.base_gp._get_price_samples_1d()
			base, extra = self._get_price_samples_1d()
			
			if option_type == 'Call':
				y_label = 'Call'
			elif option_type == 'Put':
				y_label = 'Put'
			else:
				raise ValueError("option_type must be 'Call' or 'Put'")
			if ax is None:
				fig, ax = plt.subplots(figsize=figsize)

			if x_version == 'strike':
				x_centered_loc_call = self.CallData[x_version]
				x_centered_loc_put = self.PutData[x_version]
			if x_version == 'log_m':
				x_centered_loc_call = self.CallData['m']
				x_centered_loc_put = self.PutData['m']
			if x_version == 'm':
				x_centered_loc_call = np.exp(self.CallData[x_version])
				x_centered_loc_put = np.exp(self.PutData[x_version])

			if option_type == 'Call':
				ax.plot(x_centered_loc_call, 
						self.CallData['mid'], 
						label='Call: Mid-Price', color='firebrick',
						alpha=0.6)		
				ax.fill_between(x_centered_loc_call, 
								self.CallData[self.col_map['ask']].values, 
								self.CallData[self.col_map['bid']].values,
								alpha=0.8, label='Call: Bid-Ask ' + 'Spread',
								color = 'orange')
			elif option_type == 'Put':
				ax.plot(x_centered_loc_put, 
						self.PutData['mid'], 
						label='Put: Mid-Price', color='b',
						alpha=0.6)		
				ax.fill_between(x_centered_loc_put, 
								self.PutData[self.col_map['ask']].values, 
								self.PutData[self.col_map['bid']].values,
								alpha=0.8, label='Put: Bid-Ask ' + 'Spread',
								color = 'lightblue')

			if x_version == 'strike':
				T = self.data['T'].iloc[0]
				x_extra_loc = self._strike_calculator(T = T)
				x_train_loc = self.m_scaler.inverse(self.x_train)
				x_train_loc = self._strike_calculator(T = T, m = x_train_loc)
				x_label = 'Strike Price'
			
			elif x_version == 'log_m':
				x_extra_loc = self.x_test_extra
				x_train_loc = self.m_scaler.inverse(self.x_train)
				x_label = 'Log-Moneyness'
			
			elif x_version == 'm':
				x_extra_loc = self.x_test_extra
				x_extra_loc = np.exp(x_extra_loc)
				x_train_loc = self.m_scaler.inverse(self.x_train)
				x_train_loc = np.exp(x_train_loc)
				x_label = 'Moneyness'
			else:
				raise ValueError("x_version must be 'strike', 'log_m', or 'm'")

			if option_type == 'Call':
				# Base GP
				ax.fill_between(x_extra_loc, 
								old_extra['Call']['lb'],  
								old_extra['Call']['ub'], 
								alpha=0.45, label='Unconstrained 95% CI for ' + y_label,
								color = 'indigo')		
				ax.plot(x_extra_loc, 
						extra['Call']['mean'], 
						'indigo',
						alpha = 0.45)
				
				# Constrained GP
				ax.fill_between(x_extra_loc, 
								extra['Call']['lb'],  
								extra['Call']['ub'], 
								alpha=0.30, label='Constrained 95% CI for ' + y_label,
								color = 'lime')
				ax.plot(x_extra_loc,
						extra['Call']['mean'], 
						'lime',
						alpha = 0.30)
				
				epsilon = 1e-6  
				y_train = self.CallData[
					self.CallData['m'].apply(
						lambda mid_tiv: any(abs(mid_tiv - y) <= epsilon for y in self.m_scaler.inverse(self.x_train))
					)
				].loc[:, 'mid'].values
				y_train = np.repeat(y_train, 2)
				# ax.plot(x_train_loc, 
				# 	y_train, 
				# 	'k*', label='Training Locations', markersize=1)
				
			elif option_type == 'Put':
				# Base GP
				ax.fill_between(x_extra_loc, 
								old_extra['Put']['lb'],  
								old_extra['Put']['ub'], 
								alpha=0.45, label='Unconstrained 95% CI for ' + y_label,
								color = 'indigo')		
				ax.plot(x_extra_loc, 
						extra['Put']['mean'], 
						'indigo',
						alpha = 0.45)

				# Constrained GP
				ax.fill_between(x_extra_loc, 
								extra['Put']['lb'],  
								extra['Put']['ub'], 
								alpha=0.30, label='Constrained 95% CI for ' + y_label,
								color = 'lime')
				ax.plot(x_extra_loc,
						extra['Put']['mean'], 
						'lime',
						alpha = 0.30)

				epsilon = 1e-6  
				y_train = self.PutData[
					self.PutData['m'].apply(
						lambda mid_tiv: any(abs(mid_tiv - y) <= epsilon for y in self.m_scaler.inverse(self.x_train))
					)
				].loc[:, 'mid'].values
				y_train = np.repeat(y_train, 2)
				# ax.plot(x_train_loc, 
				# 	y_train, 
				# 	'k*', label='Training Locations', markersize=1)
				
			ax.set_xlabel(x_label)
			y_lab = f"{option_type} Price"
			ax.set_ylabel(y_lab)
			ax.set_title(f"{y_lab} vs {x_label}\n Expiry Date: {self.data['expiration'].iloc[0]} -- Quote Date: {self.data['quote_date'].iloc[0]}")
			if option_type == 'Put':	
				ax.legend(loc='upper left')
			elif option_type == 'Call':
				ax.legend(loc='upper right')

			x_min, x_max = ax.get_xlim()
			if x_lim[0] is not None and x_lim[1] is not None:
				if x_lim[0] > x_min:
					x_min = x_lim[0]
				if x_lim[1] < x_max:
					x_max = x_lim[1]

			ax.set_xlim(x_min, x_max)

			def autoscale_remaining_axis(ax, respect_xlim=False, respect_ylim=False, padding=0.05):
				"""
				Autoscale the axis not manually set, with optional padding (as a fraction of range).
				Set respect_xlim=True if you set xlim manually and want to autoscale ylim.
				Set respect_ylim=True if you set ylim manually and want to autoscale xlim.
				"""
				xlim = ax.get_xlim()
				ylim = ax.get_ylim()

				visible_x = []
				visible_y = []

				for line in ax.get_lines():
					x = line.get_xdata()
					y = line.get_ydata()

					if hasattr(x, '__getitem__') and hasattr(y, '__getitem__'):
						if respect_xlim:
							mask = (x >= xlim[0]) & (x <= xlim[1])
							visible_y.extend(y[mask])

						if respect_ylim:
							mask = (y >= ylim[0]) & (y <= ylim[1])
							visible_x.extend(x[mask])

				if respect_xlim and visible_y:
					ymin, ymax = min(visible_y), max(visible_y)
					yrange = ymax - ymin or 1e-8  # avoid zero height
					pad = yrange * padding
					ax.set_ylim(ymin - pad, ymax + pad)

				if respect_ylim and visible_x:
					xmin, xmax = min(visible_x), max(visible_x)
					xrange = xmax - xmin or 1e-8  # avoid zero width
					pad = xrange * padding
					ax.set_xlim(xmin - pad, xmax + pad)

			autoscale_remaining_axis(ax, respect_xlim=True)

		return ax

	def _plot_price_centered(self, figsize=(20, 10),
							zoom=0, ax=None,
							option_type = 'Call',
							x_version = 'm'):
		with torch.no_grad():

			old_base, old_extra = self.base_gp._get_price_samples_1d()
			base, extra = self._get_price_samples_1d()
			
			if option_type == 'Call':
				center_around = base['Call']['mean']
				center_around_extra = extra['Call']['mean']
				y_label = 'Call'
			elif option_type == 'Put':
				center_around = base['Put']['mean']
				center_around_extra = extra['Put']['mean']
				y_label = 'Put'
			else:
				raise ValueError("option_type must be 'Call' or 'Put'")
			if ax is None:
				fig, ax = plt.subplots(figsize=figsize)

			if x_version == 'strike':
				x_centered_loc_call = self.CallData[x_version]
				x_centered_loc_put = self.PutData[x_version]
			if x_version == 'log_m':
				x_centered_loc_call = self.CallData['m']
				x_centered_loc_put = self.PutData['m']
			if x_version == 'm':
				x_centered_loc_call = np.exp(self.CallData[x_version])
				x_centered_loc_put = np.exp(self.PutData[x_version])

			if option_type == 'Call':
				ax.plot(x_centered_loc_call, 
						self.CallData['mid'] - center_around, 
						label='Call: Mid-Price', color='firebrick',
						alpha=0.60)		
				ax.fill_between(x_centered_loc_call, 
								self.CallData[self.col_map['ask']].values - center_around, 
								self.CallData[self.col_map['bid']].values - center_around,
								alpha=0.80, label='Call: Bid-Ask ' + 'Spread',
								color = 'orange')
			elif option_type == 'Put':
				ax.plot(x_centered_loc_put, 
						self.PutData['mid'] - center_around, 
						label='Put: Mid-Price', color='b',
						alpha=0.60)		
				ax.fill_between(x_centered_loc_put, 
								self.PutData[self.col_map['ask']].values - center_around, 
								self.PutData[self.col_map['bid']].values - center_around,
								alpha=0.80, label='Put: Bid-Ask ' + 'Spread',
								color = 'lightblue')

			if x_version == 'strike':
				T = self.data['T'].iloc[0]
				x_extra_loc = self._strike_calculator(T = T)
				x_train_loc = self.m_scaler.inverse(self.x_train)
				x_train_loc = self._strike_calculator(T = T, m = x_train_loc)
				x_label = 'Strike Price'
			
			elif x_version == 'log_m':
				x_extra_loc = self.x_test_extra
				x_train_loc = self.m_scaler.inverse(self.x_train)
				x_label = 'Log-Moneyness'
			
			elif x_version == 'm':
				x_extra_loc = self.x_test_extra
				x_extra_loc = np.exp(x_extra_loc)
				x_train_loc = self.m_scaler.inverse(self.x_train)
				x_train_loc = np.exp(x_train_loc)
				x_label = 'Moneyness'
			else:
				raise ValueError("x_version must be 'strike', 'log_m', or 'm'")

			if option_type == 'Call':
				# Base GP
				ax.fill_between(x_extra_loc, 
								old_extra['Call']['lb'] - center_around_extra,  
								old_extra['Call']['ub'] - center_around_extra, 
								alpha=0.45, label='Unconstrained 95% CI for ' + y_label,
								color = 'indigo')		
				ax.plot(x_extra_loc, 
						old_extra['Call']['mean'] - center_around_extra, 
						'indigo',
						alpha = 0.45)
				
				# Constrained GP
				ax.fill_between(x_extra_loc, 
								extra['Call']['lb'] - center_around_extra,  
								extra['Call']['ub'] - center_around_extra, 
								alpha=0.30, label='Constrained 95% CI for ' + y_label,
								color = 'lime')
				ax.plot(x_extra_loc,
						extra['Call']['mean'] - center_around_extra, 
						'lime', 
						alpha = 0.30)

			elif option_type == 'Put':
				# Base GP
				ax.fill_between(x_extra_loc, 
								old_extra['Put']['lb'] - center_around_extra,  
								old_extra['Put']['ub'] - center_around_extra, 
								alpha=0.45, label='Unconstrained 95% CI for ' + y_label,
								color = 'indigo')		
				ax.plot(x_extra_loc, 
						old_extra['Put']['mean'] - center_around_extra, 
						'indigo',
						alpha = 0.45)
		
				# Constrained GP
				ax.fill_between(x_extra_loc, 
								extra['Put']['lb'] - center_around_extra,  
								extra['Put']['ub'] - center_around_extra, 
								alpha=0.30, label='Constrained 95% CI for ' + y_label,
								color = 'lime')
				ax.plot(x_extra_loc,
						extra['Put']['mean'] - center_around_extra, 
						'lime',
						alpha = 0.30)

			ax.plot(x_train_loc, 
					np.zeros_like(self.x_train), 
					'k*', label='Training Locations', markersize=1)
			ax.set_xlabel(x_label)
			y_lab = f"{option_type} Price"
			ax.set_ylabel(y_lab)
			ax.set_title(f"{y_lab} - Centered Around Mean Prediction\n Expiry Date: {self.data['expiration'].iloc[0]} -- Quote Date: {self.data['quote_date'].iloc[0]}")
			if option_type == 'Put':	
				ax.legend(loc='upper left')
			elif option_type == 'Call':
				ax.legend(loc='upper right')
			y_min, y_max = ax.get_ylim()
			y_abs = np.max([np.abs(y_min), np.abs(y_max)])
			perc = (100-zoom) / 100 
			ax.set_ylim(-y_abs * perc, y_abs * perc)
			
			return ax

	def plot_derivatives(self, space = 'normalized', figsize=(14, 7)):

		fig, axx = plt.subplots(nrows=1, ncols=2, figsize=figsize)

		ax1 = axx[0]
		ax2 = axx[1]

		if space == 'native':
			grad_results = self.pred_deriv_fd
			grad_result_analytical = self.pred_deriv

			x_locs = self.x_test_extra
			x_label = 'Log-moneyness'

		elif space == 'normalized':
			grad_results = self.f_tilde_fd
			grad_result_analytical = self.f_tilde

			x_locs = self.m_scaler.normalize(self.x_test_extra)
			x_label = 'Normalized log-moneyness'
		else:
			raise ValueError("Invalid space. Choose 'native' or 'transformed'.")


		cmap = plt.get_cmap('Spectral', 11)
		norm = plt.Normalize(vmin=0, vmax=1)

		def _get_subplot(ax,
						x_locs, x_label, 
						grad_results, grad_result_analytical, 
						y_label = 'First Derivative',
						title = 'First Derivative'):

			ax.hlines(0, x_locs[0], x_locs[-1], color='k', lw=1, ls='--', alpha=0.25)
			for i in range(1, 11):
				lower = grad_results['quantiles'][i-1, :]
				upper = grad_results['quantiles'][i, :]
				ax.fill_between(
					x_locs,
					lower,
					upper,
					alpha=0.5,
					color=cmap(i)
				)
			
			ax.plot(x_locs, grad_result_analytical, 'k-', lw=1, label='Mean Derivative')
			ax.set_xlabel(x_label)
			ax.set_ylabel(y_label)
			ax.set_title(title)
			ax.legend()

			return ax


		_get_subplot(ax1,
					x_locs, x_label, 
					grad_results['first_derivative'],
					grad_result_analytical['first_derivative'],
					y_label = 'First Derivative',
					title = 'Distribution of First Derivatives: After '+str(self.iter_id)+' Iteration')

		_get_subplot(ax2,
					x_locs, x_label, 
					grad_results['second_derivative'],
					grad_result_analytical['second_derivative'],
					y_label = 'Second Derivative',
					title = 'Distribution of Second Derivatives: After '+str(self.iter_id)+' Iteration')


		sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
		sm.set_array([])            # gives the colorbar something to map
		cbar = fig.colorbar(sm, ax=ax2, ticks=np.linspace(0, 1, 11))
		cbar.set_label('Percentile')
		plt.tight_layout()
		plt.show()

	def plot_price_derivatives(self, option_type = 'Call', figsize=(18, 7)):

		if not (option_type == 'Call' or option_type == 'Put'):
			raise ValueError('Option type must be either Call or Put')

		results = self.price_gradient[option_type]['derivative']
		strikes = self.price_gradient[option_type]['strikes']

		fig,ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 7))
		cmap = plt.get_cmap('Spectral', 11)
		norm = plt.Normalize(vmin=0, vmax=1)

		def _get_subplot(ax,
						x_locs, x_label, 
						grad_results,
						y_label = 'First Derivative',
						title = 'First Derivative',
						center = False):

			if center:
				center_around = grad_results['mean']
			else:
				center_around = 0

			ax.hlines(0, x_locs[0], x_locs[-1], color='k', lw=1, ls='--', alpha=0.25)
			for i in range(1, 11):
				lower = grad_results['quantiles'][i-1, :] - center_around
				upper = grad_results['quantiles'][i, :] - center_around
				ax.fill_between(
					x_locs,
					lower,
					upper,
					alpha=0.5,
					color=cmap(i)
				)
			
		#	ax.plot(x_locs, grad_results['mean'], 'k-', lw=1, label='Analytical Derivative')
			ax.set_xlabel(x_label)
			ax.set_ylabel(y_label)
			ax.set_title(title)

			return ax

		def _get_subplot_2(ax,
						x_locs, x_label, 
						grad_results,
						y_label = 'First Derivative',
						title = 'First Derivative',
						center = False):

			if center:
				center_around = grad_results['mean']
			else:
				center_around = 0

			ax.hlines(0, x_locs[0], x_locs[-1], color='k', lw=1, ls='--', alpha=0.25)
			num_samples = grad_results['samples'].shape[0]
			cmap_2 = plt.get_cmap('prism', num_samples)
			for i in range(num_samples):
				ax.plot(x_locs, grad_results['samples'][i] - center_around , color = cmap_2(i), lw=1, alpha = 1 - 0.8/(num_samples+1))
			
			ax.set_xlabel(x_label)
			ax.set_ylabel(y_label)
			ax.set_title(title)

			return ax


		_get_subplot(ax[0,0],
			x_locs = strikes,
			x_label = 'Strike Price',
			grad_results = results['function_val'],
			y_label = 'Price Difference',
			title = option_type +' Price - Centered Around Mean Prediction',
			center = True
		)

		_get_subplot(ax[0,1],
			x_locs = strikes,
			x_label = 'Strike Price',
			grad_results = results['first_derivative'],
			y_label = 'First Derivative',
			title = option_type +' Price - First Derivative vs Strike Price',
		)

		_get_subplot(ax[0,2],
			x_locs = strikes,
			x_label = 'Strike Price',
			grad_results = results['second_derivative'],
			y_label = 'Second Derivative',
			title = option_type +' Price - Second Derivative vs Strike Price',
		)

		_get_subplot_2(ax[1,0],
			x_locs = strikes,
			x_label = 'Strike Price',
			grad_results = results['function_val'],
			y_label = 'Price Difference',
			title =option_type +' Price - Centered Around Mean Prediction',
			center = True
		)

		_get_subplot_2(ax[1,1],
			x_locs = strikes,
			x_label = 'Strike Price',
			grad_results = results['first_derivative'],
			y_label = 'First Derivative',
			title = option_type +' Price - First Derivative vs Strike Price',
			center = False
		)

		_get_subplot_2(ax[1,2],
			x_locs = strikes,
			x_label = 'Strike Price',
			grad_results = results['second_derivative'],
			y_label = 'Second Derivative',
			title = option_type +' Price - Second Derivative vs Strike Price',
			center = False
		)


		sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
		sm.set_array([])            # gives the colorbar something to map
		cbar = fig.colorbar(sm, ax=ax[0,2], ticks=np.linspace(0, 1, 11))
		cbar.set_label('Quantile')
		plt.tight_layout()
		plt.show()

	# Helper functions

	def _compute_gamma(self):
		m_total = np.prod(self.basis.m)
		d = self.basis.d

		if d == 1:
			knots = self.basis.knots[0]  # shape (m,)
			knots = knots[:, None]  # shape (m, 1)
			Gamma_np = self.kernel(knots, knots)  # shape (m, m)
		else:
			# Build full tensor product of knots (m_total, d)
			mesh = np.meshgrid(*self.basis.knots, indexing="ij")
			grid = np.stack(mesh, axis=-1).reshape(-1, d)  # shape (m_total, d)

			# Now compute kernel for each dimension separately
			Gamma_np = np.ones((m_total, m_total))
			for dim in range(d):
				k = self.kernel  # still assumes isotropic kernel, modify if ARD
				xi = grid[:, dim:dim+1]  # shape (m_total, 1)
				Gamma_np *= k(xi, xi.T)
		
		self.Gamma = Gamma_np
		return Gamma_np

	def _wrap_gpytorch_kernel(self, gp_kernel):
		device = next(gp_kernel.parameters()).device

		def kernel(x1, x2):
			x1_tensor = torch.tensor(x1, dtype=torch.float32, device=device)
			x2_tensor = torch.tensor(x2, dtype=torch.float32, device=device)
			with torch.no_grad():
				k_val = gp_kernel(x1_tensor, x2_tensor).evaluate()
			return k_val.cpu().numpy()

		return kernel

	def _finite_difference_derivative_calculator(self, x_test, test_samples, epsilon_m):
		N = len(x_test)
		M = test_samples.shape[1]

		# Preallocate
		first_derivative = np.zeros([N,M])
		second_derivative = np.zeros([N,M])
		function_val = np.zeros([N,M])

		# --- Left boundary---
		f_0 = test_samples[0,:]; f_1 = test_samples[1,:]; f_2 =  test_samples[2,:]; f_3 = test_samples[3,:]	
		first_derivative[0,:] = (-3*f_0 + 4*f_1 - f_2) / (2 * epsilon_m)
		second_derivative[0,:] = (2*f_0 - 5*f_1 + 4*f_2 - f_3) / (epsilon_m**2)
		function_val[0,:] = f_0

		# --- Right boundary ---
		f_N0 = test_samples[-1,:]; f_N1 = test_samples[-2,:]; f_N2 =  test_samples[-3,:]; f_N3 = test_samples[-4,:]
		first_derivative[-1,:] = (3*f_N0 - 4*f_N1 + f_N2) / (2 * epsilon_m)
		second_derivative[-1,:] = (2*f_N0 - 5*f_N1 + 4*f_N2 - f_N3) / (epsilon_m**2)
		function_val[-1,:] = f_N0

		# --- Interior (2nd-order central) ---
		f_m1 = test_samples[0:-2,:]; f_cen = test_samples[1:-1,:]; f_p1 = test_samples[2:,:]
		first_derivative[1:-1,:] = (f_p1 - f_m1) / (2 * epsilon_m)
		second_derivative[1:-1,:] = (f_p1 - 2*f_cen + f_m1) / (epsilon_m**2)
		function_val[1:-1,:] = f_cen

		return first_derivative, second_derivative, function_val

	def _strike_calculator(self, T, m = None):
		"""
		Calculate the strike price based on moneyness and time to maturity.
		"""
		if m is None:
			m = self.x_test_extra

		maturity_data = self.data[self.data['T'] == T]
		r = maturity_data['r'].iloc[0]
		q = maturity_data[self.div_type].iloc[0]
		S_0 = maturity_data[self.col_map['S_0']].iloc[0]
		# Calculate the strike price using the formula
		return S_0 * np.exp(m) * np.exp((r - q) * T)

	def _get_price_samples_1d(self):

		samples = self.pred['samples']
		extra_samples = self.pred_extra['samples']

		T,r = self.data['T'].iloc[0], self.data['r'].iloc[0] 
		q, S_0 = self.data[self.div_type].iloc[0], self.data[self.col_map['S_0']].iloc[0]

		IV_base_samples = np.sqrt(samples[:,] / T)
		IV_base_res = {
			'mean': IV_base_samples.mean(axis=0),
			'lb': np.quantile(IV_base_samples, 0.025, axis=0),
			'ub': np.quantile(IV_base_samples, 0.975, axis=0),
		}

		IV_extra_samples = np.sqrt(extra_samples[:,] / T)
		IV_extra_res = {
			'mean': IV_extra_samples.mean(axis=0),
			'lb': np.quantile(IV_extra_samples, 0.025, axis=0),
			'ub': np.quantile(IV_extra_samples, 0.975, axis=0),
		}

		K_base = self.CallData[self.col_map['strike']].values
		K_extra = self._strike_calculator(T = T, m = self.x_test_extra)

		# Call Prices
		Call_base_samples = BS.BS_Call(S_0, K_base, r, T, IV_base_samples[:,self.option_indices['base']['call']], q)
		Call_extra_samples = BS.BS_Call(S_0, K_extra, r, T, IV_extra_samples, q)

		Call_base_res = {
			'mean': Call_base_samples.mean(axis=0),
			'lb': np.quantile(Call_base_samples, 0.025, axis=0),
			'ub': np.quantile(Call_base_samples, 0.975, axis=0),
		}
		Call_extra_res = {
			'mean': Call_extra_samples.mean(axis=0),
			'lb': np.quantile(Call_extra_samples, 0.025, axis=0),
			'ub': np.quantile(Call_extra_samples, 0.975, axis=0),
			'samples': Call_extra_samples
		}


		K_base = self.PutData[self.col_map['strike']].values

		# Put Prices
		Put_base_samples = BS.BS_Put(S_0, K_base, r, T, IV_base_samples[:,self.option_indices['base']['put']], q)
		Put_extra_samples = BS.BS_Put(S_0, K_extra, r, T, IV_extra_samples, q)
		Put_base_res = {
			'mean': Put_base_samples.mean(axis=0),
			'lb': np.quantile(Put_base_samples, 0.025, axis=0),
			'ub': np.quantile(Put_base_samples, 0.975, axis=0),
		}
		Put_extra_res = {
			'mean': Put_extra_samples.mean(axis=0),
			'lb': np.quantile(Put_extra_samples, 0.025, axis=0),
			'ub': np.quantile(Put_extra_samples, 0.975, axis=0),
			'samples': Put_extra_samples
		}


		base_res = {
			'IV': IV_base_res,
			'Call': Call_base_res,
			'Put': Put_base_res
		}
		extra_res = {
			'IV': IV_extra_res,
			'Call': Call_extra_res,
			'Put': Put_extra_res
		}
		
		return base_res, extra_res
	
	def _price_finite_difference_gradient_calculator(self, test_samples, x_test = None, epsilon_m = None):

		if x_test is None or epsilon_m is None:
			x_test = self.m_scaler.normalize(self.x_test_extra)
			epsilon_m = x_test[1] - x_test[0]

		test_samples = test_samples.T
		first_derivative, second_derivative, function_val = self._finite_difference_derivative_calculator(x_test, test_samples, epsilon_m)

		# Adjust for m normalization:
		m_spread =  (self.m_scaler.inverse(1) - self.m_scaler.inverse(0))
		C_m = 1/ m_spread

		T = self.data['T'].iloc[0]
		strikes= self._strike_calculator(T=T)

		first_derivative_updated = (first_derivative.T * C_m / strikes).T
		second_derivative_updated = (second_derivative.T * C_m**2 / strikes**2 - first_derivative.T * C_m / strikes**2).T

		sec_der_neg_part = -np.minimum(second_derivative_updated, 0)
		perc_of_satisfied_condition = np.sum(second_derivative_updated >= 0, axis = 1) / second_derivative_updated.shape[1]
		
		# Randomly sample 10% of the derivatives
		rand_indices = np.random.choice(first_derivative_updated.shape[1], 5, replace=False)

		def get_results(derivative_samples, rand_indices):
			mean = np.mean(derivative_samples, axis=1)
			std = np.std(derivative_samples, axis=1)
			quantiles = np.quantile(derivative_samples, np.linspace(0,1,11), axis=1)			
			derivative_samples = derivative_samples[:, rand_indices].T
			return {
				'mean': mean,
				'std': std,
				'quantiles': quantiles,
				'samples': derivative_samples
			}
		
		# Derivative in Native space
		results = {
			'first_derivative': get_results(first_derivative_updated,rand_indices),
			'second_derivative': get_results(second_derivative_updated,rand_indices),
			'function_val': get_results(function_val,rand_indices),
			'second_derivative_neg_part': get_results(sec_der_neg_part,rand_indices),
			'perc_of_feasible_samples': perc_of_satisfied_condition
		}
		return results, strikes

	# Constraint Helper Functions

	def get_linear_constraint_coefficients(self, knots = None):

		m_spread = self.m_scaler.inverse(1) - self.m_scaler.inverse(0)
		y_max, y_min = self.y_scaler.max_, self.y_scaler.min_
		y_spread = y_max - y_min

		mean = self.base_gp.f_tilde['function_val']

		f = self.y_scaler.inverse(mean) # S_y^-1(f_tilde(m_tilde))
		f_tilde_prime = self.base_gp.f_tilde['first_derivative'] 

		C_m = 1/m_spread
		C_y = y_spread / self.y_scaling_factor

		m = self.m_scaler.inverse(knots) # S_m^-1(m_tilde)
		component_1 = ((m * C_y * C_m) / (2 * f))**2 * f_tilde_prime
		component_2 = (C_y * C_m / 2)**2 * ( 1/f + 0.25) * f_tilde_prime
		component_3 = m * C_y * C_m / f

		phi = (component_1 - component_2 - component_3)
		kappa = C_y * C_m**2 / 2

		self.const_coef = {
			'phi': phi,
			'kappa': kappa
		}

		return phi, kappa

	def build_constraint_matrix(self, knots = None):
		"""
		Construct the NxN matrix Lambda such that Lambda @ xi >= -1
		where xi = [f(m_1), ..., f(m_N)] and
		the inequality is: phi_i * f'(m_i) + kappa * f''(m_i) >= -1

		Parameters:
			N     : int        -- number of grid points
			h     : float      -- grid spacing (uniform)
			phi   : ndarray    -- shape (N,), values of phi_i at each point
			kappa : float      -- fixed scalar multiplier of second derivative

		Returns:
			Lambda : ndarray of shape (N, N)
			b      : ndarray of shape (N,), equal to -1
		"""
		phi = self.const_coef['phi']
		kappa = self.const_coef['kappa']
		N = knots.shape[0]

		# Potential Fix Later
		h = knots[1] - knots[0]


		Lambda = np.zeros((N, N))
		b = -np.ones(N)

		for i in range(N):
			if i == 0:
				# Left boundary
				Lambda[i, 0] = 2 * kappa / h**2 - 3 * phi[i] / (2*h)
				Lambda[i, 1] =  -5 * kappa / h**2 + 4 * phi[i] / (2*h)
				Lambda[i, 2] = 4 * kappa / h**2 - 1 * phi[i] / (2*h)
				Lambda[i, 3] = -  kappa / h**2 

			elif i == N - 1:
				# Right boundary
				Lambda[i, N-4] = -1 * kappa / h**2
				Lambda[i, N-3] = 4 * kappa / h**2 +  phi[i] / (2*h)
				Lambda[i, N-2] =  -5 * kappa / h**2 - 4 * phi[i] / (2*h)
				Lambda[i, N-1] = 2 * kappa / h**2 + 3 * phi[i] / (2*h)

			else:
				# Interior points
				Lambda[i, i-1] = -phi[i] / (2*h) + kappa / h**2
				Lambda[i, i]   = -2 * kappa / h**2
				Lambda[i, i+1] =  phi[i] / (2*h) + kappa / h**2

		return Lambda, b

	def make_anchor_constraint_from_min_obs(self, knots = None, epsilon=1e-6):
		"""
		Generate an anchoring constraint by identifying the training point with the minimum y-value
		and anchoring the closest knot in the basis function expansion.

		Parameters
		----------
		x_train : array-like, shape (n,) or (n, 1)
			Training inputs (unnormalized, real domain).
		y_train : array-like, shape (n,)
			Training targets.
		noise : array-like, shape (n,) or scalar
			Observation noise values for each training point.
		knots : array-like, shape (m,)
			Locations of the knots used in the basis.
		m : int
			Number of basis functions (i.e., length of `knots`).
		epsilon : float
			Minimum spread allowed for anchoring (to avoid degeneracy).

		Returns
		-------
		Lambda_anchor : ndarray of shape (1, m)
			A row vector selecting one coefficient for anchoring.
		lower_anchor : float
			Lower bound on that coefficient.
		upper_anchor : float
			Upper bound on that coefficient.
		"""
		x_train = np.asarray(self.x_train).squeeze()
		y_train = np.asarray(self.y_train).squeeze()
		noise = np.asarray(self.noise.diagonal()).squeeze()
		
		m = knots.shape[0]
		knots = np.asarray(knots).squeeze()
		

		if x_train.ndim != 1:
			raise ValueError("x_train must be 1-dimensional after squeezing.")
		if y_train.ndim != 1:
			raise ValueError("y_train must be 1-dimensional.")
		if noise.ndim == 0:
			noise = np.full_like(y_train, noise)  # broadcast scalar noise

		# Step 1: Find index of minimum y
		min_idx = np.argmin(y_train)
		x_min = x_train[min_idx]
		y_min = y_train[min_idx]
		# Step 2: Compute spread
		delta = max(5 * np.sqrt(noise[min_idx]), epsilon)

		# Step 3: Find closest knot
		closest_knot_idx = np.argmin(np.abs(knots - x_min))

		# Step 4: Build linear constraint row
		Lambda_anchor = np.zeros((1, m))
		Lambda_anchor[0, closest_knot_idx] = 1.0

		lower_anchor = y_min - delta
		upper_anchor = y_min + delta

		return Lambda_anchor, lower_anchor, upper_anchor

	def get_constraint_matrix(self, knots=None):
		"""
		Returns:
		- constraint_matrix: np.ndarray, linear inequality constraint matrix (Λ).
		- lower_bound: np.ndarray, lower bounds for Λξ.
		- upper_bound: np.ndarray, upper bounds for Λξ.
		"""
		if knots is None:
			if self.knots is None:
				raise ValueError("Knots must be provided or set in the class.")
			else:
				self.knots = self.basis.knots[0]

		phi, kappa = self.get_linear_constraint_coefficients(knots= self.knots)
		Lambda, ll = self.build_constraint_matrix(knots= self.knots)		
		Lambda_anchor, lower_anchor, upper_anchor = self.make_anchor_constraint_from_min_obs(knots=self.knots)

		constr_matrix = np.vstack((Lambda, Lambda_anchor))
		lower_limit = np.hstack((ll, lower_anchor))
		upper_limit = np.hstack((np.ones_like(ll) * np.inf, upper_anchor))

		self.constraint_matrix = constr_matrix
		self.lower_bound = lower_limit
		self.upper_bound = upper_limit

		return self






import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
from rpy2.robjects import numpy2ri

def sample_MVTN(mean, cov, lower, upper, mod, n_samples=1000):
	"""
	Sample from a multivariate truncated normal distribution using R's tmvtnorm package.
	"""

	# Enable NumPy <-> R auto-conversion
	numpy2ri.activate()

	# Import the tmvtnorm package
	tmvtnorm = importr("tmvtnorm")

	# Wrap for R
	mean_r = FloatVector(mean.tolist())
	mod_r = FloatVector(mod.tolist())
	lower_r = FloatVector(lower.tolist())
	upper_r = FloatVector(upper.tolist())

	def to_r_matrix(array):
		return robjects.r.matrix(FloatVector(array.flatten('F')), nrow=array.shape[0])

	#new_cov += 1e-3 * np.eye(new_cov.shape[0])
	cov_r = to_r_matrix(cov)

	# Call R function
	samples_r = tmvtnorm.rtmvnorm(
		n=n_samples,
		mean=mean_r,
		sigma=cov_r,
		lower=lower_r,
		upper=upper_r,
		algorithm = "gibbs",
		**{
		'burn.in.samples': 100,
		'thinning': 10,
		'start.value': mod_r
		}
	)

	return samples_r
