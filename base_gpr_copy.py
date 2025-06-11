import numpy as np
import scipy.stats as stats
import warnings
import pandas as pd

import torch
import torch.optim.lr_scheduler as lr_scheduler
import gpytorch
from linear_operator.settings import max_cg_iterations
from gpytorch.priors import LogNormalPrior

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt

from BS_utils import bs_functions as BS

to_32bit = lambda x: np.array(x, dtype=np.float32)

class MinMaxScaler:
	def __init__(self, array: np.ndarray, y_scaling_factor= None):
		self.min_ = np.min(array)
		self.max_ = np.max(array)
		self.y_scaling_factor = y_scaling_factor
		if self.max_ == self.min_:
			self.normalize = lambda x: x
			self.inverse = lambda x: x
	def normalize(self, x: np.ndarray) -> np.ndarray:
		if self.y_scaling_factor is not None:
			return self.y_scaling_factor*(x - self.min_) / (self.max_ - self.min_)
		else:
			return (x - self.min_) / (self.max_ - self.min_)
	def inverse(self, x_norm: np.ndarray) -> np.ndarray:
		if self.y_scaling_factor is not None:
			return (x_norm / self.y_scaling_factor) * (self.max_ - self.min_) + self.min_
		else:
			return x_norm * (self.max_ - self.min_) + self.min_

class FixedMultiplierKernel(gpytorch.kernels.Kernel):
	def __init__(self, base_kernel, fixed_multiplier=100.0):
		super().__init__()
		self.base_kernel = gpytorch.kernels.ScaleKernel(base_kernel)
		self.fixed_multiplier = fixed_multiplier

	def forward(self, x1, x2, **params):
		return self.fixed_multiplier**2 * self.base_kernel(x1, x2, **params)

class ScaledMean(gpytorch.means.Mean):
	def __init__(self, base_mean, fixed_multiplier=1.0):
		super().__init__()
		self.base_mean = base_mean
		self.fixed_multiplier = fixed_multiplier

	def forward(self, x):
		return self.fixed_multiplier * self.base_mean(x)

class GPY_Torch_Model(gpytorch.models.ExactGP):
	def __init__(self, train_x, train_y, likelihood, input_dim=2, fixed_multiplier=100.0):
		super(GPY_Torch_Model, self).__init__(train_x, train_y, likelihood)
		# You can choose a mean module per your requirement
		#self.base_mean = gpytorch.means.ConstantMean()
		self.base_mean = gpytorch.means.LinearMean(input_size=input_dim)
		self.mean_module = ScaledMean(self.base_mean, fixed_multiplier=1)#fixed_multiplier)
		
		self.base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=input_dim) 
		#self.base_kernel = gpytorch.kernels.MaternKernel(ard_num_dims=input_dim, nu=3.5) #+ gpytorch.kernels.LinearKernel(ard_num_dims=input_dim)
		self.covar_module = FixedMultiplierKernel(self.base_kernel, fixed_multiplier=fixed_multiplier**2)
		#self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)
		
	def forward(self, x):
		mean_x = self.mean_module(x)
		covar_x = self.covar_module(x)
		return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class Base_GP_Model:
	def __init__(
		self,
		quoteDateData,
		y_label='IV', 
		type='surface',
		train_fraction=0.6,
		c=0.05, 
		device=None, 
		**kwargs):

		# Set device: use provided device or default to CUDA if available
		self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
		print(f'Using device: {self.device}')

		# Set additional attributes from kwargs
		for key, value in kwargs.items():
			setattr(self, key, value)

		# Set default values if not provided via kwargs
		if 'lr' not in kwargs:
			self.lr = 0.05
		if 'training_iterations' not in kwargs:
			self.training_iterations = 200
		if 'sep_obs' not in kwargs:
			self.sep_obs = False
		if 'y_scaling_factor' not in kwargs:
			self.y_scaling_factor = 100
		if 'learn_additional_noise' not in kwargs:
			self.learn_additional_noise = False
		
		# Set the scaling factors for m,T and y
		self.m_scaler = MinMaxScaler(quoteDateData['m'].values)
		self.T_scaler = MinMaxScaler(quoteDateData['T'].values)
		self.y_scaler = MinMaxScaler(quoteDateData['mid_' + y_label].values, y_scaling_factor=self.y_scaling_factor)

		# Set the training data (full dataset stored for predictions later)
		self.data = quoteDateData
		self.CallData = quoteDateData[quoteDateData['option_type'] == 'C']
		self.PutData = quoteDateData[quoteDateData['option_type'] == 'P']
		self.y_label = y_label
		self.option_indices = {}

		#min_max = quoteDateData.loc[quoteDateData['m'].isin([quoteDateData['m'].min(), quoteDateData['m'].max()])]
		#rest = quoteDateData.drop(min_max.index).sample(frac=train_fraction, replace=False)
		#partialQuoteDateData = pd.concat([min_max, rest]).sort_values(by=['m', 'T'])
		partialQuoteDateData = quoteDateData.sample(frac=train_fraction, replace=False).sort_values(by=['m', 'T'])
		partialQuoteDateData['spread_' + y_label] = self.y_scaler.normalize(partialQuoteDateData['ask_' + y_label].values) - self.y_scaler.normalize(partialQuoteDateData['bid_' + y_label].values)

		partialQuoteDateData = partialQuoteDateData[partialQuoteDateData['spread_' + y_label] <= partialQuoteDateData['spread_' + y_label].quantile(0.95)]
		partialQuoteDateData = partialQuoteDateData[partialQuoteDateData[self.col_map['open_interest']] > 75]
		partialQuoteDateData = partialQuoteDateData[partialQuoteDateData['delta_1545'].abs() < 0.5]

		# Process observations: either separate (bid/ask) or using the mid price
		if self.sep_obs:
			partialQuoteDateData = partialQuoteDateData.melt(
				id_vars=['quote_date', 'expiration', 'strike', 'm', 'T', 'spread_' + y_label],
				value_vars=['bid_' + y_label, 'ask_' + y_label],
				var_name='type',
				value_name='y_var'
			)
			y_train = to_32bit(self.y_scaler.normalize(partialQuoteDateData['y_var'].values)) 
			
		else:
			y_train = to_32bit(self.y_scaler.normalize(partialQuoteDateData['mid_' + y_label].values)) 

		Z_value = stats.norm.ppf(1 - c / 2)
		noise_train = ((partialQuoteDateData['spread_' + y_label]).values) ** 2 / (Z_value ** 2)
		noise_train = to_32bit(noise_train)

		# Convert training data to torch tensors and move them to the selected device.
		self.y_train = torch.tensor(y_train, dtype=torch.float).to(self.device)
		self.noise = torch.tensor(noise_train, dtype=torch.float).to(self.device)

		if type == 'surface':
			self.m_train = to_32bit(self.m_scaler.normalize(partialQuoteDateData['m'].values))
			self.T_train = to_32bit(self.T_scaler.normalize(partialQuoteDateData['T'].values))
			self.x_train = torch.tensor(np.column_stack((self.m_train, self.T_train)), dtype=torch.float).to(self.device)
			self.input_dim = 2
		elif type == 'smile':
			m_train = to_32bit(self.m_scaler.normalize(partialQuoteDateData['m'].values))
			self.x_train = torch.tensor(m_train, dtype=torch.float).to(self.device)
			self.input_dim = 1
		else:
			raise ValueError("Invalid type. Choose 'surface' or 'smile'.")

		idx_true_call = np.where(partialQuoteDateData[self.col_map['option_type']] == 'C')[0]
		idx_true_put = np.where(partialQuoteDateData[self.col_map['option_type']] == 'P')[0]

		train_opt_indices = {
			'call': idx_true_call,
			'put': idx_true_put
		}

		self.option_indices['train'] = train_opt_indices



		self.init_lr = 1e-5  # Starting learning rate for all parameters
		self.likelihood = None
		self.model = None

	def learn(self, **kwargs):
		if 'noise' in kwargs:
			self.noise = torch.tensor(kwargs['noise'], dtype=torch.float).to(self.device)
		if 'lr' in kwargs:
			self.lr = kwargs['lr']
		if 'training_iterations' in kwargs:
			self.training_iterations = kwargs['training_iterations']

		with gpytorch.settings.min_fixed_noise(float_value=1e-6):
			with gpytorch.settings.cholesky_jitter(1e-6):
				with max_cg_iterations(1000):
					# Initialize likelihood if not already provided
					if self.likelihood is None:
						likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
							noise=self.noise,
							learn_additional_noise=self.learn_additional_noise
						)
						self.likelihood = likelihood
					else:
						likelihood = self.likelihood

					# Create the GP model if it doesn't already exist
					if self.model is None:
						model = GPY_Torch_Model(self.x_train,
												self.y_train, 
												likelihood, 
												input_dim=self.input_dim,
												fixed_multiplier=self.y_scaling_factor)
						self.model = model
					else:
						model = self.model


					model = model.to(self.device)
					likelihood = likelihood.to(self.device)

					# Set the model and likelihood to training mode
					model.train()
					likelihood.train()

					# First, get all named parameters
					#named_params = list(model.named_parameters())

					# # Any parameter in the mean module
					# mean_params = [p for n, p in named_params if 'mean_module' in n]
					# # We seperate the outputscale and kernel params
					# outputscale_params = [p for n, p in named_params if 'outputscale' in n]
					# kernel_params = [p for n, p in named_params if 'covar_module' in n and 'outputscale' not in n]

					# New

					named_params = list(model.named_parameters())

					mean_params = [p for n, p in named_params if 'mean_module' in n]
					outputscale_params = [p for n, p in named_params if 'covar_module.outputscale' in n]
					kernel_params = [p for n, p in named_params if 'covar_module.base_kernel' in n]
					# New

					# Noise params (if learn_additional_noise=True)
					if self.learn_additional_noise:
						noise_params = list(model.likelihood.parameters())
					else:
						noise_params = []

					# Define grouped learning rates
					optimizer = torch.optim.Adam([
						{'params': mean_params, 'lr': self.lr},               # normal learning rate
						{'params': kernel_params, 'lr': self.lr},             # normal learning rate
						{'params': outputscale_params, 'lr': self.lr},    # boost outputscale learning
						{'params': noise_params, 'lr': self.lr},          # boost noise learning
					])
					
					# Scheduler with warm-up (optional via LambdaLR) followed by ReduceLROnPlateau
					warmup_steps = int(0.1 * self.training_iterations)  # e.g., 10% of total
					# Save target LRs
					for g in optimizer.param_groups:
						g['initial_lr'] = g['lr']
						g['lr'] = self.init_lr  # All start small
					# Capture param references up front
					outputscale_param_set = set(outputscale_params)
					exp_decay_gamma = 1  

					def outputscale_lr_lambda(epoch):
						if epoch < warmup_steps:
							return (epoch + 1) / warmup_steps  # Linear ramp
						else:
							return exp_decay_gamma ** (epoch - warmup_steps)

					def make_lr_lambda(param_group):
						# If this param group contains outputscale (by identity), apply exponential logic
						if any(p in outputscale_param_set for p in param_group['params']):
							return  lambda epoch: (
								min(1.0, epoch / warmup_steps)
							)
						else:
							# All other groups get linear warmup and then flat
							return lambda epoch: (
								min(1.0, epoch / warmup_steps)
							)

					# First a linear warm-up scheduler
					lr_lambdas = [make_lr_lambda(pg) for pg in optimizer.param_groups]
					warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambdas)

					# Then use ReduceLROnPlateau for dynamic adjustment
					plateau_scheduler = lr_scheduler.ReduceLROnPlateau(
						optimizer,
						mode='min',
						factor=0.5,         # Reduce LR by half
						patience=100,        # Wait 10 epochs with no improvement
						threshold=1e-4,     # Consider improvement if loss drops more than this
						threshold_mode = 'abs',
						min_lr=1e-5,        # Minimum LR allowed
					)

					# Marginal Log Likelihood (negative for minimization)
					mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

					training_iterations = self.training_iterations
					losses = []

					for i in range(training_iterations):
						optimizer.zero_grad()
						output = model(self.x_train)
						loss = -mll(output, self.y_train)
						losses.append(loss.item())
						loss.backward()
						optimizer.step()                        
						# Warm-up scheduler step (only for initial phase)
						if i < warmup_steps:
							warmup_scheduler.step()  # Gradually increase LR
						else: 
							plateau_scheduler.step(loss.item())  # Adjust LR if no improvement
						
						if (i + 1) % int(self.training_iterations / 4) == 0:
							print(f'Iter {i + 1}/{training_iterations} - Loss: {loss.item():.4f}')
							print(f'  lengthscale: {model.covar_module.base_kernel.base_kernel.lengthscale}')
							print(f'  outputscale: {model.covar_module.base_kernel.outputscale.item():.4f}')
							current_lrs = [pg['lr'] for pg in optimizer.param_groups]
							print("Learning rates:", current_lrs)

						min_lr_threshold = 1e-4
						all_lrs = [group['lr'] for group in optimizer.param_groups]
						if all(lr <= min_lr_threshold for lr in all_lrs) and i > warmup_steps:
							print(f"Stopping early at iteration {i+1} due to minimal learning rate.")
							current_lrs = [pg['lr'] for pg in optimizer.param_groups]
							print("Learning rates:", current_lrs)
							break



					self.model = model
					self.loss = losses

					# Plot training curve
					plt.figure(figsize=(10, 4))
					plt.plot(losses[100:])
					plt.title("Negative Log Likelihood During Training")
					plt.xlabel("Iterations")
					plt.ylabel("NLL")
					plt.show()

					matern_lengthscale = model.covar_module.base_kernel.base_kernel.lengthscale.detach().cpu().numpy()
					#linear_variance = model.covar_module.base_kernel.base_kernel.kernels[1].variance.item()
					outputscale = model.covar_module.base_kernel.outputscale.item()


					# Store kernel parameters for further use
					if self.input_dim == 1:
						kernel_params = {
							'lengthscale': matern_lengthscale[0][0],
							'outputscale': outputscale,
						# 'linear_variance': linear_variance
						}
						self.kernel_params = kernel_params
					elif self.input_dim == 2:
						kernel_params = {
							'lengthscale_m': matern_lengthscale[0][0],
							'lengthscale_T': matern_lengthscale[0][1],
							'outputscale': outputscale,
						#  'linear_variance': linear_variance
						}
						self.kernel_params = kernel_params

					return self

	def predict(self, add_x_test=None, add_label=None):
		# Prepare test data
		if self.input_dim == 1:
			x_test = self.m_scaler.normalize(self.data['m'].unique())

			# Find indices in x_test for which there is an available Call option with that m value
			call_m_values = self.CallData['m'].unique()
			put_m_values = self.PutData['m'].unique()

			call_indices = [i for i, m_val in enumerate(self.m_scaler.inverse(x_test)) if np.isclose(m_val, call_m_values, atol=1e-8).any()]
			put_indices = [i for i, m_val in enumerate(self.m_scaler.inverse(x_test)) if np.isclose(m_val, put_m_values, atol=1e-8).any()]

			indices = {
				'call': call_indices,
				'put': put_indices
			}

			self.option_indices['base'] = indices

		elif self.input_dim == 2:
			x_test = self.m_scaler.normalize(self.data['m'].unique())
			x_test = np.column_stack((x_test, self.T_scaler.normalize(self.data['T'].unique())))
		else:
			raise ValueError("Invalid input dimension. Expected 1 or 2.")

		self.pred = self._get_pred(x_test)
		if add_x_test is not None:
			if add_label is None:
				warnings.warn("add_label is None. Setting it to 'extra'")
				add_label = 'extra'
			predictions = self._get_pred(add_x_test, label=add_label)
			setattr(self, f"pred_{add_label}", predictions)
		
		return self

	def _get_pred(self, x_test, label=None):
		# Convert test data to tensor and move to selected device
		x_test = torch.tensor(x_test, dtype=torch.float).to(self.device)
		self.model.eval()
		self.likelihood.eval()

		# Make predictions
		#with torch.no_grad():    #, gpytorch.settings.fast_pred_var():
		with gpytorch.settings.max_cholesky_size(3000):                
			observed_pred = self.model(x_test)
			if self.input_dim == 2:
				m_test = x_test[:, 0].cpu().numpy()
				T_test = x_test[:, 1].cpu().numpy()
				if label is None:
					self.m_test = self.m_scaler.inverse(m_test)
					self.T_test = self.T_scaler.inverse(T_test)
					self.x_test = np.column_stack((m_test, T_test))
				else:
					setattr(self, f"m_test_{label}", self.m_scaler.inverse(m_test))
					setattr(self, f"T_test_{label}", self.T_scaler.inverse(T_test))
			elif self.input_dim == 1:
				if label is None:
					self.x_test = self.m_scaler.inverse(x_test.cpu().numpy())
				else:
					setattr(self, f"x_test_{label}", self.m_scaler.inverse(x_test.cpu().numpy()))
			else:
				raise ValueError("Invalid input dimension. Expected 1 or 2.")

			mean = self.y_scaler.inverse(observed_pred.mean.cpu().detach().numpy())
			y_scaling_spread = (self.y_scaler.max_ - self.y_scaler.min_) / self.y_scaler.y_scaling_factor
			std = observed_pred.stddev.cpu().detach().numpy() * y_scaling_spread

			observed_sample = observed_pred.sample(torch.Size((10000,)))
			observed_sample = self.y_scaler.inverse(observed_sample.cpu().detach().numpy())

			pred = {
				'mean': mean,
				'lower': np.quantile(observed_sample, 0.025, axis=0),
				'upper': np.quantile(observed_sample, 0.975, axis=0),
				'variance': std ** 2,
				'samples': observed_sample
			}
			return pred

	def get_analytical_gradient(self, x_test):
		
		x_test = torch.tensor(x_test, dtype=torch.float32, device=self.device, requires_grad=True)

		# Forward pass: get GP predictive mean (scalar per input)
		mean_func = self.model(x_test).mean  # shape: (n,)

		# First derivative (gradient): dμ/dx
		grad_outputs = torch.ones_like(mean_func)
		first_derivative = torch.autograd.grad(
			outputs=mean_func,
			inputs=x_test,
			grad_outputs=grad_outputs,
			create_graph=True
		)[0]  # shape: (n, d)
		# Initialize second derivatives (Hessians)
		if self.input_dim == 1:
			n, d = x_test.shape, 1
			second_derivative = torch.autograd.grad(
				outputs=first_derivative,
				inputs=x_test,
				grad_outputs=torch.ones_like(first_derivative),
				create_graph=True,
				retain_graph=True
			)[0]
	
		elif self.input_dim == 2:
			n, d = x_test.shape
			second_derivative = torch.zeros(n, d, d, device=self.device)

			for i in range(d):
				grad_i = first_derivative[:, i]  # dμ/dx_i
				grad2 = torch.autograd.grad(
					outputs=grad_i,
					inputs=x_test,
					grad_outputs=torch.ones_like(grad_i),
					create_graph=True,
					retain_graph=True
				)[0]  # shape: (n, d, d)
				second_derivative[:, i, :] = grad2  # fills row i of Hessian for each sample

		y_max, y_min = self.y_scaler.max_, self.y_scaler.min_
		y_spread = y_max - y_min

		mean = self.y_scaler.inverse(mean_func.detach().cpu().numpy())  # (n,)
		# first_derivative (n , d)
		first_derivative = first_derivative.detach().cpu().numpy() * y_spread / self.y_scaling_factor

		# second_derivative (n , d , d)
		# [:, 0 , 1] Hessian's (1,2) element for each test point
		second_derivative = second_derivative.detach().cpu().numpy() * y_spread / self.y_scaling_factor

		if self.input_dim == 1:
			x_test = x_test.detach().cpu().numpy()
			m_base_scale = self.m_scaler.inverse(x_test)
			self.x_test_deriv = m_base_scale
			m_spread =  (self.m_scaler.inverse(1) - self.m_scaler.inverse(0))

			first_derivative = first_derivative / m_spread
			second_derivative = second_derivative / (m_spread**2)

			# Create a dictionary to store the results
			results = {
				'first_derivative': first_derivative,
				'second_derivative': second_derivative,
				'function_val': mean
			}
			self.pred_deriv = results

			# Results in terms of f_tilde
			results_f_tilde = {
				'first_derivative': first_derivative * m_spread / y_spread * self.y_scaling_factor,
				'second_derivative': second_derivative * (m_spread**2) / y_spread * self.y_scaling_factor,
				'function_val': self.y_scaler.normalize(mean)
			}
			self.f_tilde = results_f_tilde

			g_1 = (1  -  (m_base_scale * first_derivative) / (2*mean))**2
			g_2 = - (first_derivative**2 / 4 ) * ( 1 / mean + 0.25)
			g_3 = second_derivative / 2
			g = g_1 + g_2 + g_3

			constraint_results = {
				'g_1': g_1,
				'g_2': g_2,
				'g_3': g_3,
				'g': g
			}
			self.constraint_results = constraint_results
		
		elif self.input_dim == 2:
			x_test = x_test.detach().cpu().numpy()
			m_base_scale = self.m_scaler.inverse(x_test[:, 0])
			T_base_scale = self.T_scaler.inverse(x_test[:, 1])
			self.m_test_deriv = m_base_scale
			self.T_test_deriv = T_base_scale
			m_spread =  (self.m_scaler.inverse(1) - self.m_scaler.inverse(0))
			T_spread =  (self.T_scaler.inverse(1) - self.T_scaler.inverse(0))

			# Gradient results in m

			first_derivative_m = first_derivative[:,0] / m_spread
			second_derivative_m = second_derivative[:,0,0] / (m_spread**2)

			results_m = {
				'first_derivative': first_derivative_m,
				'second_derivative': second_derivative_m,
				'function_val': mean
			}
			self.pred_deriv_m = results_m

			# Gradient results in T
			first_derivative_T = first_derivative[:,1] / T_spread
			second_derivative_T = second_derivative[:,1,1] / (T_spread**2)
			cross_term = second_derivative[:,0,1] / (m_spread * T_spread)
			results_T = {
				'first_derivative': first_derivative_T,
				'second_derivative': second_derivative_T,
				'cross_term': cross_term
			}

			self.pred_deriv_T = results_T

			g_1 = (1  -  (m_base_scale * first_derivative_m) / (2*mean))**2
			g_2 = - (first_derivative_m**2 / 4 ) * ( 1 / mean + 0.25)
			g_3 = second_derivative_m / 2
			g = g_1 + g_2 + g_3

			constraint_results_m = {
				'g_1': g_1,
				'g_2': g_2,
				'g_3': g_3,
				'g': g
			}
			self.constraint_results_m = constraint_results_m
		
		return self

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

		test_samples = self.pred_extra['samples'].T
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
		with torch.no_grad():
			fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
			self._plot_smile(ax=ax2, x_version=x_version, y_version=y_version)
			self._plot_smile_centered(ax=ax1, zoom=zoom, x_version=x_version, y_version=y_version)
			plt.tight_layout()
			plt.show()
			return self

	def plot_price_comparison(self, figsize=(24, 10), zoom=80, 
							x_version = 'log_m',
							x_lim_call = (None, None),
							x_lim_put = (None, None)):
		with torch.no_grad():
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
			
			ax.plot(x_locs, grad_result_analytical, 'k-', lw=1, label='Analytical Derivative')
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
					title = 'Analytical Estimate and Distribution of First Derivative')

		_get_subplot(ax2,
					x_locs, x_label, 
					grad_results['second_derivative'],
					grad_result_analytical['second_derivative'],
					y_label = 'Second Derivative',
					title = 'Analytical Estimate and Distribution of Second Derivative')


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

	def plot_heatmaps(self, zoom = 20):

		with torch.no_grad():

			# Extract normalized grid values.
			m_unique = np.sort(np.unique(self.m_test_extra))
			T_unique = np.sort(np.unique(self.T_test_extra))

			n_m = len(m_unique)
			n_T = len(T_unique)

			Z_mean = np.reshape(self.pred_extra['mean'], (n_T, n_m))
			Z_lower = np.reshape(self.pred_extra['lower'], (n_T, n_m))
			Z_upper = np.reshape(self.pred_extra['upper'], (n_T, n_m))

			Z_lower_centered = -(Z_lower - Z_mean)
			Z_upper_centered = Z_upper - Z_mean

			# Helper to compute possibly zoomed min/max
			def zoom_range(z_array):
				min_val, max_val = z_array.min(), z_array.max()
				if zoom > 0:
					pad = (zoom / 100.0) * (max_val - min_val)
					return (min_val + pad, max_val - pad)
				else:
					return (min_val, max_val)

			zmin_lower, zmax_lower = zoom_range(Z_lower_centered)
			zmin_upper, zmax_upper = zoom_range(Z_upper_centered)
			
			
			# Create subplots.
			fig = make_subplots(rows=1, cols=2,
								subplot_titles=["GP Mean Prediction for "+self.y_label , 'Δ '+ self.y_label +' for 95% CI'],
								horizontal_spacing=0.1)
			
			# Add heatmaps.
			fig.add_trace(go.Heatmap(x=m_unique, y=T_unique, z=Z_mean,
									colorscale="Viridis", zsmooth='best',
									colorbar=dict(title=self.y_label, thickness=12, x=0.45)),
									row=1, col=1)
			fig.add_trace(
				go.Heatmap(
					x=m_unique,
					y=T_unique,
					z=Z_lower_centered,
					colorscale="Cividis",
					zmin=zmin_lower,
					zmax=zmax_lower,
					colorbar=dict(
						title="Δ "+self.y_label,
						thickness=12,
						x=1
					),
					showscale=True,
					zsmooth='best'
				),
				row=1, col=2
			)

			# Overlay training data.
			train_m = self.m_scaler.inverse(self.x_train[:, 0].detach().cpu().numpy())
			train_T = self.T_scaler.inverse(self.x_train[:, 1].detach().cpu().numpy())
			
			scatter_trace = go.Scatter(x=train_m, y=train_T, mode="markers",
									marker=dict(color="white", size=2, symbol="star"),
									name="Training Data")
			
			fig.add_trace(scatter_trace, row=1, col=1)
			fig.add_trace(scatter_trace, row=1, col=2)
			
			# Explicitly set axis ranges to cover the full domain.
			x_range = [np.min(m_unique)*0.90, np.max(m_unique)*1.05]
			y_range = [np.min(T_unique)*0.90, np.max(T_unique)*1.05]
			for col in [1, 2]:
				fig.update_xaxes(range=x_range, row=1, col=col)
				fig.update_yaxes(range=y_range, row=1, col=col)
			
			fig.update_layout(width=1400, height=700,
							margin=dict(l=40, r=40, t=60, b=40),
							template='plotly',  # Alternative templates: 'plotly', 'plotly_dark', 'ggplot2', 'seaborn', 'simple_white'
							title="GP Prediction Heatmaps",
							title_x=0.5)
			
			fig.show()

			return self
	
	def plot_combined_surfaces(self, zoom=0):
		"""
		Combines the GP Prediction Surface and the Centered GP Prediction Surface 
		into a single Plotly figure with subplots side by side.
		
		Assumes:
		- self.m_test_extra and self.T_test_extra contain the unnormalized grid values.
		- self.pred_extra is a dictionary with keys 'mean', 'lower', and 'upper'.
		- self.x_train and self.y_train contain the training data.
		
		Parameters:
		zoom (float): Adjusts the z-axis view for the centered plot if greater than 0.
		
		Returns:
		A Plotly figure object with two 3D subplots side by side.
		"""

		with torch.no_grad():

			# Create a subplot figure with 1 row and 2 columns (each a 3D scene).
			fig = make_subplots(
				rows=1, cols=2,
				specs=[[{'type': 'scene'}, {'type': 'scene'}]],
				subplot_titles=['GP - '+ self.y_label+' Surface', 'GP - Centered '+ self.y_label+' Surface']
			)
			
			# Create the meshgrid based on unique grid values.
			m_unique = np.sort(np.unique(self.m_test_extra))
			T_unique = np.sort(np.unique(self.T_test_extra))
			M, T = np.meshgrid(m_unique, T_unique)
			
			# Reshape the predictions to match the grid.
			Z_mean = np.reshape(self.pred_extra['mean'], M.shape)
			Z_lower = np.reshape(self.pred_extra['lower'], M.shape)
			Z_upper = np.reshape(self.pred_extra['upper'], M.shape)
			
			# ---------------------- LEFT SUBPLOT ----------------------
			# GP Mean surface.
			fig.add_trace(go.Surface(
				x=M,
				y=T,
				z=Z_mean,
				colorscale='Viridis',
				opacity=0.9,
				name='GP Mean Prediction',
				showscale=True,
				colorbar=dict(
						title="Mean "+self.y_label,
						thickness=20,
						x=-0.1
					)
			), row=1, col=1)
			
			# Overlay the training data.
			train_m = self.m_scaler.inverse(self.x_train[:, 0].detach().cpu().numpy())
			train_T = self.T_scaler.inverse(self.x_train[:, 1].detach().cpu().numpy())
			train_y = self.y_scaler.inverse(self.y_train.detach().cpu().numpy()) 
			
			fig.add_trace(go.Scatter3d(
				x=train_m,
				y=train_T,
				z=train_y,
				mode='markers',
				marker=dict(size=4, color='black'),
				name='Training Data'
			), row=1, col=1)
			
			# ---------------------- RIGHT SUBPLOT ----------------------
			# Compute centered surfaces by subtracting the GP mean.
			Z_lower_centered = Z_lower - Z_mean
			Z_upper_centered = Z_upper - Z_mean
			
			# Lower centered CI surface.
			fig.add_trace(go.Surface(
				x=M,
				y=T,
				z=Z_lower_centered,
				colorscale='Cividis',
				opacity=0.7,
				name='Lower 95% CI (Centered)',
				showscale=False
			), row=1, col=2)
			
			# Upper centered CI surface.
			fig.add_trace(go.Surface(
				x=M,
				y=T,
				z=Z_upper_centered,
				colorscale='Plasma',
				opacity=0.7,
				name='Upper 95% CI (Centered)',
				showscale=False
			), row=1, col=2)
			
			# Calculate an overall average GP mean and center the training data.
			avg_pred_mean = np.mean(self.pred_extra['mean'])
			train_y_centered = (self.y_scaler.inverse(self.y_train.detach().cpu().numpy())) - avg_pred_mean
			
			# Note: Here we plot the centered training data at z=0 for clarity.
			fig.add_trace(go.Scatter3d(
				x=train_m,
				y=train_T,
				z=np.zeros_like(train_y_centered),
				mode='markers',
				marker=dict(size=4, color='black'),
				name='Training Data (Centered)'
			), row=1, col=2)
			
			# Optionally adjust the z-axis range for the centered plot if zoom is specified.
			if zoom > 0:
				z_min = np.min([np.min(Z_lower_centered), np.min(train_y_centered)])
				z_max = np.max([np.max(Z_upper_centered), np.max(train_y_centered)])
				z_range = z_max - z_min
				pad = zoom / 100 * z_range
				scene2_zaxis = dict(range=[z_min + pad, z_max - pad])
			else:
				scene2_zaxis = {}
			
			# Update the layout with distinct scene settings for each subplot.
			fig.update_traces(marker=dict(size=4, color='black'), selector=dict(mode='markers'), showlegend=False)
			fig.update_layout(
				title=dict(
					text= self.y_label+' Surface -- Quote Date: '+self.data['quote_date'].iloc[0],
					x=0.5,  # Centers the title
					xanchor='center'
				),
				width=1400,
				height=700,
				scene=dict(
					xaxis_title="Moneyness",
					yaxis_title="Time to Expiry",
					zaxis_title="Implied Volatility"
				),
				scene2=dict(
					xaxis_title="Moneyness",
					yaxis_title="Time to Expiry",
					zaxis_title="Centered Implied Volatility",
					zaxis=scene2_zaxis
				)
			)
			
			return fig

	def plot_constraint(self, num_m=100, num_T=20):
	
		if self.input_dim == 1:
			self._plot_constraint_1d()
		elif self.input_dim == 2:
			self._plot_constraint_components(num_m = num_m, num_T = num_T)
		else:
			raise ValueError("Invalid input dimension. Expected 1 or 2.")

	def _plot_constraint_components(self, figsize=(12, 12), num_m = 100, num_T = 20):

		fig, (ax1, ax2) = plt.subplots(2, 2, figsize=figsize)

		ax11, ax12 = ax1.flatten()
		ax21, ax22 = ax2.flatten()

		r_shape = lambda x: x.reshape((num_T,num_m))
		T = self.T_test_deriv.reshape((num_T,num_m))

		for ind in range(num_T):
			color = plt.cm.viridis(ind / num_T)  # Use a colormap to vary color with ind
			ax11.plot(r_shape(self.m_test_deriv)[ind], r_shape(self.constraint_results['g_1'])[ind], label=f'g curve {ind}', color=color)
			ax12.plot(r_shape(self.m_test_deriv)[ind], r_shape(self.constraint_results['g_2'])[ind], label=f'g curve {ind}', color=color)
			ax21.plot(r_shape(self.m_test_deriv)[ind], r_shape(self.constraint_results['g_3'])[ind], label=f'g curve {ind}', color=color)
			ax22.plot(r_shape(self.m_test_deriv)[ind], r_shape(self.constraint_results['g'])[ind], label=f'g curve {ind}', color=color)
			sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=T[0,0], vmax=T[19,0]))


		ax11.set_xlabel('Moneyness')
		ax12.set_xlabel('Moneyness')
		ax21.set_xlabel('Moneyness')
		ax22.set_xlabel('Moneyness')

		ax11.set_ylabel('Constraint Value')
		ax12.set_ylabel('Constraint Value')
		ax21.set_ylabel('Constraint Value')
		ax22.set_ylabel('Constraint Value')

		ax11.set_title('Second Order Component')
		ax12.set_title('Middle Component')
		ax21.set_title('Second Derivative Component')
		ax22.set_title('Durrleman Condition')



		plt.colorbar(sm, label='Time to Maturity', ax= ax12)
		plt.colorbar(sm, label='Time to Maturity', ax= ax22)

		plt.show()

	def _plot_constraint_1d(self):

		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

		ax1.plot(self.x_test_deriv, self.constraint_results['g_1'], label='First Component')
		ax1.plot(self.x_test_deriv, self.constraint_results['g_2'], label='Second Component')
		ax1.plot(self.x_test_deriv, self.constraint_results['g_3'], label='Third Component')
		ax1.set_title('Constraint Components')
		ax1.set_xlabel('Moneyness')
		ax1.set_ylabel('Constraint Values')
		ax1.legend()

		ax2.plot(self.x_test_deriv, self.constraint_results['g'], label='Durrleman\'s Condition')
		x_min, x_max = ax2.get_xlim()
		ax2.hlines(0, x_min, x_max, color='k', linestyle='--', label='Zero Line')
		ax2.set_title('Durrleman\'s Condition')
		ax2.set_xlabel('Moneyness')
		ax2.set_ylabel('Total Constraint Value')
		ax2.legend()

		plt.tight_layout()

	def _plot_smile(self, figsize=(20, 10), ax=None,
					x_version = 'm',
					y_version = 'TIV'):

		with torch.no_grad():

			if y_version == 'IV':
				base, extra = self._get_price_samples_1d()
				y_label = 'IV'
			elif y_version == 'TIV':
				y_label = 'TIV'
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


			ax.plot(x_centered_loc_call, 
					self.CallData['mid_' + y_label], 
					label='Call: Mid-Price ' + y_label, color='firebrick',
					alpha=0.85)		
			ax.fill_between(x_centered_loc_call, 
							self.CallData['ask_' + y_label].values, 
							self.CallData['bid_' + y_label].values,
							alpha=0.85, label='Call: Bid-Ask ' + y_label + ' Spread',
							color = 'orange')

			ax.plot(x_centered_loc_put, 
					self.PutData['mid_' + y_label], 
					label='Put: Mid-Price ' + y_label, color='b',
					alpha=0.60)	

			ax.fill_between(x_centered_loc_put, 
					self.PutData['ask_' + y_label].values, 
					self.PutData['bid_' + y_label].values,
							alpha=0.60, label='Put: Bid-Ask ' + y_label + ' Spread',
							color = 'lightblue')

			if x_version == 'strike':
				T = self.data['T'].iloc[0]
				x_extra_loc = self._strike_calculator(T = T)
				x_train_loc = self.m_scaler.inverse(self.x_train.cpu().numpy())
				x_train_loc = self._strike_calculator(T = T, m = x_train_loc)
				x_label = 'Strike Price'
			
			elif x_version == 'log_m':
				x_extra_loc = self.x_test_extra
				x_train_loc = self.m_scaler.inverse(self.x_train)
				x_label = 'Log-Moneyness'
			
			elif x_version == 'm':
				x_extra_loc = self.x_test_extra
				x_extra_loc = np.exp(x_extra_loc)
				x_train_loc = self.m_scaler.inverse(self.x_train.cpu().numpy())
				x_train_loc = np.exp(x_train_loc)
				x_label = 'Moneyness'
			else:
				raise ValueError("x_version must be 'strike', 'log_m', or 'm'")

			if y_version == 'TIV':
				ax.fill_between(x_extra_loc, 
								self.pred_extra['lower'],  
								self.pred_extra['upper'], 
								alpha=0.3, label='95% CI for ' + y_label,
								color = 'indigo')		
				ax.plot(x_extra_loc, 
						self.pred_extra['mean'], 
						'indigo', label='GP Mean Prediction for ' + y_label)
				ax.plot(x_train_loc, 
					self.y_scaler.inverse(self.y_train.cpu().numpy()),
					'k*', label='Training Data', markersize=4)
			elif y_version == 'IV':
				ax.fill_between(x_extra_loc, 
								extra['IV']['lb'],  
								extra['IV']['ub'], 
								alpha=0.3, label='95% CI for ' + y_label,
								color = 'indigo')		
				ax.plot(x_extra_loc, 
						extra['IV']['mean'], 
						'indigo', label='GP Mean Prediction for ' + y_label)
				T = self.data['T'].iloc[0]
				ax.plot(x_train_loc,
					np.sqrt(self.y_scaler.inverse(self.y_train.cpu().numpy()) / T),
					'k*', label='Training Data', markersize=4)	

			
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
			if x_version == 'log_m':
				x_centered_loc_call = self.CallData['m']
				x_centered_loc_put = self.PutData['m']
			if x_version == 'm':
				x_centered_loc_call = np.exp(self.CallData[x_version])
				x_centered_loc_put = np.exp(self.PutData[x_version])


			ax.plot(x_centered_loc_call, 
					self.CallData['mid_' + y_label] - center_around[self.option_indices['base']['call']], 
					label='Call: Mid-Price ' + y_label, color='firebrick',
					alpha=0.85)		
			ax.fill_between(x_centered_loc_call, 
							self.CallData['ask_' + y_label].values - center_around[self.option_indices['base']['call']], 
							self.CallData['bid_' + y_label].values - center_around[self.option_indices['base']['call']],
							alpha=0.85, label='Call: Bid-Ask ' + y_label + ' Spread',
							color = 'orange')

			ax.plot(x_centered_loc_put, 
					self.PutData['mid_' + y_label] - center_around[self.option_indices['base']['put']], 
					label='Put: Mid-Price ' + y_label, color='b',
					alpha=0.60)	

			ax.fill_between(x_centered_loc_put, 
					self.PutData['ask_' + y_label].values - center_around[self.option_indices['base']['put']], 
					self.PutData['bid_' + y_label].values - center_around[self.option_indices['base']['put']],
							alpha=0.60, label='Put: Bid-Ask ' + y_label + ' Spread',
							color = 'lightblue')

			if x_version == 'strike':
				T = self.data['T'].iloc[0]
				x_extra_loc = self._strike_calculator(T = T)
				x_train_loc = self.m_scaler.inverse(self.x_train.cpu().numpy())
				x_train_loc = self._strike_calculator(T = T, m = x_train_loc)
				x_label = 'Strike Price'
			
			elif x_version == 'log_m':
				x_extra_loc = self.x_test_extra
				x_train_loc = self.m_scaler.inverse(self.x_train)
				x_label = 'Log-Moneyness'
			
			elif x_version == 'm':
				x_extra_loc = self.x_test_extra
				x_extra_loc = np.exp(x_extra_loc)
				x_train_loc = self.m_scaler.inverse(self.x_train.cpu().numpy())
				x_train_loc = np.exp(x_train_loc)
				x_label = 'Moneyness'
			else:
				raise ValueError("x_version must be 'strike', 'log_m', or 'm'")

			if y_version == 'TIV':
				ax.fill_between(x_extra_loc, 
								self.pred_extra['lower'] - center_around_extra,  
								self.pred_extra['upper'] - center_around_extra, 
								alpha=0.3, label='95% CI for ' + y_label,
								color = 'indigo')		
				ax.plot(x_extra_loc, 
						self.pred_extra['mean'] - center_around_extra, 
						'indigo', label='GP Mean Prediction for ' + y_label)
			elif y_version == 'IV':
				ax.fill_between(x_extra_loc, 
								extra['IV']['lb'] - center_around_extra,  
								extra['IV']['ub'] - center_around_extra, 
								alpha=0.3, label='95% CI for ' + y_label,
								color = 'indigo')		
				ax.plot(x_extra_loc, 
						extra['IV']['mean'] - center_around_extra, 
						'indigo', label='GP Mean Prediction for ' + y_label)
			
			ax.plot(x_train_loc, 
					np.zeros_like(self.x_train.cpu().numpy()), 
					'k*', label='Training Locations', markersize=3)
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

	def _plot_price(self, figsize=(20, 10),
					ax=None,
					option_type = 'Call',
					x_version = 'm',
					x_lim = (None, None)):
		with torch.no_grad():

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
						alpha=0.85)		
				ax.fill_between(x_centered_loc_call, 
								self.CallData[self.col_map['ask']].values, 
								self.CallData[self.col_map['bid']].values,
								alpha=0.85, label='Call: Bid-Ask ' + 'Spread',
								color = 'orange')
			elif option_type == 'Put':
				ax.plot(x_centered_loc_put, 
						self.PutData['mid'], 
						label='Put: Mid-Price', color='b',
						alpha=0.85)		
				ax.fill_between(x_centered_loc_put, 
								self.PutData[self.col_map['ask']].values, 
								self.PutData[self.col_map['bid']].values,
								alpha=0.85, label='Put: Bid-Ask ' + 'Spread',
								color = 'lightblue')

			if x_version == 'strike':
				T = self.data['T'].iloc[0]
				x_extra_loc = self._strike_calculator(T = T)
				x_train_loc = self.m_scaler.inverse(self.x_train.cpu().numpy())
				x_train_loc = self._strike_calculator(T = T, m = x_train_loc)
				x_label = 'Strike Price'
			
			elif x_version == 'log_m':
				x_extra_loc = self.x_test_extra
				x_train_loc = self.m_scaler.inverse(self.x_train)
				x_label = 'Log-Moneyness'
			
			elif x_version == 'm':
				x_extra_loc = self.x_test_extra
				x_extra_loc = np.exp(x_extra_loc)
				x_train_loc = self.m_scaler.inverse(self.x_train.cpu().numpy())
				x_train_loc = np.exp(x_train_loc)
				x_label = 'Moneyness'
			else:
				raise ValueError("x_version must be 'strike', 'log_m', or 'm'")

			if option_type == 'Call':
				ax.fill_between(x_extra_loc, 
								extra['Call']['lb'],  
								extra['Call']['ub'], 
								alpha=0.40, label='95% CI for ' + y_label,
								color = 'indigo')		
				ax.plot(x_extra_loc, 
						extra['Call']['mean'], 
						'indigo', label='GP Mean Prediction for ' + y_label)
			
				epsilon = 1e-6  
				y_train = self.CallData[
					self.CallData['m'].apply(
						lambda mid_tiv: any(abs(mid_tiv - y) <= epsilon for y in self.m_scaler.inverse(self.x_train.cpu().numpy()))
					)
				].loc[:, 'mid'].values
				y_train = np.repeat(y_train, 2)
				#ax.plot(x_train_loc, 
				#    y_train, 
				#    'k*', label='Training Locations', markersize=3)                
				
			elif option_type == 'Put':
				ax.fill_between(x_extra_loc, 
								extra['Put']['lb'],  
								extra['Put']['ub'], 
								alpha=0.40, label='95% CI for ' + y_label,
								color = 'indigo')		
				ax.plot(x_extra_loc, 
						extra['Put']['mean'], 
						'indigo', label='GP Mean Prediction for ' + y_label)

				epsilon = 1e-6  
				y_train = self.PutData[
					self.PutData['m'].apply(
						lambda mid_tiv: any(abs(mid_tiv - y) <= epsilon for y in self.m_scaler.inverse(self.x_train.cpu().numpy()))
					)
				].loc[:, 'mid'].values
				y_train = np.repeat(y_train, 2)
				#ax.plot(x_train_loc, 
				#    y_train, 
				#    'k*', label='Training Locations', markersize=3)
				
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
						alpha=0.85)		
				ax.fill_between(x_centered_loc_call, 
								self.CallData[self.col_map['ask']].values - center_around, 
								self.CallData[self.col_map['bid']].values - center_around,
								alpha=0.85, label='Call: Bid-Ask ' + 'Spread',
								color = 'orange')
			elif option_type == 'Put':
				ax.plot(x_centered_loc_put, 
						self.PutData['mid'] - center_around, 
						label='Put: Mid-Price', color='b',
						alpha=0.85)		
				ax.fill_between(x_centered_loc_put, 
								self.PutData[self.col_map['ask']].values - center_around, 
								self.PutData[self.col_map['bid']].values - center_around,
								alpha=0.85, label='Put: Bid-Ask ' + 'Spread',
								color = 'lightblue')

			if x_version == 'strike':
				T = self.data['T'].iloc[0]
				x_extra_loc = self._strike_calculator(T = T)
				x_train_loc = self.m_scaler.inverse(self.x_train.cpu().numpy())
				x_train_loc = self._strike_calculator(T = T, m = x_train_loc)
				x_label = 'Strike Price'
			
			elif x_version == 'log_m':
				x_extra_loc = self.x_test_extra
				x_train_loc = self.m_scaler.inverse(self.x_train)
				x_label = 'Log-Moneyness'
			
			elif x_version == 'm':
				x_extra_loc = self.x_test_extra
				x_extra_loc = np.exp(x_extra_loc)
				x_train_loc = self.m_scaler.inverse(self.x_train.cpu().numpy())
				x_train_loc = np.exp(x_train_loc)
				x_label = 'Moneyness'
			else:
				raise ValueError("x_version must be 'strike', 'log_m', or 'm'")

			if option_type == 'Call':
				ax.fill_between(x_extra_loc, 
								extra['Call']['lb'] - center_around_extra,  
								extra['Call']['ub'] - center_around_extra, 
								alpha=0.40, label='95% CI for ' + y_label,
								color = 'indigo')		
				ax.plot(x_extra_loc, 
						extra['Call']['mean'] - center_around_extra, 
						'indigo', label='GP Mean Prediction for ' + y_label)
			elif option_type == 'Put':
				ax.fill_between(x_extra_loc, 
								extra['Put']['lb'] - center_around_extra,  
								extra['Put']['ub'] - center_around_extra, 
								alpha=0.40, label='95% CI for ' + y_label,
								color = 'indigo')		
				ax.plot(x_extra_loc, 
						extra['Put']['mean'] - center_around_extra, 
						'indigo', label='GP Mean Prediction for ' + y_label)
		
			ax.plot(x_train_loc, 
					np.zeros_like(self.x_train.cpu().numpy()), 
					'k*', label='Training Locations', markersize=3)
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

	# Helper functions

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