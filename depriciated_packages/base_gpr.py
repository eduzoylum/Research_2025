import numpy as np
import scipy.stats as stats
import warnings

import torch
import gpytorch
from linear_operator.settings import max_cg_iterations

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt

to_32bit = lambda x: np.array(x, dtype=np.float32)


class GPY_Torch_Model(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, input_dim=2):
        super(GPY_Torch_Model, self).__init__(train_x, train_y, likelihood)
        # You can choose a mean module per your requirement
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_kernel = gpytorch.kernels.MaternKernel(ard_num_dims=input_dim, nu=2.5)
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class Base_GP_Model:
    def __init__(self, quoteDateData, y_label='IV', type='surface', train_fraction=0.6,
                 c=0.05, device=None, **kwargs):
        # Set device: use provided device or default to CUDA if available
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        print(f'Using device: {self.device}')

        # Set additional attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Set default values if not provided via kwargs
        if 'lr' not in kwargs:
            self.lr = 0.1
        if 'training_iterations' not in kwargs:
            self.training_iterations = 200
        if 'sep_obs' not in kwargs:
            self.sep_obs = False
        if 'y_scaling_factor' not in kwargs:
            self.y_scaling_factor = 1000
        if 'learn_additional_noise' not in kwargs:
            self.learn_additional_noise = False
        
        # Normalization functions for the 'm' and 'T' columns
        m_min, m_max = quoteDateData['m'].min(), quoteDateData['m'].max()
        T_min, T_max = quoteDateData['T'].min(), quoteDateData['T'].max()

        normalize_m = lambda x: (x - m_min) / (m_max - m_min)
        normalize_T = lambda x: (x - T_min) / (T_max - T_min)
        self.normalize_m = normalize_m
        self.normalize_T = normalize_T
        self.inv_normalize_m = lambda x: (x * (m_max - m_min)) + m_min
        self.inv_normalize_T = lambda x: (x * (T_max - T_min)) + T_min

        # Set the training data (full dataset stored for predictions later)
        self.data = quoteDateData
        self.y_label = y_label

        partialQuoteDateData = quoteDateData.sample(frac=train_fraction, replace=False).sort_values(by=['m', 'T'])
        
        # Process observations: either separate (bid/ask) or using the mid price
        if self.sep_obs:
            partialQuoteDateData = partialQuoteDateData.melt(
                id_vars=['quote_date', 'expiration', 'strike', 'm', 'T', 'spread_' + y_label],
                value_vars=['adj_bid_' + y_label, 'adj_ask_' + y_label],
                var_name='type',
                value_name='y_var'
            )
            y_train = to_32bit(partialQuoteDateData['y_var'].values) * self.y_scaling_factor
        
        else:
            y_train = to_32bit(partialQuoteDateData['adj_mid_' + y_label].values) * self.y_scaling_factor

        Z_value = stats.norm.ppf(1 - c / 2)
        noise_train = ((partialQuoteDateData['spread_' + y_label]).values * self.y_scaling_factor) ** 2 / (Z_value ** 2)
        noise_train = to_32bit(noise_train)

        # Convert training data to torch tensors and move them to the selected device.
        self.y_train = torch.tensor(y_train, dtype=torch.float).to(self.device)
        self.noise = torch.tensor(noise_train, dtype=torch.float).to(self.device)

        if type == 'surface':
            self.m_train = to_32bit(normalize_m(partialQuoteDateData['m'].values))
            self.T_train = to_32bit(normalize_T(partialQuoteDateData['T'].values))
            self.x_train = torch.tensor(np.column_stack((self.m_train, self.T_train)), dtype=torch.float).to(self.device)
            self.input_dim = 2
        elif type == 'smile':
            m_train = to_32bit(normalize_m(partialQuoteDateData['m'].values))
            self.x_train = torch.tensor(m_train, dtype=torch.float).to(self.device)
            self.input_dim = 1
        else:
            raise ValueError("Invalid type. Choose 'surface' or 'smile'.")
        
        self.likelihood = None
        self.model = None

    def learn(self, **kwargs):
        if 'noise' in kwargs:
            self.noise = torch.tensor(kwargs['noise'], dtype=torch.float).to(self.device)
        if 'lr' in kwargs:
            self.lr = kwargs['lr']
        if 'training_iterations' in kwargs:
            self.training_iterations = kwargs['training_iterations']

        with gpytorch.settings.min_fixed_noise(float_value=1e-10):
            with gpytorch.settings.cholesky_jitter(1e-6):
                with max_cg_iterations(4000):
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
                        model = GPY_Torch_Model(self.x_train, self.y_train, likelihood, input_dim=self.input_dim)
                        self.model = model
                    else:
                        model = self.model

                    # Move model and likelihood to the designated device
                    model = model.to(self.device)
                    likelihood = likelihood.to(self.device)

                    # Set the model and likelihood to training mode
                    model.train()
                    likelihood.train()

                    # Use Adam optimizer
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
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
                        
                        if (i + 1) % int(self.training_iterations / 4) == 0:
                            print(f'Iter {i + 1}/{training_iterations} - Loss: {loss.item():.4f}')
                            print(f'  lengthscale: {model.covar_module.base_kernel.lengthscale}')
                            print(f'  outputscale: {model.covar_module.outputscale.item():.4f}')

                    self.model = model
                    self.loss = losses

                    # Plot training curve
                    plt.figure(figsize=(10, 4))
                    plt.plot(losses)
                    plt.title("Negative Log Likelihood During Training")
                    plt.xlabel("Iterations")
                    plt.ylabel("NLL")
                    plt.show()

                    # Store kernel parameters for further use
                    if self.input_dim == 1:
                        kernel_params = {
                            'lengthscale': model.covar_module.base_kernel.lengthscale.item(),
                            'outputscale': model.covar_module.outputscale.item()
                        }
                        self.kernel_params = kernel_params
                    elif self.input_dim == 2:
                        kernel_params = {
                            'lengthscale_m': model.covar_module.base_kernel.lengthscale[0][0].item(),
                            'lengthscale_T': model.covar_module.base_kernel.lengthscale[0][1].item(),
                            'outputscale': model.covar_module.outputscale.item()
                        }
                        self.kernel_params = kernel_params

                    return self

    def predict(self, add_x_test=None, add_label=None):
        # Prepare test data
        if self.input_dim == 1:
            x_test = self.normalize_m(self.data['m'].values)
        elif self.input_dim == 2:
            x_test = self.normalize_m(self.data['m'].values)
            x_test = np.column_stack((x_test, self.normalize_T(self.data['T'].values)))
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
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if self.input_dim == 2:
                noise = torch.ones_like(x_test[:, 0]) * 1e-5  # adjust noise if needed
            elif self.input_dim == 1:
                noise = torch.ones_like(x_test) * 1e-5
            else:
                raise ValueError("Invalid input dimension. Expected 1 or 2.")
            observed_pred = self.likelihood(self.model(x_test), noise=noise)
        
        if self.input_dim == 2:
            m_test = x_test[:, 0].cpu().numpy()
            T_test = x_test[:, 1].cpu().numpy()
            if label is None:
                self.m_test = self.inv_normalize_m(m_test)
                self.T_test = self.inv_normalize_T(T_test)
                self.x_test = np.column_stack((m_test, T_test))
            else:
                setattr(self, f"m_test_{label}", self.inv_normalize_m(m_test))
                setattr(self, f"T_test_{label}", self.inv_normalize_T(T_test))
        elif self.input_dim == 1:
            if label is None:
                self.x_test = self.inv_normalize_m(x_test.cpu().numpy())
            else:
                setattr(self, f"x_test_{label}", self.inv_normalize_m(x_test.cpu().numpy()))
        else:
            raise ValueError("Invalid input dimension. Expected 1 or 2.")

        pred = {
            'mean': observed_pred.mean.cpu().numpy() / self.y_scaling_factor,
            'lower': (observed_pred.mean.cpu().numpy() / self.y_scaling_factor) - 1.96 * observed_pred.stddev.cpu().numpy() / self.y_scaling_factor,
            'upper': (observed_pred.mean.cpu().numpy() / self.y_scaling_factor) + 1.96 * observed_pred.stddev.cpu().numpy() / self.y_scaling_factor,
            'variance': observed_pred.variance.cpu().numpy() / (self.y_scaling_factor ** 2),
        }
        return pred

    def plot_smile_comparison(self, figsize=(24, 10), zoom=0):
        with torch.no_grad():
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            self._plot_smile(ax=ax2)
            self._plot_smile_centered(ax=ax1, zoom=zoom)
            plt.tight_layout()
            plt.show()
            return self

    def _plot_smile(self, figsize=(20, 10), ax=None):

        with torch.no_grad():

            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
                
            ax.fill_between(self.data['m'], self.data['adj_ask_' + self.y_label],
                            self.data['adj_bid_' + self.y_label], alpha=0.5,
                            label='Bid-Ask ' + self.y_label + ' Spread')
            ax.plot(self.data['m'], self.data['adj_mid_' + self.y_label],
                    label='Mid-Price ' + self.y_label, color='b')
            ax.fill_between(self.x_test_extra, self.pred_extra['lower'],
                            self.pred_extra['upper'], alpha=0.5,
                            label='95% Confidence Interval for ' + self.y_label)
            ax.plot(self.x_test_extra, self.pred_extra['mean'],
                    'r', label='GP Mean Prediction for ' + self.y_label)
            ax.plot(self.inv_normalize_m(self.x_train.cpu().numpy()), self.y_train.cpu().numpy() / self.y_scaling_factor,
                    'k*', label='Training Data', markersize=4)
            ax.set_xlabel('Moneyness')
            y_lab = 'Implied Volatility' if self.y_label == 'IV' else 'Total Implied Variance'
            obs_type = 'Two Separate Observations' if self.sep_obs else 'Single Observation'
            ax.set_ylabel(y_lab)
            ax.set_title(f"{y_lab} vs Moneyness\n Expiry Date: {self.data['expiration'].iloc[0]} -- Quote Date: {self.data['quote_date'].iloc[0]}\n Each Option: {obs_type}")
            ax.legend()
            return ax

    def _plot_smile_centered(self, figsize=(20, 10), zoom=0, ax=None):

        with torch.no_grad():
            
            center_around = self.pred['mean']
            center_around_extra = self.pred_extra['mean']
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            
            ax.fill_between(self.data['m'], 
                            self.data['adj_ask_' + self.y_label].values - center_around, 
                            self.data['adj_bid_' + self.y_label].values - center_around,
                            alpha=0.5, label='Bid-Ask ' + self.y_label + ' Spread')
            ax.plot(self.data['m'], 
                    self.data['adj_mid_' + self.y_label] - center_around, 
                    label='Mid-Price ' + self.y_label, color='b')
            ax.fill_between(self.x_test_extra, 
                            self.pred_extra['lower'] - center_around_extra,  
                            self.pred_extra['upper'] - center_around_extra, 
                            alpha=0.5, label='95% Confidence Interval for ' + self.y_label)
            ax.plot(self.x_test_extra, 
                    self.pred_extra['mean'] - center_around_extra, 
                    'r', label='GP Mean Prediction for ' + self.y_label)
            ax.plot(self.inv_normalize_m(self.x_train.cpu().numpy()), 
                    np.zeros_like(self.x_train.cpu().numpy()), 
                    'k*', label='Training Locations', markersize=3)
            ax.set_xlabel('Moneyness')
            y_lab = 'Implied Volatility' if self.y_label == 'IV' else 'Total Implied Variance'
            obs_type = 'Two Separate Observations' if self.sep_obs else 'Single Observation'
            ax.set_ylabel(y_lab)
            ax.set_title(f"{y_lab} - Centered around Mean Prediction\n Expiry Date: {self.data['expiration'].iloc[0]} -- Quote Date: {self.data['quote_date'].iloc[0]}\n Each Option: {obs_type}")
            ax.legend(loc='lower center')
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            perc = zoom / 100 * 0.5
            ax.set_ylim(y_min + perc * y_range, y_max - perc * y_range)
            return ax

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
            train_m = self.inv_normalize_m(self.x_train[:, 0].detach().cpu().numpy())
            train_T = self.inv_normalize_T(self.x_train[:, 1].detach().cpu().numpy())
            
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
        - self.inv_normalize_m and self.inv_normalize_T are the corresponding inverse normalization functions.
        - self.y_scaling_factor is used to scale the target values.
        
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
            train_m = self.inv_normalize_m(self.x_train[:, 0].detach().cpu().numpy())
            train_T = self.inv_normalize_T(self.x_train[:, 1].detach().cpu().numpy())
            train_y = self.y_train.detach().cpu().numpy() / self.y_scaling_factor
            
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
            train_y_centered = (self.y_train.detach().cpu().numpy() / self.y_scaling_factor) - avg_pred_mean
            
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

    def calculate_derivative_1d(self, x_test, epsilon_m = 1e-3, num_of_samples = 10**5):

        with torch.no_grad(), gpytorch.settings.fast_pred_var():

            num_of_samples = int(num_of_samples / 3) * 3  # Ensure it's a multiple of 3

            x_test_eps = np.repeat(x_test, 3)
            x_test_eps[0::3] = x_test - epsilon_m/2
            x_test_eps[1::3] = x_test
            x_test_eps[2::3] = x_test + epsilon_m/2
            x_test_eps = torch.tensor(x_test_eps, dtype=torch.float).to(self.device)
            self.model.eval()
            self.likelihood.eval()

            # Make predictions
            
            if self.input_dim == 2:
                noise = torch.zeros_like(x_test_eps[:, 0])   # adjust noise if needed
            elif self.input_dim == 1:
                noise = torch.zeros_like(x_test_eps) 
            else:
                raise ValueError("Invalid input dimension. Expected 1 or 2.")
            
            pred_eps = self.likelihood(self.model(x_test_eps), noise=noise)

            samples = pred_eps.sample(torch.Size([num_of_samples])).cpu().numpy() / self.y_scaling_factor

            samples_minus = samples[:,0::3]
            samples_plus = samples[:,2::3]
            samples_mid = samples[:,1::3]

            first_derivative = ((samples_plus - samples_minus) / (epsilon_m)).mean(axis=0)
            second_derivative = ((samples_plus + samples_minus - 2 * samples_mid) / (epsilon_m**2)).mean(axis=0)
            function_val = samples_mid.mean(axis=0)

            # Create a dictionary to store the results
            results = {
                'first_derivative': first_derivative,
                'second_derivative': second_derivative,
                'function_val': function_val
            }

            self.x_test_deriv = self.inv_normalize_m(x_test)
            self.pred_deriv = results

            x = self.x_test_deriv
            b_a = (self.inv_normalize_m(1) - self.inv_normalize_m(0))


            g_1_inside = 1  -  (x / b_a * self.pred_deriv['first_derivative']) / (2*self.pred_deriv['function_val'])  

            g_1 = (g_1_inside)**2
            g_1_approx = 2 * (g_1_inside) - 1

            g_2 = - (self.pred_deriv['first_derivative']**2 / (4*b_a**2) ) * ( 1 / self.pred_deriv['function_val'] + 0.25)

            g_3 = self.pred_deriv['second_derivative'] / (2 * b_a**2)

            g = g_1 + g_2 + g_3

            # Create a dictionary to store the results
            results_compenets = {
                'g_1': g_1,
                'g_1_approx': g_1_approx,
                'g_2': g_2,
                'g_3': g_3,
                'g': g
            }

            self.constraint_results = results_compenets

            return self
        
    def calculate_derivatives_2d(self, x_test, epsilon_m=1e-3, epsilon_T=1e-3, num_of_samples_m=10**5, num_of_samples_T=10**5):
        """
        Computes the first and second derivatives with respect to both m and T directions
        for a 2D input grid. The derivatives are calculated from finite differences computed
        along each coordinate individually.
        
        Parameters:
            x_test (np.array): A 2D array of normalized test inputs with shape (n, 2),
                            where each row is [m, T].
            epsilon_m (float): Finite difference step size for m direction.
            epsilon_T (float): Finite difference step size for T direction.
            num_of_samples (int): Number of Monte Carlo samples for the GP predictive distribution.
            
        The method stores:
        - self.pred_deriv_m and self.pred_deriv_T: derivative info (first, second, function value)
            for the m and T directions, respectively.
        - self.constraint_results_m and self.constraint_results_T: constraint components for each dimension.
        - Also stores the unnormalized x values as self.x_test_deriv_m and self.x_test_deriv_T.
        
        Returns:
            self (to allow for chaining).
        """
        if self.input_dim != 2:
            raise ValueError("Model is not configured for 2 dimensions.")

        # Calculate derivative estimates along m (dim=0).
        deriv_m = self._calculate_derivative_along_dim(x_test, dim=0, epsilon=epsilon_m, num_of_samples=num_of_samples_m)
        # Calculate derivative estimates along T (dim=1).
        deriv_T = self._calculate_derivative_along_dim(x_test, dim=1, epsilon=epsilon_T, num_of_samples=num_of_samples_T)

        # Store the unnormalized test inputs for each coordinate.
        self.m_test_deriv = self.inv_normalize_m(x_test[:, 0])
        self.T_test_deriv = self.inv_normalize_T(x_test[:, 1])
        self.pred_deriv_m = deriv_m
        self.pred_deriv_T = deriv_T

        # Define scaling factors using the inverse normalization functions.
        # These represent the span in the original units for each axis.
        b_a = self.inv_normalize_m(1) - self.inv_normalize_m(0)

        # Compute constraint components for m derivatives (as in your original code).
        g_1_m_inside = 1 - ((self.m_test_deriv / b_a) * deriv_m['first_derivative']) / (2 * deriv_m['function_val'])
        g_1_m = g_1_m_inside ** 2
        g_1_m_approx = 2 * (g_1_m_inside) - 1  # Approximation of g_1_m
        g_2_m = - (deriv_m['first_derivative']**2 / (4 * b_a**2)) * (1 / deriv_m['function_val'] + 0.25)
        g_3_m = deriv_m['second_derivative'] / 2
        g_m = g_1_m + g_2_m + g_3_m
        constraint_results_m = {
            'g_1': g_1_m,
            'g_1_approx': g_1_m_approx,
            'g_2': g_2_m,
            'g_3': g_3_m,
            'g': g_m
        }

        # Save both sets of results in the model.
        self.constraint_results = constraint_results_m

        return self
    
    def _calculate_derivative_along_dim(self, x_test, dim, epsilon=1e-3, num_of_samples=10**5):
        """
        Helper function: Computes finite-difference approximations of the first and second 
        derivatives with respect to the coordinate specified by 'dim'.
        
        Parameters:
            x_test (np.array): A 2D array of normalized test inputs with shape (n, d).
            dim (int): The column index to perturb (0 for m, 1 for T, etc.).
            epsilon (float): Finite difference step size.
            num_of_samples (int): Number of Monte Carlo samples.
        
        Returns:
            A dictionary with:
            - 'first_derivative': Estimated first derivative.
            - 'second_derivative': Estimated second derivative.
            - 'function_val': Estimated function value at the unperturbed points.
        """
        n = x_test.shape[0]
        # Repeat each row 3 times so we can perturb only the selected coordinate:
        x_test_eps = np.repeat(x_test, 3, axis=0)
        # Perturb the designated coordinate: 
        #    - first copy: x - epsilon/2
        #    - second copy: unperturbed x
        #    - third copy: x + epsilon/2
        x_test_eps[0::3, dim] = x_test[:, dim] - epsilon / 2
        x_test_eps[1::3, dim] = x_test[:, dim]  # unperturbed
        x_test_eps[2::3, dim] = x_test[:, dim] + epsilon / 2

        # Convert to tensor and move to the appropriate device.
        x_test_eps = torch.tensor(x_test_eps, dtype=torch.float).to(self.device)

        # Make predictions: use a small noise level as in your predict routines.
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            noise = torch.zeros_like(x_test_eps[:, 0]) 
            observed_pred = self.likelihood(self.model(x_test_eps), noise=noise)

        # Draw Monte Carlo samples (and rescale them).
        samples = observed_pred.sample(torch.Size([num_of_samples])).cpu().numpy() / self.y_scaling_factor

        # Group the samples corresponding to the three perturbations:
        samples_minus = samples[:, 0::3]
        samples_mid = samples[:, 1::3]
        samples_plus = samples[:, 2::3]

        # Compute derivatives using finite differences.
        first_derivative = ((samples_plus - samples_minus) / epsilon).mean(axis=0)
        second_derivative = ((samples_plus + samples_minus - 2 * samples_mid) / (epsilon ** 2)).mean(axis=0)
        function_val = samples_mid.mean(axis=0)

        return {
            'first_derivative': first_derivative,
            'second_derivative': second_derivative,
            'function_val': function_val
        }

    def plot_constraint(self, num_m=100, num_T=20):
    
        if self.input_dim == 1:
            self._plot_constraint_1d()
        elif self.input_dim == 2:
            self._plot_constraint_approx(num_m=num_m, num_T=num_T)
            self._plot_constraint_components(num_m = num_m, num_T = num_T)
        else:
            raise ValueError("Invalid input dimension. Expected 1 or 2.")

    def _plot_constraint_approx(self, figsize= (12, 6), num_m = 100, num_T = 20):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize= figsize)

        r_shape = lambda x: x.reshape((num_T,num_m))
        T = self.T_test_deriv.reshape((num_T,num_m))

        for ind in range(num_T):
            color = plt.cm.plasma(ind / num_T)  # Use a colormap to vary color with ind
            ax1.plot(r_shape(self.m_test_deriv)[ind], r_shape(self.constraint_results['g_1'])[ind], label=f'g curve {ind}', color=color)
            ax2.plot(r_shape(self.m_test_deriv)[ind], r_shape(self.constraint_results['g_1_approx'])[ind], label=f'g curve {ind}', color=color)
            sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=T[0, 0], vmax=T[19, 0]))

        y_min,ymax = ax1.get_ylim()
        ax2.set_ylim(y_min, ymax)


        ax1.set_xlabel('Moneyness')
        ax2.set_xlabel('Moneyness')
        ax1.set_ylabel('Constraint Value')
        ax2.set_ylabel('Constraint Value')

        ax1.set_title('Second Order Component')
        ax2.set_title('Approximate First Order Component')

        plt.colorbar(sm, label='Time to Maturity', ax = ax2)
        plt.show()

    def _plot_constraint_components(self, figsize=(12, 12), num_m = 100, num_T = 20):

        fig, (ax1, ax2) = plt.subplots(2, 2, figsize=figsize)

        ax11, ax12 = ax1.flatten()
        ax21, ax22 = ax2.flatten()

        r_shape = lambda x: x.reshape((num_T,num_m))
        T = self.T_test_deriv.reshape((num_T,num_m))

        for ind in range(num_T):
            color = plt.cm.viridis(ind / num_T)  # Use a colormap to vary color with ind
            ax11.plot(r_shape(self.m_test_deriv)[ind], r_shape(self.constraint_results['g_2'])[ind], label=f'g curve {ind}', color=color)
            ax12.plot(r_shape(self.m_test_deriv)[ind], r_shape(self.constraint_results['g_3'])[ind], label=f'g curve {ind}', color=color)
            ax21.plot(r_shape(self.m_test_deriv)[ind], r_shape(self.constraint_results['g_1_approx'])[ind] + r_shape(self.constraint_results['g_2'])[ind] + r_shape(self.constraint_results['g_3'])[ind]
                    , label=f'g curve {ind}', color=color)
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

        ax11.set_title('Middle Component')
        ax12.set_title('Second Derivative Component')
        ax21.set_title('Durrleman Condition - Approximate')
        ax22.set_title('Durrleman Condition - True')



        plt.colorbar(sm, label='Time to Maturity', ax= ax12)
        plt.colorbar(sm, label='Time to Maturity', ax= ax22)

        plt.show()

    def _plot_constraint_1d(self):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        ax1.plot(self.x_test_deriv, self.constraint_results['g_1'], label='First Component')
        ax1.plot(self.x_test_deriv, self.constraint_results['g_1_approx'], label='Approximate First Component')
        ax1.plot(self.x_test_deriv, self.constraint_results['g_2'], label='Second Component')
        ax1.plot(self.x_test_deriv, self.constraint_results['g_3'], label='Third Component')
        ax1.set_title('Constraint Components')
        ax1.set_xlabel('Moneyness')
        ax1.set_ylabel('Constraint Values')
        ax1.legend()

        ax2.plot(self.x_test_deriv, self.constraint_results['g'], label='Durrleman\'s Condition')
        ax2.plot(self.x_test_deriv, self.constraint_results['g_1_approx']+self.constraint_results['g_2']+self.constraint_results['g_3'], label='Approximate Durrleman\'s Condition')
        x_min, x_max = ax2.get_xlim()
        ax2.hlines(0, x_min, x_max, color='k', linestyle='--', label='Zero Line')
        ax2.set_title('Durrleman\'s Condition')
        ax2.set_xlabel('Moneyness')
        ax2.set_ylabel('Total Constraint Value')
        ax2.legend()

        plt.tight_layout()