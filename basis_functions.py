import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class BasisFunctions:
    """
    Defines piecewise linear basis functions for finite-dimensional Gaussian Process approximation.
    """
    def __init__(self, **kwargs):
        """
        Initialize the basis function for either a given set of knots, number of knots, or knot spacing.
        
        Parameters:
        - knots (list): Knots (d dimensional list of lists of length (m_1, m_2, ..., m_d))
        - m (int or list): Number of knots
        - delta_m (int or list): Space between knots
        - d (int): Dimension of input space
        - **kwargs: Additional keyword arguments.
        """

        # Handle keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

        # ------ Make sure that only one of the knots, m, or delta_m is provided ----- #
        if 'knots' in kwargs and 'm' in kwargs or 'knots' in kwargs and 'delta_m' in kwargs or 'm' in kwargs and 'delta_m' in kwargs:
            raise ValueError("Only one of 'knots', 'm', or 'delta_m' must be provided.")


        # ---------- Check if the dimension of the input space is provided ---------- #
        if 'd' in kwargs:
            self.d = kwargs['d']

        # If its not provided, go trough the other options to deduce it
        elif 'knots' in kwargs:
            # Make sure that what we have is a list of list, not a single list for d=1.
            if isinstance(kwargs['knots'][0], list):
                self.d = len(kwargs['knots'])
            else:
                self.d = 1
                self.knots = [kwargs['knots']]
        elif 'm' in kwargs:
            # If m is provided as an integer, the dimension is 1
            if isinstance(kwargs['m'], int):
                self.d = 1
            # If m is provided as an array, the dimension is the length of the array
            elif isinstance(kwargs['m'], list):
                self.d = len(kwargs['m'])
            else:
                raise TypeError("Invalid type for 'm'.")

        elif 'delta_m' in kwargs:
            # If delta_m is provided as an integer, the dimension is 1
            if isinstance(kwargs['delta_m'], float):
                self.d = 1
            # If delta_m is provided as an array, the dimension is the length of the array
            elif isinstance(kwargs['delta_m'], list):
                self.d = len(kwargs['delta_m'])
            else:
                raise TypeError("Invalid type for 'delta_m'.")

  
        # ------ If knots are provided, set m ------------------------- #
        if 'knots' in kwargs:
            if self.d == 1:
                self.m = len(self.knots[0])
            else:
                self.m = [len(knot_array) for knot_array in self.knots]


        # ------- If m provided, generate the knots ------- #
        if 'm' in kwargs:
            # If the dimension is 1
            if self.d == 1:
                try:
                    self.knots = [np.linspace(0, 1, self.m)]
                except:
                    raise ValueError("Invalid value for 'm'.")

            # If the dimension is higher than 1
            else:
                try:
                    self.knots = [np.linspace(0, 1, m) for m in self.m]
                except:
                    raise ValueError("Invalid value for 'm'.")


        # ------- If delta_m provided, generate the knots ------- #
        if 'delta_m' in kwargs:
            # If the dimension is 1
            if self.d == 1:
                try:
                    self.knots = [np.append(np.arange(0, 1, self.delta_m),1)]
                    self.m = len(self.knots[0])
                except:
                    raise ValueError("Invalid value for 'delta_m'.")
            # If the dimension is higher than 1
            else:
                try:
                    self.knots = [np.append(np.arange(0, 1, delta_m),1) for delta_m in self.delta_m]
                    self.m = [len(knot_array) for knot_array in self.knots]
                except:
                    raise ValueError("Invalid value for 'delta_m'.")
            

        for knot_array,ind in zip(self.knots,range(len(self.knots))):
            self.knots[ind] = np.array(knot_array)

    def phi_one_dim(self, x, j, m, knots):
        """
        Compute piecewise linear basis functions at a given point.
        
        Parameters:
        - x: Input point, float
        - m: Number of knots
        - j: Knot index, going trough (0,2,...,m-1)
        
        Returns:
        - Phi_j(x): Value of the j-th basis function at x.
        """
        if j < 0 or j >= m:
            raise ValueError("Invalid knot index.")        
        t_j = knots[j]

        # Check if this is the first or the last knot.
        if j == 0:
            t_j_minus = 0
            t_j_plus = knots[j+1]
            
            if np.abs(x - t_j_minus) < 1e-6:
                return 1
        elif j == m - 1:
            t_j_minus = knots[j-1]
            t_j_plus = 1
            
            if np.abs(x - t_j_plus) < 1e-6:
                return 1
        else:
            t_j_minus = knots[j-1]
            t_j_plus = knots[j+1]

        # Compute the basis function
        if t_j_minus <= x < t_j:
            return (x - t_j_minus) / (t_j - t_j_minus)
        elif t_j <= x <= t_j_plus:
            return (t_j_plus - x) / (t_j_plus - t_j)
        else:
            return 0
 
    def basis_matrix(self,x):
        """
        Compute the basis matrix for the given set of knots and input data.
        
        Returns:
        - Basis matrix Phi, shape (n_samples, m)
        """
        
        # If the dimension is 1
        if self.d == 1:
            knots = self.knots[0]
            m = self.m
            Phi = np.zeros((len(x), m))
            
            try: 
                for i, x_i in enumerate(x):
                    for j in range(m):  
                            Phi[i, j] = self.phi_one_dim(x_i, j, m, knots)
            except:
                raise TypeError("For 1-d input data, the input must be a list of floats.")
            self.phi = Phi
            
            return np.array(Phi)
        
        # If the dimension is higher than 1
        else:
            # Check if the input data has the right dimension
            if len(x) != self.d:
                raise ValueError("Input data must have the same dimension as the basis functions.")

            m_total = np.prod(self.m)
            Phi= np.zeros((len(x[0]), m_total))
            for i, x_i in enumerate(np.array(x).T):
                j = 0
                while j < m_total:                    
                    j_knots = self.indices_to_knots(j)
                    product = 1
                    for dim in range(self.d):
                        product *= self.phi_one_dim(x_i[dim], j_knots[dim], self.m[dim], self.knots[dim])
                    Phi[i, j] = product
                    j += 1

            self.phi = Phi
            self.x = x

            return np.array(Phi)

    def knots_to_indices(self, j):
        """
        Convert knot indices to indices of the basis functions.
        
        Parameters:
        - j: List of knot indices, list of integers
        
        Returns:
        - Indices of the basis functions, list of integers
        """
        # Check if the indices are valid
        for j_dim, m_dim in zip(j, self.m):
            if j_dim < 0 or j_dim >= m_dim:
                raise ValueError("Invalid knot index.")

        components = [np.prod(self.m[dim+1:])* j[dim]  for dim in range(self.d)]
        
        return int(np.sum(components))

    def indices_to_knots(self, basis_index):
        """
        Convert indices of the basis functions to knot indices.
        
        Parameters:
        - basis_index: Basis function index, integer
        """
        # Check if the indices are valid
        if basis_index < 0 or basis_index >= np.prod(self.m):
            raise ValueError("Invalid basis function index.")
        
        j_knots = []
        for dim in range(self.d):
            j_knots.append(int(basis_index // np.prod(self.m[dim+1:])))
            basis_index = basis_index % np.prod(self.m[dim+1:])
        
        return j_knots

    def plot_basis_functions(self, X_range=[0, 1], dim = 0):
        """
        Plot the basis functions over a given range.
        
        Parameters:
        - X_range: Tuple (min, max) defining the plotting range
        """

        if self.d == 1:
            X = np.linspace(X_range[0], X_range[1], 100).tolist()
            j_start = 0
            j_end = self.m
        else:
            X = [np.linspace(X_range[0], X_range[1], 100) for i in range(self.d)]
            j_start = int(np.sum(self.m[:dim]))
            j_end = np.sum(self.m[:dim+1])

        Phi = self.basis_matrix(X)
        

        plt.figure(figsize=(8, 5))
        
        for j in range(0, Phi.shape[1]):

            
            if self.d == 1:
                x = X
                knot_x = self.knots[0][j]
            else:
                x = X[dim]
                knot_indices = self.indices_to_knots(j)[dim]
                knot_x = self.knots[dim][knot_indices]


            plt.plot(x, Phi[:, j], label=f'Basis {j}')
            plt.vlines(knot_x, 0, 1, color='red', linestyle='--', alpha=0.15)
        
        plt.xlabel('x')
        plt.ylabel('Basis function value')
        plt.title('Piecewise Linear Basis Functions')
        #plt.legend()
        plt.grid()
        plt.show()
    
    def plot_basis_functions_2d(self,X_range=[0, 1]):
        """
        Plot the basis functions in 2D.
        """
        if self.d != 2:
            raise ValueError("This function is specifically for 2D basis functions.")
        
        # Compute the basis matrix
        X = [np.random.uniform(X_range[0], X_range[1], 10) for i in range(self.d)]

        x = X[0]
        y = X[1]

        x, y = np.meshgrid(x, y)
        xx = x.flatten()
        yy = y.flatten()


        knots_xx, knots_yy = np.meshgrid(self.knots[0], self.knots[1])
        knots_x = knots_xx.flatten()
        knots_y = knots_yy.flatten()


        Phi = self.basis_matrix(X)
        
        # Plot knots
        plt.figure(figsize=(12, 5))
        plt.scatter(xx, yy, color='blue', label='Test points', s=10,alpha=0.5)
        
        # Plot observations
        plt.scatter(knots_x, knots_y, color='green', label='Knot Locations',s =15)
        

        # Plot lines between knots and observations with color based on Phi values
        for i in range(len(xx)):
            for j in range(Phi.shape[1]):
                
                j_knots = self.indices_to_knots(j)
                phi_value = self.phi_one_dim(xx[i], j_knots[0], self.m[0], self.knots[0])*self.phi_one_dim(yy[i], j_knots[1], self.m[1], self.knots[1])
                color = cm.viridis(phi_value)  # Use the 'viridis' colormap for better contrast and readability
                if phi_value > 0:

                    knot_x = self.knots[0][j_knots[0]]
                    knot_y = self.knots[1][j_knots[1]]
                    plt.plot([knot_x, xx[i]], [knot_y, yy[i]], color=color, alpha=phi_value)


        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('2D Basis Functions')
        plt.legend()
        plt.grid()
        plt.show()

