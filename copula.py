"""
Fit a copula to standardised GARCH residuals.

From the documentation of the `copulae` package:
Internally, the fit method will convert the data to pseudo observations so there is no need to for that sort of data treatment prior. 
However, even if your data is already in pseudo observations, there will be no change to the results as the transformation is monotonic in nature.
"""

def fit_copula_gaussian(matrix_residuals_std):
    """
    Fit a Gaussian copula to the standardized residuals from GARCH models.
    Parameters:
        - matrix_residuals_std: 2D array, standardized residuals for each asset (columns) and time points (rows).
    Returns:
        - copula: Fitted Gaussian copula object.
    """
    from copulae import GaussianCopula

    copula = GaussianCopula(dim=matrix_residuals_std.shape[1]) # Initialize Gaussian copula
    copula.fit(matrix_residuals_std) # Fit the copula to the standardized residuals

    return copula

def fit_copula_t(matrix_residuals_std, df):
    """
    Fit a t-copula to the standardized residuals from GARCH models. 
    Parameters:
        - matrix_residuals_std: 2D array, standardized residuals for each asset (columns) and time points (rows).
        - df: Degrees of freedom for the t-copula.
    Returns:
        - copula: Fitted t-copula object.
    """
    from copulae import TCopula

    copula = TCopula(dim=matrix_residuals_std.shape[1], df=df) # Initialize t-copula
    copula.fit(matrix_residuals_std) # Fit the copula to the standardized residuals

    return copula

def simulate_copula(copula, n_sim):
    """
    Simulate data from a fitted copula.
    Parameters:
        - copula: Fitted copula object (GaussianCopula or TCopula).
        - n_sim: Number of simulations to generate.
    Returns:
        - samples: Simulated data from the copula.
    """
    samples = copula.sample(n_sim) # Simulate data from the fitted copula
    return samples