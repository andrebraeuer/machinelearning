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

    u = pseudo_obs(matrix_residuals_std) # Convert standardized residuals to pseudo-observations (ranks scaled to [0,1])
    copula = GaussianCopula(dim=matrix_residuals_std.shape[1]) # Initialize Gaussian copula
    copula.fit(u) # Fit the copula to the pseudo-observations

    return copula


def fit_copula_t(matrix_residuals_std):
    """
    Fit a t-copula to the standardized residuals from GARCH models. 
    Parameters:
        - matrix_residuals_std: 2D array, standardized residuals for each asset (columns) and time points (rows).
    Returns:
        - copula: Fitted t-copula object.
    """
    from copulae import TCopula

    u = pseudo_obs(matrix_residuals_std) # Convert standardized residuals to pseudo-observations (ranks scaled to [0,1])
    copula = TCopula(dim=matrix_residuals_std.shape[1]) # Initialize t-copula
    copula.fit(u) # Fit the copula to the pseudo-observations

    return copula


def fit_copula_clayton(matrix_residuals_std):
    """
    Fit a Clayton copula to the standardized residuals from GARCH models. 
    Parameters:
        - matrix_residuals_std: 2D array, standardized residuals for each asset (columns) and time points (rows).
    Returns:
        - copula: Fitted Clayton copula object.
    """
    from copulae.archimedean import ClaytonCopula

    u = pseudo_obs(matrix_residuals_std) # Convert standardized residuals to pseudo-observations (ranks scaled to [0,1])
    copula = ClaytonCopula(dim=matrix_residuals_std.shape[1]) # Initialize Clayton copula
    copula.fit(u) # Fit the copula to the pseudo-observations

    return copula


def simulate_copula(copula, n_sim=10000):
    """
    Simulate data from a fitted copula. 
    Parameters:
        - copula: Fitted copula object (GaussianCopula or TCopula).
        - n_sim: Number of simulations to generate. 
    Returns:
        - random: Simulated data from the copula.
    """
    u_sim = copula.random(n_sim) # Simulate data from the fitted copula
    return u_sim