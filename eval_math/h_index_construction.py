import numpy as np
from numba import njit
import math

@njit
def _t_pdf(x, df):
    """
    Compute the probability density function (PDF) of the t-distribution.
    """
    return (1 + x**2/df)**(-(df+1)/2) * math.gamma((df+1)/2) / (math.sqrt(df*math.pi) * math.gamma(df/2))

@njit
def _norm_pdf(x):
    """
    Compute the probability density function (PDF) of the normal distribution.
    """
    return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

@njit
def _t_cdf(x, df):
    """
    Compute the cumulative distribution function (CDF) of the t-distribution.
    """
    if x == 0:
        return 0.5
    elif x > 0:
        return 1 - 0.5 * (1 - math.erf(x/math.sqrt(2)))
    else:
        return 0.5 * (1 - math.erf(-x/math.sqrt(2)))

@njit
def _norm_cdf(x):
    """
    Compute the cumulative distribution function (CDF) of the normal distribution.
    """
    return 0.5 * (1 + math.erf(x/math.sqrt(2)))

@njit
def _norm_ppf(p):
    """
    Compute the inverse CDF (PPF) of the normal distribution.
    This is a Numba-compatible implementation.
    """
    if p < 0.5:
        return -_norm_ppf(1 - p)
    
    t = math.sqrt(-2 * math.log(1 - p))
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308
    
    return t - (c0 + c1*t + c2*t**2)/(1 + d1*t + d2*t**2 + d3*t**3)

@njit
def _t_ppf(p, df):
    """
    Compute the inverse CDF (PPF) of the t-distribution using a simple approximation.
    This is a Numba-compatible implementation.
    """
    if df <= 2:
        return math.sqrt(df * (p**(-2/df) - 1))
    
    z = _norm_ppf(p)
    return z * (1 + (1 + z**2)/(4*df) + (3 + 5*z**2 + 2*z**4)/(96*df**2))

@njit
def H_myopic_jit(recall, sigma_flag, z, k, alpha0):
    """
    Compute the myopic H-function value.
    """
    if recall == 1:
        if sigma_flag == 0:
            df = 2 * alpha0 + k
            return ((df + z**2) / (df - 1)) * _t_pdf(z, df) - z * (1 - _t_cdf(z, df))
        else:  # sigma known
            return _norm_pdf(z) - z * (1 - _norm_cdf(z))
    else:  # recall = 0
        return -z

def build_grids(G, rho=0.85, Z=30, ita=0.75):
    """
    Build the grids for c and z values.
    """
    c = np.zeros(G+2)
    z = np.zeros(G+2)
    for j in range(G+2):
        c[j] = G * rho**j
        z[j] = Z * ((1-ita)*(2*j - G - 1)/(G-1) + ita*((2*j - G - 1)/(G-1))**3)
    c[G+1] = 0
    return c, z

@njit
def h_index_recall1(mu_flag, sigma_flag, alpha0, nu0, n, G, c, z):
    """
    Compute the h-index values for the recall=1 case.
    """
    H = np.zeros((n+1, G+2, G+2))
    h = np.zeros((n+1, G+2))
    
    if sigma_flag == 0:
        k_min = max(np.floor(2 - 2*alpha0), 1)
    else:
        k_min = 1
    k_min = int(k_min)
    
    for k in range(n-1, k_min-1, -1):
        for j_z in range(G, 0, -1):
            for j_c in range(G, 0, -1):
                H[k, j_z, j_c] = H_myopic_jit(1, sigma_flag, z[j_z], k, alpha0)
                if k < n-1:
                    for j_u in range(G, 0, -1):
                        if mu_flag == 0:
                            mu_u = 1 / (nu0 + k + 1)
                        else:
                            mu_u = 0
                        if sigma_flag == 0:
                            L = math.sqrt((1 - mu_u**2) / (2*alpha0 + k + 1))
                            s = L * math.sqrt(2*alpha0 + k + z[j_u]**2)
                        else:
                            s = math.sqrt(1 - mu_u**2)
                        z_new = (max(z[j_z], z[j_u]) - z[j_u]*mu_u) / s
                        if k == n-2:
                            H_u = H_myopic_jit(1, sigma_flag, z_new, k+1, alpha0)
                        else:
                            j_1 = G
                            while j_1 > 1 and z_new < z[j_1]:
                                j_1 -= 1
                            if j_1 == G:
                                j_1 = G-1
                            j_2 = G
                            while j_2 > 1 and c[j_c]/s > c[j_2]:
                                j_2 -= 1
                            if j_2 == G:
                                j_2 = G-1
                            theta_z = (z_new - z[j_1]) / (z[j_1+1] - z[j_1])
                            theta_c = (c[j_c]/s - c[j_2]) / (c[j_2+1] - c[j_2])
                            H_u = (1-theta_c)*((1-theta_z)*H[k+1,j_1,j_2] + theta_z*H[k+1,j_1+1,j_2]) + theta_c*((1-theta_z)*H[k+1,j_1,j_2+1] + theta_z*H[k+1,j_1+1,j_2+1])
                        if sigma_flag == 0:
                            density = _t_pdf(z[j_u], df=2*alpha0+k)
                        else:
                            density = _norm_pdf(z[j_u])
                        dz = (z[j_u+1] - z[j_u-1]) / 2
                        H[k, j_z, j_c] += s * max(0, H_u - c[j_c]/s) * density * dz

            j = G
            while j > 1 and c[j] < H[k, j_z, j]:
                j -= 1
            if j == G:
                j = G-1
            if (H[k, j_z, j+1] - H[k, j_z, j] + c[j] - c[j+1]) == 0:
                h[k, j_z] = c[j]
            else:
                h[k, j_z] = c[j] + (c[j+1] - c[j]) * (c[j] - H[k,j_z,j]) / (H[k,j_z,j+1] - H[k,j_z,j] + c[j] - c[j+1])
    
    return h

def h_index_full(recall, mu_flag, sigma_flag, alpha0, nu0, n, G):
    """
    Solve the full h-index table.
    """
    c, z_grid = build_grids(G)
    if recall == 1:
        h_matrix = h_index_recall1(mu_flag, sigma_flag, alpha0, nu0, n, G, c, z_grid)
    else:
        raise NotImplementedError("recall=0 case not yet optimized in Numba version.")
    return h_matrix, z_grid

def h_index_value(h_matrix, z_grid, k, z_val):
    """
    Retrieve the interpolated h(k, z) value from h_matrix and z_grid.
    """
    G = len(z_grid) - 2
    n = len(h_matrix) - 1
    
    if k > n or k < 0:
        raise ValueError(f"k must be between 0 and {n}")

    j = G
    while j > 0 and z_val < z_grid[j]:
        j -= 1
    if j == G:
        j = G-1

    if (z_grid[j+1] - z_grid[j]) == 0:
        theta = 0
    else:
        theta = (z_val - z_grid[j]) / (z_grid[j+1] - z_grid[j])
    
    h_interp = (1-theta) * h_matrix[k, j] + theta * h_matrix[k, j+1]
    return h_interp 