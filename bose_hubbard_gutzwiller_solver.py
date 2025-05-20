import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.colors import ListedColormap

def gutzwiller_energy(f_coeffs, J, U, mu, z, n_max):
    """
    Calculate energy for Gutzwiller ansatz.

    Parameters:
    -----------
    f_coeffs : array
        Flattened array of variational coefficients
    J : float
        Hopping parameter
    U : float
        On-site interaction
    mu : float
        Chemical potential
    z : int
        Coordination number (number of nearest neighbors)
    n_max : int
        Maximum occupation number considered

    Returns:
    --------
    energy : float
        Energy per site
    """
    # Reshape coefficients
    f = f_coeffs.reshape(-1)

    # Calculate order parameter (mean-field) with numerical stability
    phi = 0
    for n in range(1, n_max + 1):
        # Use complex numbers to avoid overflow
        if n > 0 and n <= len(f) and n-1 < len(f):
            term = np.sqrt(float(n)) * np.conj(f[n-1]) * f[n]
            if not np.isnan(term) and not np.isinf(term):
                phi += term

    # Calculate on-site energy
    onsite_energy = 0
    for n in range(n_max + 1):
        if n < len(f):
            term = (U/2 * n * (n-1) - mu * n) * np.abs(f[n])**2
            if not np.isnan(term) and not np.isinf(term):
                onsite_energy += term

    # Calculate hopping energy
    hopping_energy = -J * z * np.abs(phi)**2

    # Total energy
    total_energy = onsite_energy + hopping_energy

    return np.real(total_energy)

def normalize_constraint(f_coeffs, n_max):
    """Normalization constraint for variational coefficients."""
    return np.sum(np.abs(f_coeffs)**2) - 1.0

def optimize_gutzwiller(J, U, mu, z, n_max):
    """
    Optimize Gutzwiller variational parameters for given system parameters.

    Parameters:
    -----------
    J : float
        Hopping parameter
    U : float
        On-site interaction
    mu : float
        Chemical potential
    z : int
        Coordination number
    n_max : int
        Maximum occupation number considered

    Returns:
    --------
    f_opt : array
        Optimized variational coefficients
    energy : float
        Ground state energy per site
    """
    # Initial guess - uniform distribution
    f_init = np.ones(n_max + 1) / np.sqrt(n_max + 1)

    # Define constraint for normalization
    constraint = {'type': 'eq', 'fun': lambda f: normalize_constraint(f, n_max)}

    # Use bounded optimization to improve stability
    bounds = [(0, 1) for _ in range(n_max + 1)]

    # Minimize energy
    try:
        result = minimize(
            lambda f: gutzwiller_energy(f, J, U, mu, z, n_max),
            f_init,
            constraints=[constraint],
            bounds=bounds,
            method='SLSQP',
            options={'maxiter': 1000, 'ftol': 1e-8}
        )

        f_opt = result.x
        energy = result.fun

        # Ensure normalization
        f_opt = f_opt / np.sqrt(np.sum(np.abs(f_opt)**2))

    except Exception as e:
        print(f"Optimization failed for J={J}, U={U}, mu={mu}. Using initial guess.")
        f_opt = f_init
        energy = gutzwiller_energy(f_init, J, U, mu, z, n_max)

    return f_opt, energy

def calculate_order_parameter(f_opt, n_max):
    """Calculate the superfluid order parameter."""
    phi = 0
    for n in range(1, n_max + 1):
        if n-1 < len(f_opt) and n < len(f_opt):
            term = np.sqrt(float(n)) * np.conj(f_opt[n-1]) * f_opt[n]
            if not np.isnan(term) and not np.isinf(term):
                phi += term
    return np.abs(phi)

def calculate_number_distribution(f_opt):
    """Calculate the number distribution from optimized coefficients."""
    # Make sure we're using correctly shaped array
    f_opt = f_opt.reshape(-1)
    # Handle potential numerical issues
    prob_dist = np.abs(f_opt)**2
    # Normalize if needed due to numerical errors
    if np.sum(prob_dist) > 0:
        prob_dist = prob_dist / np.sum(prob_dist)
    return prob_dist

def calculate_phase_diagram():
    """Calculate the phase diagram in the J/U vs mu/U plane."""
    # Parameters
    n_max = 5  # Maximum occupation number
    z = 6      # Coordination number (e.g., cubic lattice)

    # Range of parameters
    J_U_range = np.linspace(0.01, 0.2, 20)  # J/U ratio
    mu_U_range = np.linspace(0, 3, 40)      # mu/U ratio

    # Initialize order parameter array
    order_params = np.zeros((len(mu_U_range), len(J_U_range)))
    number_density = np.zeros_like(order_params)

    # Calculate for each point in parameter space
    for i, mu_U in enumerate(mu_U_range):
        for j, J_U in enumerate(J_U_range):
            # Set parameters
            U = 1.0  # Set U as the energy scale
            J = J_U * U
            mu = mu_U * U

            # Optimize
            f_opt, _ = optimize_gutzwiller(J, U, mu, z, n_max)

            # Calculate order parameter
            order_params[i, j] = calculate_order_parameter(f_opt, n_max)

            # Calculate average number
            n_dist = calculate_number_distribution(f_opt)
            number_density[i, j] = sum(n * p for n, p in enumerate(n_dist))

    return J_U_range, mu_U_range, order_params, number_density

def plot_phase_diagram():
    """Plot the phase diagram for the Bose-Hubbard model."""
    J_U_range, mu_U_range, order_params, number_density = calculate_phase_diagram()

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot order parameter
    im1 = ax1.imshow(order_params, extent=[min(J_U_range), max(J_U_range),
                                         min(mu_U_range), max(mu_U_range)],
                   aspect='auto', origin='lower', cmap='viridis')
    ax1.set_xlabel('J/U')
    ax1.set_ylabel('μ/U')
    ax1.set_title('Superfluid Order Parameter')
    fig.colorbar(im1, ax=ax1, label='|Φ|')

    # Plot number density
    im2 = ax2.imshow(number_density, extent=[min(J_U_range), max(J_U_range),
                                           min(mu_U_range), max(mu_U_range)],
                   aspect='auto', origin='lower', cmap='plasma')
    ax2.set_xlabel('J/U')
    ax2.set_ylabel('μ/U')
    ax2.set_title('Average Occupation Number')
    fig.colorbar(im2, ax=ax2, label='⟨n⟩')

    # Add estimated phase boundary
    # This is an approximate critical line based on mean-field theory
    J_U_crit = np.linspace(0.01, 0.2, 100)

    # Calculate boundary for first lobe (n=1)
    mu_U_crit_1 = 0.5 - 2*J_U_crit*z + np.sqrt(0.25 + 4*J_U_crit*z)
    mu_U_crit_2 = 1.5 - 2*J_U_crit*z + np.sqrt(0.25 + 4*J_U_crit*z)
    mu_U_crit_3 = 2.5 - 2*J_U_crit*z + np.sqrt(0.25 + 4*J_U_crit*z)

    # Plot critical boundaries
    ax1.plot(J_U_crit, mu_U_crit_1, 'r--', label='n=1 lobe')
    ax1.plot(J_U_crit, mu_U_crit_2, 'r--', label='n=2 lobe')
    ax1.plot(J_U_crit, mu_U_crit_3, 'r--', label='n=3 lobe')

    ax2.plot(J_U_crit, mu_U_crit_1, 'r--')
    ax2.plot(J_U_crit, mu_U_crit_2, 'r--')
    ax2.plot(J_U_crit, mu_U_crit_3, 'r--')

    # Add annotation for phases
    ax1.text(0.15, 0.5, 'Superfluid', color='white', fontsize=12)
    ax1.text(0.03, 1.0, 'Mott n=1', color='white', fontsize=10)
    ax1.text(0.03, 2.0, 'Mott n=2', color='white', fontsize=10)

    plt.tight_layout()
    plt.show()

def plot_order_parameter_vs_JU():
    """Plot the order parameter as a function of J/U for fixed chemical potential."""
    # Parameters
    n_max = 5
    z = 6
    U = 1.0
    mu_U = 0.5  # Fixed chemical potential (middle of first Mott lobe)
    mu = mu_U * U

    # Range of J/U
    J_U_range = np.linspace(0.01, 0.3, 50)

    # Calculate order parameter and number variance
    order_params = []
    number_vars = []

    for J_U in J_U_range:
        J = J_U * U
        f_opt, _ = optimize_gutzwiller(J, U, mu, z, n_max)
        order_params.append(calculate_order_parameter(f_opt, n_max))

        # Calculate number variance
        n_dist = calculate_number_distribution(f_opt)
        avg_n = sum(n * p for n, p in enumerate(n_dist))
        var_n = sum((n - avg_n)**2 * p for n, p in enumerate(n_dist))
        number_vars.append(var_n)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax1.plot(J_U_range, order_params, 'b-', linewidth=2)
    ax1.set_ylabel('Superfluid Order Parameter |Φ|')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Phase Transition at μ/U = {mu_U}')

    ax2.plot(J_U_range, number_vars, 'r-', linewidth=2)
    ax2.set_xlabel('J/U')
    ax2.set_ylabel('Number Variance ⟨(n - ⟨n⟩)²⟩')
    ax2.grid(True, alpha=0.3)

    # Add vertical line at critical point (approximate from mean-field theory)
    J_U_c = 1/(z * (1 + np.sqrt(1 - mu_U) + mu_U))
    ax1.axvline(x=J_U_c, color='k', linestyle='--', alpha=0.7, label=f'Critical point J/U ≈ {J_U_c:.3f}')
    ax2.axvline(x=J_U_c, color='k', linestyle='--', alpha=0.7)

    ax1.legend()
    plt.tight_layout()
    plt.show()

def main():
    print("Calculating Bose-Hubbard phase diagram using Gutzwiller approximation...")
    plot_phase_diagram()
    plot_order_parameter_vs_JU()
    print("Calculation complete!")

if __name__ == "__main__":
    main()
