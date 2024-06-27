import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt


# Define constants
a = 0.2957  # Example constant a
b = 6.937  # Example constant b
c = 1.676e3  # Example constant c
H =  22.380e3  # Example constant H
T_B = 779.9  # Example constant T_B
k = -1.231e8  # Example constant k
E_0 = 70e9  # Example modulus constant E0
lambda_ = 0.7012  # Example relaxation time constant lambda


# Define the function to calculate scaled time xi
def calculate_xi(t, a, b, c, H, T_B):
    term1 = (a * t + b) ** (1 - H / c)
    term2 = b ** (1 - H / c)
    constant_factor = np.exp(H / T_B) / (a * (1 - H / c))
    return constant_factor * (term1 - term2)

# Define the strain rate function dε(t)/dt
def strain_rate(t, a, b, H, c, k):
    if t < 0:
        return 0
    else:
        return k * (a * t + b) ** (-H / c)

# Define the relaxation modulus function E(ξ - ξ')
def relaxation_modulus(xi_diff, E_0, lambda_):
    return E_0 * np.exp(-xi_diff / lambda_)

# Define the convolution integral for stress σ(t)
def stress_integral(t, a, b, c, H, T_B, k, E_0, lambda_):
    def integrand(t_prime):
        xi = calculate_xi(t, a, b, c, H, T_B)
        xi_prime = calculate_xi(t_prime, a, b, c, H, T_B)
        xi_diff = xi - xi_prime
        return relaxation_modulus(xi_diff, E_0, lambda_) * strain_rate(t_prime, a, b, H, c, k)
    
    sigma, _ = integrate.quad(integrand, 0, t)
    return sigma

# Example usage
t_values = np.linspace(start=0.0, stop=50.0, num=500)
xi_values = [calculate_xi(t, a, b, c, H, T_B) for t in t_values]
epsilon_values = [strain_rate(t, a, b, H, c, k) for t in t_values]
sigma_values = [stress_integral(t, a, b, c, H, T_B, k, E_0, lambda_) for t in t_values]

# Print the stress values
for t, sigma in zip(t_values, sigma_values):
    print(f"Time t = {t:.2f}, Stress σ(t) = {sigma:.6f}")
    
# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(t_values, xi_values, label='analytical soln', color='b')

# Add title and labels
plt.title('Plot of variables vs Time')
plt.xlabel('Time (t)')
plt.ylabel('variable')

# Add a legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()


# Integrate stress for each time in t
#xi_ = np.array([quad(lambda t: phi(t, constants), 0, t_i)[0] for t_i in t])

'''def xi(t:float):
    """Integrate the scaled time by quadrature integration,
    Eq. 3 """
    T_val= T(t,constants)
    phi_val= phi(T_val)
    return phi_val

xis, _ = quad(xi, 0, t)'''


np.savetxt("data3_stress.csv",sigma_analytical(t,constants),  
              delimiter = ",")
#results = np.array([quad(lambda t: stress(xi, t, constants), 0, t_i)[0] for t_i, xi in zip(t, xi_)])
#results = np.array([quad(stress, 0, t_i, args=(constants, xi_i))[0] for t_i, xi_i in zip(t, xi_)])

    
#sigma_t = np.array([quad(lambda t_prime: E(t_prime, constants)*(constants["k"] * ((constants["a"] * t_prime) + constants["b"]) ** (-constants["H"]*constants["c"])), 0, t_i)[0] for t_i in t])

# try to compare T, phi vs time from simulation with the analytical soln here
#epsilon_ = np.array([quad(lambda t: epsilon(t, constants), 0, t_i)[0] for t_i in t])
#epsilon_ = [quad(lambda t_: AnalyticalSoln.epsilon(t_, constants), 0, t_i)[0] for t_i in t_]