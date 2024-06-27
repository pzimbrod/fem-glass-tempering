import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt


# Define constants
a = 1.0  # Example constant a
b = 1.0  # Example constant b
c = 2.0  # Example constant c
H = 1.0  # Example constant H
T_B = 1.0  # Example constant T_B

# Define the shift function phi(T)
def phi(T):
    return np.exp(H * (1 / T_B - np.log(T) / c))

# Define the temperature function T(t)
def T(t):
    return c / np.log(a * t + b)

# Define the integrand for scaled time xi
def integrand(t_prime, a, b, c, H, T_B):
    temperature = c / np.log(a * t_prime + b)
    return np.exp(H * (1 / T_B - np.log(a * t_prime + b) / c))

# Define the function to calculate scaled time xi
def calculate_xi(t, a, b, c, H, T_B):
    if t < 0:
        return 0
    else:
        xi, _ = integrate.quad(integrand, 0, t, args=(a, b, c, H, T_B))
        return xi

# Example usage
t_values = np.linspace(0, 10, 100)  # Time values from 0 to 10

xi_values = [calculate_xi(t, a, b, c, H, T_B) for t in t_values]

# Print the scaled time values
for t, xi in zip(t_values, xi_values):
    print(f"Time t = {t:.2f}, Scaled time ξ(t) = {xi:.6f}")

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