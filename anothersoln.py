import numpy as np
import matplotlib.pyplot as plt

# Constants
a = 0.2957  # 1/s
b = 6.937
c = 1.676e3  # K
H = 22.380e3  # K
TB = 779.9  # K
k = -1.231e8  # 1/s
E0 = 70e9  # Pa
λ = 0.7012  # s

# Time array
t = np.linspace(0.0, 50.0, 1000)

# Constants A1 to A5
A1 = (-1 / (λ * E0 * k)) * np.exp((H * (H - 2 * c)) / (TB * (H - c)))
A2 = c / (a * λ * TB * (H - c))
A3 = -H / c
A4 = -np.exp(-H * c / (TB * (H - c)))
A5 = A2 * np.exp(H / TB)

# Analytical solution for stress σ(t)
σ_t = A1 * (
    np.exp(A5 * b * TB * (a * t + b)**A3 - A5 * TB * b**(A3 + 1) + A5 * t * a * TB * (a * t + b)**A3 - A2 * H * a * λ) + A4
)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(t, σ_t, label='Stress $\sigma(t)$')
plt.xlabel('Time $t$ (s)')
plt.ylabel('Stress $\sigma(t)$ (Pa)')
plt.title('Stress vs. Time')
plt.legend()
plt.grid(True)
plt.show()
