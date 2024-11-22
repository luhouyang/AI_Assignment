# %%
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

data = pd.read_csv('C:/Users/User/Desktop/Python/ai_assignment/crossover_mutation_medium.txt')

crossover_rate = data['CXPB']
mutation_rate = data['MU_INDPB']
makespan = data['MAKESPAN']

makespan = np.array([math.log(ms) for ms in makespan])

# Create a grid for interpolation
x = np.linspace(0.05, 1.05, 247)
y = np.linspace(0, 0.255, 247)
X, Y = np.meshgrid(x, y)

Z = griddata((crossover_rate, mutation_rate), makespan, (X, Y), method='cubic')

plt.figure(figsize=(8, 6))

contour_filled = plt.contourf(X, Y, Z, levels=np.linspace(makespan.min(), makespan.max(), 247), cmap='viridis_r')

contour_lines = plt.contour(X, Y, Z, levels=np.linspace(makespan.min(), makespan.max(), 10), colors='black', linewidths=0.5)

plt.clabel(contour_lines, inline=True, fontsize=8, fmt="%.1f")

cbar = plt.colorbar(contour_filled)
cbar.set_label('Makespan', fontsize=12)

plt.scatter(crossover_rate, mutation_rate, c=makespan, cmap='viridis_r', edgecolor='k', s=25, label='Data Points')

plt.xlabel('Crossover Rate', fontsize=12)
plt.ylabel('Mutation Rate', fontsize=12)
plt.title('Makespan Gradient Density of Crossover vs Mutation', fontsize=14)
plt.legend(loc='upper right')
plt.tight_layout()

plt.show()


# %%
