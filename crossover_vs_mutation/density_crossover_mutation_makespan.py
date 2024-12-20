# %%
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

data = pd.read_csv(
    'C:/Users/User/Desktop/Python/ai_assignment/crossover_vs_mutation/crossover_mutation_medium.txt'
)

crossover_rate = data['CXPB']
mutation_rate = data['MU_INDPB']
makespan = data['MAKESPAN']

# makespan = np.array([math.log(ms) for ms in makespan]) # log makespan

### normal makespan with dropped values
removed = []
for i, ms in enumerate(makespan):
    if (ms > 10000):
        removed.append(i)

for i in removed:
    crossover_rate[i] = np.NaN
    mutation_rate[i] = np.NaN
    makespan[i] = np.NAN

    # makespan[i] = 48

crossover_rate = crossover_rate.dropna()
mutation_rate = mutation_rate.dropna()
makespan = makespan.dropna()

###

print(makespan.shape[0])

# Create a grid for interpolation
x = np.linspace(0.05, 1.05, makespan.shape[0])
y = np.linspace(0, 0.255, makespan.shape[0])
X, Y = np.meshgrid(x, y)

Z = griddata((crossover_rate, mutation_rate), makespan, (X, Y), method='cubic')

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111)

contour_filled = plt.contourf(X,
                              Y,
                              Z,
                              levels=np.linspace(makespan.min(),
                                                 makespan.max(),
                                                 makespan.shape[0]),
                              cmap='viridis_r')

contour_lines = plt.contour(X,
                            Y,
                            Z,
                            levels=np.linspace(makespan.min(), makespan.max(),
                                               15),
                            colors='black',
                            linewidths=0.5)

plt.clabel(contour_lines, inline=True, fontsize=8, fmt="%.1f")

ax.xaxis.set_minor_formatter(plt.FormatStrFormatter('%d'))
# defining custom minor tick locations:
ax.xaxis.set_major_locator(
    plt.FixedLocator([
        1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4,
        0.35, 0.3, 0.25, 0.2, 0.15, 0.1
    ]))
ax.yaxis.set_major_locator(
    plt.FixedLocator([
        0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23,
        0.25
    ]))
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

cbar = plt.colorbar(contour_filled)
cbar.set_label('Makespan', fontsize=12)

plt.scatter(crossover_rate,
            mutation_rate,
            c=makespan,
            cmap='viridis_r',
            edgecolor='k',
            s=25,
            label='Data Points')

plt.xlabel('Crossover Rate', fontsize=12)
plt.ylabel('Mutation Rate', fontsize=12)
# plt.title('Log Makespan Contour Map of Crossover vs Mutation', fontsize=14) # log makespan title
plt.title('Makespan Contour Map of Crossover vs Mutation',
          fontsize=14)  # normal makespan title
plt.legend(loc='upper right')
plt.tight_layout()

plt.show()

# %%
