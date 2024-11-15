# %%
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.preprocessing import StandardScaler

dataframe = pd.read_csv('C:/Users/User/Desktop/Python/ai_assignment/houyang/lab/weather_forecast_data_csv.csv')

display(dataframe.head())

X = dataframe.iloc[:, :-1]
y = dataframe.iloc[:, -1]

print(X)
print(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = LogisticRegression(random_state=0).fit(X_scaled, y)

print(clf.score(X_scaled, y))

# plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# data points
ax.scatter(dataframe['Temperature'], dataframe['Humidity'], dataframe['Cloud_Cover'], c=y, cmap='coolwarm', marker='o', label="Data Points")

# grid to plot the decision boundary
x_vals = np.linspace(dataframe['Temperature'].min(), dataframe['Temperature'].max(), 10)
y_vals = np.linspace(dataframe['Humidity'].min(), dataframe['Humidity'].max(), 10)
x_vals, y_vals = np.meshgrid(x_vals, y_vals)

# predicted values on grid
z_vals = (-clf.intercept_[0] - clf.coef_[0][0] * scaler.transform(np.c_[x_vals.ravel(), y_vals.ravel(), np.zeros(x_vals.size)])[:,0] 
          - clf.coef_[0][1] * scaler.transform(np.c_[x_vals.ravel(), y_vals.ravel(), np.zeros(x_vals.size)])[:,1]) / clf.coef_[0][2]
z_vals = z_vals.reshape(x_vals.shape)

# plot decision boundary
ax.plot_surface(x_vals, y_vals, z_vals, color='yellow', alpha=0.5, rstride=100, cstride=100)

ax.set_xlabel('Temperature')
ax.set_ylabel('Humidity')
ax.set_zlabel('Cloud Cover')
ax.set_title('3D Logistic Regression Decision Boundary')

plt.show()
# %%
