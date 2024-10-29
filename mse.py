import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Sun’iy ma’lumot yaratish
X = np.random.rand(100, 1) * 10
y = 2 * X + np.random.randn(100, 1) * 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")


epochs = list(range(1, 101))
loss_values = []
for i in epochs:
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    loss = mean_squared_error(y_train, y_train_pred)
    loss_values.append(loss)

plt.plot(epochs, loss_values, color='blue', label='Training Loss (MSE)')
plt.title('Loss Function Progression over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# 4. Tasodifiy gradient descendent usul bilan sinov
model = SGDRegressor(max_iter=1, tol=None, warm_start=True)  
epochs = 100  
loss_values = []
for epoch in range(epochs):
    model.fit(X_train, y_train.ravel()) 
    y_train_pred = model.predict(X_train)
    loss = mean_squared_error(y_train, y_train_pred)
    loss_values.append(loss)

plt.plot(range(epochs), loss_values, color='blue', label='Training Loss (MSE)')
plt.title('Loss Function Progression over Epochs (SGD)')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()
