import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from NeuralNetwork import NeuralNetwork

column_names = ['x_dist', 'y_dist', 'x_vel', 'y_vel']
df = pd.read_csv('data.csv', names=column_names)

# Drop duplicate rows
df = df.drop_duplicates()

# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

print(f'Mins: {scaler.data_min_.tolist()}')
print(f'Maxs: {scaler.data_max_.tolist()}\n')

df = pd.DataFrame(scaled_data, columns=column_names)

# Divide data into input and target
x = df[['x_dist','y_dist']].values.tolist()
y = df[['x_vel','y_vel']].values.tolist()

# Configure network
nn = NeuralNetwork(2, [4,4], 2)
nn.lam = 0.7
nn.learning_rate = 0.6
nn.momentum_rate = 0.1

# Train network
train_error, val_error = nn.train(x, y, val_split=0.3, epochs=200, min_delta=0.01, patience=3, restore_best_weights=True)

# Plot training and validation errors
plt.plot(train_error, label = "Training Error")
plt.plot(val_error, label = "Validation Error")
plt.title("Error")
plt.legend()
plt.grid()
plt.show()

# Extract weights
print(nn.weights)