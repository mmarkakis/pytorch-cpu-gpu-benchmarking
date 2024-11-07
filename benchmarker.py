import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# Define a simple neural network model
class SimpleNet(nn.Module):
    def __init__(self, input_size=784):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Function to train the model and measure time per epoch
def train_model(device, input_size, num_points, num_epochs):
    model = SimpleNet(input_size=input_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Generate random data for benchmarking
    inputs = torch.randn(num_points, input_size).to(device)
    labels = torch.randint(0, 10, (num_points,)).to(device)
    
    # Training loop (single epoch)
    model.train()
    optimizer.zero_grad()
    start_time = time.time()

    for _ in range(num_epochs):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    end_time = time.time()
    
    return (end_time - start_time)/num_epochs  

# Parameters for benchmarking
num_points_list = [1000, 5000, 10000, 50000, 100000]  # Different numbers of points
dimensionality_list = [128, 256, 512, 1024, 2048]  # Different input dimensionalities
num_epochs_list = [5,10,20]  # Single epoch to get time per epoch

num_points_fixpoint = num_points_list[len(num_points_list)//2]
dimensionality_fixpoint = dimensionality_list[len(dimensionality_list)//2]
num_epochs_fixpoint = num_epochs_list[len(num_epochs_list)//2]


# Results containers
cpu_times_points = []
gpu_times_points = []
cpu_times_dims = []
gpu_times_dims = []
cpu_times_epochs = []
gpu_times_epochs = []

# Benchmark varying input dimensionality (with fixed number of points and epochs).
print("Starting experiment 1")
for input_dim in dimensionality_list:
    cpu_time = train_model(torch.device("cpu"), input_dim, num_points_fixpoint, num_epochs_fixpoint)
    cpu_times_dims.append(cpu_time)

    if torch.cuda.is_available():
        gpu_time = train_model(torch.device("cuda"), input_dim, num_points_fixpoint, num_epochs_fixpoint)
    else:
        gpu_time = None  # No GPU available

    gpu_times_dims.append(gpu_time)


# Benchmark varying number of points (with fixed dimensionality and epochs).
print("Starting experiment 2")
for num_points in num_points_list:
    cpu_time = train_model(torch.device("cpu"), dimensionality_fixpoint, num_points, num_epochs_fixpoint)
    cpu_times_points.append(cpu_time)

    if torch.cuda.is_available():
        gpu_time = train_model(torch.device("cuda"), dimensionality_fixpoint, num_points, num_epochs_fixpoint)
    else:
        gpu_time = None  # No GPU available

    gpu_times_points.append(gpu_time)


# Benchmark varying number of epochs (with fixed dimensionality and number of points).
print("Starting experiment 3")
for num_epochs in num_epochs_list:
    cpu_time = train_model(torch.device("cpu"), dimensionality_fixpoint, num_points_fixpoint, num_epochs)
    cpu_times_epochs.append(cpu_time)

    if torch.cuda.is_available():
        gpu_time = train_model(torch.device("cuda"), dimensionality_fixpoint, num_points_fixpoint, num_epochs)
    else:
        gpu_time = None  # No GPU available

    gpu_times_epochs.append(gpu_time)


# Save results

# Plotting
plt.figure(figsize=(12, 6))
# Plot 1: Varying input dimensionality
plt.subplot(1, 3, 1)
plt.plot(dimensionality_list, cpu_times_dims, label="CPU", marker='o')
if None not in gpu_times_dims:
    plt.plot(dimensionality_list, gpu_times_dims, label="GPU", marker='o')
plt.xlabel("Input Dimensionality")
plt.ylabel("Time per Epoch (s)")
plt.title("Time per Epoch vs Input Dimensionality")
plt.legend()

# Plot 2: Varying number of points
plt.subplot(1, 3, 2)
plt.plot(num_points_list, cpu_times_points, label="CPU", marker='o')
if None not in gpu_times_points:
    plt.plot(num_points_list, gpu_times_points, label="GPU", marker='o')
plt.xlabel("Number of Points")
plt.ylabel("Time per Epoch (s)")
plt.title("Time per Epoch vs Number of Points")
plt.legend()

# Plot 3: Varying number of epochs
plt.subplot(1, 3, 3)
plt.plot(num_epochs_list, cpu_times_epochs, label="CPU", marker='o')
if None not in gpu_times_epochs:
    plt.plot(num_epochs_list, gpu_times_epochs, label="GPU", marker='o')
plt.xlabel("Epochs")
plt.ylabel("Time per Epoch (s)")
plt.title("Time per Epoch vs Number of Epochs")
plt.legend()


plt.tight_layout()
plt.show()



