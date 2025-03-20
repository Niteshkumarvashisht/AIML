import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# PyTorch Basics - Machine Learning Library
print("PyTorch Basics Examples:")

# 1. Creating Tensors
print("\n1. Creating Tensors:")
# Create a tensor from a list
tensor1 = torch.tensor([[1, 2], [3, 4]])
print("Basic tensor:")
print(tensor1)

# Create random tensor
random_tensor = torch.rand(2, 3)
print("\nRandom tensor:")
print(random_tensor)

# Create tensor of zeros
zeros = torch.zeros(2, 2)
print("\nZeros tensor:")
print(zeros)

# 2. Basic Operations
print("\n2. Basic Operations:")
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

print("Addition:")
print(a + b)

print("\nMultiplication:")
print(a * b)

print("\nMatrix multiplication:")
print(torch.matmul(a, b))

# 3. Simple Neural Network
print("\n3. Simple Neural Network Example:")
# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(4, 2)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.softmax(x)
        return x

# Create model instance
model = SimpleNN()
print("\nModel structure:")
print(model)

# 4. Working with Gradients
print("\n4. Working with Gradients:")
x = torch.tensor([[1., 2.]], requires_grad=True)
y = x ** 2
z = y.sum()

print("Forward pass:")
print(f"x: {x}")
print(f"y = x²: {y}")
print(f"z = sum(y): {z}")

# Compute gradients
z.backward()
print("\nGradients:")
print(f"dz/dx: {x.grad}")

# 5. Simple Training Example
print("\n5. Simple Training Example:")
# Generate random data
X = torch.randn(100, 2)
y = torch.randint(0, 2, (100,))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop (just 5 epochs as example)
print("\nTraining the model:")
for epoch in range(5):
    # Forward pass
    outputs = model(X.float())
    loss = criterion(outputs, y)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f'Epoch [{epoch+1}/5], Loss: {loss.item():.4f}')

print("\nNote: This is a basic example. Real-world applications would require more data, tuning, and validation.")
