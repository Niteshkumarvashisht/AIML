import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

print("PyTorch Basics Tutorial")
print("======================")

# 1. Tensors
print("\n1. Basic Tensor Operations:")
# Create tensors
tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])

print("Tensor 1:")
print(tensor1)
print("\nTensor 2:")
print(tensor2)

# Basic operations
print("\nAddition:")
print(tensor1 + tensor2)

print("\nMultiplication:")
print(tensor1 * tensor2)

print("\nMatrix multiplication:")
print(torch.matmul(tensor1, tensor2))

# 2. Tensor Operations
print("\n2. Advanced Tensor Operations:")
# Create random tensor
random_tensor = torch.randn(3, 4)
print("Random tensor:")
print(random_tensor)

print("\nMean:", random_tensor.mean())
print("Sum:", random_tensor.sum())
print("Standard deviation:", random_tensor.std())

# 3. Autograd (Automatic Differentiation)
print("\n3. Autograd Example:")
x = torch.ones(2, 2, requires_grad=True)
print("Input tensor:")
print(x)

y = x + 2
z = y * y * 3
out = z.mean()
print("\nOutput value:")
print(out)

out.backward()
print("\nGradients:")
print(x.grad)

# 4. Neural Network Basics
print("\n4. Simple Neural Network:")
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNN()
print("\nModel structure:")
print(model)

# 5. Loss Functions and Optimization
print("\n5. Loss Functions and Optimization:")
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Generate sample data
X = torch.randn(10, 2)
y = torch.randn(10, 1)

# Training loop
print("\nTraining loop example:")
for epoch in range(5):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f'Epoch [{epoch+1}/5], Loss: {loss.item():.4f}')

# 6. Data Loading
print("\n6. Data Loading Example:")
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, size=100):
        self.x = torch.randn(size, 2)
        self.y = torch.randn(size, 1)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print("\nDataLoader example:")
for i, (inputs, targets) in enumerate(dataloader):
    print(f"Batch {i+1}:")
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    if i == 0:  # Print only first batch
        break

# 7. Model Saving and Loading
print("\n7. Model Saving and Loading:")
# Save model
torch.save(model.state_dict(), 'simple_model.pth')
print("Model saved to 'simple_model.pth'")

# Load model
new_model = SimpleNN()
new_model.load_state_dict(torch.load('simple_model.pth'))
print("Model loaded successfully")

# 8. GPU Support (if available)
print("\n8. GPU Support:")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move model to device
model.to(device)
print(f"Model moved to {device}")

print("\nNote: This is a basic introduction to PyTorch. Real-world applications would require more data, proper model architecture, and validation procedures.")
