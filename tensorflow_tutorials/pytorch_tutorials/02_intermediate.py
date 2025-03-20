import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

print("PyTorch Intermediate Tutorial")
print("============================")

# 1. CNN Implementation
print("\n1. Convolutional Neural Network:")
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

cnn_model = ConvNet()
print("CNN Model Structure:")
print(cnn_model)

# 2. RNN Implementation
print("\n2. Recurrent Neural Network:")
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros(1, x.size(0), self.hidden_size)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

rnn_model = RNN(10, 20, 2)
print("\nRNN Model Structure:")
print(rnn_model)

# 3. Custom Dataset with Transforms
print("\n3. Custom Dataset with Transforms:")
class TransformDataset(Dataset):
    def __init__(self, size=100, transform=None):
        self.data = torch.randn(size, 1, 28, 28)
        self.labels = torch.randint(0, 10, (size,))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

# Custom transform
class RandomNoise:
    def __call__(self, x):
        return x + 0.1 * torch.randn_like(x)

# 4. Custom Loss Function
print("\n4. Custom Loss Function:")
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, pred, target, weight=1.0):
        return weight * torch.mean((pred - target) ** 2)

# 5. Learning Rate Scheduler
print("\n5. Learning Rate Scheduling:")
model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

print("\nInitial LR:", optimizer.param_groups[0]['lr'])
for epoch in range(2):
    scheduler.step()
    print(f"Epoch {epoch+1} LR:", optimizer.param_groups[0]['lr'])

# 6. Model with Multiple Inputs
print("\n6. Multiple Input Model:")
class MultiInputNet(nn.Module):
    def __init__(self):
        super(MultiInputNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(5, 20)
        self.fc3 = nn.Linear(40, 2)

    def forward(self, x1, x2):
        out1 = F.relu(self.fc1(x1))
        out2 = F.relu(self.fc2(x2))
        combined = torch.cat((out1, out2), dim=1)
        return self.fc3(combined)

multi_model = MultiInputNet()
print("\nMulti-Input Model Structure:")
print(multi_model)

# 7. Custom Activation Function
print("\n7. Custom Activation Function:")
class SwishActivation(nn.Module):
    def __init__(self, beta=1.0):
        super(SwishActivation, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

# Test custom activation
swish = SwishActivation()
test_input = torch.randn(5)
print("\nSwish activation test:")
print("Input:", test_input)
print("Output:", swish(test_input))

# 8. Checkpoint Management
print("\n8. Checkpoint Management:")
def save_checkpoint(model, optimizer, epoch, loss, path='checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Checkpoint saved: {path}")

def load_checkpoint(model, optimizer, path='checkpoint.pth'):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

# Example usage
model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=0.01)
save_checkpoint(model, optimizer, 1, 0.5)

print("\nNote: This intermediate tutorial covers more complex PyTorch concepts. Real applications would require proper data and validation procedures.")
