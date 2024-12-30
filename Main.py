# Full Implementation for MNIST Classification Assignment

# 1. Import Required Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

# 2. Data Preparation
# Define transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and preprocess MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Split training data into train and validation
train_data, val_data = train_test_split(train_dataset, test_size=0.25, random_state=42)

# DataLoaders
batch_sizes = 64
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_sizes, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_sizes, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_sizes, shuffle=False)

# 3. Define Softmax Regression Model
class SoftmaxRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SoftmaxRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)

# Initialize model, optimizer, and loss function
input_size = 28 * 28
num_classes = 10
learning_rate = 0.01
epochs = 10

model = SoftmaxRegression(input_size, num_classes)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 4. Training and Validation Loop
def train_and_validate(model, train_loader, val_loader, epochs):
    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images = images.view(-1, 28 * 28)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))

        # Validation phase
        model.eval()
        val_loss, correct = 0, 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.view(-1, 28 * 28)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(correct / total)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}")

    return train_losses, val_losses, val_accuracies

train_losses, val_losses, val_accuracies = train_and_validate(model, train_loader, val_loader, epochs)

# 5. Visualizations
# Plot losses and accuracies
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Losses')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.show()

# 6. Test Model and Evaluate Performance
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(-1, 28 * 28)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())

# Confusion Matrix and Classification Report
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_true, y_pred)
print("Classification Report:\n", class_report)

# BONUS: Feedforward Neural Network
class FeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize Feedforward Neural Network
hidden_sizes = [128, 64]
ffnn_model = FeedforwardNN(input_size, hidden_sizes, num_classes)
optimizer = optim.SGD(ffnn_model.parameters(), lr=learning_rate)

# Train and Validate Feedforward Neural Network
train_losses, val_losses, val_accuracies = train_and_validate(ffnn_model, train_loader, val_loader, epochs)

# Compare Models (Optional Step)
print("Softmax Regression vs. Feedforward NN Accuracy:")
print(f"Softmax Regression Test Accuracy: {val_accuracies[-1]:.4f}")

