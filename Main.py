import time
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 0. Set Variables
random_data_split_seed = 123456789
batch_sizes = 64
epochs = 10
learning_rate = 0.01
input_size = 28 * 28 # the image dimentions are 28x28
num_classes = 10 # the number of digits (0-9)

# 2. Data Preparation

# Define transformation variable to apply to images
transform = transforms.Compose([
    transforms.ToTensor(), # Convert image to tensor and set range between 0 and 1
    transforms.Normalize((0.5,), (0.5,)) # Normalize image with mean and standard deviation into range -1 to 1
])

# Download and preprocess MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True) # 60,000 examples
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True) # 10,000 examples

# Combine train and test datasets for splitting
full_dataset = train_dataset + test_dataset
# Split data into 80% training/validation and 20% testing
train_val_data, test_data = train_test_split(full_dataset, test_size=0.2, random_state=random_data_split_seed)
# Split the 80% training/validation data into 60% training and 20% validation
train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=random_data_split_seed)

# DataLoaders
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

# 5. Train and Validate Softmax Regression Model
print("\nSoftmax Regression Model Training:")
start_time = time.time()
train_losses, val_losses, val_accuracies = train_and_validate(model, train_loader, val_loader, epochs)
print("\nTraining Complete!")
training_time = time.time() - start_time

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
report = classification_report(y_true, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report_df)
print(f"\nAccuracy: {report['accuracy']:.4f}")
print(f"Training Time: {training_time:.2f} seconds")

def plot_metrics(train_losses, val_losses, val_accuracies, figure_name):
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Plot losses
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Validation Loss')
    axes[0].set_title('Losses')
    axes[0].legend()

    # Plot accuracies
    axes[1].plot(val_accuracies, label='Validation Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()

    # Adjust layout
    plt.tight_layout(pad=1)
    # Adjust font size
    for ax in axes.flat:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(8)

    # Save the figure
    plt.savefig(figure_name)
    plt.show()
    
plot_metrics(train_losses, val_losses, val_accuracies, f'Softmax Regression Metrics.{learning_rate}.{batch_sizes}.png')

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
print("\nFeedforward NN Training:")
start_time = time.time()
train_losses, val_losses, val_accuracies = train_and_validate(ffnn_model, train_loader, val_loader, epochs)
print("\nTraining Complete!")
training_time = time.time() - start_time

# Test Feedforward Neural Network and Evaluate Performance
ffnn_model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(-1, 28 * 28)
        outputs = ffnn_model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())

# Confusion Matrix and Classification Report
conf_matrix = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report_df)
print(f"\nAccuracy: {report['accuracy']:.4f}")
print(f"Training Time: {training_time:.2f} seconds")

plot_metrics(train_losses, val_losses, val_accuracies, f'Feedforward NN Metrics.{learning_rate}.{batch_sizes}.png')

