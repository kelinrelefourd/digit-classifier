import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        
        # --- HIDDEN LAYER 1 (Convolutional) ---
        # Receives data from the Input Layer (28x28 grayscale images, so 1 channel).
        # We use 32 filters (nodes) to detect initial patterns like edges.
        # Mathematically, each filter performs a dot product across the input.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        
        # --- HIDDEN LAYER 2 (Convolutional) ---
        # Takes the 32 feature maps from the previous layer and expands to 64 nodes.
        # This allows the network to identify complex shapes (corners, arcs).
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Dropout randomly zeros out nodes during training (Bernoulli dist. p=0.25).
        # This prevents specific nodes from becoming too dominant (overfitting).
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # --- HIDDEN LAYER 3 (Fully Connected) ---
        # First, we flatten the 2D tensors into a 1D vector of nodes.
        # The input size 3136 comes from: 64 channels * 7 * 7 pixels.
        # This layer has 128 nodes.
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        
        # --- OUTPUT LAYER ---
        # This layer has 10 nodes, representing the digits 0-9.
        # It performs the linear transformation: z = xW^T + b
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Pass input through first hidden layer.
        # Apply ReLU activation function: f(x) = max(0, x).
        # Apply Max Pooling to reduce spatial dimensions (28x28 -> 14x14).
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Pass through second hidden layer.
        # Max Pooling reduces dimensions again (14x14 -> 7x7).
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Apply dropout to the nodes.
        x = self.dropout1(x)
        
        # Flatten the data for the fully connected nodes.
        # Reshapes tensor from (Batch, 64, 7, 7) to (Batch, 3136).
        x = torch.flatten(x, 1)
        
        # Pass through the third hidden layer (128 nodes) with ReLU.
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Pass through the Output Layer (10 nodes).
        x = self.fc2(x)
        
        # Apply Log Softmax to output nodes to get probabilities.
        # Formula: log( exp(x_i) / sum(exp(x_j)) )
        output = F.log_softmax(x, dim=1)
        return output