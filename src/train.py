import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import DigitClassifier
import os

def train():
    # We use a batch size of 64 images (m = 64).
    BATCH_SIZE = 64
    EPOCHS = 5
    LEARNING_RATE = 0.001
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {DEVICE}")

    # The Input Data is converted to Tensors.
    # Normalization applies the formula: z = (x - mean) / std.
    # Here, mean=0.1307 and std=0.3081. This scales input node values to approx [-1, 1].
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print("Downloading MNIST data...")
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = DigitClassifier().to(DEVICE)
    # Adam optimizer updates the weights of the nodes using adaptive learning rates.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    model.train() 
    
    for epoch in range(1, EPOCHS + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            # Gradients are accumulated by default, so we zero them: grad = 0.
            optimizer.zero_grad()
            
            # Forward pass: Pass inputs through the nodes to get predictions.
            output = model(data)
            
            # Calculate Negative Log Likelihood loss at the Output Layer.
            # With reduction='mean', this computes: Loss = (1/m) * Sum( -log(y_pred[target]) )
            loss = F.nll_loss(output, target)
            
            # Backward pass: Compute gradient of Loss w.r.t weights (Backpropagation).
            loss.backward()
            
            # Optimization step: Update weights (W = W - learning_rate * gradient).
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch} | Processed: {batch_idx * len(data)}/{len(train_loader.dataset)} images | Loss: {loss.item():.6f}')

    if not os.path.exists('../models'):
        os.makedirs('../models')
    torch.save(model.state_dict(), "../models/mnist_cnn.pt")
    print("Finished! Model saved to ../models/mnist_cnn.pt")

if __name__ == '__main__':
    train()