import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import DigitClassifier

def test():
    # Configuration
    BATCH_SIZE = 1000
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Prepare Data (Only the Test set this time)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Load the Model
    model = DigitClassifier().to(DEVICE)
    try:
        model.load_state_dict(torch.load("../models/mnist_cnn.pt"))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: Train the model first using 'python src/train.py'!")
        return

    model.eval() # Important: Sets the model to evaluation mode (turns off Dropout)
    
    test_loss = 0
    correct = 0

    # 3. Evaluation Loop (No gradient calculation needed)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            
            # Sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            
            # Get the index of the max log-probability (the predicted digit)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nTest set results:\nAverage loss: {test_loss:.4f}')
    print(f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

if __name__ == '__main__':
    test()