import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, num_layers, layers_size, output_size):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        
        self.linears = nn.ModuleList([nn.Linear(input_size, layers_size)])
        self.linears.extend([nn.Linear(layers_size, layers_size) for i in range(1, self.num_layers-1)])
        self.linears.append(nn.Linear(layers_size, output_size))
    
    def forward(self, x):
        for i in range(len(self.linears) - 1):
            x = torch.relu(self.linears[i](x))
        
        x = self.linears[-1](x)
        return x
    
    def loss(reference, predicted):
        loss = torch.mean((reference - predicted)**2)
        return loss
        
    
# import torch
# import torchvision
# from torchvision import transforms

# # Define the transformations to apply to the data
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# # Load the MNIST dataset
# train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# # Create the data loaders
# batch_size = 64
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# from utils.mlp_utilities import MLP
# import torch
# import torch.nn as nn

# model = MLP(input_size=5, num_layers=10, layers_size=50, output_size=5)
# # x = torch.randn(5, 1, 28, 28)
# x = torch.randn(32, 5)
# print(model)
# print(model(x).shape)

# # print the number of parameters
# num_params = sum(p.numel() for p in model.parameters())
# # Use comma to print the number in a more readable format
# print(f"Number of parameters: {num_params:,}")

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

# # Define the loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# from tqdm import tqdm

# num_epochs = 10

# model.train()
# for epoch in tqdm(range(num_epochs)):
#     total_loss = 0
    
#     for images, labels in train_loader:
#         images = images.to(device)
#         labels = labels.to(device)
        
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)
        
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item()
    
#     avg_loss = total_loss / len(train_loader)
#     print(f"Epoch: {epoch+1}, Loss: {avg_loss:.4f}")

# model.eval()  # Set the model to evaluation mode

# correct = 0
# total = 0

# with torch.no_grad():
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
        
#         # Forward pass
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
        
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# accuracy = correct / total
# print(f"Test Accuracy: {accuracy:.4f}")

# import random
# import matplotlib.pyplot as plt

# # Set the model to evaluation mode
# model.eval()

# # Select a random image from the test dataset
# random_index = random.randint(0, len(test_dataset) - 1)
# image, label = test_dataset[random_index]

# # Move the image to the device
# image = image.to(device)

# # Forward pass to get the predicted label
# output = model(image.unsqueeze(0))
# _, predicted_label = torch.max(output, 1)

# # Convert the image tensor to a numpy array
# image_np = image.cpu().numpy()

# # Display the image, its label, and the predicted label
# plt.imshow(image_np.squeeze(), cmap='gray')
# plt.title(f"Label: {label}, Predicted: {predicted_label.item()}")
# plt.axis('off')
# plt.show()