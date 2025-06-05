"""
@File: AE.py
@Author: zhang
@Time: 11/1/22 2:24 PM
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor

"""
Implementation of Autoencoder in Pytorch
"""
torch.manual_seed(1)  # 为了使用同样的随机初始化种子以形成相同的随机效果

"""
1. Loading the Dataset
"""

# Download the MNIST Dataset
training_data = datasets.MNIST(root="../data",
                               train=True,
                               download=True,
                               transform=ToTensor())

test_data = datasets.MNIST(root="../data",
                           train=False,
                           download=True,
                           transform=ToTensor())

batch_size = 32

# DataLoader is used to load the dataset for training
training_dataloader = DataLoader(dataset=training_data,
                                 batch_size=batch_size,
                                 shuffle=True)
test_dataloader = DataLoader(test_data,
                             batch_size=batch_size)

"""
2. Create Autoencoder Class
- encoder:28*28 = 784 ==> 128 ==> 64 ==> 36 ==> 18 ==> 9
- decoder:9 ==> 18 ==> 36 ==> 64 ==> 128 ==> 784 ==> 28*28 = 784

"""
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# Creating a PyTorch class
# 28*28 ==> 9 ==> 28*28
class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Building an linear encoder with Linear layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9)
        )

        # Building an linear decoder with Linear layer followed by Relu activation function
        # The Sigmoid activation function outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 28 * 28),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Model Initialization
model = AE()
print(model)

"""
Optimizing the Model Parameters
"""
# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()  # 定义均方误差为损失函数

# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(),
                             lr=1e-1,
                             weight_decay=1e-8)  # 为防止过拟合，使用权重衰减(L2正则化)

"""
4. Create Output Generation
"""

epochs = 5
outputs = []
losses = []
for epoch in range(epochs):
    for (image, _) in training_dataloader:
        # Reshaping the image to (-1, 784)
        image = image.reshape(-1, 28 * 28)

        # Output of Autoencoder
        reconstructed = model(image)

        # Calculating the loss function
        loss = loss_function(reconstructed, image)

        # The gradients are set to zero, the gradient is computed and stored.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # .step() performs parameter update

        # Storing the losses in a list for plotting
        losses.append(loss.detach())
    print('Epoch :', epoch, '|', 'train_loss:%.4f' % loss.data)
    outputs.append((epochs, image, reconstructed))

# save the model
torch.save(model, './AE.pth')

# Defining the Plot Style
plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')

# Plotting the last 100 values
plt.plot(losses[-100:])
# plt.plot(losses)
plt.show()
"""
5. Input/Reconstructed Input to/from Autoencoder
"""
for i, item in enumerate(image):
    # Reshape the array for plotting
    item = item.reshape(-1, 28, 28)
    plt.imshow(item[0].detach())

    for i, item in enumerate(reconstructed):
        item = item.reshape(-1, 28, 28)
        plt.imshow(item[0].detach())

"""

"""
model = AE()
model = torch.load('./AE.pth')


