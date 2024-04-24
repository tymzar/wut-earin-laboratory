import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Implement a multilayer perceptron for image classification. The neural network should be trained with the mini-batch gradient descent method. Remember to split the dataset into training and validation sets.
# The main point of this task is to evaluate how various components and hyperparameters of a neural network and the training process affect the network’s performance in terms of its ability to converge, the speed of convergence, and final accuracy on the training and validation sets.
# For evaluation, please create plots visualizing:
# • The loss value for every learning step,
# • Accuracy on the training and validation set after each epoch.
# Make sure to include conclusions and observations based on the conducted experiments in your report.
# The details are described in your variant of the project.

# Use the MNIST dataset. Evaluate at least 3 different numbers/values/types of:
# • Learning rate,
# • Mini-batch size (including a batch containing only 1 example),
# • Number of hidden layers (including 0 hidden layers - a linear model),
# • Width (number of neurons in hidden layers),
# • Loss functions (e.g., Mean Squared Error, Mean Absolute Error, Cross Entropy).

# Tips
# The network can be implemented with a library offering neural network layers, optimizers, and error backpropagation. However, you must implement the learning procedure yourself. I highly recommend using PyTorch. This course may be useful for starting.
# Setting a fixed seed will make your results reproducible across different runs. Check this article or others for details.
# To input an image (28 x 28 matrix for all the given datasets) into the multilayer perceptron network, you need to flatten the image - represent it as a vector.
# Training of neural networks can take a while, and you will train several of them in this assessment. Keep this in mind and do not start working on the lab at the last minute!


class NeuralNetwork(nn.Module):
    def __init__(self, num_classes, num_hidden):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(28 * 28, num_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_hidden, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def present_dataset_sample(dataset):
    import matplotlib.pyplot as plt
    import numpy as np

    # Display the first image
    plt.imshow(np.array(dataset[0][0]).reshape(28, 28), cmap="gray")
    plt.title("Label: {}".format(dataset[0][1]))
    plt.show()


def main():

    # Define the hyperparameters
    learning_rate = 0.01
    batch_size = 64
    num_epochs = 10
    num_classes = 10
    num_hidden = 128

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transforms.ToTensor()
    )

    # Display the first image
    present_dataset_sample(train_dataset)

    # Create the dataloaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    # Create the neural network
    model = NeuralNetwork(num_classes, num_hidden)

    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for index, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, 28 * 28)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (index + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, num_epochs, index + 1, total_step, loss.item()
                    )
                )

    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28 * 28)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            "Accuracy of the network on the 10000 test images: {} %".format(
                100 * correct / total
            )
        )

    # Save the model checkpoint
    torch.save(model.state_dict(), "model.ckpt")


if __name__ == "__main__":
    main()
