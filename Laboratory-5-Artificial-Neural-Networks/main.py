import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms

import ray
from ray.train import torch as raytorch
from ray import tune
from ray.air import session
from ray.train import Checkpoint
from ray.tune.analysis import ExperimentAnalysis
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import random_split
from functools import partial
import tempfile
import os
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np


# Implement a multilayer perceptron for image classification. The neural network should be trained with the mini-batch gradient descent method. Remember to split the dataset into training and validation sets.
# The main point of this task is to evaluate how various components and hyperparameters of a neural network and the training process affect the network’s performance in terms of its ability to converge, the speed of convergence, and final accuracy on the training and validation sets.
# For evaluation, please create plots visualizing:
# • The loss value for every learning step,
# • Accuracy on the training and validation set after each epoch.


class NeuralNetwork(nn.Module):
    def __init__(
        self,
        num_classes,
        num_hidden_neurons,
        num_hidden_layers=1,
        loss_finction_name="CrossEntropyLoss",
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, num_hidden_neurons),
            nn.ReLU(),
        )

        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(num_hidden_neurons, num_hidden_neurons)
                for _ in range(num_hidden_layers)
            ]
        )

        potential_decoder = [
            nn.Linear(num_hidden_neurons, num_classes),
        ]

        if loss_finction_name != "CrossEntropyLoss":
            potential_decoder.append(nn.LogSoftmax(dim=1))

        self.decoder = nn.Sequential(*potential_decoder)

    def forward(self, x):
        """Forward pass"""

        x = self.encoder(x)
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        x = self.decoder(x)
        return x


def present_dataset_sample(dataset):

    # Display the first image
    plt.imshow(np.array(dataset[0][0]).reshape(28, 28), cmap="gray")
    plt.title("Label: {}".format(dataset[0][1]))
    plt.show()

    # Display the target distribution
    labels = [label for _, label in dataset]
    plt.hist(labels, bins=10)
    plt.title("Target distribution")
    plt.show()


def load_dataset(data_dir="./data") -> tuple:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, transform=transform, download=True
    )

    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform)

    return train_dataset, test_dataset


def calculate_loss(outputs, labels, criterion_name, criterion):
    return criterion(outputs, labels)
    # match criterion_name:
    #     case "CrossEntropyLoss":
    #         return criterion(outputs, labels)
    #     case "MSELoss":
    #         softmaxed_outputs_prob = nn.Softmax(dim=1)(outputs)
    #         y_pred = torch.argmax(softmaxed_outputs_prob, dim=1)
    #         loss = criterion(y_pred.float(), labels.float())
    #         return loss


def train_mnist(config, trainset):
    raytorch.enable_reproducibility(0)

    net = NeuralNetwork(
        config["num_classes"],
        config["num_hidden"],
        config["num_hidden_layers"],
        config["criterion"]["name"],
    )
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    net.to(device)

    criterion = config["criterion"]["criterion"]
    optimizer = optim.SGD(net.parameters(), lr=config["lr"])

    # checkpoint = session.get_checkpoint()

    # if checkpoint:
    #     checkpoint_state = checkpoint.to_dict()
    #     start_epoch = checkpoint_state["epoch"]
    #     net.load_state_dict(checkpoint_state["net_state_dict"])
    #     optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    # else:
    #     start_epoch = 0

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs]
    )

    trainloader = DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        num_workers=8,
    )
    valloader = DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8,
    )

    total_step = len(trainloader)

    learning_step_loss_per_epoch = []

    for epoch in range(0, config["epochs"]):
        current_step_loss = []
        running_loss = 0.0
        epoch_steps = 0
        for index, (images, labels) in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]

            images = images.reshape(-1, 28 * 28)
            images, labels = images.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(images)

            loss = calculate_loss(
                outputs, labels, config["criterion"]["name"], criterion
            )

            loss.backward()
            optimizer.step()

            current_step_loss.append(loss.item())

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if (index + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, config["epochs"], index + 1, total_step, loss.item()
                    )
                )

        learning_step_loss_per_epoch.append(deepcopy(current_step_loss))

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, (images, labels) in enumerate(valloader):
            with torch.no_grad():

                images = images.reshape(-1, 28 * 28)
                images, labels = images.to(device), labels.to(device)

                outputs = net(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = calculate_loss(
                    outputs, labels, config["criterion"]["name"], criterion
                )

                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(net.state_dict(), os.path.join(tmpdir, "model.pt"))
            torch.save(optimizer.state_dict(), os.path.join(tmpdir, "optim.pt"))
            torch.save({"epoch": epoch}, os.path.join(tmpdir, "extra_state.pt"))

            session.report(
                dict(
                    loss=running_loss / epoch_steps,
                    accuracy=correct / total,
                    learning_step_loss_per_epoch=learning_step_loss_per_epoch,
                ),
                # checkpoint=Checkpoint.from_directory(tmpdir),
            )

    print("Finished Training")


def config_to_string(config: dict):
    return f"lr: {config['lr']}, batch_size: {config['batch_size']}, num_hidden: {config['num_hidden']}, num_hidden_layers: {config['num_hidden_layers']}, criterion: {config['criterion']['name']}"


def plot_loss_learning_step(results: list[list[float]], config: dict):

    fig = plt.figure().add_subplot()
    plt.title(f"Loss value for every step in best trial")

    colormap = plt.cm.nipy_spectral
    colors = colormap(np.linspace(0, 1, len(results)))
    fig.set_prop_cycle('color', colors)

    # make one plot for all epochs in the trial
    for epoch in results:
        fig.plot(epoch, label=f"Epoch {results.index(epoch)}")

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()


def plot_accuracy_epoch(results: ExperimentAnalysis):

    fig = plt.figure().add_subplot()

    colormap = plt.cm.nipy_spectral
    colors = colormap(np.linspace(0, 1, len(results.trials)))
    fig.set_prop_cycle('color', colors)
    for trial in results.trials:
        print(trial.trial_id)
        accuracy = trial.metric_n_steps["accuracy"]["10"]
        epochs = list(range(len(accuracy)))

        fig.plot(
            epochs,
            accuracy,
            label=f"Trial {trial.trial_id}",
        )
        fig.text(epochs[-1], accuracy[-1], f'{trial.trial_id[-3:]}')
    plt.title(f"Convergence and accuracy for all trials")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()


def main(num_samples=1, max_num_epochs=5):

    torch.manual_seed(0)
    np.random.seed(0)
    ray.init(object_store_memory=10**9)
    data_dir = os.path.abspath("./data")

    trainset, _ = load_dataset(data_dir)

    # Use the MNIST dataset. Evaluate at least 3 different numbers/values/types of:
    # • Learning rate, DONE
    # • Mini-batch size (including a batch containing only 1 example),
    # • Number of hidden layers (including 0 hidden layers - a linear model),
    # • Width (number of neurons in hidden layers), DONE
    # • Loss functions (e.g., Mean Squared Error, Mean Absolute Error, Cross Entropy). DONE
    config = {
        "epochs": max_num_epochs,
        "num_classes": tune.choice([10]),
        "num_hidden": tune.choice([2**number for number in range(6, 9)]),
        "num_hidden_layers": tune.choice([0, 1, 5]),
        "lr": tune.loguniform(1e-4, 1e-2, 1e-1),
        "batch_size": tune.choice([1, 16, 32]),
        "criterion": tune.choice(
            [
                {"name": "CrossEntropyLoss", "criterion": nn.CrossEntropyLoss()},
                {"name": "MultiMarginLoss", "criterion": nn.MultiMarginLoss()},
                {"name": "NLLLoss", "criterion": nn.NLLLoss()},
            ]
        ),
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=3,
        reduction_factor=2,
    )
    result = tune.run(
        partial(train_mnist, trainset=trainset),
        resources_per_trial={"cpu": 1.5, "gpu":0},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    plot_accuracy_epoch(result)
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")
    plot_loss_learning_step(
        best_trial.last_result["learning_step_loss_per_epoch"], best_trial.config
    )

    plt.show()
    print("Done")


if __name__ == "__main__":
    main(num_samples=16, max_num_epochs=3)
