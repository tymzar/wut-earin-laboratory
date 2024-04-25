import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms

from ray import tune
from ray.air import session
from ray.train import Checkpoint
from ray.tune.analysis import ExperimentAnalysis
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import random_split
from functools import partial
import tempfile
import os


# Implement a multilayer perceptron for image classification. The neural network should be trained with the mini-batch gradient descent method. Remember to split the dataset into training and validation sets.
# The main point of this task is to evaluate how various components and hyperparameters of a neural network and the training process affect the network’s performance in terms of its ability to converge, the speed of convergence, and final accuracy on the training and validation sets.
# For evaluation, please create plots visualizing:
# • The loss value for every learning step,
# • Accuracy on the training and validation set after each epoch.


def plot_loss_accuracy(loss, accuracy, params):
    import matplotlib.pyplot as plt

    print(f"Plotting loss and accuracy for {len(loss)} epochs")

    plt.plot(loss, label="loss")
    plt.plot(accuracy, label="accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    # save the plot
    plot_name = f"./plot-{len(loss)}-epochs-{len(accuracy)}-accuracy-{params['num_hidden']}-hidden-{params['lr']}-lr-{params['batch_size']}-batch_size.png"
    plt.savefig(plot_name)


class NeuralNetwork(nn.Module):
    def __init__(self, num_classes, num_hidden_neurons, num_hidden_layers=1):

        super(NeuralNetwork, self).__init__()
        self.num_classes = num_classes
        self.num_hidden_neurons = num_hidden_neurons
        self.num_hidden_layers = num_hidden_layers

        self.input_layer = nn.Linear(28 * 28, self.num_hidden_neurons)
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(self.num_hidden_neurons, self.num_hidden_neurons)
                for _ in range(self.num_hidden_layers)
            ]
        )
        self.output_layer = nn.Linear(self.num_hidden_neurons, self.num_classes)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        x = self.output_layer(x)
        return x


def present_dataset_sample(dataset):
    import matplotlib.pyplot as plt
    import numpy as np

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


def train_mnist(config, trainset):

    net = NeuralNetwork(
        config["num_classes"], config["num_hidden"], config["num_hidden_layers"]
    )
    device = "cpu"
    net.to(device)

    criterion = config["criterion"]["criterion"]
    optimizer = optim.SGD(net.parameters(), lr=config["lr"])

    checkpoint = session.get_checkpoint()

    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        net.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

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

    for epoch in range(start_epoch, config["epochs"]):
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

            loss = None
            # depending on the loss function, you may need to change the next line
            match config["criterion"]["name"]:
                case "CrossEntropyLoss":
                    loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if (index + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, config["epochs"], index + 1, total_step, loss.item()
                    )
                )

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

                loss = None
                # depending on the loss function, you may need to change the next line
                match config["criterion"]["name"]:
                    case "CrossEntropyLoss":
                        loss = criterion(outputs, labels)

                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(net.state_dict(), os.path.join(tmpdir, "model.pt"))
            torch.save(optimizer.state_dict(), os.path.join(tmpdir, "optim.pt"))
            torch.save({"epoch": epoch}, os.path.join(tmpdir, "extra_state.pt"))
            # save loss per step

            session.report(
                dict(
                    loss=running_loss / epoch_steps,
                    accuracy=correct / total,
                ),
                checkpoint=Checkpoint.from_directory(tmpdir),
            )

    print("Finished Training")


def test_accuracy(net, testset, config, device="cpu"):

    testloader = DataLoader(
        testset, batch_size=int(config["batch_size"]), shuffle=False
    )

    # Test the model
    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testloader:
            images = images.reshape(-1, 28 * 28)
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            "Accuracy of the network on the 10000 test images: %d %%"
            % (100 * correct / total)
        )
        return correct / total


def plot_loss_learning_step():
    pass


def plot_accuracy_epoch(results: ExperimentAnalysis):

    import matplotlib.pyplot as plt

    for trial in results.trials:

        # {"10": deque([0.87775, 0.901, 0.91125], maxlen=10)}
        accuracy = trial.metric_n_steps["accuracy"]["10"]
        epochs = list(range(len(accuracy)))

        plt.plot(epochs, accuracy, label=f"Trial {trial.trial_id}")

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()


#     import matplotlib.pyplot as plt

#     for trial in results:


def main(num_samples=1, max_num_epochs=5):

    data_dir = os.path.abspath("./data")
    device = "cpu"

    trainset, testset = load_dataset(data_dir)

    # Use the MNIST dataset. Evaluate at least 3 different numbers/values/types of:
    # • Learning rate, DONE
    # • Mini-batch size (including a batch containing only 1 example),
    # • Number of hidden layers (including 0 hidden layers - a linear model),
    # • Width (number of neurons in hidden layers), DONE
    # • Loss functions (e.g., Mean Squared Error, Mean Absolute Error, Cross Entropy). DONE
    config = {
        "epochs": 10,
        "num_classes": tune.choice([10]),
        "num_hidden": tune.choice([2**number for number in range(6, 9)]),
        "num_hidden_layers": tune.choice([0, 1, 5]),
        "lr": tune.loguniform(1e-4, 1e-2, 1e-1),
        "batch_size": tune.choice([1, 16, 32]),
        "criterion": tune.choice(
            [
                {"name": "CrossEntropyLoss", "criterion": nn.CrossEntropyLoss()},
            ]
        ),
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    result = tune.run(
        partial(train_mnist, trainset=trainset),
        resources_per_trial={"cpu": 3},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
    )

    plot_accuracy_epoch(result)

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    best_trained_model = NeuralNetwork(
        best_trial.config["num_classes"], best_trial.config["num_hidden"]
    )
    best_trained_model.to(device)

    best_checkpoint = best_trial.checkpoint.to_directory()
    best_checkpoint_data = torch.load(f"{best_checkpoint}/model.pt")

    best_trained_model.load_state_dict(best_checkpoint_data)

    test_acc = test_accuracy(best_trained_model, best_trial.config, testset, device)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    main(num_samples=10, max_num_epochs=10)
