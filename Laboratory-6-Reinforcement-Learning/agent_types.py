from typing import TypedDict


class Statistics(TypedDict):
    convergence_period: int


class Hyperparameters(TypedDict):
    patience: int
    n_episodes: int
    min_epsilon: float
    max_epsilon: float
    decay_rate: float
    discount: float
    learning_rate: float
    discretized_space: tuple[int, int, int, int]
    


default_hyperparameters = {
    # Number of steps to wait before stopping the training
    "patience": None,
    # Number of episodes - how many times we train the agent
    "n_episodes": 1200,
    # Minimum epsilon value - exploration vs exploitation
    "min_epsilon": 0.0,
    # Maximum epsilon value - exploration vs exploitation
    "max_epsilon": 1.0,
    # Decay rate - how fast epsilon decays
    "decay_rate": 5e-3,
    # Discount factor - how much we value future rewards
    "discount": 1,
    # Learning rate - how much we value the difference between the predicted and the target Q value
    "learning_rate": 0.1,
    # Shape of the observation space after discretization
    "discretized_space": (3, 3, 6, 6)
}
