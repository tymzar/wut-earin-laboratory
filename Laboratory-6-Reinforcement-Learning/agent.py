import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import math
from agent_types import Statistics, Hyperparameters


class CartPoleAgent:

    def __init__(
        self,
        visualation_rate=299,
        statistics: Statistics = None,
        hyperparameters: Hyperparameters = None,
        logger=None,
    ):
        if logger is None:
            raise ValueError("Logger is not set")

        self.__logger = logger

        # Setting hyperparameters
        self.n_episodes = hyperparameters["n_episodes"]
        self.min_epsilon = hyperparameters["min_epsilon"]
        self.max_epsilon = hyperparameters["max_epsilon"]

        self.decay_rate = hyperparameters["decay_rate"]
        self.discount = hyperparameters["discount"]
        self.learning_rate = hyperparameters["learning_rate"]

        if hyperparameters["patience"] == None:
            self.patience = self.n_episodes
        else:
            self.patience = hyperparameters["patience"]

        self.discretized_space = hyperparameters["discretized_space"]

        self.visualation_rate = visualation_rate

        self.steps = np.zeros(self.n_episodes)

        self.training_environment = gym.make("CartPole-v1", render_mode="rgb_array")

        self.action_space = self.training_environment.action_space.n

        self.__logger.info(
            f"Action space: {self.action_space} (possible actions - 0, 1)"
        )

        self.state_space = self.training_environment.observation_space.shape[0]

        self.__logger.info(
            f"Observation space: {self.state_space} (Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity)"
        )

        self.lowest_values = self.training_environment.observation_space.low
        self.highest_values = self.training_environment.observation_space.high

        self.__logger.info(f"Lowest values in observation space: {self.lowest_values}")
        self.__logger.info(
            f"Highest values in observation space: {self.highest_values}"
        )

        self.highest_values = [
            self.highest_values[0],
            0.5,
            self.highest_values[2],
            math.radians(50) / 1.0,
        ]
        self.lowest_values = [
            self.lowest_values[0],
            -0.5,
            self.lowest_values[2],
            -math.radians(50) / 1.0,
        ]

        self.statistics = statistics
        self.epsilon_history = [1]
        self.convergence_episode = None

        self.initialize_q_table()

    def __del__(self):
        cv2.destroyAllWindows()
        self.training_environment.close()
        self.__logger.info("Closing the environment")

    def choose_action(self, state):
        """Function to choose action based on epsilon greedy policy"""

        if np.random.random() < self.epsilon:
            return self.training_environment.action_space.sample()
        else:
            return np.argmax(self.Qtable[state])

    def initialize_q_table(self):
        """Function to initialize Q table with 0 values"""

        self.Qtable = np.zeros((self.discretized_space + (self.action_space,)))
        self.__logger.info(f"Q table shape: {self.Qtable.shape}")

    def discretize(self, state):
        """Function to discretize the state space from continuous"""

        state = [
            max(min(state[i], self.highest_values[i]), self.lowest_values[i])
            for i in range(0, self.state_space)
        ]
        scales = [
            (self.discretized_space[i] - 1)
            / (self.highest_values[i] - self.lowest_values[i])
            for i in range(0, self.state_space)
        ]
        discretized = [
            int(np.round(scales[i] * (state[i] - self.lowest_values[i])))
            for i in range(0, self.state_space)
        ]

        return tuple(discretized)

    def calculate_episilon(self, episode):
        """Function to calculate epsilon based on decay rate"""

        self.epsilon = self.min_epsilon + (
            self.max_epsilon - self.min_epsilon
        ) * np.exp(-self.decay_rate * episode)

        self.epsilon_history.append(self.epsilon)

    def update_table(self, state, action, reward, new_state):
        """Function uses Q learning to update Q table"""

        self.Qtable[state][action] += self.learning_rate * (
            reward
            + self.discount * np.max(self.Qtable[new_state])
            - self.Qtable[state][action]
        )

    def plot_performance(self):
        """Plotting the performance of the agent, step vs episode and epsilon vs episode"""

        plt.figure(figsize=(12, 6))
        plt.plot(range(len(self.steps)), self.steps)
        plt.xlabel("Episode")
        plt.ylabel("Steps")

        plt.figure(figsize=(12, 6))
        plt.plot(range(len(self.epsilon_history)), self.epsilon_history)
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")

        plt.show()

    def save_table(self, name="Qtable"):
        self.__logger.info(f"Saving table to tables/{name}.npy")
        np.save(f"tables/{name}", self.Qtable)

    def load_table(self, name="Qtable"):
        self.__logger.info(f"Loading table from tables/{name}.npy")
        self.Qtable = np.load(f"tables/{name}.npy")

    def preview_table(self):
        """Function generates a preview of the table be creating a new environment and using the Q table to take actions"""

        self.__logger.info("Previewing table")

        preview_environment = gym.make("CartPole-v1", render_mode="rgb_array")
        state, _ = preview_environment.reset()
        state = self.discretize(state)
        preview_environment.render()
        while True:
            action = np.argmax(self.Qtable[state])
            new_state, _, terminated, truncated, _ = preview_environment.step(action)
            new_state = self.discretize(new_state)
            rgb_array = preview_environment.render()
            img = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            cv2.imshow(f"Animation", img)
            cv2.waitKey(50)
            if terminated or truncated:
                break
            state = new_state

    def calculate_convergence(self, episode):

        if (
            episode > self.statistics["convergence_period"]
            and self.convergence_episode == None
        ):
            start = episode - self.statistics["convergence_period"]
            convergence_period = self.steps[start:episode]
            if np.mean(convergence_period) >= 0.99 * 500:
                self.__logger.info(f"Convergence at episode {episode}")
                self.convergence_episode = episode

    def patience_runout(self, episode):

        if episode > self.patience:
            start = episode - self.patience
            pacience_period = self.steps[start:episode]

            if np.mean(pacience_period) == 500:
                self.__logger.info(f"Training stopped at episode {episode}")
                self.steps = self.steps[:episode]
                return True

        return False

    def train(self):

        for episode in tqdm(range(self.n_episodes)):

            self.calculate_convergence(episode)

            if self.patience_runout(episode):
                break

            self.calculate_episilon(episode)

            state, _ = self.training_environment.reset()
            state = self.discretize(state)

            while True:
                self.steps[episode] += 1
                action = self.choose_action(state)
                new_state, reward, terminated, truncated, _ = (
                    self.training_environment.step(action)
                )
                if episode % self.visualation_rate == 0:
                    rgb_array = self.training_environment.render()
                    img = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                    cv2.imshow(f"Animation", img)
                    cv2.waitKey(50)

                new_state = self.discretize(new_state)

                self.update_table(state, action, reward, new_state)

                if terminated or truncated:
                    if episode % 1e3 == 3:
                        self.__logger.info(
                            f"Episode: {episode}, Steps: {self.steps[episode]}, Epsilon: {self.epsilon}"
                        )
                    break

                state = new_state
