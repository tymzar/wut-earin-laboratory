from agent import CartPoleAgent
from logger import setup_logger
from datetime import datetime
from agent_types import default_hyperparameters

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logger = setup_logger("main", f"logs/main-{timestamp}.log")


def main():

    agent = CartPoleAgent(
        logger=logger,
        statistics={
            "convergence_period": 100,
        },
        hyperparameters={
            **default_hyperparameters,
        },
        # hyperparameters={
        #     "patience": 800,
        #     "n_episodes": 6000,
        #     "min_epsilon": 0,
        #     "max_epsilon": 1.0,
        #     "decay_rate": 1e-3,
        #     "discount": 1,
        #     "learning_rate": 0.1,
        #     "discretized_space": (3, 3, 8, 8),
        # },
    )

    agent.train()
    agent.plot_performance()

    logger.info("Saving model")

    agent.save_table("test_table")
    agent.load_table("test_table")

    agent.preview_table()

    del agent


if __name__ == "__main__":
    main()
