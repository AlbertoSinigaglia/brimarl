from brimarl_masked.environment.environment import BriscolaLogger, BriscolaGame
import numpy as np
from brimarl_masked.agents.q_agent import DeepQAgent
from brimarl_masked.agents.ac_agent_quick import ACAgentQuick
from brimarl_masked.algorithms.dqn import QLearningAlgorithm
from brimarl_masked.scritps.training import *


def main(argv=None):
    np.set_printoptions(linewidth=500, threshold=np.inf)
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TRAIN)
    game = BriscolaGame(2, logger, win_extra_points=0)

    agent = DeepQAgent(0.99, 0.05, 3e-4)
    enemy = ACAgentQuick(training=False, name="FINAL")
    enemy.load_model("../models_savings/2/FINAL/latest/")

    episodes = 10000
    evaluate_every = 500
    num_evaluation = 1000
    training = TrainingBestResponse(
        num_epochs=episodes,
        num_game_per_epoch=1,
        game=game, agent=agent,
        agent_algorithm=QLearningAlgorithm(
            num_players=2,
            batch_size=512,
            discount=1.,
            replace_every=500,
            num_learning_per_epoch=1,
            replay_memory_capacity=10_000
        ),
        evaluate_every=evaluate_every,
        num_evaluations=num_evaluation,
        from_savings=False,
        enemy=enemy
    )
    training.train()


if __name__ == "__main__":
    main()