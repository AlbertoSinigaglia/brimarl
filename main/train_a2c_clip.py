from brimarl_masked.agents.ac_agent_quick import ACAgentQuick
from brimarl_masked.scritps.self_train import *
from brimarl_masked.agents.ac_agent import ACAgent
from brimarl_masked.algorithms.a2c_clipped import A2CClippedAlgorithm
from brimarl_masked.environment.environment import BriscolaLogger, BriscolaGame
import numpy as np

from brimarl_masked.scritps.training import *


def main(argv=None):
    np.set_printoptions(linewidth=500, threshold=np.inf)
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TRAIN)
    game = BriscolaGame(2, logger, win_extra_points=0)

    agent = ACAgentQuick()
    episodes = 50000
    evaluate_every = 1000
    num_evaluation = 1000
    training = TrainingFictitiousSelfPlay(
        num_epochs=episodes,
        num_game_per_epoch=1,
        game=game, agent=agent,
        agent_algorithm=A2CClippedAlgorithm(
            num_players=game.num_players,
            discount=1.,
            min_samples=512,
            epsilon=1e-8,
            num_learning_per_epoch_actor=8,
            num_learning_per_epoch_critic=3,
            clip_eps=0.1,
            lr_actor=6e-4,
            lr_critic=3e-4
        ),
        evaluate_every=evaluate_every,
        num_evaluations=num_evaluation,
        from_savings=False,
        max_old_agents=10,
        store_every=10,
        current_agent_prob=0.
    )
    training.train()


if __name__ == "__main__":
    main()