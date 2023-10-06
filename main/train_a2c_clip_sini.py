from brimarl_masked.scritps.self_train import *
from brimarl_masked.agents.ac_agent_quick import ACAgentQuick
from brimarl_masked.algorithms.a2c_clipped_sini import A2CSiniClippedAlgorithm
from brimarl_masked.environment.environment import BriscolaLogger, BriscolaGame
import numpy as np

from brimarl_masked.scritps.training import *


def main(argv=None):
    np.set_printoptions(linewidth=500, threshold=np.inf)
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TRAIN)
    game = BriscolaGame(2, logger, win_extra_points=0)

    agent = ACAgentQuick(name="A2CClipSiniACAgent")
    episodes = 1000000
    evaluate_every = 1000
    num_evaluation = 500
    # 384 418
    # 510 486

    training = TrainingSelfPlay(
        num_epochs=episodes,
        num_game_per_epoch=1,
        game=game,
        agent=agent,
        agent_algorithm=A2CSiniClippedAlgorithm(
            num_players=2,
            discount=1.,
            min_samples=128,
            num_learning_per_epoch_actor=8,
            num_learning_per_epoch_critic=3,
            clip_eps=0.1,
            epsilon=1e-8
        ),
        evaluate_every=evaluate_every,
        num_evaluations=num_evaluation,
        from_savings=False,

    )
    training.train()


if __name__ == "__main__":
    main()