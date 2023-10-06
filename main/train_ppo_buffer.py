from brimarl_masked.scritps.self_train import *
from brimarl_masked.agents.ac_agent_quick import ACAgentQuick
from brimarl_masked.algorithms.ppo_buffer import PPOBufferAlgorithm
from brimarl_masked.environment.environment import BriscolaLogger, BriscolaGame
import numpy as np

from brimarl_masked.scritps.training import TrainingSelfPlay


def main(argv=None):
    np.set_printoptions(linewidth=500, threshold=np.inf)
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TRAIN)
    game = BriscolaGame(2, logger, win_extra_points=0)

    agent = ACAgentQuick(name="BufferPPOAgent")
    episodes = 1000000
    evaluate_every = 1000
    num_evaluation = 500

    training = TrainingSelfPlay(
        num_epochs=episodes,
        num_game_per_epoch=1,
        game=game,
        agent=agent,
        agent_algorithm=PPOBufferAlgorithm(
            num_players=2,
            discount=1.,
            buffer_size=2048,
            batch_size=256,
            num_learning_per_epoch_actor=15,
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