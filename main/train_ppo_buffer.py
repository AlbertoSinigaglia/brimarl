from brimarl_masked.scritps.self_train import *
from brimarl_masked.agents.ac_agent import ACAgent
from brimarl_masked.algorithms.ppo_buffer import PPOBufferAlgorithm
from brimarl_masked.environment.environment import BriscolaLogger, BriscolaGame
import numpy as np

def main(argv=None):
    np.set_printoptions(linewidth=500, threshold=np.inf)
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TRAIN)
    game = BriscolaGame(2, logger, win_extra_points=0)

    # agent = DeepQAgent(0.99, 0.05, 1e-3)
    agent = ACAgent()
    # agent = RecurrentDeepQAgent(40, 0.9, 0.05, 1e-3)
    # agent = ACQAgent()
    episodes = 1000000
    evaluate_every = 1000
    num_evaluation = 500
    # 0.456 0.462
    # 0.438 0.398
    # 0.448 0.410
    _, rewards_per_episode = train_against_self(
        game,
        agent,
        PPOBufferAlgorithm(
            num_players=2,
            discount=1.,
            buffer_size=2048,
            batch_size=256,
            num_learning_per_epoch_actor=15,
            num_learning_per_epoch_critic=3,
            clip_eps=0.1,
            epsilon=1e-8
        ),
        num_epochs=episodes,
        evaluate_every=evaluate_every,
        num_evaluations=num_evaluation,
        from_savings=False,
        num_game_per_epoch=1
    )


if __name__ == "__main__":
    main()