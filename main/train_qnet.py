from brimarl_masked.scritps.self_train import *
from brimarl_masked.environment.environment import BriscolaLogger, BriscolaGame
import numpy as np

def main(argv=None):
    np.set_printoptions(linewidth=500, threshold=np.inf)
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TRAIN)
    game = BriscolaGame(2, logger, win_extra_points=0)

    agent = DeepQAgent(0.99, 0.05, 1e-3)
    episodes = 1000000
    evaluate_every = 1000
    num_evaluation = 500
    _, rewards_per_episode = train_against_scripted(
        game,
        agent,
        QLearningAlgorithm(2, 512, 1., 500, 1, 100_000),
        num_epochs=episodes,
        evaluate_every=evaluate_every,
        num_evaluations=num_evaluation,
        from_savings=False,
        num_game_per_epoch=1
    )


if __name__ == "__main__":
    main()