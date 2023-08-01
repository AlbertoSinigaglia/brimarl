from brimarl_masked.scritps.self_train import train_against_self, train_against_scripted
from brimarl_masked.agents.ac_agent import ACAgent
from brimarl_masked.algorithms.a2c import A2CAlgorithm
from brimarl_masked.environment.environment import BriscolaLogger, BriscolaGameID
import numpy as np

def main(argv=None):
    np.set_printoptions(linewidth=500, threshold=np.inf)
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TRAIN)
    game = BriscolaGameID(2, logger, win_extra_points=0)
    agent = ACAgent()
    episodes = 10000
    evaluate_every = 1000
    num_evaluation = 500
    print("A2C 1e-4 3e-4 128 samples")
    _, rewards_per_episode = train_against_scripted(
        game,
        agent,
        # A2CClippedAlgorithm(2, 1., 1, ),
        # A2CAlgorithm(2, 1., num_learning_per_epoch=10, min_samples=128),
        # A2CGAEAlgorithm(2, 1., 5, lambda_coefficient=0.),
        # QLearningAlgorithm(2, 512, 1., 500, 1, 100_000),
        A2CAlgorithm(num_players=2, discount=1., num_learning_per_epoch=1),
        # PPOAlgorithm(num_players=2, ppo_clip_eps=0.1, min_samples=128),
        # PPOGAEAlgorithm(2,1., 10,),
        # A2CAlgorithm(2, 1., 1, min_samples=64),
        # A2CQAlgorithm(2,1., 1),
        # RQLearningAlgorithm(2, 10_000, False, 4*4, 256, 1., 500, 1),
        num_epochs=episodes,
        evaluate_every=evaluate_every,
        num_evaluations=num_evaluation,
        from_savings=False,
        num_game_per_epoch=1
    )


if __name__ == "__main__":
    main()