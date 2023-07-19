from brimarl_masked.agents.ac_agent import ACAgent
from brimarl_masked.algorithms.ppo import PPOAlgorithm
from brimarl_masked.scritps.training import *
from brimarl_masked.environment.environment import BriscolaLogger, BriscolaGame


def main(argv=None):
    np.set_printoptions(linewidth=500, threshold=np.inf)
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TRAIN)
    game = BriscolaGame(4, logger, win_extra_points=0)

    agent = ACAgent()
    episodes = 1000000
    evaluate_every = 1000
    num_evaluation = 500
    training = TrainingSelfPlay(
        num_epochs=episodes,
        num_game_per_epoch=1,
        game=game, agent=agent,
        agent_algorithm=PPOAlgorithm(
            num_players=game.num_players,
            discount=1.,
            min_samples=512,
            epsilon=1e-8,
            num_learning_per_epoch_actor=10,
            num_learning_per_epoch_critic=3,
            clip_eps=0.1,
            lr_actor=1e-4,
            lr_critic=3e-4
        ),
        evaluate_every=evaluate_every,
        num_evaluations=num_evaluation,
        from_savings=False
    )
    training.train()


if __name__ == "__main__":
    main()