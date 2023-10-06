from brimarl_masked.agents.maac_oracle_agent import OracleAgentQuick
from brimarl_masked.algorithms.ppo import PPOAlgorithm
from brimarl_masked.scritps.training import *
from brimarl_masked.environment.environment import BriscolaLogger, BriscolaGame


def main(argv=None):
    np.set_printoptions(linewidth=500, threshold=np.inf, suppress=False)
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TRAIN)
    game = BriscolaGame(4, logger, win_extra_points=0)

    agent = OracleAgentQuick(name="OracleACAgent")
    #agent2 = OracleAgentQuick(name="OracleACAgent")
    episodes = 100_000
    evaluate_every = 1000
    num_evaluation = 500
    training = TrainingSelfPlayOracle(
        num_epochs=episodes,
        num_game_per_epoch=1,
        game=game,
        agent1=agent,
        #agent2=agent2,
        agent_algorithm1=PPOAlgorithm(
            num_players=game.num_players,
            discount=1.,
            min_samples=512,
            epsilon=1e-8,
            num_learning_per_epoch_actor=10, # 10
            num_learning_per_epoch_critic=5, # 5
            clip_eps=0.1,
            lr_actor=3e-4,
            lr_critic=3e-4
        ),
        # agent_algorithm2=PPOAlgorithm(
        #     num_players=game.num_players,
        #     discount=1.,
        #     min_samples=512,
        #     epsilon=1e-8,
        #     num_learning_per_epoch_actor=10,
        #     num_learning_per_epoch_critic=5,
        #     clip_eps=0.1,
        #     lr_actor=1e-3,
        #     lr_critic=1e-3
        # ),
        evaluate_every=evaluate_every,
        num_evaluations=num_evaluation,
        from_savings=False
    )

    import time
    t = time.time()
    training.train()
    print(time.time() - t)


if __name__ == "__main__":
    main()