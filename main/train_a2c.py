from brimarl_masked.agents.ac_agent_quick import ACAgentQuick
from brimarl_masked.algorithms.a2c import A2CAlgorithm
from brimarl_masked.environment.environment import BriscolaLogger, BriscolaGameID
import numpy as np
from brimarl_masked.scritps.training import *

def main(argv=None):
    np.set_printoptions(linewidth=500, threshold=np.inf)
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TRAIN)
    game = BriscolaGameID(2, logger, win_extra_points=0)
    agent = ACAgentQuick(name="A2CACAgent")
    episodes = 10000
    evaluate_every = 1000
    num_evaluation = 500
    print("A2C 1e-4 3e-4 128 samples")
    training = TrainingSelfPlay(
        num_epochs=episodes,
        num_game_per_epoch=1,
        game=game,
        agent=agent,
        agent_algorithm=A2CAlgorithm(
            num_players=2,
            discount=1.,
            num_learning_per_epoch=1
        ),
        evaluate_every=evaluate_every,
        num_evaluations=num_evaluation,
        from_savings=False,

    )
    training.train()


if __name__ == "__main__":
    main()