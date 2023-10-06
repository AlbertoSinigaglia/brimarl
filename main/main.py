import numpy as np
from line_profiler import LineProfiler

from brimarl_masked.agents.ac_agent_quick import ACAgentQuick
from brimarl_masked.algorithms.ppo import PPOAlgorithm
from brimarl_masked.environment.emulate import play_episode
from brimarl_masked.environment.environment import BriscolaGame
from brimarl_masked.environment.utils import BriscolaLogger
from brimarl_masked.main.train_mappo_parallel import main as mappo_parallel_main
from brimarl_masked.main.train_a2c import main as a2c_main
from brimarl_masked.main.train_ppo import main as ppo_main
from brimarl_masked.main.train_mappo import main as mappo_main
from brimarl_masked.scritps.training import TrainingSelfPlay

if __name__ == "__main__":
    """import time
    t = time.time()
    ppo_main()
    print("\n\n\n\n\n")
    print(time.time()-t)"""


    np.set_printoptions(linewidth=500, threshold=np.inf)
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TRAIN)
    game = BriscolaGame(2, logger, win_extra_points=0)

    agent = ACAgentQuick(name="Dumb")
    episodes = 50
    evaluate_every = 1000
    num_evaluation = 1000




    training = TrainingSelfPlay(
        num_epochs=episodes,
        num_game_per_epoch=1,
        game=game,
        agent=agent,
        agent_algorithm=PPOAlgorithm(
            num_players=game.num_players,
            discount=1.,
            min_samples=512,
            epsilon=1e-8,
            num_learning_per_epoch_actor=15,
            num_learning_per_epoch_critic=5,
            clip_eps=0.1,
            lr_actor=5e-4,
            lr_critic=5e-4
        ),
        evaluate_every=evaluate_every,
        num_evaluations=num_evaluation,
        from_savings=False
    )
    lp = LineProfiler()
    lp.add_function(ACAgentQuick.action)
    lpw = lp(training.train)
    lpw()
    lp.print_stats()