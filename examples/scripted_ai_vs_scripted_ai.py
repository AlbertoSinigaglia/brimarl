import random

from brimarl_masked.environment.utils import BriscolaLogger
from brimarl_masked.environment.emulate import play_episode
from brimarl_masked.environment.environment import BriscolaGame
from brimarl_masked.agents.recurrent_q_agent import RecurrentDeepQAgent
from brimarl_masked.agents.dumb_agent import DumbAgent
from brimarl_masked.agents.ac_agent import ACAgent
from brimarl_masked.agents.scripted_ai_agent import ScriptedAIAgent
from brimarl_masked.agents.maac_oracle_agent import OracleAgentQuick
from brimarl_masked.agents.q_agent import DeepQAgent
import numpy as np
import tensorflow as tf
def main(argv=None):
    np.random.seed(0)
    random.seed(0)
    tf.random.set_seed(0)
    """Play against one of the intelligent agents."""
    # initialize the environment
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.PVP)
    game = BriscolaGame(4, logger, win_extra_points=0, binary_reward=True)

    # initialize the agents
    agents = [OracleAgentQuick(), OracleAgentQuick(),OracleAgentQuick(),OracleAgentQuick()]

    states, actions, masks, rewards, dones = play_episode(game, agents, train=True)
    #[print(r) for r in rewards]
    #[print(s[-1].round(1)) for s in states]
    #print(states[0][-1].round(1))
    #print(states[0], actions[0], rewards[0], dones[0])


if __name__ == '__main__':
    main()
