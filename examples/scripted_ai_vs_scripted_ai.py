from brimarl_masked.environment.utils import BriscolaLogger
from brimarl_masked.environment.emulate import play_episode
from brimarl_masked.environment.environment import BriscolaGame
from brimarl_masked.agents.recurrent_q_agent import RecurrentDeepQAgent
from brimarl_masked.agents.dumb_agent import DumbAgent
from brimarl_masked.agents.ac_agent import ACAgent
from brimarl_masked.agents.scripted_ai_agent import ScriptedAIAgent
from brimarl_masked.agents.q_agent import DeepQAgent
import numpy as np

def main(argv=None):
    """Play against one of the intelligent agents."""
    # initialize the environment
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.PVP)
    game = BriscolaGame(2, logger)

    # initialize the agents
    agents = [DeepQAgent(3, 0., 0., 0.), ScriptedAIAgent()]

    states, actions, rewards, dones = play_episode(game, agents, train=True)
    print(np.array(states[0])[:,0])
    print(actions[0])
    print(rewards[0])
    #print(states[0], actions[0], rewards[0], dones[0])


if __name__ == '__main__':
    main()
