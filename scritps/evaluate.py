import argparse
from statistics import mean
from tqdm.auto import tqdm
from brimarl_masked.agents.random_agent import RandomAgent
from brimarl_masked.agents.scripted_ai_agent import ScriptedAIAgent as AIAgent
from brimarl_masked.scritps.graphic_visualizations import stats_plotter
import brimarl_masked.environment.environment as brisc
from brimarl_masked.environment.utils import BriscolaLogger, NetworkTypes
from brimarl_masked.environment.emulate import play_episode
import time
def evaluate(game, agents, num_evaluations):
    """Play num_evaluations games and report statistics."""
    total_wins = [0] * len(agents)
    points_history = [[0] for _ in range(len(agents))]
    print()
    time.sleep(0.1)
    for _ in tqdm(range(num_evaluations), desc="\tEval", position=0, smoothing=0.):
        game_winner_id, winner_points = play_episode(game, agents, train=False)
        for player in game.players:
            points_history[player.id].append(player.points)
            if player.id == game_winner_id or player.id == game.get_teammate(game_winner_id):
                total_wins[player.id] += 1

    print(f"\tTotal wins: {total_wins}.")
    for i in range(len(agents)):
        print(f"\t{agents[i].name} {i} won {total_wins[i]/num_evaluations:.2%} with an average of {mean(points_history[i]):.2f} points.")

    return total_wins, points_history


def main(argv=None):
    """Evaluate agent performance against RandomAgent and AIAgent."""
    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.TEST)
    game = brisc.BriscolaGame(2, logger)

    # agent to be evaluated is RandomAgent or QAgent if a model is provided
    if FLAGS.model_dir:
        eval_agent = QAgent(network=FLAGS.network)
        eval_agent.load_model(FLAGS.model_dir)
        eval_agent.make_greedy()
    else:
        eval_agent = RandomAgent()

    # test agent against RandomAgent
    agents = [eval_agent, RandomAgent()]
    total_wins, points_history = evaluate(game, agents, FLAGS.num_evaluations)
    stats_plotter(agents, points_history, total_wins)

    # test agent against AIAgent
    agents = [eval_agent, AIAgent()]
    total_wins, points_history = evaluate(game, agents, FLAGS.num_evaluations)
    stats_plotter(agents, points_history, total_wins)


if __name__ == '__main__':

    # parameters
    # ==================================================

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", default=None, help="Provide a trained model path if you want to play against a deep agent", type=str)
    parser.add_argument("--network", default=NetworkTypes.DRQN, choices=[NetworkTypes.DQN, NetworkTypes.DRQN], help="Neural network used for approximating value function")
    parser.add_argument("--num_evaluations", default=30, help="Number of evaluation games against each type of opponent for each test", type=int)

    FLAGS = parser.parse_args()

    main()
