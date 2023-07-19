import math
import os
import argparse
import pickle
import time
import random
import numpy as np
import brimarl_masked.environment.environment as brisc
from brimarl_masked.agents.q_agent import DeepQAgent
from brimarl_masked.agents.recurrent_q_agent import RecurrentDeepQAgent
from brimarl_masked.algorithms.dqn import QLearningAlgorithm
from brimarl_masked.environment.emulate import play_episode
import matplotlib.pyplot as plt
from brimarl_masked.agents.random_agent import RandomAgent
from brimarl_masked.agents.dumb_agent import DumbAgent
from brimarl_masked.agents.scripted_ai_agent import ScriptedAIAgent
from brimarl_masked.agents.ac_agent import ACAgent
from brimarl_masked.agents.ac_q_agent import ACQAgent
from brimarl_masked.scritps.evaluate import evaluate
from brimarl_masked.environment.utils import BriscolaLogger
import tensorflow as tf
from brimarl_masked.algorithms.algorithm import Algorithm
from brimarl_masked.environment.environment import Agent, BriscolaGame
from brimarl_masked.algorithms.a2c import A2CAlgorithm
from brimarl_masked.algorithms.a2c_q import A2CQAlgorithm
from brimarl_masked.algorithms.a2c_gae import A2CGAEAlgorithm
from brimarl_masked.algorithms.ppo import PPOAlgorithm
from brimarl_masked.algorithms.a2c_clipped import A2CClippedAlgorithm
from brimarl_masked.algorithms.ppo_gae import PPOGAEAlgorithm
from brimarl_masked.algorithms.rdqn import RQLearningAlgorithm
import json
from brimarl_masked.scritps.parallel_simulation import GamePool


def train(
        game: BriscolaGame,
        agent: Agent,
        exploiter: Agent,
        league: Agent,
        agent_algorithm: Algorithm,
        exploiter_algorithm: Algorithm,
        league_algorithm: Algorithm,
        num_epochs: int,
        evaluate_every: int,
        num_evaluations: int,
        num_old_versions_to_store: int = 20,
        num_game_per_epoch: int = 2,
        from_savings=False
):
    """
    The agent is trained for num_epochs number of episodes following an
    epsilon-greedy policy. Every evaluate_every number of episodes the agent
    is evaluated by playing num_evaluations number of games.
    The win-rate is obtained from these evaluations is used to select the best
    model and its weights are saved.
    """

    current_win_rate = 0
    win_rates = []

    old_agents = []
    old_exploiters = []
    old_leagues = []

    # prepare folder to store model
    save_dir = f"../models_savings/{type(agent).__name__}"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if from_savings:
        try:
            with open(save_dir+"/win_rates.txt", "r") as file:
                win_rates = json.load(file)
                if len(win_rates) > 0:
                    current_win_rate = win_rates[-1]
            agent.load_model(path=save_dir+"/latest")
            exploiter.load_model(path=save_dir+"/latest")
            league.load_model(path=save_dir+"/latest")
            print("Loaded from memory")
        except Exception as e:
            print(f"Unable to load or find files: {e.args[1]}")

    for epoch in range(0, num_epochs + 1):
        print(f"Epoch {epoch+1} / {num_epochs}")
        old_agents.append(agent.clone())
        old_exploiters.append(exploiter.clone())
        old_leagues.append(league.clone())
        if len(old_agents) > num_old_versions_to_store: del old_agents[random.randrange(0, len(old_agents))]
        if len(old_exploiters) > num_old_versions_to_store: del old_exploiters[random.randrange(0, len(old_exploiters))]
        if len(old_leagues) > num_old_versions_to_store: del old_leagues[random.randrange(0, len(old_leagues))]
        start_train = time.time()
        print(f"Playing... ")
        # main agents training, on old agents, exploiters and leagues
        for i in range(num_game_per_epoch):
            enemy = random.choice({
                  0: old_exploiters,
                  1: old_leagues,
                  2: old_agents,
                  3: [agent],
                  4: [exploiter],
                  5: [league],
              }[random.randint(0, 5)])
            states, actions, rewards, dones = play_episode(
                game,
                [agent.clone(training=True), enemy.clone(), agent.clone(training=True), enemy.clone()],
            )
            agent_algorithm.store_game(states[0], actions[0], rewards[0], dones[0])
            agent_algorithm.store_game(states[2], actions[2], rewards[2], dones[2])
        # exploiters training
        for i in range(num_game_per_epoch):
            states, actions, rewards, dones = play_episode(
                game,
                [agent.clone(), exploiter.clone(training=True), agent.clone(), exploiter.clone(training=True)],
            )
            agent_algorithm.store_game(states[0], actions[0], rewards[0], dones[0])
            agent_algorithm.store_game(states[2], actions[2], rewards[2], dones[2])
            exploiter_algorithm.store_game(states[1], actions[1], rewards[1], dones[1])
            exploiter_algorithm.store_game(states[3], actions[3], rewards[3], dones[3])
        # leagues training
        for i in range(num_game_per_epoch):
            enemy = random.choice({
                  0: old_exploiters,
                  1: old_leagues,
                  2: old_agents,
              }[random.randint(0, 2)])
            states, actions, rewards, dones = play_episode(
                game,
                [enemy.clone(), league.clone(training=True), enemy.clone(), league.clone(training=True)],
            )
            league_algorithm.store_game(states[1], actions[1], rewards[1], dones[1])
            league_algorithm.store_game(states[3], actions[3], rewards[3], dones[3])

        print(f" {(time.time() - start_train):.2f}s - ")

        print(f"Learning... ")
        start_learning = time.time()
        # once experience has been acquired, learn
        for player, algorithm in [[agent, agent_algorithm], [exploiter, exploiter_algorithm], [league, league_algorithm]]:
            algorithm.learn(player)

        print(f"{(time.time() - start_learning):.2f}s - ")
        if epoch % evaluate_every == 0:
            print(f"Evaluating... ")
            total_wins, points_history = evaluate(
                game,
                [agent.clone(training=False), RandomAgent(), agent.clone(training=False), RandomAgent()],
                num_evaluations
            )

            current_win_rate = total_wins[0] / num_evaluations
            win_rates.append(current_win_rate)

            print("Saving... ")
            agent.save_model(path=save_dir+"/latest")
            exploiter.save_model(path=save_dir+"/latest")
            league.save_model(path=save_dir+"/latest")
            agent.save_model(path=save_dir+f"/history/{len(win_rates)}")
            exploiter.save_model(path=save_dir+f"/history/{len(win_rates)}")
            league.save_model(path=save_dir+f"/history/{len(win_rates)}")
            with open(save_dir+"/win_rates.txt", "w+") as file:
                json.dump(win_rates, file)
        print()
        print(f"Episode: {epoch} - Win rate: {current_win_rate}")
        if isinstance(agent, RecurrentDeepQAgent):
            print(agent.epsilon)


    return max(win_rates)
def train_simple(
        game: BriscolaGame,
        agent: Agent,
        agent_algorithm: Algorithm,
        num_epochs: int,
        evaluate_every: int,
        num_evaluations: int,
        num_old_versions_to_store: int = 20,
        num_game_per_epoch: int = 2,
        from_savings=False
):
    """
    The agent is trained for num_epochs number of episodes following an
    epsilon-greedy policy. Every evaluate_every number of episodes the agent
    is evaluated by playing num_evaluations number of games.
    The win-rate is obtained from these evaluations is used to select the best
    model and its weights are saved.
    """

    current_win_rate = 0
    win_rates = []

    old_agents = []

    # prepare folder to store model
    save_dir = f"../models_savings/{type(agent).__name__}"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if from_savings:
        try:
            with open(save_dir+"/win_rates.txt", "r") as file:
                win_rates = json.load(file)
                if len(win_rates) > 0:
                    current_win_rate = win_rates[-1]
            agent.load_model(path=save_dir+"/latest")
            print("Loaded from memory")
        except Exception as e:
            print(f"Unable to load or find files: {e.args[1]}")

    for epoch in range(0, num_epochs + 1):
        print(f"Epoch {epoch+1} / {num_epochs}")
        old_agents.append(agent.clone())
        if len(old_agents) > num_old_versions_to_store: del old_agents[random.randrange(0, len(old_agents))]
        start_train = time.time()
        print(f"Playing... ")
        for i in range(num_game_per_epoch):
            enemy = random.choice({
                  0: old_agents,
                  1: [agent],
              }[random.randint(0, 1)])
            states, actions, rewards, dones = play_episode(
                game, [agent.clone(training=True), enemy.clone(), agent.clone(training=True), enemy.clone()],
            )
            agent_algorithm.store_game(states[0], actions[0], rewards[0], dones[0])
            agent_algorithm.store_game(states[2], actions[2], rewards[2], dones[2])
        print(f"{(time.time() - start_train):.2f}s - ")

        print(f"Learning... ")
        start_learning = time.time()
        # once experience has been acquired, learn
        agent_algorithm.learn(agent)

        print(f"{(time.time() - start_learning):.2f}s - ")
        if epoch % evaluate_every == 0:
            print(f"Evaluating... ")
            total_wins, points_history = evaluate(
                game,
                [agent.clone(training=False), RandomAgent(), agent.clone(training=False), RandomAgent()],
                num_evaluations
            )

            current_win_rate = total_wins[0] / num_evaluations
            win_rates.append(current_win_rate)

            print("Saving... ")
            agent.save_model(path=save_dir+"/latest")
            agent.save_model(path=save_dir+f"/history/{len(win_rates)}")
            with open(save_dir+"/win_rates.txt", "w+") as file:
                json.dump(win_rates, file)
        print()
        print(f"Episode: {epoch} - Win rate: {current_win_rate}")

    return max(win_rates)




def train_against_scripted(
        game: BriscolaGame,
        agent: Agent,
        agent_algorithm: Algorithm,
        num_epochs: int,
        evaluate_every: int,
        num_evaluations: int,
        num_game_per_epoch: int = 2,
        from_savings=False
):
    win_rates = []
    loss = None
    for epoch in range(1, num_epochs + 1):
        if epoch % 50 == 0: print(f"\nEpoch {epoch}")
        else: print('.', end='')
        for i in range(num_game_per_epoch):
            states, actions, masks, rewards, dones = play_episode(game, [agent, ScriptedAIAgent()], train=True)
            agent_algorithm.store_game(states[0], actions[0], masks[0], rewards[0], dones[0])
        loss = agent_algorithm.learn(agent) or loss
        if epoch % evaluate_every == 0:
            print('-'*30)
            print(f"Epoch {epoch} / {num_epochs}")
            print(f"Loss : {loss:.2f}")
            total_wins, points_history = evaluate(game, [agent.clone(training=False), ScriptedAIAgent()], num_evaluations)
            current_win_rate = total_wins[0] / num_evaluations
            win_rates.append(current_win_rate)
            print(f"Episode: {epoch} - Win rate Scripted: {current_win_rate}")# - {agent.epsilon}")
            total_wins, points_history = evaluate(game, [agent.clone(training=False), RandomAgent()], num_evaluations)
            current_win_rate = total_wins[0] / num_evaluations
            win_rates.append(current_win_rate)
            print(f"Episode: {epoch} - Win rate Random: {current_win_rate}")# - {agent.epsilon}")
    return max(win_rates)


def train_against_self(
        game: BriscolaGame,
        agent: Agent,
        agent_algorithm: Algorithm,
        num_epochs: int,
        evaluate_every: int,
        num_evaluations: int,
        num_game_per_epoch: int = 2,
        from_savings=False
):
    win_rates = []

    loss = None
    for epoch in range(1, num_epochs + 1):
        if epoch % 50 == 0: print(f"\nEpoch {epoch}")
        else: print('.', end='')
        for i in range(num_game_per_epoch):
            states, actions, masks, rewards, dones = play_episode(game, [agent.clone(training=True), agent.clone(training=True)], train=True)
            agent_algorithm.store_game(states[0], actions[0], masks[0], rewards[0], dones[0])
            agent_algorithm.store_game(states[1], actions[1], masks[1], rewards[1], dones[1])
        loss = agent_algorithm.learn(agent) or loss
        if epoch % evaluate_every == 0:
            print('-'*30)
            print(f"Epoch {epoch} / {num_epochs}")
            print(f"Loss : {loss:.2f}")
            total_wins, points_history = evaluate(game, [agent.clone(training=False), ScriptedAIAgent()], num_evaluations)
            current_win_rate = total_wins[0] / num_evaluations
            win_rates.append(current_win_rate)
            print(f"Episode: {epoch} - Win rate Scripted: {current_win_rate}")# - {agent.epsilon}")
            total_wins, points_history = evaluate(game, [agent.clone(training=False), RandomAgent()], num_evaluations)
            current_win_rate = total_wins[0] / num_evaluations
            win_rates.append(current_win_rate)
            print(f"Episode: {epoch} - Win rate Random: {current_win_rate}")# - {agent.epsilon}")
    return max(win_rates)


def train_against_self_fict(
        game: BriscolaGame,
        agent: Agent,
        agent_algorithm: Algorithm,
        num_epochs: int,
        evaluate_every: int,
        num_evaluations: int,
        num_game_per_epoch: int = 2,
        max_old_agents=20,
        append_every=25,
        from_savings=False
):
    win_rates = []

    loss = None
    old_agents = [agent.clone(training=False)]

    for epoch in range(1, num_epochs + 1):
        if epoch % 50 == 0: print(f"\nEpoch {epoch}")
        else: print('.', end='')

        if epoch % append_every == 0:
            old_agents.append(agent.clone(training=False))
            if len(old_agents) > max_old_agents:
                del old_agents[random.randrange(0, len(old_agents))]

        for i in range(num_game_per_epoch):
            index = np.random.random() < 0.3
            enemy = {
                0: lambda: np.random.choice(old_agents,1)[0],
                1: lambda: agent.clone(training=False)
            }[index]()
            states, actions, masks, rewards, dones = play_episode(game, [agent.clone(training=True), enemy], train=True)
            agent_algorithm.store_game(states[0], actions[0], masks[0], rewards[0], dones[0])
            if index == 1:
                agent_algorithm.store_game(states[1], actions[1], masks[1], rewards[1], dones[1])

        loss = agent_algorithm.learn(agent) or loss
        if epoch % evaluate_every == 0:
            print('-'*30)
            print(f"Epoch {epoch} / {num_epochs}")
            if game.num_players == 2:
                print(f"Episode: {epoch} - Win rate Scripted: {evaluation_scripted(game, agent, num_evaluations)}")
            print(f"Episode: {epoch} - Win rate Random: {evaluation_random(game, agent, num_evaluations)}")# - {agent.epsilon}")

    return max(win_rates)


def evaluation_scripted(game, player, num_evaluations):
    players = [player.clone(training=False), ScriptedAIAgent()] if game.num_players == 2 else \
        [player.clone(training=False), ScriptedAIAgent(), player.clone(training=False), ScriptedAIAgent()]
    total_wins, points_history = evaluate(game, players, num_evaluations)
    return total_wins[0] / num_evaluations


def evaluation_random(game, player, num_evaluations):
    players = [player.clone(training=False), RandomAgent()] if game.num_players == 2 else \
        [player.clone(training=False), RandomAgent(), player.clone(training=False), RandomAgent()]
    total_wins, points_history = evaluate(game, players, num_evaluations)
    return total_wins[0] / num_evaluations
