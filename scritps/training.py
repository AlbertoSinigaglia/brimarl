import os
import shutil
import random
import numpy as np
from brimarl_masked.environment.emulate import play_episode
from brimarl_masked.agents.random_agent import RandomAgent
from brimarl_masked.agents.scripted_ai_agent import ScriptedAIAgent
from brimarl_masked.scritps.evaluate import evaluate
from brimarl_masked.algorithms.algorithm import Algorithm
from brimarl_masked.environment.environment import Agent, BriscolaGame
import json


def print_win_rate(epoch, name, win_rate):
    print(f"Episode: {epoch} - Win rate {name}: {win_rate}")


class Training:
    def __init__(self, num_epochs: int, num_game_per_epoch: int, game: BriscolaGame, agent: Agent,
                 agent_algorithm: Algorithm, evaluate_every: int, num_evaluations: int, from_savings: bool = True,
                 save_dir: str = None):
        self.num_epochs = num_epochs
        self.num_game_per_epoch = num_game_per_epoch
        self.game = game
        self.agent_algorithm = agent_algorithm
        self.agent = agent
        self.evaluate_every = evaluate_every
        self.num_evaluations = num_evaluations
        self.from_savings = from_savings
        self.save_dir = save_dir or f"../models_savings/{game.num_players}/{type(self.agent).__name__}"

    def train(self):
        win_rates = []
        self.load_models()
        self.prepare_saving_folder()

        loss = None
        for epoch in range(1, self.num_epochs + 1):
            if epoch % 50 == 0:
                print(f"\nEpoch {epoch}")
            else:
                print('.', end='')

            self.on_new_epoch(epoch)
            for i in range(self.num_game_per_epoch):
                self.simulate_and_store()
            loss = self.agent_algorithm.learn(self.agent) or loss

            if epoch % self.evaluate_every == 0:
                win_rates.append(self.evaluate_step(epoch))
                print(f"Loss : {loss:.2f}")
                self.save_models_and_win_rate(win_rates)

        return win_rates

    def save_models_and_win_rate(self, win_rates):
        print("Saving... ")
        self.agent.save_model(path=self.save_dir + "/latest")
        self.agent.save_model(path=self.save_dir + f"/history/{len(win_rates)}")
        with open(self.save_dir + "/win_rates.txt", "w+") as file:
            json.dump(win_rates, file)

    def evaluate_step(self, epoch):
        print('-' * 30)
        print(f"Epoch {epoch} / {self.num_epochs}")
        return self.evaluate(epoch)

    def evaluation_score(self, players):
        total_wins, points_history = evaluate(self.game, players, self.num_evaluations)
        return total_wins[0] / self.num_evaluations

    def prepare_saving_folder(self):
        print("Deleting old models")
        if os.path.isdir(self.save_dir):
            shutil.rmtree(self.save_dir)
        print("Preparing folder to store models and win-rates")
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
            print("Created folder to store")
        print("Done")

    def load_models(self):
        if self.from_savings:
            print("Loading models from memory")
            try:
                self.agent.load_model(path=self.save_dir + "/latest")
                print("Loaded from memory")
            except Exception as e:
                print(f"\n!!!! Unable to load or find files: {e.args[1]} !!!\n")
        else:
            print("Skipping loading models from memory")

    def on_new_epoch(self, epoch):
        pass

    def simulate_and_store(self):
        raise NotImplementedError

    def evaluate(self, epoch):
        raise NotImplementedError


class TrainingScripted(Training):
    def evaluate(self, epoch):
        scripted_game_players = [self.agent.clone(training=False), ScriptedAIAgent()] if self.game.num_players == 2 else \
            [self.agent.clone(training=False), ScriptedAIAgent(), self.agent.clone(training=False), ScriptedAIAgent()]
        random_game_players = [self.agent.clone(training=False), RandomAgent()] if self.game.num_players == 2 else \
            [self.agent.clone(training=False), RandomAgent(), self.agent.clone(training=False), RandomAgent()]
        win_rate_script = self.evaluation_score(scripted_game_players)
        print_win_rate(epoch, "ScriptedAIAgent", win_rate_script)
        win_rate_random = self.evaluation_score(random_game_players)
        print_win_rate(epoch, "RandomAgent", win_rate_random)
        return win_rate_script, win_rate_random

    def simulate_and_store(self):
        if self.game.num_players == 2:
            states, actions, masks, rewards, dones = play_episode(self.game, [self.agent.clone(), ScriptedAIAgent()],
                                                                  train=True)
            self.agent_algorithm.store_game(states[0], actions[0], masks[0], rewards[0], dones[0])
        else:
            states, actions, masks, rewards, dones = play_episode(
                self.game,
                [self.agent.clone(), ScriptedAIAgent(), self.agent.clone(), ScriptedAIAgent()],
                train=True
            )
            self.agent_algorithm.store_game(states[0], actions[0], masks[0], rewards[0], dones[0])
            self.agent_algorithm.store_game(states[2], actions[2], masks[2], rewards[2], dones[2])


class TrainingSelfPlay(Training):
    def evaluate(self, epoch):
        if self.game.num_players == 2:
            win_rate_script = self.evaluation_score([self.agent.clone(training=False), ScriptedAIAgent()])
            print_win_rate(epoch, "ScriptedAIAgent", win_rate_script)
            win_rate_random = self.evaluation_score([self.agent.clone(training=False), RandomAgent()])
            print_win_rate(epoch, "RandomAgent", win_rate_random)
            return win_rate_script, win_rate_random
        else:
            win_rate_random = self.evaluation_score(
                [self.agent.clone(training=False), RandomAgent(), self.agent.clone(training=False), RandomAgent()]
            )
            print_win_rate(epoch, "RandomAgent", win_rate_random)
            return win_rate_random

    def simulate_and_store(self):
        if self.game.num_players == 2:
            states, actions, masks, rewards, dones = play_episode(self.game, [self.agent.clone(), self.agent.clone()],
                                                                  train=True)
            self.agent_algorithm.store_game(states[0], actions[0], masks[0], rewards[0], dones[0])
            self.agent_algorithm.store_game(states[1], actions[1], masks[1], rewards[1], dones[1])
        else:
            states, actions, masks, rewards, dones = play_episode(
                self.game,
                [self.agent.clone(), self.agent.clone(), self.agent.clone(), self.agent.clone()],
                train=True
            )
            self.agent_algorithm.store_game(states[0], actions[0], masks[0], rewards[0], dones[0])
            self.agent_algorithm.store_game(states[1], actions[1], masks[1], rewards[1], dones[1])
            self.agent_algorithm.store_game(states[2], actions[2], masks[2], rewards[2], dones[2])
            self.agent_algorithm.store_game(states[3], actions[3], masks[3], rewards[3], dones[3])

class TrainingSelfPlayMAPPO(TrainingSelfPlay):
    def simulate_and_store(self):
        states, actions, masks, rewards, dones = play_episode(
            self.game,
            [self.agent.clone(), self.agent.clone(), self.agent.clone(), self.agent.clone()],
            train=True
        )
        self.agent_algorithm.store_game((states[0], states[2]), actions[0], masks[0], rewards[0], dones[0])
        self.agent_algorithm.store_game((states[1], states[3]), actions[1], masks[1], rewards[1], dones[1])
        self.agent_algorithm.store_game((states[2], states[0]), actions[2], masks[2], rewards[2], dones[2])
        self.agent_algorithm.store_game((states[3], states[1]), actions[3], masks[3], rewards[3], dones[3])


class TrainingFictitiousSelfPlay(Training):
    def __init__(self, num_epochs: int, num_game_per_epoch: int, game: BriscolaGame, agent: Agent,
                 agent_algorithm: Algorithm, evaluate_every: int, num_evaluations: int, from_savings: bool = False,
                 save_dir: str = None, store_every: int = 50, max_old_agents: int = 20,
                 current_agent_prob: float = 0.5):
        super().__init__(num_epochs, num_game_per_epoch, game, agent, agent_algorithm, evaluate_every, num_evaluations,
                         from_savings, save_dir)
        self.store_every = store_every
        self.max_old_agents = max_old_agents
        self.current_agent_prob = current_agent_prob
        self.old_agents = [agent.clone()]

    def on_new_epoch(self, epoch):
        if epoch % self.store_every == 0:
            if len(self.old_agents) >= self.max_old_agents:
                del self.old_agents[random.randrange(0, len(self.old_agents))]
            self.old_agents.append(self.agent.clone())

    def simulate_and_store(self):
        index = np.random.random() < self.current_agent_prob
        enemy = {
            0: lambda: np.random.choice(self.old_agents, 1)[0],
            1: lambda: self.agent.clone(training=False)
        }[index]()

        if self.game.num_players == 2:
            states, actions, masks, rewards, dones = play_episode(self.game, [self.agent.clone(), enemy.clone()],
                                                                  train=True)
            self.agent_algorithm.store_game(states[0], actions[0], masks[0], rewards[0], dones[0])
            if index == 1:
                self.agent_algorithm.store_game(states[1], actions[1], masks[1], rewards[1], dones[1])
        else:
            states, actions, masks, rewards, dones = play_episode(
                self.game,
                [self.agent.clone(), enemy.clone(), self.agent.clone(), enemy.clone()],
                train=True
            )
            self.agent_algorithm.store_game(states[0], actions[0], masks[0], rewards[0], dones[0])
            self.agent_algorithm.store_game(states[2], actions[2], masks[2], rewards[2], dones[2])
            if index == 1:
                self.agent_algorithm.store_game(states[1], actions[1], masks[1], rewards[1], dones[1])
                self.agent_algorithm.store_game(states[3], actions[3], masks[3], rewards[3], dones[3])

    def evaluate(self, epoch):
        if self.game.num_players == 2:
            win_rate_script = self.evaluation_score([self.agent.clone(training=False), ScriptedAIAgent()])
            print_win_rate(epoch, "ScriptedAIAgent", win_rate_script)
            win_rate_random = self.evaluation_score([self.agent.clone(training=False), RandomAgent()])
            print_win_rate(epoch, "RandomAgent", win_rate_random)
            return win_rate_script, win_rate_random
        else:
            win_rate_random = self.evaluation_score(
                [self.agent.clone(training=False), RandomAgent(), self.agent.clone(training=False), RandomAgent()]
            )
            print_win_rate(epoch, "RandomAgent", win_rate_random)
            return win_rate_random
