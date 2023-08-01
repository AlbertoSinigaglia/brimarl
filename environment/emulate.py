from brimarl_masked.environment.environment import BriscolaGame, Agent
from typing import List


def play_episode(game: BriscolaGame, agents: List[Agent], train=True):
    """Play an entire game updating the environment at each step.
    rewards_log will contain as key the agent's name and as value a list
    containing all the rewards that the agent has received at each step.
    """
    game.reset()
    for agent in agents:
        agent.reset()

    states = [[] for _ in range(len(agents))]
    actions = [[] for _ in range(len(agents))]
    masks = [[] for _ in range(len(agents))]
    rewards = [[] for _ in range(len(agents))]
    dones = [[] for _ in range(len(agents))]

    while not game.check_end_game():
        for j in range(len(agents)):
            dones[j].extend([0] * len(agents))

        # action step
        players_order = game.get_players_order()

        for player_id in players_order:
            player = game.players[player_id]
            agent = agents[player_id]

            # each player/agent observe the state of the game
            for j, (a, p) in enumerate(zip(agents, game.players)):
                states[j].append(a.state(game, p, player))

            # available_actions = game.get_player_actions(player_id)
            game_action, action, mask = agent.action(game, player)
            actions[player_id].extend([action] * len(agents))
            masks[player_id].extend([mask] * len(agents))

            game.play_step(game_action, player_id)

        # just to add the last player card to the state of the agents that are storing a history of player cards
        # FIXMEEEEE
        for a, p in zip(agents, game.players):
            a.state(game, p, player)

        # update the environment
        turn_rewards = game.get_rewards_from_step()
        game.draw_step()
        for j, reward in turn_rewards.items():
            rewards[j].extend([reward] * len(agents))

    # update for the terminal state
    # we need to add 4 transitions to all 4 players
    for player_id in game.get_players_order():
        for j, (a, p) in enumerate(zip(agents, game.players)):
            states[j].append(a.state(game, p, game.players[player_id]))
    for i in range(len(agents)):
        actions[i].extend([[1]+[0]*39] * len(agents))
        masks[i].extend([[1]+[0]*39] * len(agents))
        rewards[i].extend([0] * len(agents))
        dones[i].extend([1] * len(agents))

    assert len(states[0]) == len(actions[0]) and len(actions[0]) == len(rewards[0]) and len(rewards[0]) == len(dones[0]) \
           and len(rewards[0]) == len(masks[0])

    end_game = game.end_game()
    for agent, player in zip(agents, game.players):
        agent.end_game(game, player)
    if train:
        return states, actions, masks, rewards, dones
    else:
        return end_game