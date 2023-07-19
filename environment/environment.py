import random
from typing import List

from brimarl_masked.environment.utils import BriscolaLogger






class BriscolaCard:
    """Create a Briscola card with its attributes."""

    def __init__(self):
        self.id = -1  # index of the one-hot encoded card in the deck [0, len(deck) - 1]
        self.name = ""  # name to display
        self.seed = -1  # seed/suit of the card [0, 3]
        self.number = -1  # face value of the card [0, 9]
        self.strength = -1  # card rank during the game [0, 9]
        self.points = -1  # point value of the card [0, 11]

    def __str__(self):
        return self.name


class BriscolaDeck:
    """Initialize a deck of Briscola cards with its attributes."""

    def __init__(self):
        self.deck = None
        self.current_deck = None
        self.end_deck = None
        self.briscola = None
        self.create_decklist()
        self.reset()

    def create_decklist(self):
        """Create all the BriscolaCards and add them to the deck."""
        points = [11, 0, 10, 0, 0, 0, 0, 2, 3, 4]
        strengths = [9, 0, 8, 1, 2, 3, 4, 5, 6, 7]
        seeds = ["Bastoni", "Coppe", "Denari", "Spade"]
        names = [
            "Asso",
            "Due",
            "Tre",
            "Quattro",
            "Cinque",
            "Sei",
            "Sette",
            "Fante",
            "Cavallo",
            "Re",
        ]

        self.deck = []
        card_id = 0
        for s, seed in enumerate(seeds):
            for n, name in enumerate(names):
                card = BriscolaCard()
                card.id = card_id
                card.name = f"{name} di {seed}"
                card.seed = s
                card.number = n
                card.strength = strengths[n]
                card.points = points[n]
                self.deck.append(card)
                card_id += 1

    def reset(self):
        """Prepare the deck for a new game."""
        self.briscola = None
        self.end_deck = False
        self.current_deck = self.deck.copy()
        self.shuffle()

    def shuffle(self):
        """Shuffle the deck."""
        random.shuffle(self.current_deck)

    def place_briscola(self, briscola: BriscolaCard):
        """Set a card as briscola and allow to draw it after the last card of the deck."""
        if self.briscola is not None:
            raise ValueError(
                "Trying BriscolaDeck.place_briscola, but BriscolaDeck.briscola is not None"
            )
        self.briscola = briscola

    def draw_card(self):
        """If the deck is not empty, draw a card, otherwise return the briscola or nothing."""
        if self.current_deck:
            drawn_card = self.current_deck.pop()
        else:
            drawn_card = self.briscola
            self.briscola = None
            self.end_deck = True
        return drawn_card

    def get_deck_size(self):
        """Size of the full deck."""
        return len(self.deck)

    def get_current_deck_size(self):
        """Size of the current deck."""
        current_deck_size = len(self.current_deck)
        current_deck_size += 1 if self.briscola else 0
        return current_deck_size

    def __len__(self):
        return len(self.current_deck)

    def __str__(self):
        str_ = ""
        for el in self.current_deck:
            str_ += el.__str__() + ", "
        return str_


class BriscolaPlayer:
    """Create basic player actions."""

    def __init__(self, _id):
        self.points = None
        self.hand = None
        self.id = _id
        self.reset()

    def reset(self):
        """Reset hand and points when starting a new game."""
        self.hand = []
        self.points = 0

    def draw(self, deck: BriscolaDeck):
        """Try to draw a card from the deck."""
        new_card = deck.draw_card()
        if new_card is not None:
            self.hand.append(new_card)
        if len(self.hand) > 3:
            raise ValueError(
                "Calling BriscolaPlayer.draw caused the player to have more than 3 cards in hand!"
            )

    def play_card(self, hand_index: int):
        """Try to play a card from the hand and return the chosen card or an exception if invalid index."""
        try:
            card = self.hand[hand_index]
            del self.hand[hand_index]
            return card
        except Exception as _:
            raise ValueError("BriscolaPlayer.play_card called with invalid hand_index!")


class BriscolaGame:
    """Create the game environment with all the game stages."""

    def __init__(self, num_players=2, logger=BriscolaLogger(), win_extra_points=100, binary_reward=False):
        self.briscola = None
        self.players_order = None
        self.turn_player = None
        self.players = None
        self.history = None
        self.played_cards = None
        self.num_players = num_players
        self.deck = BriscolaDeck()
        self.logger = logger
        self.counter = 1
        self.screen = None
        self.win_extra_points = win_extra_points
        self.binary_reward = binary_reward

    def clone(self):
        return BriscolaGame(self.num_players, self.logger, self.win_extra_points, binary_reward=self.binary_reward)

    def reset(self):
        """Start a new game."""
        # initialize the deck
        self.deck.reset()
        self.history = []
        self.played_cards = []

        # initialize the players
        self.players = [BriscolaPlayer(i) for i in range(self.num_players)]
        self.turn_player = random.randint(0, self.num_players - 1)
        self.players_order = self.get_players_order()

        # initialize the briscola
        self.briscola = self.deck.draw_card()
        self.deck.place_briscola(self.briscola)
        self.logger.PVP(f"Briscola of the game: {self.briscola.name}.")
        self.logger.PVP(f"-" * 20)

        # initialize players' hands
        for _ in range(0, 3):
            for i in self.players_order:
                self.players[i].draw(self.deck)

        self.counter = 1
        self.screen = None

    def reorder_hand(self, player_id: int):
        """Rearrange the cards in a player's hand from strongest to weakest,
        taking into account the Briscola seed.
        """
        player = self.players[player_id]
        # bubble sort algorithm using scoring() as a comparator
        for passnum in range(len(player.hand) - 1, 0, -1):
            for i in range(passnum):
                if scoring(
                        self.briscola.seed,
                        player.hand[i],
                        player.hand[i + 1],
                        keep_order=False,
                ):
                    temp = player.hand[i]
                    player.hand[i] = player.hand[i + 1]
                    player.hand[i + 1] = temp

    def get_player_actions(self, player_id: int):
        """Get the list of available actions for a player."""
        player = self.players[player_id]
        return list(range(len(player.hand)))

    def get_players_order(self):
        """Compute the clockwise players order starting from the current turn player."""
        players_order = [
            i % self.num_players
            for i in range(self.turn_player, self.turn_player + self.num_players)
        ]
        return players_order

    def draw_step(self):
        """Each player, in order, tries to draw a card from the deck."""
        self.logger.PVP(f"\n----------- NEW TURN -----------[{self.counter}]")
        # clear the table for the play_step
        self.played_cards = []
        # draw the cards in order
        for player_id in self.players_order:
            player = self.players[player_id]
            player.draw(self.deck)

    def play_step(self, action: int, player_id: int):
        """A player executes a chosen action."""
        player = self.players[player_id]

        self.logger.DEBUG(
            f"Player {player_id} hand: {[card.name for card in player.hand]}."
        )
        self.logger.DEBUG(f"Player {player_id} chose action {action}.")

        card = player.play_card(action)
        self.logger.PVP(f"Player {player_id} played {card.name}.")

        self.played_cards.append(card)
        self.history.append(card)

    def evaluate_step(self):
        """Look at the cards played and decide which player won the hand."""
        ordered_winner_id, strongest_card = get_strongest_card(
            self.briscola.seed, self.played_cards
        )
        winner_player_id = self.players_order[ordered_winner_id]

        points = sum([card.points for card in self.played_cards])
        winner_player = self.players[winner_player_id]

        self.update_game(winner_player, points)
        for player in self.players:
            if player.id in [winner_player_id, self.get_teammate(winner_player_id)]:
                self.logger.PVP(f"Player {player.id} wins the turn with {points} points with {strongest_card.name}.")
            else:
                self.logger.PVP(f"Player {player.id} loses the turn with {points} points with {strongest_card.name}.")
        # self.logger.PVP(
        #     f"Player {winner_player_id} wins {points} points with {strongest_card.name}."
        # )

        return winner_player_id, points

    def get_rewards_from_step(self):
        """Compute the reward for each player based on the cards just played.
        Note that this is a reward in RL terms, not actual game points.
        """
        winner_player_id, points = self.evaluate_step()

        rewards = {}
        # check if any won during this turn
        win_bonus = self.win_extra_points if (self.players[winner_player_id].points - points) <= 60 and \
                                             (self.players[winner_player_id].points > 60) else 0
        # add the bonus to the reward of this step
        if self.binary_reward:
            points = 1 if win_bonus else 0
        else:
            points = points + (self.win_extra_points if win_bonus else 0)

        for player_id in self.get_players_order():
            reward = points if player_id in [winner_player_id, self.get_teammate(winner_player_id)] else -points
            rewards[player_id] = reward

        return rewards

    def check_end_game(self):
        """Check if the game is over."""
        end_deck = self.deck.end_deck
        player_has_cards = False
        for player in self.players:
            if player.hand:
                player_has_cards = True
                break

        return end_deck and not player_has_cards

    def get_winner(self):
        """Return the player with the most points and the winning amount."""
        winner_player_id = -1
        winner_points = -1

        for player in self.players:
            if player.points > winner_points:
                winner_player_id = player.id
                winner_points = player.points

        return winner_player_id, winner_points

    def end_game(self):
        """End the game and return the winner."""
        if not self.check_end_game():
            raise ValueError(
                "Calling BriscolaGame.end_game when the game has not ended!"
            )

        winner_player_id, winner_points = self.get_winner()
        for player in self.players:
            if player.id in [winner_player_id, self.get_teammate(winner_player_id)]:
                self.logger.PVP(f"Player {player.id} wins with {player.points} points!!")
            else:
                self.logger.PVP(f"Player {player.id} loses with {player.points} points!!")

        return winner_player_id, winner_points

    # EDIT
    def get_teammate(self, player_id):
        if self.num_players <= 2:
            return None
        elif player_id >= self.num_players / 2:
            return player_id - (self.num_players // 2)
        elif player_id < self.num_players / 2:
            return player_id + (self.num_players // 2)

    def update_game(self, winner_player: BriscolaPlayer, points: int):
        """Update the scores and the order based on who won the previous hand."""
        winner_player_id = winner_player.id
        winner_player.points += points

        # EDIT
        if self.num_players > 2:
            self.players[self.get_teammate(winner_player.id)].points += points

        self.turn_player = winner_player_id
        self.players_order = self.get_players_order()
        self.counter += 1





def get_weakest_card(briscola_seed, cards):
    """Return the weakest card in the given set,
    taking into account the Briscola seed.
    """
    ordered_loser_id = 0
    weakest_card = cards[0]

    for ordered_id, card in enumerate(cards[1:]):
        ordered_id += 1  # adjustment since we are starting from the first element
        pair_winner = scoring(briscola_seed, weakest_card, card, keep_order=False)
        if pair_winner == 0:
            ordered_loser_id = ordered_id
            weakest_card = card

    return ordered_loser_id, weakest_card


def scoring(briscola_seed, card_0, card_1, keep_order=True):
    """Compare a pair of cards and decide which one wins.
    The keep_order argument indicates whether the first card played has a priority.
    """
    # only one card is of the briscola seed
    if briscola_seed != card_0.seed and briscola_seed == card_1.seed:
        winner = 1
    elif briscola_seed == card_0.seed and briscola_seed != card_1.seed:
        winner = 0
    # same seed, briscola or not
    elif card_0.seed == card_1.seed:
        winner = 1 if card_1.strength > card_0.strength else 0
    # if of different seeds and none of them is briscola, the first one wins
    else:
        winner = 0 if keep_order or card_0.strength > card_1.strength else 1

    return winner

def get_strongest_card(briscola_seed, cards):
    """Return the strongest card in the given set,
    taking into account the Briscola seed.
    """
    ordered_winner_id = 0
    strongest_card = cards[0]

    for ordered_id, card in enumerate(cards[1:]):
        ordered_id += 1  # adjustment since we are starting from the first element
        pair_winner = scoring(briscola_seed, strongest_card, card)
        if pair_winner == 1:
            ordered_winner_id = ordered_id
            strongest_card = card

    return ordered_winner_id, strongest_card


class Agent:
    def __init__(self, name):
        self.name = name

    def reset(self):
        raise NotImplementedError(f"method reset not implemented for agent {self.name}")

    def state(self, env: BriscolaGame, player: BriscolaPlayer, current_player: BriscolaPlayer):
        raise NotImplementedError(f"method state not implemented for agent {self.name}")

    def action(self, game: BriscolaGame, player: BriscolaPlayer):
        raise NotImplementedError(f"method select_action not implemented for agent {self.name}")

    def save_model(self, path: str):
        raise NotImplementedError(f"method save_model not implemented for agent {self.name}")

    def load_model(self, path: str):
        raise NotImplementedError(f"method load_model not implemented for agent {self.name}")

    def clone(self, training=False):
        raise NotImplementedError(f"method clone not implemented for agent {self.name}")
