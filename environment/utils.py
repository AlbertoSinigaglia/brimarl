class BriscolaLogger:
    """Adjust verbosity on four levels as needed."""

    class LoggerLevels:
        DEBUG = 0
        PVP = 1
        TRAIN = 2
        TEST = 3
        NONE = 4

    def __init__(self, verbosity=3):
        self.TEST = None
        self.TRAIN = None
        self.PVP = None
        self.DEBUG = None
        self.verbosity = None
        self.configure_logger(verbosity)

    def configure_logger(self, verbosity):

        self.verbosity = verbosity

        if self.verbosity > self.LoggerLevels.DEBUG:
            self.DEBUG = lambda *args: None
        else:
            self.DEBUG = print

        if self.verbosity > self.LoggerLevels.PVP:
            self.PVP = lambda *args: None
        else:
            self.PVP = print

        if self.verbosity > self.LoggerLevels.TRAIN:
            self.TRAIN = lambda *args: None
        else:
            self.TRAIN = print

        self.TEST = print
        if verbosity == self.LoggerLevels.NONE:
            self.TRAIN = lambda *args: None
            self.DEBUG = lambda *args: None
            self.TEST = lambda *args: None
            self.PVP = lambda *args: None


class CardsEncoding:
    HOT_ON_DECK = 'hot_on_deck'
    HOT_ON_NUM_SEED = 'hot_on_num_seed'


class CardsOrder:
    APPEND = 'append'
    REPLACE = 'replace'
    VALUE = 'value'


class NetworkTypes:
    DQN = 'dqn'
    DRQN = 'drqn'
    AC = 'actor_critic'


class PlayerState:
    HAND_PLAYED_BRISCOLA = 'hand_played_briscola'
    HAND_PLAYED_BRISCOLASEED = 'hand_played_briscolaseed'
    HAND_PLAYED_BRISCOLA_HISTORY = 'hand_played_briscola_history'