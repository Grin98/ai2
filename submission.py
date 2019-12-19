import math

from environment import Player, GameState, GameAction, get_next_state, SnakeAgentsList
from utils import get_fitness
import numpy as np
from enum import Enum


def manhaten_dist(p1: tuple, p2: tuple):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def heuristic(state: GameState, player_index: int) -> float:
    """
    Computes the heuristic value for the agent with player_index at the given state
    :param state:
    :param player_index: integer. represents the identity of the player. this is the index of the agent's snake in the
    state.snakes array as well.
    :return:
    """
    # Insert your code here...
    snake_length = state.snakes[player_index].length
    if not state.snakes[player_index].alive or len(state.living_agents) <= 1:
        return snake_length

    snakes: SnakeAgentsList = state.snakes
    fruits_locations = state.fruits_locations
    dists = [[manhaten_dist(fruit, snake.head) for fruit in fruits_locations] for snake in snakes]
    return 1 / (1 + min(dists[player_index])) + snake_length


class MinimaxAgent(Player):
    """
    This class implements the Minimax algorithm.
    Since this algorithm needs the game to have defined turns, we will model these turns ourselves.
    Use 'TurnBasedGameState' to wrap the given state at the 'get_action' method.
    hint: use the 'agent_action' property to determine if it's the agents turn or the opponents' turn. You can pass
    'None' value (without quotes) to indicate that your agent haven't picked an action yet.
    """

    class Turn(Enum):
        AGENT_TURN = 'AGENT_TURN'
        OPPONENTS_TURN = 'OPPONENTS_TURN'

    class TurnBasedGameState:
        """
        This class is a wrapper class for a GameState. It holds the action of our agent as well, so we can model turns
        in the game (set agent_action=None to indicate that our agent has yet to pick an action).
        """

        def __init__(self, game_state: GameState, agent_action: GameAction):
            self.game_state = game_state
            self.agent_action = agent_action

        @property
        def turn(self):
            return MinimaxAgent.Turn.AGENT_TURN if self.agent_action is None else MinimaxAgent.Turn.OPPONENTS_TURN

    def get_action(self, state: GameState) -> GameAction:
        # Insert your code here...
        search_depth = 2
        val, action = self.minmax(MinimaxAgent.TurnBasedGameState(state, None), search_depth, self.player_index)

        return action

    """
    return tuple(float, GameAction) where float is the minmax value
    """

    def minmax(self, state: TurnBasedGameState, depth: int, player_index: int):
        if depth == 0 or state.game_state.is_terminal_state or not state.game_state.snakes[player_index].alive:
            return heuristic(state.game_state, player_index), state.agent_action

        if state.turn == MinimaxAgent.Turn.AGENT_TURN:
            succ = [self.minmax(
                MinimaxAgent.TurnBasedGameState(state.game_state, action),
                depth, player_index)
                for action in state.game_state.get_possible_actions(player_index)]

            best_val = -math.inf
            best_action: GameAction
            for s in succ:
                if best_val < s[0]:
                    best_val = s[0]
                    best_action = s[1]
            return best_val, best_action
        else:
            succ = [self.minmax(
                MinimaxAgent.TurnBasedGameState(
                    get_next_state(state.game_state, {**opponents_actions, **{player_index: state.agent_action}}),
                    None),
                depth - 1, player_index)
                for opponents_actions in
                state.game_state.get_possible_actions_dicts_given_action(state.agent_action,
                                                                         player_index=self.player_index)]
            best_val = math.inf
            for s in succ:
                if best_val > s[0]:
                    best_val = s[0]
            return best_val, state.agent_action


class AlphaBetaAgent(MinimaxAgent):
    def get_action(self, state: GameState) -> GameAction:
        # Insert your code here...
        pass


def SAHC_sideways():
    """
    Implement Steepest Ascent Hill Climbing with Sideways Steps Here.
    We give you the freedom to choose an initial state as you wish. You may start with a deterministic state (think of
    examples, what interesting options do you have?), or you may randomly sample one (you may use any distribution you
    like). In any case, write it in your report and describe your choice.

    an outline of the algorithm can be
    1) pick an initial state
    2) perform the search according to the algorithm
    3) print the best moves vector you found.
    :return:
    """
    pass


def local_search():
    """
    Implement your own local search algorithm here.
    We give you the freedom to choose an initial state as you wish. You may start with a deterministic state (think of
    examples, what interesting options do you have?), or you may randomly sample one (you may use any distribution you
    like). In any case, write it in your report and describe your choice.

    an outline of the algorithm can be
    1) pick an initial state/states
    2) perform the search according to the algorithm
    3) print the best moves vector you found.
    :return:
    """
    pass


class TournamentAgent(Player):

    def get_action(self, state: GameState) -> GameAction:
        pass


if __name__ == '__main__':
    SAHC_sideways()
    local_search()
