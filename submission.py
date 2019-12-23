import math
import time

from agents import RandomPlayer
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
    if not state.snakes[player_index].alive:
        return 0

    snakes = state.living_agents
    fruits_locations = state.fruits_locations
    num_total_friuts = len(fruits_locations) + sum([state.snakes[snake].length-1 for snake in snakes])

    # # Invincible mode activate!!!
    # if len(fruits_locations) == 0 or (state.snakes[player_index].length > num_total_friuts/2 and state.current_winner.player_index == player_index):
    #     return snake_length

    agent_dists = [manhaten_dist(fruit, state.snakes[player_index].head) for fruit in fruits_locations]
    if not agent_dists:  # no fruits
        return snake_length
    if len(snakes) <= 1:  # no opponents
        return 1 / (1 + min(agent_dists)) + snake_length

    opponents_dists = [min([manhaten_dist(fruit, state.snakes[snake].head) for snake in snakes if snake != player_index]) for fruit in fruits_locations]
    attainable = [agent_dists[i] for i in range(len(fruits_locations)) if agent_dists[i] < opponents_dists[i]]
    if not attainable:  # give the lowest value possible to the fruit
        return 1 / (state.board_size.height + state.board_size.width + min(agent_dists)) + snake_length

    return 1 / (1 + min(attainable)) + snake_length


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
        _, action = self.minmax(MinimaxAgent.TurnBasedGameState(state, None), search_depth, self.player_index)

        return action

    """
    return tuple(float, GameAction) where float is the minmax value
    """

    def minmax(self, state: TurnBasedGameState, depth: int, player_index: int):
        if depth == 0 or state.game_state.is_terminal_state or not state.game_state.snakes[player_index].alive:
            return heuristic(state.game_state, player_index), state.agent_action

        if state.turn == MinimaxAgent.Turn.AGENT_TURN:
            best_val = -math.inf
            best_action: GameAction
            for action in state.game_state.get_possible_actions(player_index):
                val, action = self.minmax(MinimaxAgent.TurnBasedGameState(state.game_state, action), depth, player_index)
                if best_val < val:
                    best_val = val
                    best_action = action
            return best_val, best_action
        else:
            best_val = math.inf
            for opponents_actions in state.game_state.get_possible_actions_dicts_given_action(state.agent_action, player_index=self.player_index):
                val, _ = self.minmax(MinimaxAgent.TurnBasedGameState(get_next_state(state.game_state, {**opponents_actions, **{player_index: state.agent_action}}), None), depth - 1, player_index)
                if best_val > val:
                    best_val = val
            return best_val, state.agent_action


class AlphaBetaAgent(MinimaxAgent):
    def get_action(self, state: GameState) -> GameAction:
        # Insert your code here...
        search_depth = 2
        _, action = self.abminmax(MinimaxAgent.TurnBasedGameState(state, None), search_depth, self.player_index, -math.inf, math.inf)
        return action

    def abminmax(self, state: MinimaxAgent.TurnBasedGameState, depth: int, player_index: int, alpha: float, beta: float):
        if depth == 0 or state.game_state.is_terminal_state or not state.game_state.snakes[player_index].alive:
            return heuristic(state.game_state, player_index), state.agent_action

        if state.turn == MinimaxAgent.Turn.AGENT_TURN:
            curr_max = -math.inf
            best_action: GameAction
            for action in state.game_state.get_possible_actions(player_index):
                val, action = self.abminmax(MinimaxAgent.TurnBasedGameState(state.game_state, action), depth, player_index, alpha, beta)
                if curr_max <= val:
                    curr_max = val
                    best_action = action
                    alpha = max(curr_max, alpha)
                    if curr_max >= beta and curr_max != -math.inf:
                        return math.inf, best_action
            return curr_max, best_action
        else:
            curr_min = math.inf
            for opponents_actions in state.game_state.get_possible_actions_dicts_given_action(state.agent_action, player_index=self.player_index):
                val, _ = self.abminmax(MinimaxAgent.TurnBasedGameState(
                    get_next_state(state.game_state, {**opponents_actions, **{player_index: state.agent_action}}),
                    None), depth - 1, player_index, alpha, beta)
                if curr_min >= val:
                    curr_min = val
                    beta = min(curr_min, beta)
                    if curr_min <= alpha and curr_min != math.inf:
                        return -math.inf, state.agent_action
            return curr_min, state.agent_action

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
    moves = SAHC_algoritem(50)
    get_fitness(tuple(moves), True, True)


def SAHC_algoritem(sideway_limit: int, num_moves: int = 50):
    sideways = 0
    moves = []
    current_best = get_fitness(tuple(moves))
    for i in range(num_moves):
        best_val = current_best
        best_actions = []
        for action in GameAction:
            moves.append(action)
            new_val = get_fitness(tuple(moves))
            if new_val > best_val:
                best_val = new_val
                best_actions = [action]
            elif new_val == best_val:
                best_actions.append(action)
            moves.pop(len(moves)-1)

        if best_val > current_best:
            i = np.random.randint(low=0, high=len(best_actions))
            moves.append(best_actions[i])
            current_best = best_val
        elif best_val == current_best and sideways <= sideway_limit:
            i = np.random.randint(low=0, high=len(best_actions))
            moves.append(best_actions[i])
            sideways += 1
        else:
            print("no more sideways/better options")
            print("exited on iteration =", i)
            return moves
    return moves



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

    # command for testing: python main.py --custom_game --p1 TournamentAgent --p2 GreedyAgent

    total_time = 0
    turn_time_limit = 58/500
    start = 0

    def get_action(self, state: GameState) -> GameAction:
        actionmm: GameAction = self.get_actionMM(state)
        self.start = time.time()
        # Insert your code here...
        action: GameAction = self.mmBFS(state, self.player_index)
        self.total_time += time.time() - self.start
        if actionmm != action:
            print(actionmm, action)
        return action

    def mmBFS(self, s: GameState, player_index: int):
        states = [(s, None)]
        curr_max = -math.inf
        best_action: GameAction = GameAction.STRAIGHT
        # print(s.turn_number)
        counter = 0
        while True:# time.time() - self.start < self.turn_time_limit:
            if not s.snakes[player_index].alive or s.is_terminal_state or not states:
                return best_action
            if counter == 2:
                return best_action
            counter += 1
            state, state_action = states.pop()
            for action in state.get_possible_actions(player_index):
                first_action = state_action
                if state_action is None:
                    first_action = action
                val, succ_states = self.mmBFSaux((state, first_action), action, player_index)
                if val > curr_max:
                    curr_max = val
                    best_action = first_action
                states += succ_states
        print(len(states))
        return best_action

    def mmBFSaux(self, state_and_first_action, agent_action: GameAction, player_index: int):
        curr_min = math.inf
        opened_states = []
        state, first_action = state_and_first_action
        for opponents_actions in state.get_possible_actions_dicts_given_action(agent_action, player_index=self.player_index):
            next_state = get_next_state(state, {**opponents_actions, **{player_index: agent_action}})
            val = heuristic(next_state, player_index)
            opened_states.append((next_state, first_action))
            if curr_min > val:
                curr_min = val
        return curr_min, opened_states

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

    def get_actionMM(self, state: GameState) -> GameAction:
        # Insert your code here...
        search_depth = 2
        _, action = self.minmax(MinimaxAgent.TurnBasedGameState(state, None), search_depth, self.player_index)

        return action

    """
    return tuple(float, GameAction) where float is the minmax value
    """

    def minmax(self, state: TurnBasedGameState, depth: int, player_index: int):
        if depth == 0 or state.game_state.is_terminal_state or not state.game_state.snakes[player_index].alive:
            return heuristic(state.game_state, player_index), state.agent_action

        if state.turn == MinimaxAgent.Turn.AGENT_TURN:
            best_val = -math.inf
            best_action: GameAction
            for action in state.game_state.get_possible_actions(player_index):
                val, action = self.minmax(MinimaxAgent.TurnBasedGameState(state.game_state, action), depth,
                                          player_index)
                if best_val < val:
                    best_val = val
                    best_action = action
            return best_val, best_action
        else:
            best_val = math.inf
            for opponents_actions in state.game_state.get_possible_actions_dicts_given_action(state.agent_action,
                                                                                              player_index=self.player_index):
                val, _ = self.minmax(MinimaxAgent.TurnBasedGameState(
                    get_next_state(state.game_state, {**opponents_actions, **{player_index: state.agent_action}}),
                    None), depth - 1, player_index)
                if best_val > val:
                    best_val = val
            return best_val, state.agent_action


if __name__ == '__main__':
    pass
    # SAHC_sideways()
    # local_search()
