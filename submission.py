import copy
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
    print(moves)


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
    moves = beam_SAHC_algorithm(15, 10, 50)
    print(moves)
    get_fitness(moves, True, True)


def beam_SAHC_algorithm(k: int, initial_num_moves: int, maximum_num_moves: int = 50):
    move_sets = [list(np.random.choice(list(GameAction), initial_num_moves)) for _ in range(k)]
    new_beam = [(move_set, get_fitness(move_set)) for move_set in move_sets]
    while True:
        beam = new_beam
        improving_move_sets = []
        equal_move_sets = []
        for move_set, val in beam:
            possible_actions = list(GameAction)
            for action in possible_actions:
                new_move_set = copy.deepcopy(move_set)
                new_move_set.append(action)
                if len(new_move_set) == maximum_num_moves:
                    return new_move_set
                new_val = get_fitness(new_move_set)
                improvement_delta = new_val - val
                if improvement_delta > 0:
                    improving_move_sets.append((new_move_set, new_val))
                elif improvement_delta == 0:
                    equal_move_sets.append((new_move_set, new_val))
        if not improving_move_sets:
            if equal_move_sets:  # populate improving_move_sets with best equal states
                equal_move_sets.sort(key=lambda x: x[1], reverse=True)
                new_beam = [(move_set, val) for move_set, val in equal_move_sets if val == equal_move_sets[0][1]]
                if len(new_beam) > k:
                    new_beam = new_beam[:k]
                    np.random.shuffle(new_beam)
            else:  # the only option is death
                move_set, _ = beam[0]
                move_set.append(GameAction.STRAIGHT)
                new_beam = [(move_set, get_fitness(move_set))]
        elif len(improving_move_sets) <= k:
            new_beam = improving_move_sets
        else:
            total = sum([val for _, val in improving_move_sets])
            probabilities = [val/total for _, val in improving_move_sets]
            print(improving_move_sets)
            print(len(improving_move_sets))
            chosen_improvments = list(np.random.choice(range(len(improving_move_sets)), k, False, probabilities))
            new_beam = [improving_move_sets[i] for i in chosen_improvments]



class TournamentAgent(Player):

    # command for testing: python main.py --custom_game --p1 TournamentAgent --p2 GreedyAgent

    total_time = 0
    turn_time_limit = 60/500
    start = 0

    def get_action(self, state: GameState) -> GameAction:
        self.start = time.time()
        search_depth = 2
        _, action = self.abminmax(MinimaxAgent.TurnBasedGameState(state, None), search_depth, self.player_index, -math.inf, math.inf)
        self.total_time += time.time() - self.start
        return action

    def abminmax(self, state: MinimaxAgent.TurnBasedGameState, depth: int, player_index: int, alpha: float,
                 beta: float):
        if depth == 0 or state.game_state.is_terminal_state or not state.game_state.snakes[player_index].alive:
            return heuristic(state.game_state, player_index), state.agent_action

        if state.turn == MinimaxAgent.Turn.AGENT_TURN:
            curr_max = -math.inf
            best_action: GameAction
            for action in state.game_state.get_possible_actions(player_index):
                val, action = self.abminmax(MinimaxAgent.TurnBasedGameState(state.game_state, action), depth,
                                            player_index, alpha, beta)
                if curr_max <= val:
                    curr_max = val
                    best_action = action
                    alpha = max(curr_max, alpha)
                    if curr_max >= beta and curr_max != -math.inf:
                        return math.inf, best_action
                if time.time() - self.start >= self.turn_time_limit:
                    return curr_max, best_action
            return curr_max, best_action
        else:
            curr_min = math.inf
            for opponents_actions in state.game_state.get_possible_actions_dicts_given_action(state.agent_action,
                                                                                              player_index=self.player_index):
                val, _ = self.abminmax(MinimaxAgent.TurnBasedGameState(
                    get_next_state(state.game_state, {**opponents_actions, **{player_index: state.agent_action}}),
                    None), depth - 1, player_index, alpha, beta)
                if curr_min >= val:
                    curr_min = val
                    beta = min(curr_min, beta)
                    if curr_min <= alpha and curr_min != math.inf:
                        return -math.inf, state.agent_action
                if time.time() - self.start >= self.turn_time_limit:
                    return curr_min, state.agent_action
            return curr_min, state.agent_action


if __name__ == '__main__':
    pass
    # SAHC_sideways()
    local_search()
