import sys
import time

from agents import GreedyAgent, BetterGreedyAgent
from environment import Grid2DSize, SnakesBackendSync
from submission import MinimaxAgent, AlphaBetaAgent

players = ["GreedyAgent", "BetterGreedyAgent", "MinimaxAgent", "AlphaBetaAgent"]
depths = [2, 3, 4]


def start_game_with_players(players, game_duration: int, board_width: int, board_height: int, n_fruits: int,
                            fast_run: bool = False, graphics_off: bool = False):
    if len(players) < 1:
        print("The number of agents must be at least 1.")

    env = SnakesBackendSync(players,
                            grid_size=Grid2DSize(board_width, board_height),
                            n_fruits=n_fruits,
                            game_duration_in_turns=game_duration)
    env.run_game(human_speed=not fast_run, render=not graphics_off)
    return env


def start_custom_game(p1: str, game_duration: int, board_width: int, board_height: int,
                      n_fruits: int, depth: int):
    def get_player(p: str):
        if p == 'GreedyAgent':
            return GreedyAgent()
        elif p == 'BetterGreedyAgent':
            return BetterGreedyAgent()
        elif p == 'MinimaxAgent':
            return MinimaxAgent(depth=depth)
        elif p == 'AlphaBetaAgent':
            return AlphaBetaAgent(depth=depth)

    players = [get_player(p1), GreedyAgent()]

    return start_game_with_players(players,
                                   game_duration,
                                   board_width,
                                   board_height,
                                   n_fruits,
                                   fast_run=True,
                                   graphics_off=True), players[0].avg_turn_time


def run_game(p1: str, depth=2):
    scores = []
    times = []
    for i in range(10):
        print("Now running: ", p1, "Game number: ", i+1)
        game, turn_time_total = start_custom_game(p1, game_duration=500, board_width=50, board_height=50, n_fruits=51,
                                                  depth=depth)
        scores.append(game.game_state.snakes[0].length)
        times.append(turn_time_total / float(game.game_state.turn_number))
    avg_score = sum(scores) / float(len(scores))
    avg_time = sum(times) / float(len(times))
    line = (p1 + ',' +
            str(depth) + ',' +
            '%.2f' % avg_score + ',' +
            '%.2f' % (avg_time * 1e6) + 'E-06\n')
    file_ptr.write(line)
    return


if __name__ == '__main__':
    with open(sys.argv[1]+'.csv', 'w+') as file_ptr:
        if sys.argv[2] in ["GreedyAgent", "BetterGreedyAgent"]:
            run_game(sys.argv[2], depth=1)
        else:
            for d in depths:
                run_game(sys.argv[2], depth=d)
        file_ptr.write('\n')
    file_ptr.close()
