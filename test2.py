# Name: Lamees Alwasil
# UNI: laa2203
"""
This program is an implementation of IntelligentAgent that plays the puzzle game 2048
using expectiminimax with alpha-beta pruning. This game uses heuristics to make optimal
moves and consistently achieve high scores.
"""

import time
from BaseAI import BaseAI
import math


class IntelligentAgent(BaseAI):
    def __init__(self):
        # Maximum depth for the search tree
        self.max_depth = 6
        # self.max_depth = 5
        # Time limit for each move in seconds
        self.time_limit = 0.2
        self.start_time = 0

    def getMove(self, grid):
        """
        find the most optimal move by calling expectiminmax
        """
        self.start_time = time.time()
        return self.expectiminimax(grid, depth=0, alpha=float('-inf'), beta=float('inf'), maximizing=True)[1]

    def expectiminimax(self, grid, depth, alpha, beta, maximizing):
        """
        See whether you need to maximize or minimize the move
        """
        # Check if we've hit the time limit or maximum depth
        if time.time() - self.start_time > self.time_limit or depth >= self.max_depth:
            return self.evaluate(grid), None

        if maximizing:
            return self.maximize(grid, depth, alpha, beta)
        else:
            return self.minimize(grid, depth, alpha, beta)

    def maximize(self, grid, depth, alpha, beta):
        """
        Maximizes utility by finding the optimal move in the given grid configuration.
        """
        max_utility = float('-inf')
        max_move = None

        # Get all available moves and sort them by a preliminary evaluation
        moves = grid.getAvailableMoves()
        moves.sort(key=lambda x: self.evaluate(x[1]), reverse=True)  # Move ordering optimization

        for move, new_grid in moves:
            utility = self.expectiminimax(new_grid, depth + 1, alpha, beta, False)[0]

            if utility > max_utility:
                max_utility = utility
                max_move = move

            alpha = max(alpha, max_utility)
            if beta <= alpha:
                break

        return max_utility, max_move

    def minimize(self, grid, depth, alpha, beta):
        """
        Minimizes utility by finding the optimal move in the given grid configuration.
        """
        min_utility = float('inf')
        available_cells = grid.getAvailableCells()

        # Calculate probabilities for 2 and 4 tiles
        prob_2 = 0.9  # 90% chance of 2
        prob_4 = 0.1  # 10% chance of 4

        for cell in available_cells:
            # Try placing a 2 (90% probability)
            grid_copy = grid.clone()
            grid_copy.insertTile(cell, 2)
            utility_2 = self.expectiminimax(grid_copy, depth + 1, alpha, beta, True)[0]

            # Try placing a 4 (10% probability)
            grid_copy = grid.clone()
            grid_copy.insertTile(cell, 4)
            utility_4 = self.expectiminimax(grid_copy, depth + 1, alpha, beta, True)[0]

            # Calculate expected utility
            utility = (prob_2 * utility_2) + (prob_4 * utility_4)
            min_utility = min(min_utility, utility)

            beta = min(beta, min_utility)
            if beta <= alpha:
                break

        return min_utility, None

    def evaluate(self, grid):
        """
        Evaluates the grid state using multiple heuristics
        """
        if not grid.canMove():
            return float('-inf')

        # FIRST TEST: Weights for different heuristics
        SCORE_WEIGHT = 1.0
        EMPTY_WEIGHT = 2.7
        MONOTONICITY_WEIGHT = 1.0
        SMOOTHNESS_WEIGHT = 0.1
        MAX_TILE_WEIGHT = 1.0

        """
        # SECOND TEST: Weights for different heuristics
        SCORE_WEIGHT = 1.2
        EMPTY_WEIGHT = 3
        MONOTONICITY_WEIGHT = 1.5
        SMOOTHNESS_WEIGHT = 0.2
        MAX_TILE_WEIGHT = 1.5

        ## THIRD TEST: Weights for different heuristics
        SCORE_WEIGHT = 0.6
        EMPTY_WEIGHT = 1.4
        MONOTONICITY_WEIGHT = 0.5
        SMOOTHNESS_WEIGHT = 0.05
        MAX_TILE_WEIGHT = 0.5
        """
        score = 0
        score += len(grid.getAvailableCells()) * EMPTY_WEIGHT
        score += self.monotonicity_heuristic(grid) * MONOTONICITY_WEIGHT
        score += self.smoothness_heuristic(grid) * SMOOTHNESS_WEIGHT
        score += self.max_tile_heuristic(grid) * MAX_TILE_WEIGHT

        return score * SCORE_WEIGHT

    def monotonicity_heuristic(self, grid):
        """
        Measures how monotonic the grid is. This tries to ensure that the
        values of the tiles are all either increasing or decreasing along both
        the left/right and up/down directions
        """
        scores = [0, 0, 0, 0]  # up, down, left, right

        # Vertical monotonicity
        for x in range(4):
            current = 0
            next_val = current + 1
            while next_val < 4:
                while next_val < 4 and not grid.map[next_val][x]:
                    next_val += 1
                if next_val >= 4:
                    next_val -= 1
                current_value = grid.map[current][x]
                next_value = grid.map[next_val][x]
                if current_value and next_value:
                    scores[0] += next_value - current_value  # up
                    scores[1] += current_value - next_value  # down
                current = next_val
                next_val += 1

        # Horizontal monotonicity
        for y in range(4):
            current = 0
            next_val = current + 1
            while next_val < 4:
                while next_val < 4 and not grid.map[y][next_val]:
                    next_val += 1
                if next_val >= 4:
                    next_val -= 1
                current_value = grid.map[y][current]
                next_value = grid.map[y][next_val]
                if current_value and next_value:
                    scores[2] += next_value - current_value  # left
                    scores[3] += current_value - next_value  # right
                current = next_val
                next_val += 1

        return max(scores[0], scores[1]) + max(scores[2], scores[3])

    def smoothness_heuristic(self, grid):
        """
        Measures how smooth the grid is (difference between neighboring tiles)
        """
        smoothness = 0
        for x in range(4):
            for y in range(4):
                if grid.map[x][y]:
                    value = math.log2(grid.map[x][y])
                    # Check horizontal neighbor
                    if x < 3 and grid.map[x + 1][y]:
                        smoothness -= abs(value - grid.map[x + 1][y])
                    # Check vertical neighbor
                    if y < 3 and grid.map[x][y + 1]:
                        smoothness -= abs(value - grid.map[x][y + 1])
        return smoothness

    def max_tile_heuristic(self, grid):
        """Return the value of the highest tile on the board"""
        return max(max(row) for row in grid.map)