
import time
from BaseAI import BaseAI
import math


class IntelligentAgent(BaseAI):
    def __init__(self):
        self.timeLimit = 0.2
        self.startTime = 0
        #self.maxDepth = 4  # Increased depth for better lookahead
        self.maxDepth = 6

        # Adjusted weights to prioritize important strategies
        self.weights = {
            'monotonicity': 1.5,
            #'monotonicity': 1.2,
            'smoothness': 0.2,
            #'smoothness': 0.1,
            'free_cells': 4.0,
            'merge_potential': 1.2,
            #'max_value': 0,
            'max_value': 1.5,
            'corner_max': 3.0
            #'corner_max': 0
        }

    def getMove(self, grid):
        """
        See if there are any available moves
        If there is an available move find the most optimal move to take
        Return the best move
        """
        self.startTime = time.process_time()
        moves = self.getAvailableMoves(grid)

        """
        # Dynamically adjust search depth based on empty cells
        empty_cells = len(grid.getAvailableCells())
        if empty_cells > 5:
            self.maxDepth = 4
        elif empty_cells > 2:
            self.maxDepth = 5
        else:
            self.maxDepth = 6
        """

        if not moves:
            return None

        best_move = None
        best_score = float('-inf')

        for move in moves:
            grid_copy = grid.clone()
            if grid_copy.move(move):
                score = self.minimize(grid_copy, 0, float('-inf'), float('inf'))
                if score > best_score:
                    best_score = score
                    best_move = move

        return best_move

    def getAvailableMoves(self, grid):
        """
        Get the available moves in the grid.
        If a move results in a valid grid change, add it to the moves list.
        Return a list of available moves (0-UP, 1-DOWN, 2-LEFT, 2-RIGHT)
        """

        moves = []
        for move in range(4):
            grid_copy = grid.clone()
            if grid_copy.move(move):
                moves.append(move)
        return moves

    def minimize(self, grid, depth, alpha, beta):
        """
        Find the minimizable optimal score,
        If there are available cells and either a 2 or a 4 depending on the probability ( 2 has a 90% chance of being chosen and 4 has 10% chance)
        Do Beta pruning
        return the lowest optimal score achievable from the current state of the grid
        """
        if self.isTerminal(grid, depth):
            return self.evaluateGrid(grid)

        minUtility = float('inf')
        available_cells = grid.getAvailableCells()

        if not available_cells:
            return self.evaluateGrid(grid)

        # To calculate average, we'll sum all utilities and divide by count
        total_utility = 0
        cell_count = len(available_cells)

        for cell in available_cells:
            # Place a "2" in the cell, representing a possible computer move
            grid_copy_2 = grid.clone()
            grid_copy_2.insertTile(cell, 2)
            utility_2 = self.maximize(grid_copy_2, depth + 1, alpha, beta)

            #Place a "4" in the cell, with lower probability
            grid_copy_4 = grid.clone()
            grid_copy_4.insertTile(cell, 4)
            utility_4 = self.maximize(grid_copy_4, depth + 1, alpha, beta)

            #Expectiminimax step: calculate the weighted average for chance node
            utility = (0.9 * utility_2) + (0.1 * utility_4)
            total_utility += utility
            #instead of looking for the min all we need to do is to add them up and find their average
            #minUtility = min(minUtility, utility)

            #remove this since we do not need to do pruning here
            # beta pruning
            #beta = min(beta, utility)
            #if beta <= alpha:
                #break #prune remaining branches

        #return minUtility
        return total_utility / cell_count if cell_count > 0 else self.evaluateGrid(grid)

    def maximize(self, grid, depth, alpha, beta):
        """
        Find the maximizable optimal score,
        This function simulates the player’s turn, selecting moves that maximize the score
        After exploring all possible moves (or cutting off), maximize returns maxUtility, which represents the optimal score achievable for this branch
        """
        if self.isTerminal(grid, depth):
            return self.evaluateGrid(grid)

        maxUtility = float('-inf')
        moves = self.getAvailableMoves(grid)

        if not moves:
            return self.evaluateGrid(grid)

        # Sort moves based on a simple heuristic for better pruning
        move_scores = []
        for move in moves:
            grid_copy = grid.clone()
            if grid_copy.move(move):
                score = self.evaluateGrid(grid_copy)
                move_scores.append((score, move))

        # Evaluate moves in descending order of their scores
        for score, move in sorted(move_scores, reverse=True):
            grid_copy = grid.clone()
            if grid_copy.move(move):
                utility = self.minimize(grid_copy, depth + 1, alpha, beta)
                maxUtility = max(maxUtility, utility)

                #alpha pruning
                alpha = max(alpha, maxUtility)
                if beta <= alpha:
                    break #prune remaining branches

        return maxUtility

    def isTerminal(self, grid, depth):
        if time.process_time() - self.startTime >= self.timeLimit:
            return True
        if depth >= self.maxDepth:
            return True
        if not grid.getAvailableCells() and not self.getAvailableMoves(grid):
            return True
        return False

    def evaluateGrid(self, grid):
        if not self.getAvailableMoves(grid):
            return float('-inf')

        score = 0

        #Smoothness Heuristic
        score += self.smoothnessHeuristic(grid) * self.weights['smoothness']
        #Monotonicity Heuristic
        score += self.monotonicityHeuristic(grid) * self.weights['monotonicity']
        #Empty Cell Heuristic
        score += self.emptyCellHeuristic(grid) * self.weights['free_cells']
        #Merging Heuristic
        score += self.mergingHeuristic(grid) * self.weights['merge_potential']
        #Max Value Heuristic
        score += self.maxValueHeuristic(grid) * self.weights['max_value']
        #Corner Max Heuristic
        score += self.cornerMaxHeuristic(grid) * self.weights['corner_max']

        return score

    def smoothnessHeuristic(self, grid):
        """
        Calculates grid smoothness by measuring similarity between adjacent tile values.
        For each tile, subtracts the absolute difference with its right and bottom neighbors.
        Higher (less negative) smoothness values indicate a grid where adjacent tiles are more similar, making merges easier.
        Returns: smoothness (int): A score where less difference between adjacent tiles gives a higher value.
        """
        smoothness = 0
        for i in range(grid.size):
            for j in range(grid.size):
                if grid.map[i][j]:
                    value = grid.map[i][j]
                    if j + 1 < grid.size and grid.map[i][j + 1]:
                        smoothness -= abs(value - grid.map[i][j + 1])
                    if i + 1 < grid.size and grid.map[i + 1][j]:
                        smoothness -= abs(value - grid.map[i + 1][j])
        return smoothness

    def monotonicityHeuristic(self, grid):
        """
        Calculates grid monotonicity, rewarding rows and columns that increase or decrease consistently.
        Higher monotonicity scores indicate a grid where tile values consistently rise or fall along rows
        or columns, which helps organize high-value tiles.
        Returns: monotonicity (int): A score favoring configurations with smooth increases or decreases in value.
        """
        # Check for monotonic decrease from top-left to bottom-right
        scores = [0, 0, 0, 0]  # up, down, left, right

        # Vertical traversal
        for i in range(grid.size - 1):
            for j in range(grid.size):
                if grid.map[i][j] and grid.map[i + 1][j]:
                    if grid.map[i][j] > grid.map[i + 1][j]:
                        #scores[0] += grid.map[i][j] - grid.map[i + 1][j]
                        scores[0] += math.log2(grid.map[i][j]) - math.log2(grid.map[i + 1][j])
                    else:
                        #scores[1] += grid.map[i + 1][j] - grid.map[i][j]
                        scores[1] += math.log2(grid.map[i + 1][j]) - math.log2(grid.map[i][j])

        # Horizontal traversal
        for i in range(grid.size):
            for j in range(grid.size - 1):
                if grid.map[i][j] and grid.map[i][j + 1]:
                    if grid.map[i][j] > grid.map[i][j + 1]:
                        #scores[2] += grid.map[i][j] - grid.map[i][j + 1]
                        scores[2] += math.log2(grid.map[i][j]) - math.log2(grid.map[i][j + 1])
                    else:
                        #scores[3] += grid.map[i][j + 1] - grid.map[i][j]
                        scores[3] += math.log2(grid.map[i][j + 1]) - math.log2(grid.map[i][j])

        return max(scores[0], scores[1]) + max(scores[2], scores[3])

    def emptyCellHeuristic(self, grid):
        """
        Calculates a score based on the number of empty cells in the grid.
        Returns: empty_cell_score (int): A score that increases with the number of empty cells.
        """
        empty_cells = len(grid.getAvailableCells())
        return math.pow(empty_cells, 1.5) * 2.7


    def mergingHeuristic(self, grid):
        """
        Calculates a heuristic based on merging information from multiple grid aspects.
        Likely evaluates proximity to goal states or merges adjacent cells with similar properties.
        Returns: score (int): A numerical heuristic value representing how close the current grid configuration is to the solution.
        """
        score = 0
        for i in range(grid.size):
            for j in range(grid.size):
                if not grid.map[i][j]:
                    continue
                value = grid.map[i][j]
                # Check horizontal merges
                if j + 1 < grid.size and grid.map[i][j + 1] == value:
                    score += value * 2
                # Check vertical merges
                if i + 1 < grid.size and grid.map[i + 1][j] == value:
                    score += value * 2
        return score


    def maxValueHeuristic(self, grid):
        """
        Calculates a heuristic based on maximum value information from multiple grid aspects.
        Prioritizes configurations with the highest possible values to maximize progress or minimize cost.
        :returns: A numerical value representing the maximum value of the criterion being evaluated
        """
        max_value = 0
        for i in range(grid.size):
            for j in range(grid.size):
                if grid.map[i][j] > max_value:
                    max_value = grid.map[i][j]
        return math.log2(max_value) if max_value > 0 else 0


    def cornerMaxHeuristic(self, grid):
        """
        Calculates a heuristic focusing on the maximum values or features in the corners of the grid.
        Emphasizes corner cells and their importance in guiding the agent, especially in pathfinding or decision-making tasks.
        returns: A numerical heuristic value based on the maximum value or feature in the corner cells of the grid.
        """
        #max_value = max(max(row) for row in grid.map)
        corners = [
            grid.map[0][0],
            grid.map[0][grid.size - 1],
            grid.map[grid.size - 1][0],
            grid.map[grid.size - 1][grid.size - 1]
        ]
        max_value = max(max(row) for row in grid.map)
        return 2.0 * (max_value in corners)
        #return 2.0 if max_value in corners else -1.0
