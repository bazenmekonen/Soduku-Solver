#Bazen Mekonen
from queue import Queue
import copy
import time



class Problem(object):

    def __init__(self, initial):
        self.initial = initial
        self.size = len(initial)  # Define board type
        self.height = int(self.size / 3)

    # Return set of valid numbers from values that do not appear in used
    def filter_values(self, values, used):
        return [number for number in values if number not in used]

    # Return first empty spot on grid (marked with 0)
    def get_spot(self, board, state):
        for row in range(board):
            for column in range(board):
                if state[row][column] == 0:
                    return row, column

    def actions(self, state):
        number_set = range(1, self.size + 1)  # Defines set of valid numbers that can be placed on board
        in_column = []  # List of valid values in spot's column
        in_block = []  # List of valid values in spot's quadrant

        row, column = self.get_spot(self.size, state)  # Get first empty spot on board

        # Filter valid values based on row
        in_row = [number for number in state[row] if (number != 0)]
        options = self.filter_values(number_set, in_row)

        # Filter valid values based on column
        for column_index in range(self.size):
            if state[column_index][column] != 0:
                in_column.append(state[column_index][column])
        options = self.filter_values(options, in_column)

        # Filter with valid values based on quadrant
        row_start = int(row / self.height) * self.height
        column_start = int(column / 3) * 3

        for block_row in range(0, self.height):
            for block_column in range(0, 3):
                in_block.append(state[row_start + block_row][column_start + block_column])
        options = self.filter_values(options, in_block)

        for number in options:
            yield number, row, column

    # Returns updated board after adding new valid value
    def result(self, state, action):

        play = action[0]
        row = action[1]
        column = action[2]

        # Add new valid value to board
        new_state = copy.deepcopy(state)
        new_state[row][column] = play

        return new_state

    # Use sums of each row, column and quadrant to determine validity of board state
    def goal_test(self, state):

        # Expected sum of each row, column or quadrant.
        total = sum(range(1, self.size + 1))

        # Check rows and columns and return false if total is invalid
        for row in range(self.size):
            if (len(state[row]) != self.size) or (sum(state[row]) != total):
                return False

            column_total = 0
            for column in range(self.size):
                column_total += state[column][row]

            if (column_total != total):
                return False

        # Check quadrants and return false if total is invalid
        for column in range(0, self.size, 3):
            for row in range(0, self.size, self.height):

                block_total = 0
                for block_row in range(0, self.height):
                    for block_column in range(0, 3):
                        block_total += state[row + block_row][column + block_column]

                if (block_total != total):
                    return False

        return True


class Node:

    def __init__(self, state, action=None):
        self.state = state
        self.action = action

    # Use each action to create a new board state
    def expand(self, problem):
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    # Return node with new board state
    def child_node(self, problem, action):
        next = problem.result(self.state, action)
        return Node(next, action)


def BFS(problem):
    # Create initial node of problem tree holding original board
    node = Node(problem.initial)
    # Check if original board is correct and immediately return if valid
    if problem.goal_test(node.state):
        return node

    frontier = Queue()
    frontier.put(node)

    # Loop until all nodes are explored or solution found
    while (frontier.qsize() != 0):

        node = frontier.get()
        for child in node.expand(problem):
            if problem.goal_test(child.state):
                return child

            frontier.put(child)

    return None

def DFS(problem):
    start = Node(problem.initial)
    if problem.goal_test(start.state):
        return start.state

    stack = []
    stack.append(start)  # Place initial node onto the stack

    while stack:
        node = stack.pop()  # Pops the last node, tests legality, then expands the same popped node
        if problem.goal_test(node.state):
            return node.state
        stack.extend(node.expand(problem))

    return None




def SOLVE_SUDOKU_USING_BFS(board):
    print("\nSolving with BFS...")
    letters = False
    if check_if_letters(board):  # Checks of the board contains letters instead of numbers
        board = to_numbers(board)  # Transforms letter puzzles to numeric puzzles
        letters = True

    start_time = time.time()

    problem = Problem(board)
    solution = BFS(problem)
    elapsed_time = time.time() - start_time

    if solution:
        if letters:
            solution.state = to_letters(
                solution.state)  # Transforms back numeric puzzles to original letter puzzle type of true
        print("Found solution")
        for row in solution.state:
            print(row)
    else:
        print("No possible solutions")

    print("Elapsed time: " + str(elapsed_time) + " seconds")



def SOLVE_SUDOKU_USING_DFS(board):
    print("\nSolving with DFS...")
    letters = False
    if check_if_letters(board):  # Checks of the board contains letters instead of numbers
        board = to_numbers(board)  # Transforms letter puzzles to numeric puzzles
        letters = True

    start_time = time.time()
    problem = Problem(board)
    solution = DFS(problem)
    elapsed_time = time.time() - start_time

    if solution:
        if letters:
            solution = to_letters(solution)  # Transforms back numeric puzzles to original letter puzzle type of true
        print("Found solution")
        for row in solution:
            print(row)
    else:
        print("No possible solutions")

    print("Elapsed time: " + str(elapsed_time) + " seconds")


def H_Solve(board):
    print("\nSolving with DFS and heuristics...")
    letters = False
    if check_if_letters(board):  # Checks of the board contains letters instead of numbers
        board = to_numbers(board)  # Transforms letter puzzles to numeric puzzles
        letters = True

    start_time = time.time()
    problem = Problem(board)
    solution = DFS(problem)
    elapsed_time = time.time() - start_time

    if solution:
        if letters:
            solution = to_letters(solution)  # Transforms back numeric puzzles to original letter puzzle type of true
        print("Found solution")
        for row in solution:
            print(row)
    else:
        print("No possible solutions")

    print("Elapsed time: " + str(elapsed_time) + " seconds")


def get_key(dictionary, val):
    for key, value in dictionary.items():
        if val == value:
            return key

    return "Error: key doesn't exist"


def get_val(dictionary, k):
    for key, value in dictionary.items():
        if k == key:
            return value

    return "Error: value doesn't exist"


def dots_spaces(lettergrid):
    for row in range(len(lettergrid)):
        for column in range(len(lettergrid)):
            if lettergrid[row][column] == '.':
                return True
            elif lettergrid[row][column] == ' ':
                return False


def check_if_letters(grid):
    for row in range(len(grid)):
        for column in range(len(grid)):
            if isinstance(grid[row][column], str):
                return True
                # else:
                # return False


def to_numbers(lettergrid):
    gridSize = len(lettergrid)
    numKeys = [num for num in range(1, gridSize + 1)]
    letterValues = list(map(chr, range(ord('A'), ord('A') + gridSize + 1)))
    alphanumDict = dict(zip(numKeys, letterValues))
    if dots_spaces(lettergrid) == True:
        alphanumDict[0] = '.'
    elif dots_spaces(lettergrid) == False:
        alphanumDict[0] = ' '

    numbergrid = [[] for new_list in range(gridSize)]
    blanklist = []
    for row in range(gridSize):
        for column in range(gridSize):
            blanklist.append(get_key(alphanumDict, lettergrid[row][column]))
        numbergrid[row].extend(blanklist)
        blanklist.clear()
    return numbergrid


def to_letters(numbergrid):
    gridSize = len(numbergrid)
    numKeys = [num for num in range(1, gridSize + 1)]
    letterValues = list(map(chr, range(ord('A'), ord('A') + gridSize + 1)))
    alphanumDict = dict(zip(numKeys, letterValues))
    alphanumDict[0] = '.'

    lettergrid = [[] for new_list in range(gridSize)]
    blanklist = []
    for row in range(gridSize):
        for column in range(gridSize):
            blanklist.append(get_val(alphanumDict, numbergrid[row][column]))
        lettergrid[row].extend(blanklist)
        blanklist.clear()
    return lettergrid

if __name__ == "__main__":
    ## TEST CASES
    # 6x6 board
    board_6x6 = [[0, 0, 0, 0, 4, 0],
                 [5, 6, 0, 0, 0, 0],
                 [3, 0, 2, 6, 5, 4],
                 [0, 4, 0, 2, 0, 3],
                 [4, 0, 0, 0, 6, 5],
                 [1, 5, 6, 0, 0, 0]]
    SOLVE_SUDOKU_USING_BFS(board_6x6)
    SOLVE_SUDOKU_USING_DFS(board_6x6)

    # 9x9 board
    board_9x9 = [[3, 0, 6, 5, 0, 8, 4, 0, 0],
                 [5, 2, 0, 0, 0, 0, 0, 0, 0],
                 [0, 8, 7, 0, 0, 0, 0, 3, 1],
                 [0, 0, 3, 0, 1, 0, 0, 8, 0],
                 [9, 0, 0, 8, 6, 3, 0, 0, 5],
                 [0, 5, 0, 0, 9, 0, 6, 0, 0],
                 [1, 3, 0, 0, 0, 0, 2, 5, 0],
                 [0, 0, 0, 0, 0, 0, 0, 7, 4],
                 [0, 0, 5, 2, 0, 6, 3, 0, 0]]
    SOLVE_SUDOKU_USING_BFS(board_9x9)
    SOLVE_SUDOKU_USING_DFS(board_9x9)


