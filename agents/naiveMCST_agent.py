# Student agent: Add your own agent here
from copy import deepcopy
from agents.agent import Agent
from store import register_agent
import numpy as np
from collections import deque
from random import randint
from math import sqrt, log
import time
from random import choice
'''
Note to self: chess_board coordinate system is [x_pos, y_pos, direction]
directions:
0 = up
1 = right
2 = down
3 = left
x increases down column, y increases across row:
|(0,0) (0,1) (0,2)|
|(1,0) (1,1) (1,2)|
|(2,0) (2,1) (2,2)|
'''


dir_map = {
    "u": 0,
    "r": 1,
    "d": 2,
    "l": 3,
}

class MCTSnode :

    def __init__(self, pos, adv_pos, board, agent = True):

        #agent's position
        self.pos = deepcopy(pos)

        #adversary position
        self.adv_pos = deepcopy(adv_pos)
        self.children = []
        self.wins = 0.0
        self.visits = 0.0
        self.parent = None
        self.board = deepcopy(board)

        #keeps track of the player (True is us, false is adversary)
        self.agent = agent

    def add_child(self, child):
        self.children.append(child)
    
    def add_parent(self, node):
        self.parent = node
        
    def update_value(self, win):
        if win :
            self.wins +=1.0
            self.visits +=1.0
        else : 
            self.visits +=1.0
    
    def set_adv(self, val = True):
        self.agent = val
    
    def set_pos (self, pos):
        self.pos = deepcopy(pos)
    
    def set_advpos (self, pos):
        self.adv_pos = deepcopy(pos)
    
    def set_wall(self, wall):
        self.wall = wall

    def update_board(self, board):
        self.board = deepcopy(board)

# finds squares of the board that can be reached from a given node
# i.e., they are adjacent and don't have a wall between them
def get_neighbors(chess_board, pos, adv_pos):
    #print(pos, end='\t')
    adv_pos = tuple(adv_pos)
    x, y = pos
    output = []
    if not chess_board[x, y, 0] and (x - 1, y) != adv_pos:
        coord = (x - 1, y)
        output.append(np.array(coord))    # no wall above
    if not chess_board[x, y, 1] and (x, y + 1) != adv_pos:
        coord = (x, y + 1)
        output.append(np.array(coord))    # no wall to the right
    if not chess_board[x, y, 2] and (x + 1, y) != adv_pos:
        coord = (x + 1, y)
        output.append(np.array(coord))    # no wall below
    if not chess_board[x, y, 3] and (x, y - 1) != adv_pos:
        coord = (x, y - 1)
        output.append(np.array(coord))    # no wall to the left
    '''
    print('neighbors = ', end='')
    for x in output:
        print(x, end='\t')
    print()
    '''
    return output


# finds all moves locations (squares) an agent can move to using Breadth-first search
# pos = agents position, max_step = maximum step size
def find_positions(chess_board, pos, adv_pos, max_step):
    size = len(chess_board)
    cost_lookup = np.full((size, size), -1, dtype=int)        # the cost to reach a given tile
    visited_nodes = []
    visit_queue = deque()
    visit_queue.append(pos)
    cost_lookup[tuple(pos)] = 0

    while len(visit_queue) > 0:
        #print('queue length = ' + str(len(visit_queue)))
        cur_pos = visit_queue.popleft()
        cur_step = cost_lookup[tuple(cur_pos)]
        visited_nodes.append(cur_pos)
        # print('cur step = ' + str(cur_step))
        # print('max step = ' + str(max_step))

        # if we can still move further, visit the node's neighbors
        if cur_step < max_step:
            for new_pos in get_neighbors(chess_board, cur_pos, tuple(adv_pos)):
                if cost_lookup[tuple(new_pos)] == -1:
                    cost_lookup[tuple(new_pos)] = cur_step + 1
                    visit_queue.append(new_pos)
    return visited_nodes

def check_endgame(board, p0_pos, p1_pos):
        """
        Check if the game ends and compute the current score of the agents.
        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        # Union-Find
        father = dict()
        for r in range(len(board)):
            for c in range(len(board)):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(len(board)):
            for c in range(len(board)):
                for dir, move in enumerate(
                    moves[1:3]
                ):  # Only check down and right
                    if board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(len(board)):
            for c in range(len(board)):
                find((r, c))
        p0_r = find(tuple(p0_pos))
        p1_r = find(tuple(p1_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score
        player_win = None
        if p0_score > p1_score:
            player_win = 0
        elif p0_score < p1_score:
            player_win = 1
        else:
            player_win = -1  # Tie
        return True, player_win

# finds all moves possible moves an agent can make (including wall placements)
# pos = agents position, max_step = maximum step size
def find_moves(chess_board, pos, adv_pos, max_step):
    moves = []
    for cur_pos in find_positions(chess_board, pos, adv_pos, max_step):
        x, y = cur_pos
        if not chess_board[x, y, 0]:
            moves.append((x, y, dir_map['u']))     # place wall above
        if not chess_board[x, y, 1]:
            moves.append((x, y, dir_map['r']))     # place wall to the right
        if not chess_board[x, y, 2]:
            moves.append((x, y, dir_map['d']))     # place wall below
        if not chess_board[x, y, 3]:
            moves.append((x, y, dir_map['l']))     # place wall to the left
    return moves

#calculates best UCT value of children
def UCT (node, c = sqrt(2)):
        log_ns = log(node.visits,10)
        best_score = float('-inf')
        best_node = node
        for child in node.children:
            if child.visits == 0: 
                val = float('inf')
            else : 
                val = child.wins/child.visits + c * sqrt(log_ns/child.visits)

            if val > best_score:
                best_score = val
                best_node = child

        return best_node

#traverses explored nodes until leaf
def tree_policy (root):
    curr = root
    while len(curr.children) > 0:
        curr = UCT(curr)
    return curr

#adapted from world.py
def set_barrier(r, c, dir, board):
        # Set the barrier to True
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        board[r, c, dir] = True
        # Set the opposite barrier to True
        move = moves[dir]
        board[r + move[0], c + move[1], opposites[dir]] = True
        return board


def print_board(chess_board, p0_pos, p1_pos):
    board_size = len(chess_board)
    for x in range(0, board_size):
        for y in range(0, board_size):
            if chess_board[x, y, 0]:
                print(' -', end='')
            else:
                print('  ', end='')
        print()
        for y in range(0, board_size):
            if chess_board[x, y, 3]:
                print('|O', end='')
            else:
                print(' O', end='')
            if x == p0_pos[0] and y == p0_pos[1]:
                print('\bP', end='')
            if x == p1_pos[0] and y == p1_pos[1]:
                print('\bX', end='')
        print('|')
    print(' -' * board_size + '\n')

def distance_to_center_heuristic(chess_board, pos):
    x, y = tuple(pos)
    center = len(chess_board)/2
    return abs(x - center) + abs(y - center)


# calculates the average number of walls in a 'rad' radius area centered on the agent,
# with the idea that being in a more open space is better than an enclosed space
def open_space_heuristic(chess_board, pos, rad):
    x_pos, y_pos = tuple(pos)
    num_squares = 0
    num_walls = 0
    for x in range(max(0, x_pos - rad), min(len(chess_board), x_pos + rad)):
        for y in range(max(0, y_pos - rad), min(len(chess_board), y_pos + rad)):
            num_squares += 1
            for i in range(0, 4):
                if chess_board[x, y, i]:
                    num_walls += 1
    return num_walls/num_squares


def heuristic(chess_board, pos):
    '''
    WEIGHT VALUES ARE NOT FINAL
    '''
    center_w = -1                 # should be negative, so distances farther to center are worse
    space_w = -1                  # should be negative, so areas with more walls are worse
    open_space = space_w * open_space_heuristic(chess_board, pos, 1)
    center = center_w * distance_to_center_heuristic(chess_board, pos)
    return center + open_space

@register_agent("naiveMCST_agent")
class NaiveMCSTAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(NaiveMCSTAgent, self).__init__()
        self.name = "HeuristicAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.first_move = True

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer
        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.
        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        #add root node 
        root = MCTSnode(my_pos, adv_pos, chess_board)
        #timer
        endpoint = time.time() + 1.9
        if self.first_move:
            endpoint = time.time() + 1.9
            self.first_move = False

        while(1):

            #use UCT to reach a leaf
            simNode = tree_policy(root)

            if simNode.agent :
                end, score = check_endgame(simNode.board, simNode.pos, simNode.adv_pos)
            else :
                end, score = check_endgame(simNode.board, simNode.adv_pos, simNode.pos)
            
            #if not
            if not end:
                
                #find all new moves from simNode which is a leaf
                move_list = find_moves(simNode.board, simNode.pos, simNode.adv_pos, max_step)
                for r,c,wall in move_list:

                    #create a new board for each possible move
                    new_board = deepcopy(simNode.board)
                    new_board = set_barrier(r,c,wall, new_board)
                    temp = MCTSnode(simNode.adv_pos, (r,c), new_board, (not simNode.agent))

                    #saves which wall was added (only for return at the end of step)
                    temp.set_wall(wall)

                    #add the new node to the tree
                    simNode.add_child(temp)
                    temp.add_parent(simNode)

                # for s in simNode.children:
                    # print (s.adv_pos)
                #randomly select new node
                simNode = choice(simNode.children)
                #create copy of it
                testcpy = MCTSnode(simNode.pos, simNode.adv_pos, simNode.board, simNode.agent)
                testcpy.set_wall(simNode.wall)

                #while not game is not over
                while(1):
                    if testcpy.agent :
                        end, score = check_endgame(testcpy.board, testcpy.pos, testcpy.adv_pos)
                    else :
                        end, score = check_endgame(testcpy.board, testcpy.adv_pos, testcpy.pos)

                    if end: 
                        break
                    
                    #simulate game with copy of new randomly selected node
                    move_list = find_moves(testcpy.board, testcpy.pos, testcpy.adv_pos, max_step)
                    r,c,wall = choice(move_list)
                    nb = testcpy.board
                    nb = set_barrier(r,c,wall, nb)
                    testcpy.set_adv(not testcpy.agent)
                    testcpy.set_pos(testcpy.adv_pos)
                    testcpy.set_advpos((r,c))
                    testcpy.update_board(nb)
                    testcpy.set_wall(wall)

            #check winner here 
            if (score == 0): 
                while simNode != None: 
                    simNode.update_value(1)
                    simNode = simNode.parent
            
            else:
                while simNode != None: 
                    simNode.update_value(0)
                    simNode = simNode.parent

            if (time.time() > endpoint):
                break
        
        max = 0.0
        bestnode = root.children[0]
        for c in root.children: 
            ratio = c.wins/(c.visits +1)
            if ratio > max:
                bestnode = c
                max = ratio

        return bestnode.adv_pos, bestnode.wall
