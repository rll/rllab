import numpy as np
from .base import Env
from rllab.spaces import Discrete, Box
from rllab.envs.base import Step
from rllab.core.serializable import Serializable

MAPS = {
    "chain": [ #MAX 2 agents
        ".xxx.xxx.",
        "........."
    ],
    "4x4": [
        "....",
        ".xx.",
        ".xx.",
        "...."
    ],
    "5x5": [
        ".....",
        ".xxx.",
        ".....",
        ".xxx.",
        "....."
    ],
    "7x7": [
        "..xxx..",
        ".x...x.",
        "...x...",
        "..xxx..",
        "...x...",
        ".x...x.",
        "..xxx.."
    ],
    "9x9_danger": [
        "o...o...o",
        "..x...x..",
        "....o....",
        "..xxxxx..",
        ".........",
        "..xxxxx..",
        "....o....",
        "..x...x..",
        "o...o...o"
    ],
}


class MultiAgentGridWorldEnv(Env, Serializable):
    """
    'A','B',..., 'F' : starting points of agents A, B, C,...,F
    'a', 'b', ..., 'f': goals of agents A, B, ..., F
    '.': free space
    'x': wall
    'o': hole (terminates episode)
    """

    # n: number of agents
    def __init__(self, desc='4x4', n=2, seed=0):
        
        assert(n <= 6 and n >= 2);
        
        Serializable.quick_init(self, locals())
        self.input_desc = desc
        if isinstance(desc, str):
            desc = MAPS[desc]
        desc = np.array(list(map(list, desc)))
        self.raw_desc = desc
        self.n_row, self.n_col = desc.shape
        self.n_agent = n # number of agents
        # generate starting locations and goals
        self.gen_start_and_goal()
        self.gen_initial_state()
        self.domain_fig = None

    # generate start positions and goal positions for agents
    #  --> assume every map is fully connected
    def get_empty_location(self):
        while True:
            x = np.random.randint(0, self.n_row)
            y = np.random.randint(0, self.n_col)
            if self.desc[x,y] == '.':
              return (x, y)
        return (-1, -1)
    def gen_start_and_goal(self):
        self.desc = self.raw_desc.copy()
        self.cur_pos = []
        self.cur_pos_map = [[-1] * self.n_col] * self.n_row
        self.tar_pos = []
        self.is_done = [False] * self.n_agent
        for agent in range(self.n_agent):
            x_s, y_s = self.get_empty_location() # start position
            self.desc[x_s,y_s] = chr(ord('A')+agent)
            self.cur_pos.append((x_s,y_s))
            self.cur_pos_map[x_s][y_s] = agent
            x_g, y_g = self.get_empty_location() # goal position
            self.desc[x_g,y_g] = chr(ord('a')+agent)
            self.tar_pos.append((x_g,y_g))

    # generate initial states
    #  --> must be called after gen_start_and_goal()
    #  states[0]: global map
    #  states[i+1] for i >= 0: position and goal for agent i
    def gen_initial_state(self):
        self.state = np.zeros((self.n_agent+1,self.n_row, self.n_col))
        for i in range(self.n_agent):
            cp = self.cur_pos[i]
            tp = self.tar_pos[i]
            self.state[i+1][cp[0]][cp[1]]=1 #current position
            self.state[i+1][tp[0]][tp[1]]=-1 #goal
            self.state[0][cp[0]][cp[1]]=0.5 #agent
        for x in range(self.n_row):
            for y in range(self.n_col):
                if self.raw_desc[x][y] == 'x':
                    self.state[0][x][y] = 1 # wall
                elif self.raw_desc[x][y] == 'o':
                    self.state[0][x][y] = -1 # hole

    def reset(self):
        # re-generate all the positions of the agents
        self.gen_start_and_goal()
        self.gen_initial_state()
        
        assert self.observation_space.contains(self.state)
        
        return self.state

    @staticmethod
    def action_from_direction(d):
        """
        Return the action corresponding to the given direction. This is a helper method for debugging and testing
        purposes.
        :return: the action index corresponding to the given direction
        """
        return dict(
            left=0,
            down=1,
            right=2,
            up=3,
            stay=4
        )[d]
    
    # Helper Method
    def plot_current_map(self):
        ret = self.raw_desc.copy()
        for i in range(self.n_agent):
            if self.cur_pos[i][0] > -1:
                ret[self.tar_pos[i]] = chr(ord('a') + i)
        for i in range(self.n_agent):
            if self.cur_pos[i][0] > -1:
                ret[self.cur_pos[i]] = chr(ord('A') + i)
        return ret


    def step(self, action):
        """
        action map:
        0: left
        1: down
        2: right
        3: up
        4: stay
        :param action: a list of 0~4 in length of self.n_agent
        :return:
        """
        # return [(next_state, next_pos, reward, done, prob)]
        possible_next_states = self.get_possible_next_states(action)

        probs = [x[4] for x in possible_next_states]
        next_state_idx = np.random.choice(len(probs), p=probs)
        next_state_info = possible_next_states[next_state_idx]

        reward = next_state_info[2]
        done = next_state_info[3]
        self.cur_pos = next_state_info[1]
        next_state = next_state_info[0]

        self.state = next_state
        return Step(observation=self.state, reward=reward, done=done)

    
    
    def get_possible_next_states(self, joint_action):
        """
        Given the action, return a list of possible next states and their probabilities. Only next states
        with nonzero probabilities will be returned
        :param action: action, a list of integers in [0, 4]
        :return: a list of tuples (s', p(s'|s,a))
                 here s' = (observation, next_agent_positions, reward, done)
        """
        assert self.action_space.contains(joint_action)
        # convert into a list of actions
        action = []
        for i in range(self.n_agent):
            action.append(joint_action % 5)
            joint_action /= 5

        # if agent_i escapes, pos[i] = [-1, -1]
        coors = self.cur_pos
        next_state = self.state.copy()
        reward = 0
        done = False
        remain = 0
        increments = [[0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]]
        next_coors = []
        mark = [False] * self.n_agent
        
        # move agents
        for i in range(self.n_agent):
            x, y = coors[i]
            if x > 0:
                remain += 1 # an active agent
            # single move
            tx, ty = x + increments[action[i]][0], y + increments[action[i]][1]
            # already escape || out of range || hit wall || stay
            if x < 0 or \
               tx < 0 or ty < 0 or tx >= self.n_row or ty >= self.n_col \
               or self.raw_desc[tx][ty] == 'x' \
               or action[i] == 4:
                mark[i] = True
                next_coors.append(coors[i])
                if x > -1 and action[i] != 4: # move to wall or out of range
                    reward -= 1
            else:
                next_coors.append((tx, ty))
                if self.raw_desc[tx][ty] == 'o': # check holes
                    reward -= 1
                    remain -= 1 # dead
                    mark[i] = True
                    done = True # game over
        
        # check collisions
        while True:
            has_colide = False
            for i in range(self.n_agent):
                if not mark[i]:
                    for j in range(i+1, self.n_agent):
                        # move to the same cell || move in opposite dir
                        if next_coors[j] == next_coors[i] or \
                           (next_coors[j] == coors[i] and coors[j] == next_coor[i]):
                            has_colide = True
                            next_coors[i] = coors[i]
                            mark[i] = True
                            reward -= 1
                            if not mark[j]:
                                mark[j] = True
                                reward -= 1
                                next_coors[j] = coors[j]
                            break
            if not has_colide:
                break
            
        # update next_state and check escape    
        for i in range(self.n_agent):
            if next_coors[i] == coors[i]: # do nothing
                continue
            x, y = coors[i]
            next_state[0][x][y] = 0
            next_state[i + 1][x][y] = 0
            # check escape
            x, y = next_coors[i]
            if self.raw_desc[x][y] == chr(ord('a') + i): # reach goal
                remain -= 1
                reward += 1
                next_state[i + 1][x][y] = 0 # clear the whole channel
                next_coors[i] = (-1, -1)
            else: # normal move
                next_state[0][x][y] = 0.5
                next_state[i + 1][x][y] = 1
        
        # check if finished
        if remain == 0:
            done = True

        # return, currently assume deterministic transition
        return [(next_state, next_coors, reward, done, 1.)]

    @property
    def action_space(self):
        return Discrete(5 ** self.n_agent) # convert multi-actions into a single joint action

    @property
    def observation_space(self):
        # channel 0: raw map, -1~hole, 0~free, 0.5~agent, 1~wall
        # channel i+1: agent_i, -1~goal, 0~free, 1~agent
        return Box(-1.0, 1.0, (self.n_agent+1,self.n_row,self.n_col))

