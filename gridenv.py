import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy

# rep'd by a matrix:
#  [ x, 0,  0, 0]
#  [ 0, 0,  0, G]
#  [ 0, 0,  G, 0]
#  [ G, 0,  0, 0]
# note that top left is 0,0
#
# Gs are goals, x is agent starting pos

class GridEnvSim:
    def __init__(self, w, h, rand_init, num_goals, rand_goals, task_only, rep_type):
        self.w = w
        self.h = h
        self.init_grid = np.zeros((h, w))
        self.grid = np.copy(self.init_grid)
        if rand_init:
            starting_loc = np.unravel_index(np.random.choice(self.grid.size, 1), (h, w))
            self.starting_loc = np.transpose(np.asarray(starting_loc))[0]
        else:
            self.starting_loc = np.asarray([0, 0])
        #print("starting_loc:")
        #print(self.starting_loc)
        #input("wait")
        #goal_locs = np.unravel_index(np.random.choice(self.grid.size, num_goals, replace=False), (h, w))
        #self.goal_locs = np.transpose(np.asarray(goal_locs))
        #while (self.starting_loc == self.goal_locs).all(1).any():
        #    #print("starting loc conflict, re-rolling")
        #    goal_locs = np.unravel_index(np.random.choice(self.grid.size, num_goals, replace=False), (h, w))
        #    self.goal_locs = np.transpose(np.asarray(goal_locs))
        #self.goal_locs = np.asarray([[18, 18]])
        self.goal_locs = np.asarray([[8, 8]])
        #print("goal_locs:")
        #print(self.goal_locs)
        #input("wait")
        self.grid[self.starting_loc[0], self.starting_loc[1]] = 1
        for g_loc in self.goal_locs:
            self.grid[g_loc[0], g_loc[1]] = 9
        #print("grid:")
        #print(self.grid)
        #input("wait")
        self.count_grid = np.copy(self.init_grid)
        self.count_grid[self.starting_loc[0], self.starting_loc[1]] = 1
        self.game_counter = 0
        #self.max_game_counter = 750
        self.max_game_counter = 200
        self.rand_init = rand_init
        self.num_goals = num_goals
        self.rand_goals = rand_goals

        self.stage = 0
        self.task_only = task_only
        if self.task_only:
            self.stage = 1 # 0 is char select, 1 is actual task
        self.chosen_char = -1
        self.rep_type = rep_type # 0 is pos, 1 is flattened grid, 2 is img-style grid

        if self.task_only:
            self.pos = np.asarray([self.starting_loc[0], self.starting_loc[1]])
        else:
            self.pos = np.asarray([0, 0])
        self.goals_left = copy.deepcopy(self.goal_locs)

    def get_state(self):
        if 0 == self.rep_type:
            return self.pos
        elif 1 == self.rep_type:
            return self.grid.flatten()
        elif 2 == self.rep_type:
            return self.grid
        else:
            return self.pos

    def get_state_shape(self):
        if 0 == self.rep_type:
            return np.asarray(self.pos).shape
        elif 1 == self.rep_type:
            return self.grid.flatten().shape
        elif 2 == self.rep_type:
            return self.grid.shape
        else:
            return np.asarray(self.pos).shape

    def get_actions(self):
        a_set = []

        if 0 == self.stage:
            # NOTE only the first 3 actions do anything!
            a_set = [0, 1, 2, -1, -1] # 0 = char 1 (standard), 1 = char 2 (good), 2 = char 3 (bad)
        else:
            #a_set = [0, 1, 2, 3, 4] # (for now) 0 = right, 1 = down, 2 = left, 3 = up, 4 = special move
            a_set = [0, 1, 2, 3]

        return a_set

    def get_actions_shape(self):
        a_set = []

        if 0 == self.stage:
            # NOTE only the first 3 actions do anything!
            a_set = [0, 1, 2, -1, -1] # *see above
        else:
            #a_set = [0, 1, 2, 3, 4] # *see above
            a_set = [0, 1, 2, 3]

        return np.asarray(a_set).shape

    def do_action(self, action):
        if 0 == self.stage:
            if action < 3:
                chosen_char = action
            else:
                chosen_char = -1

            return [chosen_char, -1]
        else:
            curr_y, curr_x = self.pos

            if action < 4:
                possible_a = [(curr_y, max(curr_x-1, 0)),
                         #(curr_y+1, curr_x-1),
                         (min(curr_y+1, self.h-1), curr_x),
                         #(curr_y+1, curr_x+1),
                         (curr_y, min(curr_x+1, self.w-1)),
                         #(curr_y-1, curr_x+1),
                         (max(curr_y-1, 0), curr_x)]
                         #(curr_y-1, curr_x-1)]

                new_pos = [possible_a[action][0], possible_a[action][1]]
            else:
                if 1 == self.chosen_char:
                    # teleports agent next to a random goal
                    cand_goal = self.goals_left[np.random.choice(len(self.goals_left), 1)[0]]

                    cand_locs = [(cand_goal[0], max(cand_goal[1]-1, 0)),
                            (min(cand_goal[0]+1, self.h-1), cand_goal[1]),
                            (cand_goal[0], min(cand_goal[1]+1, self.w-1)),
                            (max(cand_goal[0]-1, 0), cand_goal[1])]

                    cand_loc = cand_locs[np.random.choice(len(cand_locs), 1)[0]]
                    while (list(cand_loc) == self.goals_left).all(1).any():
                        cand_loc = cand_locs[np.random.choice(len(cand_locs), 1)[0]]

                    new_pos = [cand_loc[0], cand_loc[1]]
                else:
                    new_pos = self.pos

            return new_pos

    def update(self, action):
        s_prime = None
        r = None
        done = None

        if 0 == self.stage:
            chosen_char = self.do_action(action)[0]
            if -1 != chosen_char:
                #print("Character chosen!")
                self.chosen_char = chosen_char
                self.stage = 1
                r = 0.1
                if 0 == self.rep_type:
                    s_prime = self.starting_loc
                elif 1 == self.rep_type:
                    s_prime = self.grid.flatten()
                elif 2 == self.rep_type:
                    s_prime = self.grid
                else:
                    s_prime = self.starting_loc
                done = False
            else:
                #print("Invalid action!")
                r = 0
                if 0 == self.rep_type:
                    s_prime = [-1, -1]
                elif 1 == self.rep_type:
                    s_prime = self.init_grid.flatten()
                elif 2 == self.rep_type:
                    s_prime = self.init_grid
                else:
                    s_prime = [-1, -1]
                done = False
        else:
            next_pos = np.asarray(self.do_action(action))
            self.grid[self.pos[0], self.pos[1]] = 0
            self.grid[next_pos[0], next_pos[1]] = 1

            if (next_pos == self.goals_left).all(1).any():
                #print("reached goal!")
                r = 10
                #r = 100
                goal_idx = np.where((next_pos == self.goals_left).all(1))
                g_loc = self.goals_left[goal_idx]
                self.goals_left = np.delete(self.goals_left, goal_idx, 0)
            else:
                #r = -0.1
                r = 0

            if (0 == len(self.goals_left)) or (self.game_counter == self.max_game_counter):
                #if 0 == len(self.goals_left):
                #    print("All goals found!")
                #else:
                #    print("Time's up! Game over!")
                self.grid = np.copy(self.init_grid)
                if self.rand_init:
                    starting_loc = np.unravel_index(np.random.choice(self.grid.size, 1), (self.h, self.w))
                    self.starting_loc = np.transpose(np.asarray(starting_loc))[0]
                #print("RESET starting_loc:")
                #print(self.starting_loc)
                if self.rand_goals:
                    goal_locs = np.unravel_index(np.random.choice(self.grid.size, self.num_goals, replace=False), (self.h, self.w))
                    self.goal_locs = np.transpose(np.asarray(goal_locs))
                    while (self.starting_loc == self.goal_locs).all(1).any():
                        goal_locs = np.unravel_index(np.random.choice(self.grid.size, self.num_goals, replace=False), (self.h, self.w))
                        self.goal_locs = np.transpose(np.asarray(goal_locs))
                #print("RESET goal_locs:")
                #print(self.goal_locs)
                self.goals_left = copy.deepcopy(self.goal_locs)
                self.grid[self.starting_loc[0], self.starting_loc[1]] = 1
                for g_loc in self.goal_locs:
                    self.grid[g_loc[0], g_loc[1]] = 9
                #print("RESET grid:")
                #print(self.grid)
                #input("wait")
                self.game_counter = 0
                self.pos = self.starting_loc
                if not self.task_only:
                    self.stage = 0
                    if 0 == self.rep_type:
                        s_prime = [-1, -1]
                    elif 1 == self.rep_type:
                        s_prime = self.init_grid.flatten()
                    elif 2 == self.rep_type:
                        s_prime = self.init_grid
                    else:
                        s_prime = [-1, -1]
                else:
                    if 0 == self.rep_type:
                        s_prime = self.starting_loc
                    elif 1 == self.rep_type:
                        s_prime = self.grid.flatten()
                    elif 2 == self.rep_type:
                        s_prime = self.grid
                    else:
                        s_prime = self.starting_loc
                done = True
            else:
                self.game_counter += 1
                self.pos = next_pos
                if 0 == self.rep_type:
                    s_prime = next_pos
                elif 1 == self.rep_type:
                    s_prime = self.grid.flatten()
                elif 2 == self.rep_type:
                    s_prime = self.grid
                else:
                    s_prime = next_pos
                done = False

            self.count_grid[self.pos[0], self.pos[1]] += 1

        assert (s_prime is not None) and (r is not None) and (done is not None)

        #print("s_prime:")
        #print(s_prime)
        #print("r:")
        #print(r)
        #print("done:")
        #print(done)
        #input("wait")

        return (s_prime, r, done)

    def get_vis_states(self):
        vis_states = []

        for i in range(self.w):
            for j in range(self.h):
                cand_grid = np.zeros((self.h, self.w))
                cand_starting_loc = [j, i]
                #cand_goal_locs = [[18, 18]]
                cand_goal_locs = [[8, 8]]
                for g_loc in cand_goal_locs:
                    cand_grid[g_loc[0], g_loc[1]] = 9
                cand_grid[cand_starting_loc[0], cand_starting_loc[1]] = 1

                vis_states.append(cand_grid.tolist())

        return vis_states
