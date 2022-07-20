from arguments import get_arguments
from state_encodings import ENCODING_SCHEMES
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, OvercookedGridworld, Direction, Action
from overcooked_ai_py.planning.planners import MotionPlanner

from copy import deepcopy
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class Heuristic():
    def __init__(self, grid, state, mlam):
        self.state = state
        self.players_pos_or_list = self.state.players_pos_and_or
        self.grid = grid
        self.terrain_mtx = self.grid.terrain_mtx
        self.planner = mlam
    def heuristic_1(self):
        heuristic1 = []
        for i,player in enumerate(self.state.players):
            distance_heur = dict()
            # current player position
            player_pos_or = self.players_pos_or_list[i]
            #calculate heuristics for onion class
            onion_disp_loc = self.grid.get_onion_dispenser_locations()
            min_steps_onion = self.planner.min_cost_to_feature(player_pos_or, onion_disp_loc)
            #calculate heuristics for dish class
            dish_disp_loc = self.grid.get_dish_dispenser_locations()
            min_steps_dish = self.planner.min_cost_to_feature(player_pos_or, dish_disp_loc)
            #calculate heuristics for soup class
            pot_disp_loc = self.grid.get_pot_locations()
            serving_locations = self.grid.get_serving_locations()
            # find closest pot
            min_steps_to_pot, best_pot_pos = self.planner.min_cost_to_feature(player_pos_or, pot_disp_loc, True)
            # distance between closest pot and serving locations
            min_steps_to_serving = self.planner.min_cost_between_features([best_pot_pos], serving_locations)
            min_steps_soup = min_steps_to_pot + min_steps_to_serving
            distance_heur["onion"] = min_steps_onion
            distance_heur["dish"] = min_steps_dish
            distance_heur["soup"] = min_steps_soup
            heuristic1.append(distance_heur)
        return heuristic1

    def heuristic2(self, history0, history1):
        # dividing by history to find task probabiity
        task_counter_sum = [ [sum(x)/ len(history1) for x in zip(*history0)], [sum(x)/ len(history1) for x in zip(*history1)]]
        return task_counter_sum
    
    def compute_heuristics(self, history0, history1):
        heuristics = []
        heuristics.append(self.heuristic_1())
        heuristics.append(self.heuristic2(history0, history1))
        return heuristics
        

