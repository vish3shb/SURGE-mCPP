import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
import sys
import json5
import pandas as pd
sys.path.append('DARP')
from multiRobotPathPlanner import MultiRobotPathPlanner
from optuna.exceptions import TrialPruned
from optuna.trial import TrialState
from global_to_local import process_polygon
from time_sampling import time_sample_paths
import plotly  # Ensure plotly is imported

class optimize():
    def __init__(self, rows, cols, number_of_trials, nep, portions, obstacles_positions, num_drones, top_n):
        # DARP Parameters
        self.number_of_drones = num_drones
        self.cols = cols
        self.rows = rows
        self.nep = nep
        self.portions = portions
        self.obstacles_positions = obstacles_positions
        self.obstacles_coords = obstacles_positions
        
        # Optimization Parameters
        self.sampler = optuna.samplers.TPESampler()
        self.number_of_trials = number_of_trials
        self.results = []
        self.best_avg = sys.maxsize
        self.all_instances = []
        self.top_n = top_n
        self.top_trials = []

    def optimize(self):        
        study = optuna.create_study(
            study_name="study",
            direction="minimize",
            sampler=self.sampler,
            pruner=optuna.pruners.MedianPruner()
        )

        study.optimize(self.objective, n_trials=self.number_of_trials)
        self.study = study
        
        sorted_trials = sorted(study.trials, key=lambda x: x.value if x.value is not None else float('inf'))
        self.top_trials = sorted_trials[:self.top_n]
        
        print("=========== Optimization Results ===========")        
        print("Number of finished trials: ", len(self.study.trials))
        for i, trial in enumerate(self.top_trials):
            print(f"Top {i+1} trial: {trial.number} with value: {trial.value}")
        print("============================================")        

    def objective(self, trial):
        positions = []
        for i in range(self.number_of_drones):
            positions.append(trial.suggest_int(f"p{i}", 0, self.rows*self.cols-1))

        if len(positions) != len(set(positions)):
            self.all_instances.append("Pruned")
            raise TrialPruned()
        
        for obstacle in self.obstacles_positions:
            for p in positions:
                if p == obstacle:
                    self.all_instances.append("Pruned")
                    raise TrialPruned()

        multiRobotPathPlanner_instance = MultiRobotPathPlanner(self.rows, self.cols, self.nep, positions, self.portions, self.obstacles_coords, False)
        self.all_instances.append(multiRobotPathPlanner_instance)

        if not multiRobotPathPlanner_instance.DARP_success:
            raise TrialPruned()
        else:
            return max(multiRobotPathPlanner_instance.best_case.turns)

    def save_top_paths(self, results_folder):
        for i, trial in enumerate(self.top_trials):
            initial_positions = []
            for robot, position in trial.params.items():
                initial_positions.append(position)

            multiRobotPathPlanner_instance = MultiRobotPathPlanner(self.rows, self.cols, self.nep, initial_positions, 
                                                                   self.portions, self.obstacles_coords, False)
            path_folder = os.path.join(results_folder, f"Top_{i+1}_trial_paths")
            path_files = multiRobotPathPlanner_instance.save_paths_txt(multiRobotPathPlanner_instance.best_case.paths, path_folder)
            
            # Time sampling the paths
            time_sampled_folder = os.path.join(path_folder, "time_sampled")
            time_sample_paths(path_files, time_sampled_folder)

    def export_results(self):
        if not self.obstacles_positions:
            obs_flag = "No_obstacles"
        else:
            obs_flag = "With_obstacles"
        
        results_folder = f'/home/samshad/Documents/multi_uav/CA_DARP/sampled_paths/x={self.rows}_y={self.cols}_num_drones={self.number_of_drones}_{obs_flag}'
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        with open(f'{results_folder}/optimization_results.txt', "a") as txt_file:
            for i, trial in enumerate(self.top_trials):
                txt_file.write(f"Top {i+1} trial: {trial.number}\n")
                txt_file.write(f"Value: {trial.value}\n")
                txt_file.write(f"Initial positions: {trial.params}\n")
                txt_file.write(f"Initial positions in cartesian coordinates: {self.all_instances[trial.number].darp_instance.initial_positions}\n")
                txt_file.write("\n")
            txt_file.close()

        # Reintroducing contour plot generation
        fig = optuna.visualization.plot_contour(self.study)
        fig.write_image(f'{results_folder}/Results_contour.png')

        self.save_top_paths(results_folder)

    def run_DARP_with_best_results(self, visualization):
        for i, trial in enumerate(self.top_trials):
            initial_positions = []
            for robot, position in trial.params.items():
                initial_positions.append(position)

            multiRobotPathPlanner_instance = MultiRobotPathPlanner(self.rows, self.cols, self.nep, initial_positions, 
                                                                   self.portions, self.obstacles_coords, visualization)
            if visualization:
                plt.title(f'Top {i+1} trial: {trial.number}')
                plt.savefig(f'{results_folder}/Top_{i+1}_path.png')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '-grid',
        default=(15, 10),
        type=int,
        nargs=2,
        help='Dimensions of the Grid (default: (10, 10))')
    argparser.add_argument(
        '-obs_pos',
        default=[],
        nargs='*',
        type=int,
        help='Dimensions of the Grid (default: (10, 10))')
    argparser.add_argument(
        '-nep',
        action='store_true',
        help='Not Equal Portions shared between the Robots in the Grid (default: False)')
    argparser.add_argument(
        '-portions',
        default=[0.2, 0.3, 0.5],
        nargs='*',
        type=float,
        help='Portion for each Robot in the Grid (default: (0.2, 0.7, 0.1))')
    argparser.add_argument(
        '-vis',
        action='store_true',
        help='Visualize results (default: False)')
    argparser.add_argument(
        '-num_drones',
        default=[3],
        type=int,
        nargs=1,
        help='Insert desired number of drones')
    argparser.add_argument(
        '-number_of_trials',
        default=[200],
        type=int,
        nargs=1,
        help='Insert desired number of trials')
    argparser.add_argument(
        '-top_n',
        default=[5],
        type=int,
        nargs=1,
        help='Number of top trials to save')
    
    args = argparser.parse_args()
    rows, cols = args.grid
    num_drones = args.num_drones[0]
    top_n = args.top_n[0]

    # Sanity checks:
    for obstacle in args.obs_pos:
        if obstacle < 0 or obstacle >= rows*cols:
            print("Obstacles should be inside the Grid.")
            sys.exit(3)

    portions = []
    if args.nep:
        portions.extend(args.portions)
    else:
        for drone in range(num_drones):
            portions.append(1/num_drones)

    if num_drones != len(portions):
        print("Portions should be defined for each drone")
        sys.exit(4)

    s = sum(portions)
    if abs(s-1) >= 0.0001:
        print("Sum of portions should be equal to 1.")
        sys.exit(1)

    print("\nInitial Conditions Defined:")
    print("Grid Dimensions:", rows, cols)
    print("Robot Number:", num_drones)
    print("Portions for each Robot:", portions, "\n")

    # Paths
    input_file_path = '/home/samshad/Documents/multi_uav/CA_DARP/flyzon.poly'

    # Parameters
    cell_size = 30  # size of the grid cell
    coverage_threshold = 0.2  # minimum coverage percentage to include a cell

    # Process the polygon
    result = process_polygon(input_file_path, cell_size, coverage_threshold)

    # Extract canvas size and obstacle positions
    rows, cols = result['canvas_size']
    obstacle_cells = result['obstacle_cells']
    polygon_cells = result['polygon_cells']
    
    optimization = optimize(rows, cols, args.number_of_trials[0], args.nep, portions, obstacle_cells, num_drones, top_n)
    optimization.optimize()
    optimization.export_results()
   
    if args.vis:
        optimization.run_DARP_with_best_results(args.vis)
