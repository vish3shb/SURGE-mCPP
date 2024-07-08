import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
import sys
import time
sys.path.append('DARP')
from multiRobotPathPlanner import MultiRobotPathPlanner
from optuna.exceptions import TrialPruned
from optuna.trial import TrialState

# Import the global to local functions
from global_to_local import process_polygon
# Import the time sampling function
from time_sampling import time_sample_paths
# Import the network check function
from network_check import plot_paths_from_files, read_path_file, shift_path, reverse_shift_path, ensure_closed_path

class optimize():
    def __init__(self, rows, cols, number_of_trials, nep, portions, obstacles_positions, num_drones, allowed_comm_distance):
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
        self.allowed_comm_distance = allowed_comm_distance
        self.best_allowed_avg_turns = sys.maxsize
        self.best_allowed_path_info = None

    def optimize(self):        
        study = optuna.create_study(
            study_name="study",
            direction="minimize",
            sampler=self.sampler,
            pruner=optuna.pruners.MedianPruner()
        )

        study.optimize(self.objective, n_trials=self.number_of_trials)
        self.study = study
        
        # In case more than one trial has the same minimum required distance, pick the trial with the smaller average number of turns
        self.best_trial_number = self.study.best_trial.number
        self.min_number_of_turns = self.study.best_trial.value
        self.best_trial = self.all_instances[self.best_trial_number]
        self.best_avg = self.best_trial.best_case.avg
        
        for trial in study.trials:
            if trial.state == TrialState.PRUNED:
                continue

            if trial.value <= self.min_number_of_turns and (self.all_instances[trial.number].best_case.avg < self.best_avg):
                self.best_trial_number = trial.number
                self.min_number_of_turns = trial.value
                self.best_trial = self.all_instances[self.best_trial_number]
                self.best_avg = self.best_trial.best_case.avg
        
        print("=========== Optimization Results ===========")        
        print("Number of finished trials: ", len(self.study.trials))
        print(f"Best trial: {self.best_trial_number}")
        print(f'Best value: {self.min_number_of_turns}')
        print(f'Best Initial positions: {self.study.best_trial.params}')
        print(f'Best Initial positions in cartesian coordinates: {self.best_trial.darp_instance.initial_positions}')
        print("============================================")        

        # Save overall results to file
        with open("overall_results.txt", "w") as file:
            file.write("=========== Optimization Results ===========\n")
            file.write(f"Number of finished trials: {len(self.study.trials)}\n")
            file.write(f"Best trial: {self.best_trial_number}\n")
            file.write(f"Best value: {self.min_number_of_turns}\n")
            file.write(f"Best Initial positions: {self.study.best_trial.params}\n")
            file.write(f"Best Initial positions in cartesian coordinates: {self.best_trial.darp_instance.initial_positions}\n")
            file.write("============================================\n")

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

        # Save number of turns
        turns = [len(path) for path in multiRobotPathPlanner_instance.best_case.paths]
        avg_turns = multiRobotPathPlanner_instance.best_case.avg
        std_turns = multiRobotPathPlanner_instance.best_case.std

        # Time sample the paths
        path_files = multiRobotPathPlanner_instance.path_files
        output_directory = "sampled"
        os.makedirs(output_directory, exist_ok=True)
        sampled_path_files = time_sample_paths(path_files, output_directory)
        print(f"Sampled paths: {sampled_path_files}")

        # Network check with the sampled paths
        start_time = time.time()
        best_distance, best_combination = plot_paths_from_files(sampled_path_files, step_size=1, n_trials=200, verbose=False)
        connectivity_check_time = time.time() - start_time

        # Handle NoneType for best_distance
        best_distance_str = f"{best_distance:.3f}" if best_distance is not None else "N/A"

        # Save results to file
        overall_time = time.time() - start_time
        iteration_number = len(self.all_instances) - 1
        with open("iteration_results.txt", "a") as file:
            file.write(f"Iteration: {iteration_number}\n")
            file.write(f"Turns: {turns}\n")
            file.write(f"Average Turns: {avg_turns:.3f}\n")
            file.write(f"Standard Deviation Turns: {std_turns:.3f}\n")
            file.write(f"Minimum Required Distance: {best_distance_str}\n")
            file.write(f"Connectivity Check Time: {connectivity_check_time:.3f} seconds\n")
            file.write(f"Overall Time: {overall_time:.3f} seconds\n")
            file.write("\n")

        # Save paths and results to a separate file if the minimum required distance is less than the allowed communication distance
        if best_distance is not None and best_distance < self.allowed_comm_distance:
            allowed_result_dir = os.path.join("Results", "allowed_distance_results")
            os.makedirs(allowed_result_dir, exist_ok=True)
            with open(os.path.join(allowed_result_dir, "allowed_distance_results.txt"), "a") as allowed_file:
                allowed_file.write(f"Iteration: {iteration_number}\n")
                allowed_file.write(f"Turns: {turns}\n")
                allowed_file.write(f"Average Turns: {avg_turns:.3f}\n")
                allowed_file.write(f"Standard Deviation Turns: {std_turns:.3f}\n")
                allowed_file.write(f"Minimum Required Distance: {best_distance_str}\n")
                allowed_file.write(f"Connectivity Check Time: {connectivity_check_time:.3f} seconds\n")
                allowed_file.write(f"Overall Time: {overall_time:.3f} seconds\n")
                allowed_file.write(f"Shift Combination: {best_combination}\n")
                allowed_file.write("\n")
            
            # Apply shifts and save the modified path files
            shifted_path_files = []
            for path_file in sampled_path_files:
                path_filename = os.path.basename(path_file)
                robot_id = int(path_filename.split('_')[1])
                shift = best_combination[f'shift_{path_file}']
                
                # Read and apply shift to the path
                coordinates = read_path_file(path_file)
                coordinates = ensure_closed_path(coordinates)
                if shift >= 0:
                    shifted_coordinates = shift_path(coordinates, shift)
                else:
                    shifted_coordinates = reverse_shift_path(coordinates, -shift)
                
                # Save the shifted path
                shifted_path_file = os.path.join(allowed_result_dir, f"{iteration_number}_{path_filename}")
                with open(shifted_path_file, 'w') as spf:
                    for coord in shifted_coordinates:
                        spf.write(f"{coord[0]} {coord[1]} {coord[2]}\n")
                shifted_path_files.append(shifted_path_file)

            # Check if this path has the least turns among allowed paths
            if avg_turns < self.best_allowed_avg_turns:
                self.best_allowed_avg_turns = avg_turns
                self.best_allowed_path_info = {
                    "iteration": iteration_number,
                    "turns": turns,
                    "avg_turns": avg_turns,
                    "std_turns": std_turns,
                    "best_distance": best_distance_str,
                    "connectivity_check_time": connectivity_check_time,
                    "overall_time": overall_time,
                    "shift_combination": best_combination,
                    "shifted_path_files": shifted_path_files
                }

        # Use minimum required distance as the objective
        if best_distance is None:
            raise TrialPruned()
        return best_distance

    def export_results(self):
        if not self.obstacles_positions:
            obs_flag = "No_obstacles"
        else:
            obs_flag = "With_obstacles"
        
        if not os.path.exists("Results"):
            os.mkdir("Results")

        if not os.path.exists(f'Results/x={self.rows}_y={self.cols}_num_drones={self.number_of_drones}_{obs_flag}'):
            os.mkdir(f'Results/x={self.rows}_y={self.cols}_num_drones={self.number_of_drones}_{obs_flag}')
    
        with open(f'Results/x={self.rows}_y={self.cols}_num_drones={self.number_of_drones}_{obs_flag}/optimization_results.txt', "a") as txt_file:
            txt_file.write(f"Best trial: {self.best_trial_number}\n")
            txt_file.write(f"Best value: {self.min_number_of_turns}\n")
            txt_file.write(f"Best Initial positions: {self.study.best_trial.params}\n")
            txt_file.write(f"Best Initial positions in cartesian coordinates: {self.best_trial.darp_instance.initial_positions}\n")
            txt_file.write(f"Average number of turns: {self.best_avg}\n")
            txt_file.close()

        all_results = dict()
        for i in range(len(self.study.trials)):
            temp_all_results = list()
            for index, value in self.study.trials[i].params.items():
                temp_all_results.append(value)
            
            temp_all_results = tuple(temp_all_results)
            
            if self.study.trials[i].value is not None:
                all_results[temp_all_results] = self.study.trials[i].value
            else:
                all_results[temp_all_results] = None

        with open(f'Results/x={self.rows}_y={self.cols}_num_drones={self.number_of_drones}_{obs_flag}/all_results.txt', "a") as txt_file:            
            for positions, results in all_results.items():
                txt_file.write(f'Initial positions: {positions}, Number of turns: {results}\n')
            txt_file.close()

        fig = optuna.visualization.plot_contour(self.study)
        fig.write_image(f'Results/x={self.rows}_y={self.cols}_num_drones={self.number_of_drones}_{obs_flag}/Results_contour.png')

        # Save the best allowed distance result with least turns
        if self.best_allowed_path_info:
            least_turns_result_dir = os.path.join("Results", "least_turns_results")
            os.makedirs(least_turns_result_dir, exist_ok=True)
            for path_file in self.best_allowed_path_info["shifted_path_files"]:
                path_filename = os.path.basename(path_file)
                new_path_file = os.path.join(least_turns_result_dir, path_filename)
                os.rename(path_file, new_path_file)
            with open(os.path.join(least_turns_result_dir, "least_turns_results.txt"), "w") as least_turns_file:
                least_turns_file.write(f"Iteration: {self.best_allowed_path_info['iteration']}\n")
                least_turns_file.write(f"Turns: {self.best_allowed_path_info['turns']}\n")
                least_turns_file.write(f"Average Turns: {self.best_allowed_path_info['avg_turns']:.3f}\n")
                least_turns_file.write(f"Standard Deviation Turns: {self.best_allowed_path_info['std_turns']:.3f}\n")
                least_turns_file.write(f"Minimum Required Distance: {self.best_allowed_path_info['best_distance']}\n")
                least_turns_file.write(f"Connectivity Check Time: {self.best_allowed_path_info['connectivity_check_time']:.3f} seconds\n")
                least_turns_file.write(f"Overall Time: {self.best_allowed_path_info['overall_time']:.3f} seconds\n")
                least_turns_file.write(f"Shift Combination: {self.best_allowed_path_info['shift_combination']}\n")
                least_turns_file.write("\n")

    def run_DARP_with_best_results(self, visualization):
        initial_positions = []
        
        for robot, position in self.study.best_trial.params.items():
            initial_positions.append(position)

        multiRobotPathPlanner_instance = MultiRobotPathPlanner(self.rows, self.cols, self.nep, initial_positions, 
                                                               self.portions, self.obstacles_coords, visualization)

        # Time sample the paths
        path_files = multiRobotPathPlanner_instance.path_files
        output_directory = "sampled"
        os.makedirs(output_directory, exist_ok=True)
        sampled_path_files = time_sample_paths(path_files, output_directory)
        print(f"Sampled paths: {sampled_path_files}")

        # Network check with the sampled paths
        plot_paths_from_files(sampled_path_files, step_size=1, n_trials=100, verbose=False)


if __name__ == '__main__':
    start_time = time.time()

    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '-polygon_file',
        default=os.path.join(os.path.dirname(__file__), 'flyzon.poly'),
        type=str,
        help='Path to the polygon file (default: flyzon.poly in the same directory)')
    argparser.add_argument(
        '-nep',
        action='store_true',
        help='Not Equal Portions shared between the Robots in the Grid (default: False)')
    argparser.add_argument(
        '-portions',
        default=[0.2, 0.3, 0.5],
        nargs='*',
        type=float,
        help='Portion for each Robot in the Grid (default: (0.2, 0.3, 0.5))')
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
        '-allowed_comm_distance',
        default=850.0,
        type=float,
        help='Allowed communication distance threshold')
    
    args = argparser.parse_args()
    polygon_file_path = args.polygon_file
    num_drones = args.num_drones[0]

    # Process the polygon file
    polygon_data = process_polygon(polygon_file_path, cell_size=50, coverage_threshold=0.2)
    rows, cols = polygon_data["canvas_size"]
    obstacles_positions = polygon_data["obstacle_cells"]

    # Sanity checks:
    for obstacle in obstacles_positions:
        if obstacle < 0 or obstacle >= rows * cols:
            print("Obstacles should be inside the Grid.")
            sys.exit(3)

    portions = []
    if args.nep:
        portions.extend(args.portions)
    else:
        for drone in range(num_drones):
            portions.append(1 / num_drones)

    if num_drones != len(portions):
        print("Portions should be defined for each drone")
        sys.exit(4)

    s = sum(portions)
    if abs(s - 1) >= 0.0001:
        print("Sum of portions should be equal to 1.")
        sys.exit(1)

    print("\nInitial Conditions Defined:")
    print("Grid Dimensions:", rows, cols)
    print("Robot Number:", num_drones)
    print("Portions for each Robot:", portions, "\n")

    optimization = optimize(rows, cols, args.number_of_trials[0], args.nep, portions, obstacles_positions, num_drones, args.allowed_comm_distance)
    optimization.optimize()
    optimization.export_results()

    if args.vis:
        optimization.run_DARP_with_best_results(args.vis)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.3f} seconds")
