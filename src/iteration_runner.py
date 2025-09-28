"""
Iteration runner module for Sailfish VRP Optimizer.
Handles the execution of optimization iterations.
"""

import logging
from typing import Dict, List
from .population_manager import PopulationManager
from .position_updater import PositionUpdater
from .replacement_manager import ReplacementManager
from .results_reporter import ResultsReporter

logger = logging.getLogger(__name__)


class IterationRunner:
    """Handles the execution of optimization iterations."""
    
    def __init__(self, epsilon: float = 0.001):
        """
        Initialize iteration runner.
        
        Args:
            epsilon: Epsilon parameter for convergence checking
        """
        self.epsilon = epsilon
        self.fitness_history = []
    
    def run_iteration_zero(self, 
                          population_manager: PopulationManager,
                          position_updater: PositionUpdater,
                          results_reporter: ResultsReporter,
                          max_iter: int,
                          A: float) -> None:
        """
        Run iteration zero (initial population setup).
        
        Args:
            population_manager: Population manager instance
            position_updater: Position updater instance
            results_reporter: Results reporter instance
            max_iter: Maximum number of iterations
            A: Sailfish optimizer parameter
        """
        logger.info("="*80)
        logger.info("SAILFISH VRP OPTIMIZATION - ITERATION 0")
        logger.info("="*80)
        
        # Generate initial populations
        population_manager.generate_initial_populations()
        
        # Save original positions
        population_manager.save_original_positions()
        
        # Convert to routes
        population_manager.convert_populations_to_routes()
        
        # Calculate fitness
        population_manager.calculate_population_fitness()
        
        # Update elite positions
        population_manager.update_elite_positions()
        
        # Save sorted positions for updates
        population_manager.save_sorted_positions()
        
        # Print detailed results to file
        results_reporter.print_initial_parameters(
            problem_size=population_manager.problem_size,
            n_sailfish=population_manager.n_sailfish,
            n_sardines=population_manager.n_sardines,
            max_iter=max_iter,
            A=A,
            epsilon=self.epsilon,
            depot_data=population_manager.depot_data,
            customers_data=population_manager.customers_data,
            max_capacity=population_manager.max_capacity,
            max_vehicles=population_manager.max_vehicles
        )
        
        results_reporter.print_random_populations(
            population_manager.sailfish_random_values,
            population_manager.sardine_random_values,
            population_manager.problem_size
        )
        
        results_reporter.print_routes_and_solutions(
            population_manager.sailfish_random_values,
            population_manager.sardine_random_values,
            population_manager.sailfish_routes,
            population_manager.sardine_routes,
            population_manager.depot_data,
            population_manager.customers_data,
            population_manager.max_capacity,
            population_manager.max_vehicles
        )
        
        results_reporter.print_fitness_summary(
            population_manager.sailfish_fitness,
            population_manager.sardine_fitness,
            population_manager.best_fitness,
            population_manager.best_routes
        )
        
        results_reporter.print_comprehensive_results_table(
            population_manager.sailfish_random_values,
            population_manager.sardine_random_values,
            population_manager.sailfish_routes,
            population_manager.sardine_routes,
            population_manager.sailfish_fitness,
            population_manager.sardine_fitness,
            population_manager.best_fitness
        )
        
        # Print terminal output
        algorithm_params = {
            'n_sailfish': population_manager.n_sailfish,
            'n_sardines': population_manager.n_sardines,
            'max_iter': max_iter,
            'A': A,
            'epsilon': self.epsilon
        }
        
        vrp_params = {
            'max_capacity': population_manager.max_capacity,
            'max_vehicles': population_manager.max_vehicles,
            'problem_size': population_manager.problem_size,
            'data_file': population_manager.data_file
        }
        
        results_reporter.print_terminal_data_table(
            population_manager.depot_data,
            population_manager.customers_data,
            population_manager.max_capacity,
            population_manager.max_vehicles
        )
        
        results_reporter.print_terminal_parameters(algorithm_params, vrp_params)
        
        # Find best sailfish for terminal output
        best_sailfish_idx = population_manager.sailfish_fitness.index(min(population_manager.sailfish_fitness))
        results_reporter.print_terminal_iteration_best(
            0, population_manager.best_fitness, best_sailfish_idx, population_manager.best_routes
        )
        
        # Store fitness history
        self.fitness_history.append(population_manager.best_fitness)
    
    def run_iteration(self, 
                     iteration_num: int,
                     population_manager: PopulationManager,
                     position_updater: PositionUpdater,
                     replacement_manager: ReplacementManager,
                     results_reporter: ResultsReporter,
                     A: float) -> None:
        """
        Run a single optimization iteration.
        
        Args:
            iteration_num: Current iteration number
            population_manager: Population manager instance
            position_updater: Position updater instance
            replacement_manager: Replacement manager instance
            results_reporter: Results reporter instance
            A: Sailfish optimizer parameter
        """
        logger.info("="*80)
        logger.info(f"ITERATION {iteration_num}")
        logger.info("="*80)
        
        # Calculate PD and lambda values
        PD, lambda_k_values = position_updater.calculate_pd_and_lambda_values(
            population_manager.n_sailfish, population_manager.n_sardines
        )
        
        # Update positions
        position_updater.update_sailfish_positions(
            population_manager.sailfish_random_values,
            population_manager.sorted_sailfish_positions,
            population_manager.elite_sailfish_fitness,
            population_manager.injured_sardine_fitness,
            lambda_k_values,
            is_iteration_zero=False,
            population_manager=population_manager
        )
        
        # Calculate Attack Power (AP) - simplified for now
        AP = 2.0  # Default AP value
        
        position_updater.update_all_sardines(
            population_manager.sardine_random_values,
            population_manager.sorted_sardine_positions,
            population_manager.elite_sailfish_fitness,
            AP
        )
        
        # Convert updated positions to routes
        population_manager.convert_populations_to_routes()
        
        # Calculate fitness for updated populations
        population_manager.calculate_population_fitness()
        
        # Perform sailfish-sardine replacement
        replacement_stats = replacement_manager.perform_sailfish_sardine_replacement(
            population_manager, iteration_num
        )
        
        # Update elite positions
        population_manager.update_elite_positions()
        
        # Save sorted positions for next iteration
        population_manager.save_sorted_positions()
        
        # Print detailed results to file
        results_reporter.print_routes_and_solutions(
            population_manager.sailfish_random_values,
            population_manager.sardine_random_values,
            population_manager.sailfish_routes,
            population_manager.sardine_routes,
            population_manager.depot_data,
            population_manager.customers_data,
            population_manager.max_capacity,
            population_manager.max_vehicles,
            iteration_num
        )
        
        results_reporter.print_fitness_summary(
            population_manager.sailfish_fitness,
            population_manager.sardine_fitness,
            population_manager.best_fitness,
            population_manager.best_routes,
            iteration_num
        )
        
        results_reporter.print_comprehensive_results_table(
            population_manager.sailfish_random_values,
            population_manager.sardine_random_values,
            population_manager.sailfish_routes,
            population_manager.sardine_routes,
            population_manager.sailfish_fitness,
            population_manager.sardine_fitness,
            population_manager.best_fitness,
            iteration_num
        )
        
        # Print terminal output
        best_sailfish_idx = population_manager.sailfish_fitness.index(min(population_manager.sailfish_fitness))
        results_reporter.print_terminal_iteration_best(
            iteration_num, population_manager.best_fitness, best_sailfish_idx, population_manager.best_routes
        )
        
        # Store fitness history
        self.fitness_history.append(population_manager.best_fitness)
    
    def get_final_results(self, population_manager: PopulationManager) -> Dict:
        """
        Get final optimization results.
        
        Args:
            population_manager: Population manager instance
            
        Returns:
            Dictionary containing final results
        """
        return {
            'algorithm_parameters': {
                'n_sailfish': population_manager.original_n_sailfish,
                'n_sardines': population_manager.original_n_sardines,
                'max_iter': len(self.fitness_history) - 1,
                'A': 4.0,  # Default value
                'epsilon': self.epsilon
            },
            'vrp_parameters': {
                'max_capacity': population_manager.max_capacity,
                'max_vehicles': population_manager.max_vehicles,
                'problem_size': population_manager.problem_size,
                'data_file': population_manager.data_file
            },
            'best_solution': {
                'routes': population_manager.best_routes,
                'total_distance': population_manager.best_fitness
            },
            'fitness_evolution': {
                'initial': self.fitness_history[0] if self.fitness_history else 0,
                'final': population_manager.best_fitness,
                'improvement': self.fitness_history[0] - population_manager.best_fitness if self.fitness_history else 0,
                'history': self.fitness_history
            }
        }