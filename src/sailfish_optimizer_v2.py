"""
Modularized Sailfish VRP Optimizer.
This is a refactored version that uses separate modules for better organization.
"""

import logging
from typing import Dict

from .population_manager import PopulationManager
from .position_updater import PositionUpdater
from .replacement_manager import ReplacementManager
from .results_reporter import ResultsReporter
from .iteration_runner import IterationRunner

logger = logging.getLogger(__name__)


class SailfishVRPOptimizerV2:
    """
    Modularized Sailfish VRP Optimizer implementing the Sailfish Optimizer algorithm.
    
    This class orchestrates the optimization process using separate modules for:
    - Population management
    - Position updates
    - Replacement operations
    - Results reporting
    - Iteration execution
    """
    
    def __init__(self, 
                 n_sailfish: int,
                 n_sardines: int,
                 depot_data: Dict,
                 customers_data: list,
                 max_capacity: float = 200,
                 max_vehicles: int = 2,
                 max_iter: int = 100,
                 A: float = 4,
                 epsilon: float = 0.001,
                 log_to_file: bool = True,
                 output_mode: str = 'steps',
                 data_file: str = 'kecil.csv'):
        """
        Initialize Modularized Sailfish VRP Optimizer.
        
        Args:
            n_sailfish: Number of sailfish
            n_sardines: Number of sardines
            depot_data: Depot information dictionary
            customers_data: List of customer information dictionaries
            max_capacity: Maximum vehicle capacity
            max_vehicles: Maximum number of vehicles
            max_iter: Maximum number of iterations
            A: Sailfish optimizer parameter
            epsilon: Attack Power calculation parameter (not used for convergence)
            log_to_file: Whether to log output to file
        """
        # Store algorithm parameters
        self.max_iter = max_iter
        self.A = A
        self.epsilon = epsilon
        
        # Initialize modules
        self.population_manager = PopulationManager(
            n_sailfish=n_sailfish,
            n_sardines=n_sardines,
            depot_data=depot_data,
            customers_data=customers_data,
            max_capacity=max_capacity,
            max_vehicles=max_vehicles,
            log_to_file=log_to_file,
            output_mode=output_mode,
            data_file=data_file
        )
        
        self.position_updater = PositionUpdater(
            problem_size=len(customers_data)
        )
        
        self.replacement_manager = ReplacementManager()
        self.results_reporter = ResultsReporter()
        self.iteration_runner = IterationRunner(epsilon=epsilon)
    
    def run_optimization(self) -> Dict:
        """
        Run the complete optimization process.
        
        Returns:
            Dictionary containing optimization results
        """
        # Run iteration zero
        self.iteration_runner.run_iteration_zero(
            population_manager=self.population_manager,
            position_updater=self.position_updater,
            results_reporter=self.results_reporter,
            max_iter=self.max_iter,
            A=self.A
        )
        
        # Run subsequent iterations
        stop_reason = "max_iterations_reached"
        final_iteration = self.max_iter
        
        for iteration in range(1, self.max_iter + 1):
            if self.population_manager.n_sardines == 0:
                logger.info(f"\nNo sardines remaining after iteration {iteration-1}. Stopping.")
                stop_reason = "no_sardines_remaining"
                final_iteration = iteration - 1
                break
            
            self.iteration_runner.run_iteration(
                iteration_num=iteration,
                population_manager=self.population_manager,
                position_updater=self.position_updater,
                replacement_manager=self.replacement_manager,
                results_reporter=self.results_reporter,
                A=self.A
            )
            
            final_iteration = iteration
            
            # Only stop if sardines are empty or max iterations reached
            # Convergence check removed as per user request
        
        # Get final results
        final_results = self.iteration_runner.get_final_results(self.population_manager)
        
        # Add stop reason and final sardine count to results
        final_results['optimization_info'] = {
            'stop_reason': stop_reason,
            'final_sardine_count': self.population_manager.n_sardines,
            'final_iteration': final_iteration
        }
        
        # Print final results to file
        self.results_reporter.print_final_results(
            final_results=final_results,
            depot_data=self.population_manager.depot_data,
            customers_data=self.population_manager.customers_data,
            max_capacity=self.population_manager.max_capacity,
            max_vehicles=self.population_manager.max_vehicles
        )
        
        return final_results
    
    def get_population_summary(self) -> Dict:
        """Get summary of current population state."""
        return self.population_manager.get_population_summary()
    
    def get_best_solution(self) -> Dict:
        """Get the best solution found so far."""
        return {
            'routes': self.population_manager.best_routes,
            'total_distance': self.population_manager.best_fitness,
            'fitness_history': self.population_manager.fitness_history
        }

