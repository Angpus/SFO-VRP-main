"""
Iteration runner module for Sailfish VRP Optimizer.
Handles iteration execution and convergence checking.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class IterationRunner:
    """Handles iteration execution and convergence checking."""
    
    def __init__(self, epsilon: float):
        """
        Initialize iteration runner.
        
        Args:
            epsilon: Attack Power calculation parameter
        """
        self.epsilon = epsilon
    
    def run_iteration_zero(self, 
                          population_manager,
                          position_updater,
                          results_reporter,
                          max_iter: int,
                          A: float) -> None:
        """Run the initial iteration."""
        logger.info("STARTING SAILFISH VRP OPTIMIZATION ALGORITHM")
        logger.info("="*80)
        
        # Step 1: Print initial parameters
        results_reporter.print_initial_parameters(
            population_manager.problem_size,
            population_manager.n_sailfish,
            population_manager.n_sardines,
            max_iter,
            A,
            self.epsilon,
            population_manager.depot_data,
            population_manager.customers_data,
            population_manager.max_capacity,
            population_manager.max_vehicles
        )
        
        # Step 2: Generate initial populations
        population_manager.generate_initial_populations()
        results_reporter.print_random_populations(
            population_manager.sailfish_random_values,
            population_manager.sardine_random_values,
            population_manager.problem_size
        )
        
        # Step 3: Save original positions
        population_manager.save_original_positions()
        
        # Step 4: Convert to routes and calculate fitness
        population_manager.convert_populations_to_routes()
        results_reporter.print_routes_and_solutions(
            population_manager.sailfish_random_values,
            population_manager.sardine_random_values,
            population_manager.sailfish_routes,
            population_manager.sardine_routes,
            population_manager.depot_data,
            population_manager.customers_data,
            population_manager.max_capacity,
            population_manager.max_vehicles,
            current_iteration=0
        )
        
        # Step 5: Calculate fitness
        population_manager.calculate_population_fitness(show_details=True)
        population_manager.update_elite_positions()
        results_reporter.print_fitness_summary(
            population_manager.sailfish_fitness,
            population_manager.sardine_fitness,
            population_manager.best_fitness,
            population_manager.best_routes,
            current_iteration=0
        )
        
        # Step 6: Save sorted positions for iteration 0
        population_manager.save_sorted_positions()
        
        # Step 7: Print comprehensive results
        results_reporter.print_comprehensive_results_table(
            population_manager.sailfish_random_values,
            population_manager.sardine_random_values,
            population_manager.sailfish_routes,
            population_manager.sardine_routes,
            population_manager.sailfish_fitness,
            population_manager.sardine_fitness,
            population_manager.best_fitness,
            current_iteration=0
        )
        
        # Step 8: Calculate PD and lambda values
        PD, lambda_k_values = position_updater.calculate_pd_and_lambda_values(
            population_manager.n_sailfish, 
            population_manager.n_sardines
        )
        
        # Step 9: Update sailfish positions (iteration 0 - use sorted positions)
        population_manager.sailfish_random_values = position_updater.update_sailfish_positions(
            population_manager.sailfish_random_values,
            population_manager.original_sailfish_positions,
            population_manager.elite_sailfish_fitness,
            population_manager.injured_sardine_fitness,
            lambda_k_values,
            is_iteration_zero=True,
            population_manager=population_manager
        )
        
        # Step 9: Calculate AP and update sardines
        AP = position_updater.calculate_attack_power(A, 0, self.epsilon)
        
        if AP >= 0.5:
            logger.info(f"AP >= 0.5: Update ALL sardine positions")
            population_manager.sardine_random_values = position_updater.update_all_sardines(
                population_manager.sardine_random_values,
                population_manager.original_sardine_positions,
                population_manager.elite_sailfish_fitness,
                AP
            )
        else:
            logger.info(f"AP < 0.5: Partial sardine update")
            population_manager.sardine_random_values = position_updater.update_partial_sardines(
                population_manager.sardine_random_values,
                population_manager.original_sardine_positions,
                population_manager.elite_sailfish_fitness,
                AP
            )
        
        # Record fitness history
        population_manager.fitness_history.append(population_manager.best_fitness)
        
        logger.info(f"\n" + "="*80)
        logger.info("ITERATION 0 COMPLETED")
        logger.info("="*80)
        logger.info(f"Best fitness: {population_manager.best_fitness:.3f}")
        logger.info(f"Best routes: {population_manager.best_routes}")
    
    def run_iteration(self, 
                     iteration_num: int,
                     population_manager,
                     position_updater,
                     replacement_manager,
                     results_reporter,
                     A: float) -> None:
        """Run a single iteration."""
        logger.info(f"\n" + "="*100)
        logger.info(f"STARTING ITERATION {iteration_num}")
        logger.info("="*100)
        
        # Step 1: Save original positions and clear replacement tracking
        population_manager.save_original_positions()
        population_manager.sailfish_replacement_map.clear()
        population_manager.sardine_positions_before_removal.clear()
        
        # Step 2: Convert to routes and calculate fitness
        population_manager.convert_populations_to_routes()
        results_reporter.print_routes_and_solutions(
            population_manager.sailfish_random_values,
            population_manager.sardine_random_values,
            population_manager.sailfish_routes,
            population_manager.sardine_routes,
            population_manager.depot_data,
            population_manager.customers_data,
            population_manager.max_capacity,
            population_manager.max_vehicles,
            current_iteration=iteration_num
        )
        
        # Step 3: Calculate fitness
        population_manager.calculate_population_fitness(show_details=True)
        population_manager.update_elite_positions()
        results_reporter.print_fitness_summary(
            population_manager.sailfish_fitness,
            population_manager.sardine_fitness,
            population_manager.best_fitness,
            population_manager.best_routes,
            current_iteration=iteration_num
        )
        
        # Step 4: Perform replacement
        replacement_manager.perform_sailfish_sardine_replacement(population_manager, iteration_num)
        
        # Step 5: Print comprehensive results
        results_reporter.print_comprehensive_results_table(
            population_manager.sailfish_random_values,
            population_manager.sardine_random_values,
            population_manager.sailfish_routes,
            population_manager.sardine_routes,
            population_manager.sailfish_fitness,
            population_manager.sardine_fitness,
            population_manager.best_fitness,
            current_iteration=iteration_num
        )
        
        # Step 6: Calculate PD and lambda values
        PD, lambda_k_values = position_updater.calculate_pd_and_lambda_values(
            population_manager.n_sailfish, 
            population_manager.n_sardines
        )
        
        # Step 7: Update sailfish positions (iterations > 0 - use replacement-specific positions)
        population_manager.sailfish_random_values = position_updater.update_sailfish_positions(
            population_manager.sailfish_random_values,
            population_manager.original_sailfish_positions,
            population_manager.elite_sailfish_fitness,
            population_manager.injured_sardine_fitness,
            lambda_k_values,
            is_iteration_zero=False,
            population_manager=population_manager
        )
        
        # Step 8: Calculate AP and update sardines
        AP = position_updater.calculate_attack_power(A, iteration_num, self.epsilon)
        
        if population_manager.n_sardines > 0:
            if AP >= 0.5:
                logger.info(f"AP >= 0.5: Update ALL sardine positions")
                population_manager.sardine_random_values = position_updater.update_all_sardines(
                    population_manager.sardine_random_values,
                    population_manager.original_sardine_positions,
                    population_manager.elite_sailfish_fitness,
                    AP
                )
            else:
                logger.info(f"AP < 0.5: Partial sardine update")
                population_manager.sardine_random_values = position_updater.update_partial_sardines(
                    population_manager.sardine_random_values,
                    population_manager.original_sardine_positions,
                    population_manager.elite_sailfish_fitness,
                    AP
                )
        
        # Record fitness history
        population_manager.fitness_history.append(population_manager.best_fitness)
        
        logger.info(f"\n" + "="*80)
        logger.info(f"ITERATION {iteration_num} COMPLETED")
        logger.info("="*80)
        logger.info(f"Best fitness: {population_manager.best_fitness:.3f}")
        logger.info(f"Best routes: {population_manager.best_routes}")
    

    
    def get_final_results(self, population_manager) -> Dict:
        """
        Get final optimization results.
        
        Returns:
            Dictionary containing final results
        """
        final_results = {
            'algorithm_parameters': {
                'initial_sailfish': population_manager.original_n_sailfish,
                'initial_sardines': population_manager.original_n_sardines,
                'final_sailfish': population_manager.n_sailfish,
                'final_sardines': population_manager.n_sardines,
                'iterations': len(population_manager.fitness_history)
            },
            'vrp_parameters': {
                'customers': population_manager.problem_size,
                'max_capacity': population_manager.max_capacity,
                'max_vehicles': population_manager.max_vehicles,
                'data_file': getattr(population_manager, 'data_file', 'kecil.csv')
            },
            'best_solution': {
                'routes': population_manager.best_routes,
                'total_distance': population_manager.best_fitness
            },
            'fitness_evolution': {
                'initial': population_manager.fitness_history[0],
                'final': population_manager.fitness_history[-1],
                'improvement': population_manager.fitness_history[0] - population_manager.fitness_history[-1],
                'history': population_manager.fitness_history
            }
        }
        
        return final_results
