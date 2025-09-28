"""
Position updater module for Sailfish VRP Optimizer.
Handles position updates for sailfish and sardines using mathematical formulas.
"""

import random
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class PositionUpdater:
    """Handles position updates for sailfish and sardines."""
    
    def __init__(self, problem_size: int):
        """
        Initialize position updater.
        
        Args:
            problem_size: Size of the problem (number of customers)
        """
        self.problem_size = problem_size
    
    def calculate_pd_and_lambda_values(self, n_sailfish: int, n_sardines: int) -> tuple:
        """
        Calculate PD and lambda values for sailfish position updates.
        
        Args:
            n_sailfish: Number of sailfish
            n_sardines: Number of sardines
            
        Returns:
            Tuple of (PD, lambda_values)
        """
        total_population = n_sailfish + n_sardines
        PD = 1 - (n_sailfish / total_population)
        
        logger.info(f"Population Decline (PD) = 1 - ({n_sailfish} / {total_population}) = {PD:.6f}")
        logger.info("")
        
        lambda_k_values = []
        logger.info("Lambda Calculations:")
        
        for k in range(n_sailfish):
            random_val = round(random.random(), 3)
            lambda_k = (2 * random_val * PD) - PD
            lambda_k_values.append(lambda_k)
            
            logger.info(f"SF{k+1}: λ_{k+1} = (2 × {random_val} × {PD:.6f}) - {PD:.6f} = {lambda_k:.6f}")
        
        return PD, lambda_k_values
    
    def update_sailfish_positions(self, 
                                sailfish_positions: List[List[float]],
                                original_positions: List[List[float]],
                                elite_sailfish_fitness: float,
                                injured_sardine_fitness: float,
                                lambda_k_values: List[float],
                                is_iteration_zero: bool = False,
                                population_manager = None) -> List[List[float]]:
        """
        Update sailfish positions using the Sailfish Optimizer formula.
        
        Args:
            sailfish_positions: Current sailfish positions
            original_positions: Original positions before updates
            elite_sailfish_fitness: Elite sailfish fitness value
            injured_sardine_fitness: Injured sardine fitness value
            lambda_k_values: Lambda values for each sailfish
            is_iteration_zero: Whether this is iteration 0
            population_manager: Population manager for replacement tracking
            
        Returns:
            Updated sailfish positions
        """
        logger.info("FORMULA: SF[i] = elite_sailfish_fitness - lambda[k] × ((random × (elite_sailfish_fitness + injured_sardine_fitness)/2) - old_sailfish)")
        logger.info(f"Elite Sailfish Fitness: {elite_sailfish_fitness:.3f}")
        logger.info(f"Injured Sardine Fitness: {injured_sardine_fitness:.3f}")
        
        if is_iteration_zero:
            logger.info("ITERATION 0: Using sorted positions for updates...")
            update_positions = population_manager.sorted_sailfish_positions
        else:
            logger.info("ITERATION > 0: Using replacement-specific positions for updates...")
            # Use sorted positions as the base for non-replaced sailfish
            update_positions = self._get_replacement_positions(population_manager, population_manager.sorted_sailfish_positions)
        
        new_sailfish_positions = []
        
        for k in range(len(sailfish_positions)):
            logger.info(f"\nUpdating SF{k+1}:")
            new_position = []
            
            for j in range(self.problem_size):
                rand = round(random.random(), 3)
                elite_sf_fitness = elite_sailfish_fitness
                injured_sardine_fitness = injured_sardine_fitness
                old_sailfish_j = update_positions[k][j]
                
                # Sailfish Optimizer formula
                avg_elite_injured = (elite_sf_fitness + injured_sardine_fitness) / 2
                bracket_term = (rand * avg_elite_injured) - old_sailfish_j
                lambda_term = lambda_k_values[k] * bracket_term
                new_val = elite_sf_fitness - lambda_term
                
                new_position.append(new_val)
                
                logger.info(f"  Pos[{j+1}]: {elite_sf_fitness:.3f} - {lambda_k_values[k]:.6f} × (({rand:.3f} × {avg_elite_injured:.3f}) - {old_sailfish_j:.3f}) = {new_val:.3f}")
            
            new_sailfish_positions.append(new_position)
            logger.info(f"New position: {[f'{x:.3f}' for x in new_position]}")
        
        logger.info("\nAll sailfish positions updated!")
        return new_sailfish_positions
    
    def _get_replacement_positions(self, population_manager, sorted_positions: List[List[float]]) -> List[List[float]]:
        """
        Get the appropriate positions for sailfish updates based on replacements.
        
        Args:
            population_manager: Population manager with replacement tracking
            sorted_positions: Sorted positions (original sailfish positions sorted by fitness)
            
        Returns:
            List of positions to use for updates
        """
        replacement_positions = []
        
        for k in range(len(sorted_positions)):
            if k in population_manager.sailfish_replacement_map:
                # This sailfish was replaced by a sardine, use the sardine's position
                sardine_idx = population_manager.sailfish_replacement_map[k]
                if sardine_idx in population_manager.sardine_positions_before_removal:
                    # Use the stored sardine position from before removal
                    replacement_positions.append(population_manager.sardine_positions_before_removal[sardine_idx])
                    logger.info(f"SF{k+1} was replaced by S{sardine_idx+1}, using S{sardine_idx+1} position")
                else:
                    # Fallback to sorted position if sardine position not found
                    replacement_positions.append(sorted_positions[k])
                    logger.info(f"SF{k+1} was replaced by S{sardine_idx+1}, but S{sardine_idx+1} position not found, using sorted position")
            else:
                # This sailfish was not replaced, use its original sorted position
                replacement_positions.append(sorted_positions[k])
                logger.info(f"SF{k+1} was not replaced, using original sorted position")
        
        return replacement_positions
    
    def calculate_attack_power(self, A: float, current_iteration: int, epsilon: float) -> float:
        """
        Calculate Attack Power (AP) for sardine updates.
        
        Args:
            A: Sailfish optimizer parameter
            current_iteration: Current iteration number
            epsilon: Convergence parameter
            
        Returns:
            Attack Power value
        """
        AP = A * (1 - (2 * (current_iteration + 1) * epsilon))
        logger.info(f"Attack Power (AP) = {A} × (1 - (2 × {current_iteration + 1} × {epsilon})) = {AP:.6f}")
        return AP
    
    def update_all_sardines(self, 
                           sardine_positions: List[List[float]],
                           original_positions: List[List[float]],
                           elite_sailfish_fitness: float,
                           AP: float) -> List[List[float]]:
        """
        Update all sardine positions when AP >= 0.5.
        
        Args:
            sardine_positions: Current sardine positions
            original_positions: Original positions before updates
            elite_sailfish_fitness: Elite sailfish fitness value
            AP: Attack Power value
            
        Returns:
            Updated sardine positions
        """
        logger.info("\nFORMULA: S[i] = random × (elite_sailfish_fitness - old_sardine + AP)")
        logger.info(f"Elite Sailfish Fitness: {elite_sailfish_fitness:.3f}")
        logger.info("Updating ALL sardines:")
        
        new_sardine_positions = []
        
        for i in range(len(sardine_positions)):
            logger.info(f"\nUpdating S{i+1}:")
            new_position = []
            
            for j in range(self.problem_size):
                rand = round(random.random(), 3)
                elite_sf_fitness = elite_sailfish_fitness
                old_sardine_j = original_positions[i][j]
                
                # Sardine update formula
                bracket_term = elite_sf_fitness - old_sardine_j + AP
                new_val = rand * bracket_term
                
                new_position.append(new_val)
                
                logger.info(f"  Pos[{j+1}]: {rand:.3f} × ({elite_sf_fitness:.3f} - {old_sardine_j:.3f} + {AP:.6f}) = {new_val:.3f}")
            
            new_sardine_positions.append(new_position)
            logger.info(f"New position: {[f'{x:.3f}' for x in new_position]}")
        
        logger.info("\nAll sardine positions updated!")
        return new_sardine_positions
    
    def update_partial_sardines(self, 
                               sardine_positions: List[List[float]],
                               original_positions: List[List[float]],
                               elite_sailfish_fitness: float,
                               AP: float) -> List[List[float]]:
        """
        Update partial sardines when AP < 0.5.
        
        Args:
            sardine_positions: Current sardine positions
            original_positions: Original positions before updates
            elite_sailfish_fitness: Elite sailfish fitness value
            AP: Attack Power value
            
        Returns:
            Updated sardine positions
        """
        logger.info("\nFORMULA: S[i] = random × (elite_sailfish_fitness - old_sardine + AP)")
        logger.info(f"Elite Sailfish Fitness: {elite_sailfish_fitness:.3f}")
        logger.info("Partial sardine update:")
        
        n_sardines = len(sardine_positions)
        alpha = int(n_sardines * AP)
        beta = int(self.problem_size * AP)
        
        logger.info(f"alpha = {n_sardines} × {AP:.6f} = {alpha}")
        logger.info(f"beta = {self.problem_size} × {AP:.6f} = {beta}")
        
        if alpha == 0 or beta == 0:
            logger.info("Alpha or beta is 0, no sardines updated.")
            return sardine_positions
        
        sardines_to_update = random.sample(range(n_sardines), min(alpha, n_sardines))
        logger.info(f"Selected sardines to update: {[f'S{i+1}' for i in sardines_to_update]}")
        
        new_sardine_positions = [pos.copy() for pos in sardine_positions]
        
        for i in sardines_to_update:
            logger.info(f"\nUpdating S{i+1} (partial):")
            positions_to_update = random.sample(range(self.problem_size), min(beta, self.problem_size))
            logger.info(f"Updating positions: {[j+1 for j in positions_to_update]}")
            
            for j in positions_to_update:
                rand = round(random.random(), 3)
                elite_sf_fitness = elite_sailfish_fitness
                old_sardine_j = original_positions[i][j]
                
                # Sardine update formula
                bracket_term = elite_sf_fitness - old_sardine_j + AP
                new_val = rand * bracket_term
                
                new_sardine_positions[i][j] = new_val
                
                logger.info(f"  Pos[{j+1}]: {rand:.3f} × ({elite_sf_fitness:.3f} - {old_sardine_j:.3f} + {AP:.6f}) = {new_val:.3f}")
            
            logger.info(f"New position: {[f'{x:.3f}' for x in new_sardine_positions[i]]}")
        
        return new_sardine_positions
