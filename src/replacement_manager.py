"""
Replacement Manager for Sailfish VRP Optimizer.

This module handles the replacement of sailfish with better sardines during the optimization process.
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class ReplacementManager:
    """
    Manages sailfish-sardine replacement operations in the Sailfish VRP Optimizer.
    
    This class handles:
    - Identifying sardines that are better than the worst sailfish
    - Executing replacements
    - Tracking replacement statistics
    - Updating population after replacements
    """
    
    def __init__(self):
        """Initialize the Replacement Manager."""
        self.replacement_history = []
    
    def perform_sailfish_sardine_replacement(self, population_manager, iteration_num: int) -> Dict:
        """
        Perform sailfish-sardine replacement for the current iteration.
        
        Args:
            population_manager: Population manager instance
            iteration_num: Current iteration number
            
        Returns:
            Dictionary containing replacement statistics
        """
        logger.info(f"\n" + "="*80)
        logger.info(f"ITERATION {iteration_num} - STEP 4: SAILFISH-SARDINE REPLACEMENT")
        logger.info("="*80)
        
        # Find worst sailfish fitness
        worst_sailfish_fitness = max(population_manager.sailfish_fitness)
        
        # Identify sardines better than worst sailfish
        better_sardines = []
        for i, sardine_fitness in enumerate(population_manager.sardine_fitness):
            if sardine_fitness < worst_sailfish_fitness:
                better_sardines.append((i, sardine_fitness))
        
        logger.info(f"Analysis:")
        logger.info(f"- Worst sailfish fitness: {worst_sailfish_fitness:.3f}")
        logger.info(f"- Sardines better than worst sailfish: {len(better_sardines)}")
        
        if not better_sardines:
            logger.info("- No replacement will occur")
            return {
                'replacements_made': 0,
                'sardines_removed': 0,
                'new_best_found': False
            }
        
        # Sort better sardines by fitness (ascending)
        better_sardines.sort(key=lambda x: x[1])
        sardines_to_remove = []
        replacements_made = []
        new_best_found = False
        
        for sardine_idx, sardine_fitness in better_sardines:
            # Find current worst sailfish
            worst_sf_idx = population_manager.sailfish_fitness.index(max(population_manager.sailfish_fitness))
            worst_sf_fitness = population_manager.sailfish_fitness[worst_sf_idx]
            
            if sardine_fitness < worst_sf_fitness:
                logger.info(f"\nReplacement {len(replacements_made) + 1}:")
                logger.info(f"- Sardine S{sardine_idx+1} (fitness: {sardine_fitness:.3f}) -> Sailfish SF{worst_sf_idx+1} (fitness: {worst_sf_fitness:.3f})")
                
                # Replace sailfish with sardine data
                population_manager.sailfish_random_values[worst_sf_idx] = population_manager.sardine_random_values[sardine_idx].copy()
                population_manager.sailfish_routes[worst_sf_idx] = population_manager.sardine_routes[sardine_idx]
                population_manager.sailfish_fitness[worst_sf_idx] = population_manager.sardine_fitness[sardine_idx]
                
                # Track replacement for position updates - store sardine position before removal
                population_manager.sailfish_replacement_map[worst_sf_idx] = sardine_idx
                # Store the sardine's position data before we remove it
                population_manager.sardine_positions_before_removal[sardine_idx] = population_manager.sardine_random_values[sardine_idx].copy()
                
                sardines_to_remove.append(sardine_idx)
                replacements_made.append({
                    'sardine_idx': sardine_idx,
                    'sailfish_idx': worst_sf_idx,
                    'new_fitness': sardine_fitness,
                    'old_fitness': worst_sf_fitness
                })
                
                # Check if this is a new overall best solution
                if sardine_fitness < population_manager.best_fitness:
                    population_manager.best_fitness = sardine_fitness
                    population_manager.best_routes = population_manager.sardine_routes[sardine_idx]
                    new_best_found = True
                    logger.info(f"  NEW OVERALL BEST SOLUTION! Fitness: {sardine_fitness:.3f}")
            else:
                break
        
        # Remove replaced sardines
        sardines_to_remove.sort(reverse=True)
        for sardine_idx in sardines_to_remove:
            del population_manager.sardine_random_values[sardine_idx]
            del population_manager.sardine_routes[sardine_idx]
            del population_manager.sardine_fitness[sardine_idx]
            del population_manager.original_sardine_positions[sardine_idx]
            population_manager.n_sardines -= 1
        
        logger.info(f"\nReplacement Summary: {len(replacements_made)} replacements made")
        
        # Update elite positions
        self._update_elite_positions(population_manager)
        
        # Record replacement in history
        replacement_stats = {
            'iteration': iteration_num,
            'replacements_made': len(replacements_made),
            'sardines_removed': len(sardines_to_remove),
            'new_best_found': new_best_found,
            'details': replacements_made
        }
        self.replacement_history.append(replacement_stats)
        
        return replacement_stats
    
    def _update_elite_positions(self, population_manager):
        """
        Update elite sailfish and injured sardine positions after replacement.
        
        Args:
            population_manager: Population manager instance
        """
        # Update elite sailfish position
        if population_manager.sailfish_fitness:
            best_sailfish_idx = population_manager.sailfish_fitness.index(min(population_manager.sailfish_fitness))
            population_manager.elite_sailfish_fitness = min(population_manager.sailfish_fitness)
            population_manager.elite_sailfish_position = population_manager.sailfish_random_values[best_sailfish_idx].copy()
        
        # Update injured sardine position
        if population_manager.sardine_fitness:
            best_sardine_idx = population_manager.sardine_fitness.index(min(population_manager.sardine_fitness))
            population_manager.injured_sardine_fitness = min(population_manager.sardine_fitness)
            population_manager.injured_sardine_position = population_manager.sardine_random_values[best_sardine_idx].copy()
        else:
            # If no sardines remain, use elite sailfish as injured sardine
            population_manager.injured_sardine_fitness = population_manager.elite_sailfish_fitness
            population_manager.injured_sardine_position = population_manager.elite_sailfish_position.copy()
    
    def get_replacement_statistics(self) -> Dict:
        """
        Get comprehensive replacement statistics.
        
        Returns:
            Dictionary containing replacement statistics
        """
        if not self.replacement_history:
            return {
                'total_replacements': 0,
                'total_iterations_with_replacements': 0,
                'average_replacements_per_iteration': 0,
                'iterations_with_replacements': []
            }
        
        total_replacements = sum(stats['replacements_made'] for stats in self.replacement_history)
        iterations_with_replacements = [stats['iteration'] for stats in self.replacement_history if stats['replacements_made'] > 0]
        
        return {
            'total_replacements': total_replacements,
            'total_iterations_with_replacements': len(iterations_with_replacements),
            'average_replacements_per_iteration': total_replacements / len(self.replacement_history) if self.replacement_history else 0,
            'iterations_with_replacements': iterations_with_replacements,
            'replacement_history': self.replacement_history
        }
    
    def reset_history(self):
        """Reset replacement history."""
        self.replacement_history.clear()
