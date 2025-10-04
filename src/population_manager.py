"""
Population management module for Sailfish VRP Optimizer.
Handles sailfish and sardine populations, their storage, and basic operations.
"""

import random
import logging
from typing import Dict, List, Tuple
from datetime import datetime

from .utils import generate_random_values, convert_random_to_route, calculate_vrp_fitness
from .logger import VRPLogger

logger = logging.getLogger(__name__)


class PopulationManager:
    """Manages sailfish and sardine populations for the VRP optimizer."""
    
    def __init__(self, 
                 n_sailfish: int,
                 n_sardines: int,
                 depot_data: Dict,
                 customers_data: List[Dict],
                 max_capacity: float,
                 max_vehicles: int,
                 log_to_file: bool = True,
                 output_mode: str = 'steps',
                 data_file: str = 'kecil.csv'):
        """
        Initialize population manager.
        
        Args:
            n_sailfish: Number of sailfish
            n_sardines: Number of sardines
            depot_data: Depot information dictionary
            customers_data: List of customer information dictionaries
            max_capacity: Maximum vehicle capacity
            max_vehicles: Maximum number of vehicles
            log_to_file: Whether to log output to file
        """
        if n_sardines <= n_sailfish:
            raise ValueError("Number of sardines must be greater than number of sailfish")
        
        # Store original parameters
        self.original_n_sailfish = n_sailfish
        self.original_n_sardines = n_sardines
        
        # Current population sizes
        self.n_sailfish = n_sailfish
        self.n_sardines = n_sardines
        
        # VRP problem data
        self.depot_data = depot_data
        self.customers_data = customers_data
        self.max_capacity = max_capacity
        self.max_vehicles = max_vehicles
        self.problem_size = len(customers_data)
        self.data_file = data_file
        
        # Setup logging
        self.logger = VRPLogger(log_to_file=log_to_file, output_mode=output_mode)
        if log_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sailfish_vrp_SF{n_sailfish}_S{n_sardines}_{timestamp}.txt"
            self.logger.setup_output_logger(filename)
            
            # Add header to file
            from .utils import print_system_header
            print_system_header("file")
        
        # Population storage
        self.sailfish_random_values = []
        self.sailfish_routes = []
        self.sailfish_fitness = []
        
        self.sardine_random_values = []
        self.sardine_routes = []
        self.sardine_fitness = []
        
        # Original positions for updates
        self.original_sailfish_positions = []
        self.original_sardine_positions = []
        
        # Sorted positions for iteration 0
        self.sorted_sailfish_positions = []
        self.sorted_sardine_positions = []
        
        # Replacement tracking for iterations > 0
        self.sailfish_replacement_map = {}  # Maps sailfish_idx to sardine_idx that replaced it
        self.sardine_positions_before_removal = {}  # Stores sardine positions before removal
        
        # Best solution tracking
        self.best_routes = None
        self.best_fitness = float('inf')
        self.fitness_history = []
        
        # Elite tracking
        self.elite_sailfish_fitness = None
        self.elite_sailfish_position = None
        self.injured_sardine_fitness = None
        self.injured_sardine_position = None
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'logger'):
            self.logger.cleanup()
    
    def generate_initial_populations(self) -> None:
        """Generate initial random populations for sailfish and sardines."""
        self.sailfish_random_values = generate_random_values(self.n_sailfish, self.problem_size)
        self.sardine_random_values = generate_random_values(self.n_sardines, self.problem_size)
    
    def save_original_positions(self) -> None:
        """Save original positions before sorting and replacement."""
        self.original_sailfish_positions = [pos.copy() for pos in self.sailfish_random_values]
        self.original_sardine_positions = [pos.copy() for pos in self.sardine_random_values]
    
    def save_sorted_positions(self) -> None:
        """Save sorted positions for iteration 0 updates."""
        # Sort sailfish by fitness (best first)
        sailfish_with_fitness = list(zip(self.sailfish_random_values, self.sailfish_fitness))
        sailfish_with_fitness.sort(key=lambda x: x[1])
        self.sorted_sailfish_positions = [pos.copy() for pos, _ in sailfish_with_fitness]
        
        # Sort sardines by fitness (best first)
        sardine_with_fitness = list(zip(self.sardine_random_values, self.sardine_fitness))
        sardine_with_fitness.sort(key=lambda x: x[1])
        self.sorted_sardine_positions = [pos.copy() for pos, _ in sardine_with_fitness]
    
    def convert_populations_to_routes(self) -> None:
        """Convert random values to routes for both populations."""
        self.sailfish_routes = []
        self.sardine_routes = []
        
        # Convert sailfish random values to routes
        for random_vals in self.sailfish_random_values:
            routes, _ = convert_random_to_route(random_vals, self.depot_data, 
                                             self.customers_data, self.max_capacity, self.max_vehicles)
            self.sailfish_routes.append(routes)
        
        # Convert sardine random values to routes
        for random_vals in self.sardine_random_values:
            routes, _ = convert_random_to_route(random_vals, self.depot_data, 
                                             self.customers_data, self.max_capacity, self.max_vehicles)
            self.sardine_routes.append(routes)
    
    def calculate_population_fitness(self, show_details: bool = False) -> None:
        """Calculate fitness for all individuals in both populations."""
        self.sailfish_fitness = []
        self.sardine_fitness = []
        
        # Calculate sailfish fitness
        for i, routes in enumerate(self.sailfish_routes):
            fitness = calculate_vrp_fitness(routes, self.depot_data, self.customers_data, show_details=show_details)
            self.sailfish_fitness.append(fitness)
            
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_routes = routes
        
        # Calculate sardine fitness
        for i, routes in enumerate(self.sardine_routes):
            fitness = calculate_vrp_fitness(routes, self.depot_data, self.customers_data, show_details=show_details)
            self.sardine_fitness.append(fitness)
            
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_routes = routes
    
    def update_elite_positions(self) -> None:
        """Update elite sailfish and injured sardine positions."""
        if self.sailfish_fitness:
            best_sailfish_idx = self.sailfish_fitness.index(min(self.sailfish_fitness))
            self.elite_sailfish_fitness = min(self.sailfish_fitness)
            self.elite_sailfish_position = self.sailfish_random_values[best_sailfish_idx].copy()
        
        if self.sardine_fitness:
            best_sardine_idx = self.sardine_fitness.index(min(self.sardine_fitness))
            self.injured_sardine_fitness = min(self.sardine_fitness)
            self.injured_sardine_position = self.sardine_random_values[best_sardine_idx].copy()
        else:
            self.injured_sardine_fitness = self.elite_sailfish_fitness
            self.injured_sardine_position = self.elite_sailfish_position.copy()
    
    def remove_sardine(self, sardine_idx: int) -> None:
        """Remove a sardine from the population."""
        del self.sardine_random_values[sardine_idx]
        del self.sardine_routes[sardine_idx]
        del self.sardine_fitness[sardine_idx]
        del self.original_sardine_positions[sardine_idx]
        self.n_sardines -= 1
    
    def replace_sailfish_with_sardine(self, sailfish_idx: int, sardine_idx: int) -> None:
        """Replace a sailfish with a sardine."""
        self.sailfish_random_values[sailfish_idx] = self.sardine_random_values[sardine_idx].copy()
        self.sailfish_routes[sailfish_idx] = self.sardine_routes[sardine_idx]
        self.sailfish_fitness[sailfish_idx] = self.sardine_fitness[sardine_idx]
        
        # Track which sardine replaced this sailfish for position updates
        self.sailfish_replacement_map[sailfish_idx] = sardine_idx
    
    def get_population_summary(self) -> Dict:
        """Get summary of current population state."""
        return {
            'sailfish_count': self.n_sailfish,
            'sardine_count': self.n_sardines,
            'best_fitness': self.best_fitness,
            'best_routes': self.best_routes,
            'elite_sailfish_fitness': self.elite_sailfish_fitness,
            'injured_sardine_fitness': self.injured_sardine_fitness
        }
