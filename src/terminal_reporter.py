"""
Terminal reporter module for Sailfish VRP Optimizer.
Handles terminal output with simplified information display.
"""

import logging
from typing import Dict, List
from .utils import format_route_string, calculate_route_demand, calculate_vrp_fitness

logger = logging.getLogger(__name__)


class TerminalReporter:
    """Handles terminal output with simplified information display."""
    
    def __init__(self):
        """Initialize terminal reporter."""
        pass
    
    def print_data_table(self, depot_data: Dict, customers_data: List[Dict], 
                        max_capacity: float, max_vehicles: int) -> None:
        """Print VRP data in table format for terminal."""
        print("\n" + "="*80)
        print("📊 VRP DATA TABLE")
        print("="*80)
        
        # Depot information
        print(f"📍 DEPOT:")
        print(f"   Location: ({depot_data['x']}, {depot_data['y']})")
        print(f"   Max Capacity: {max_capacity}")
        print(f"   Max Vehicles: {max_vehicles}")
        print()
        
        # Customers table
        print("👥 CUSTOMERS:")
        print(f"{'ID':<4} {'X':<8} {'Y':<8} {'Demand':<8}")
        print("-" * 32)
        for customer in customers_data:
            print(f"{customer['id']:<4} {customer['x']:<8.1f} {customer['y']:<8.1f} {customer['demand']:<8.1f}")
        print()
    
    def print_parameters(self, algorithm_params: Dict, vrp_params: Dict) -> None:
        """Print algorithm and VRP parameters for terminal."""
        print("="*80)
        print("⚙️  PARAMETERS")
        print("="*80)
        
        print("🔧 Algorithm Parameters:")
        for key, value in algorithm_params.items():
            print(f"   {key.replace('_', ' ').title()}: {value}")
        
        print("\n🚚 VRP Parameters:")
        for key, value in vrp_params.items():
            print(f"   {key.replace('_', ' ').title()}: {value}")
        print()
    
    def print_iteration_best(self, iteration: int, best_fitness: float, 
                           best_sailfish_idx: int, best_routes: List[List[int]]) -> None:
        """Print best solution for each iteration."""
        print(f"🔄 Iteration {iteration:3d}: Best Fitness = {best_fitness:.3f} (Sailfish {best_sailfish_idx + 1})")
    
    def print_final_routes_table(self, best_routes: List[List[int]], 
                                depot_data: Dict, customers_data: List[Dict], 
                                max_capacity: float) -> None:
        """Print final routes in table format for terminal."""
        print("\n" + "="*80)
        print("🚛 FINAL ROUTES TABLE")
        print("="*80)
        
        if not best_routes:
            print("No routes found.")
            return
        
        print(f"{'Route':<6} {'Customers':<50} {'Demand':<10} {'Distance':<12}")
        print("-" * 80)
        
        total_distance = 0
        for i, route in enumerate(best_routes, 1):
            if not route:
                continue
                
            # Format customers string without C prefix
            customers_str = " -> ".join([str(c) for c in route])
            
            # Calculate demand and distance
            demand = calculate_route_demand(route, customers_data)
            distance = calculate_vrp_fitness([route], depot_data, customers_data)
            total_distance += distance
            
            # Check if customers string is too long for the column
            if len(customers_str) > 50:
                # Split into multiple lines
                words = customers_str.split(" -> ")
                lines = []
                current_line = ""
                
                for word in words:
                    if len(current_line + word) <= 50:
                        if current_line:
                            current_line += " -> " + word
                        else:
                            current_line = word
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                
                if current_line:
                    lines.append(current_line)
                
                # Print first line with route info
                print(f"R{i:<5} {lines[0]:<50} {demand:<10.1f} {distance:<12.3f}")
                
                # Print additional lines with proper indentation
                for line in lines[1:]:
                    print(f"      {line:<50} {'':<10} {'':<12}")
            else:
                print(f"R{i:<5} {customers_str:<50} {demand:<10.1f} {distance:<12.3f}")
        
        print("-" * 80)
        print(f"{'TOTAL':<6} {'':<50} {'':<10} {total_distance:<12.3f}")
        print()
    
    def print_optimization_summary(self, final_results: Dict) -> None:
        """Print final optimization summary for terminal."""
        print("="*80)
        print("🎯 OPTIMIZATION SUMMARY")
        print("="*80)
        
        print(f"✅ Best Total Distance: {final_results['best_solution']['total_distance']:.3f}")
        print(f"📈 Fitness Improvement: {final_results['fitness_evolution']['improvement']:.3f}")
        print(f"🔄 Total Iterations: {final_results['algorithm_parameters']['max_iter']}")
        print(f"🚛 Vehicles Used: {len([r for r in final_results['best_solution']['routes'] if r])}")
        print()

