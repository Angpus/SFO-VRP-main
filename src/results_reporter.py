"""
Results reporter module for Sailfish VRP Optimizer.
Handles results reporting, formatting, and output generation.
"""

import logging
from typing import Dict, List
from .utils import print_vrp_data, format_route_string, calculate_route_demand, calculate_vrp_fitness

logger = logging.getLogger(__name__)


class ResultsReporter:
    """Handles results reporting and formatting for the VRP optimizer."""
    
    def __init__(self):
        """Initialize results reporter."""
        pass
    
    def print_initial_parameters(self, 
                               problem_size: int,
                               n_sailfish: int,
                               n_sardines: int,
                               max_iter: int,
                               A: float,
                               epsilon: float,
                               depot_data: Dict,
                               customers_data: List[Dict],
                               max_capacity: float,
                               max_vehicles: int) -> None:
        """Print initial parameters and VRP data."""
        logger.debug("\n" + "="*80)
        logger.debug("1. INITIAL VARIABLES AND VRP DATA")
        logger.debug("="*80)
        
        logger.debug(f"Initial Parameters:")
        logger.debug(f"- Problem size: {problem_size} customers")
        logger.debug(f"- Sailfish population: {n_sailfish}")
        logger.debug(f"- Sardine population: {n_sardines}")
        logger.debug(f"- Maximum iterations: {max_iter}")
        logger.debug(f"- Parameter A: {A}")
        logger.debug(f"- Epsilon: {epsilon}")
        logger.debug("")
        
        print_vrp_data(depot_data, customers_data, max_capacity, max_vehicles)
    
    def print_random_populations(self, 
                               sailfish_random_values: List[List[float]],
                               sardine_random_values: List[List[float]],
                               problem_size: int) -> None:
        """Print random values for sailfish and sardines."""
        logger.debug("\n" + "="*80)
        logger.debug("2. RANDOM SAILFISH AND SARDINES")
        logger.debug("="*80)
        
        logger.debug("SAILFISH Random Values:")
        header = f"{'ID':<8}"
        for i in range(problem_size):
            header += f"C{i+1:2}   "
        logger.debug(header)
        
        for i, values in enumerate(sailfish_random_values):
            row = f"SF{i+1:<6}"
            for val in values:
                row += f"{val:5.3f} "
            logger.debug(row)
        
        logger.debug("\nSARDINE Random Values:")
        header = f"{'ID':<8}"
        for i in range(problem_size):
            header += f"C{i+1:2}   "
        logger.debug(header)
        
        for i, values in enumerate(sardine_random_values):
            row = f"S{i+1:<7}"
            for val in values:
                row += f"{val:5.3f} "
            logger.debug(row)
    
    def print_routes_and_solutions(self, 
                                 sailfish_random_values: List[List[float]],
                                 sardine_random_values: List[List[float]],
                                 sailfish_routes: List[List[List[int]]],
                                 sardine_routes: List[List[List[int]]],
                                 depot_data: Dict,
                                 customers_data: List[Dict],
                                 max_capacity: float,
                                 max_vehicles: int,
                                 current_iteration: int = 0) -> None:
        """Print routes for each sailfish and sardine."""
        logger.debug("\n" + "="*80)
        if current_iteration == 0:
            logger.debug("3. ROUTES FOR EACH SAILFISH AND SARDINE")
        else:
            logger.debug(f"ITERATION {current_iteration} - STEP 1: GENERATING NEW ROUTES")
        logger.debug("="*80)
        
        logger.debug("SAILFISH Routes:")
        for i, (random_vals, routes) in enumerate(zip(sailfish_random_values, sailfish_routes)):
            logger.debug(f"\n===== SF{i+1} ====================================================")
            
            logger.debug(f"Random values: {random_vals}")
            route_info = f"Routes (with depot returns): "
            for j, route in enumerate(routes):
                route_str = format_route_string(route)
                route_info += f"Route{j+1}: {route_str}  "
            logger.debug(route_info)
            
            # Calculate route capacities
            for j, route in enumerate(routes):
                total_demand = calculate_route_demand(route, customers_data)
                logger.debug(f"Route{j+1} demand: {total_demand}/{max_capacity}")
        
        logger.debug("\nSARDINE Routes:")
        for i, (random_vals, routes) in enumerate(zip(sardine_random_values, sardine_routes)):
            logger.debug(f"\n===== S{i+1} ====================================================")
            
            logger.debug(f"Random values: {random_vals}")
            route_info = f"Routes (with depot returns): "
            for j, route in enumerate(routes):
                route_str = format_route_string(route)
                route_info += f"Route{j+1}: {route_str}  "
            logger.debug(route_info)
            
            # Calculate route capacities
            for j, route in enumerate(routes):
                total_demand = calculate_route_demand(route, customers_data)
                logger.debug(f"Route{j+1} demand: {total_demand}/{max_capacity}")
    
    def print_fitness_summary(self, 
                            sailfish_fitness: List[float],
                            sardine_fitness: List[float],
                            best_fitness: float,
                            best_routes: List[List[int]],
                            current_iteration: int = 0) -> None:
        """Print summary of all fitness scores."""
        logger.debug(f"\n" + "="*80)
        if current_iteration == 0:
            logger.debug("FITNESS SUMMARY")
        else:
            logger.debug(f"ITERATION {current_iteration} - STEP 3: FITNESS SUMMARY")
        logger.debug("="*80)
        
        logger.debug("SAILFISH FITNESS SCORES:")
        logger.debug("-" * 30)
        for i, fitness in enumerate(sailfish_fitness):
            marker = " ⭐ BEST" if fitness == min(sailfish_fitness) else ""
            logger.debug(f"SF{i+1}: {fitness:.3f}{marker}")
        
        logger.debug("\nSARDINE FITNESS SCORES:")
        logger.debug("-" * 25)
        for i, fitness in enumerate(sardine_fitness):
            marker = " ⭐ BEST" if fitness == min(sardine_fitness) else ""
            logger.debug(f"S{i+1}: {fitness:.3f}{marker}")
        
        logger.info(f"\nOVERALL SUMMARY:")
        logger.info("-" * 20)
        logger.info(f"Best Sailfish Fitness: {min(sailfish_fitness):.3f}")
        logger.info(f"Best Sardine Fitness: {min(sardine_fitness):.3f}")
        logger.info(f"Overall Best Fitness: {best_fitness:.3f}")
        logger.info(f"Best Routes: {best_routes}")
    
    def print_comprehensive_results_table(self, 
                                        sailfish_random_values: List[List[float]],
                                        sardine_random_values: List[List[float]],
                                        sailfish_routes: List[List[List[int]]],
                                        sardine_routes: List[List[List[int]]],
                                        sailfish_fitness: List[float],
                                        sardine_fitness: List[float],
                                        best_fitness: float,
                                        current_iteration: int = 0) -> None:
        """Print comprehensive results table."""
        logger.debug(f"\n" + "="*120)
        if current_iteration == 0:
            logger.debug("COMPREHENSIVE RESULTS TABLE")
        else:
            logger.debug(f"ITERATION {current_iteration} - COMPREHENSIVE RESULTS TABLE")
        logger.debug("="*120)
        
        logger.debug(f"{'ID':<4} {'Random Values':<30} {'Routes':<40} {'Fitness':<10}")
        logger.debug("-" * 120)
        
        # Sailfish table
        for i, (random_vals, routes, fitness) in enumerate(zip(sailfish_random_values, sailfish_routes, sailfish_fitness)):
            random_str = str([f"{x:.3f}" for x in random_vals])
            routes_str = str(routes)
            
            marker = " 🎯" if abs(fitness - best_fitness) < 1e-6 else ""
            logger.debug(f"SF{i+1:<2} {random_str:<30} {routes_str:<40} {fitness:<10.3f}{marker}")
        
        # Sardine table
        for i, (random_vals, routes, fitness) in enumerate(zip(sardine_random_values, sardine_routes, sardine_fitness)):
            random_str = str([f"{x:.3f}" for x in random_vals])
            routes_str = str(routes)
            
            marker = " 🎯" if abs(fitness - best_fitness) < 1e-6 else ""
            logger.debug(f"S{i+1:<3} {random_str:<30} {routes_str:<40} {fitness:<10.3f}{marker}")
        
        logger.info(f"\nBest Solution: {best_fitness:.3f}")
    
    def print_final_results(self, 
                          final_results: Dict,
                          depot_data: Dict,
                          customers_data: List[Dict],
                          max_capacity: float,
                          max_vehicles: int) -> None:
        """Print final optimization results."""
        logger.info(f"\n" + "="*100)
        logger.info("FINAL VRP OPTIMIZATION RESULTS")
        logger.info("="*100)
        
        logger.info(f"Algorithm Parameters:")
        for key, value in final_results['algorithm_parameters'].items():
            logger.info(f"- {key.replace('_', ' ').title()}: {value}")
        logger.info("")
        
        logger.info(f"VRP Parameters:")
        for key, value in final_results['vrp_parameters'].items():
            logger.info(f"- {key.replace('_', ' ').title()}: {value}")
        logger.info("")
        
        logger.info(f"Best Solution:")
        logger.info(f"- Routes: {final_results['best_solution']['routes']}")
        logger.info(f"- Total Distance: {final_results['best_solution']['total_distance']:.3f}")
        logger.info("")
        
        best_routes = final_results['best_solution']['routes']
        if best_routes:
            # Print detailed route analysis
            self._print_detailed_route_analysis(best_routes, depot_data, customers_data, max_capacity, max_vehicles)
            
            logger.info(f"\nSummary:")
            total_customers = sum(len(route) for route in best_routes)
            total_demand = sum(calculate_route_demand(route, customers_data) for route in best_routes)
            logger.info(f"- Total customers served: {total_customers}/{len(customers_data)}")
            logger.info(f"- Total demand served: {total_demand}")
            logger.info(f"- Number of vehicles used: {len(best_routes)}/{max_vehicles}")
        
        logger.info(f"\nFitness Evolution:")
        logger.info(f"- Initial: {final_results['fitness_evolution']['initial']:.3f}")
        logger.info(f"- Final: {final_results['fitness_evolution']['final']:.3f}")
        logger.info(f"- Improvement: {final_results['fitness_evolution']['improvement']:.3f}")
        logger.info(f"- History: {[f'{f:.3f}' for f in final_results['fitness_evolution']['history']]}")
        
        logger.info("\n" + "="*100)
        logger.info("VRP OPTIMIZATION COMPLETED!")
        logger.info("="*100)
    
    def _print_detailed_route_analysis(self, 
                                     best_routes: List[List[int]], 
                                     depot_data: Dict, 
                                     customers_data: List[Dict], 
                                     max_capacity: float, 
                                     max_vehicles: int) -> None:
        """Print detailed analysis of the optimal routes with step-by-step explanations."""
        from .utils import calculate_distance
        
        logger.info("\n" + "="*100)
        logger.info("🚚 DETAILED ROUTE ANALYSIS - VEHICLE FLOW EXPLANATION")
        logger.info("="*100)
        
        depot_x, depot_y = depot_data['x'], depot_data['y']
        logger.info(f"📍 DEPOT LOCATION: ({depot_x}, {depot_y})")
        logger.info("")
        
        total_distance = 0
        
        for vehicle_id, route in enumerate(best_routes, 1):
            if not route:  # Skip empty routes
                continue
                
            logger.info(f"🚛 VEHICLE {vehicle_id} ROUTE ANALYSIS")
            logger.info("-" * 60)
            
            # Calculate route demand
            route_demand = calculate_route_demand(route, customers_data)
            logger.info(f"📦 Route Demand: {route_demand}/{max_capacity} ({route_demand/max_capacity*100:.1f}% capacity used)")
            logger.info(f"👥 Customers to visit: {len(route)}")
            logger.info("")
            
            # Step-by-step route explanation
            current_x, current_y = depot_x, depot_y
            route_distance = 0
            step = 1
            current_capacity = max_capacity
            
            logger.info("🔄 ROUTE FLOW:")
            logger.info(f"   Step {step}: Vehicle starts at DEPOT ({depot_x}, {depot_y})")
            logger.info(f"            Initial capacity: {current_capacity} units")
            
            for customer_id in route:
                # Find customer data
                customer = next(c for c in customers_data if c['id'] == customer_id)
                customer_x, customer_y = customer['x'], customer['y']
                customer_demand = customer['demand']
                
                # Calculate distance to customer
                distance_to_customer = calculate_distance(
                    {'x': current_x, 'y': current_y}, 
                    {'x': customer_x, 'y': customer_y}
                )
                route_distance += distance_to_customer
                
                step += 1
                logger.info(f"   Step {step}: Travel to Customer {customer_id} ({customer_x}, {customer_y})")
                logger.info(f"            Distance: {distance_to_customer:.3f} units")
                logger.info(f"            Deliver demand: {customer_demand} units")
                logger.info(f"            Remaining capacity: {current_capacity - customer_demand:.1f} units")
                
                current_capacity -= customer_demand
                current_x, current_y = customer_x, customer_y
            
            # Return to depot
            distance_to_depot = calculate_distance(
                {'x': current_x, 'y': current_y}, 
                {'x': depot_x, 'y': depot_y}
            )
            route_distance += distance_to_depot
            total_distance += route_distance
            
            step += 1
            logger.info(f"   Step {step}: Return to DEPOT ({depot_x}, {depot_y})")
            logger.info(f"            Distance: {distance_to_depot:.3f} units")
            logger.info(f"            Vehicle unloaded and ready for next route")
            
            logger.info("")
            logger.info(f"📊 VEHICLE {vehicle_id} SUMMARY:")
            logger.info(f"   Total distance: {route_distance:.3f} units")
            logger.info(f"   Customers served: {len(route)}")
            logger.info(f"   Demand delivered: {route_demand} units")
            logger.info(f"   Route efficiency: {route_demand/route_distance:.3f} units/distance")
            logger.info("")
        
        logger.info("="*100)
        logger.info("🎯 OVERALL OPTIMIZATION SUMMARY")
        logger.info("="*100)
        logger.info(f"Total distance traveled by all vehicles: {total_distance:.3f} units")
        logger.info(f"Total customers served: {sum(len(route) for route in best_routes)}")
        logger.info(f"Total demand delivered: {sum(calculate_route_demand(route, customers_data) for route in best_routes)} units")
        logger.info(f"Vehicles used: {len([r for r in best_routes if r])}/{max_vehicles}")
        logger.info(f"Average distance per vehicle: {total_distance/len([r for r in best_routes if r]):.3f} units")
        logger.info("="*100)

