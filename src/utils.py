"""
Utility functions for VRP optimization.
Includes distance calculations, route conversions, and other helper functions.
"""

import math
import random
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_distance(point1: Dict, point2: Dict, show_calculation: bool = False, point1_name: str = "", point2_name: str = "") -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: Dictionary with 'x' and 'y' coordinates
        point2: Dictionary with 'x' and 'y' coordinates
        show_calculation: If True, shows detailed calculation
        point1_name: Name of first point for display
        point2_name: Name of second point for display
        
    Returns:
        Euclidean distance between the points
    """
    dx = point1['x'] - point2['x']
    dy = point1['y'] - point2['y']
    distance = math.sqrt(dx**2 + dy**2)
    
    if show_calculation:
        logger.info(f"    Distance calculation: {point1_name} -> {point2_name}")
        logger.info(f"      Point 1: ({point1['x']}, {point1['y']})")
        logger.info(f"      Point 2: ({point2['x']}, {point2['y']})")
        logger.info(f"      dx = {point1['x']} - {point2['x']} = {dx}")
        logger.info(f"      dy = {point1['y']} - {point2['y']} = {dy}")
        logger.info(f"      Distance = √(dx² + dy²) = √({dx}² + {dy}²) = √({dx**2} + {dy**2}) = √{dx**2 + dy**2} = {distance:.3f}")
        logger.info("")
    
    return distance


def convert_random_to_route(random_values: List[float], 
                          depot_data: Dict, 
                          customers_data: List[Dict], 
                          max_capacity: float, 
                          max_vehicles: int) -> Tuple[List[List[int]], List[int]]:
    """
    Convert random values to VRP solution (routes with capacity constraints).
    
    Args:
        random_values: List of random values [0,1]
        depot_data: Depot information dictionary
        customers_data: List of customer information dictionaries
        max_capacity: Maximum vehicle capacity
        max_vehicles: Maximum number of vehicles
        
    Returns:
        Tuple of (routes, visit_order) where:
        - routes: List of routes, each route is a list of customer IDs
        - visit_order: List of customer IDs in visit order
    """
    n = len(random_values)
    
    # Create pairs of (value, customer_index)
    value_index_pairs = [(random_values[i], i) for i in range(n)]
    
    # Sort by value to get customer visit order
    sorted_pairs = sorted(value_index_pairs)
    visit_order = [customers_data[idx]['id'] for val, idx in sorted_pairs]
    
    # Build routes considering capacity constraints
    routes = []
    current_route = []
    current_capacity = 0
    
    for customer_id in visit_order:
        customer = customers_data[customer_id - 1]  # customer_id is 1-based, list is 0-based
        
        # Check if adding this customer exceeds capacity
        if current_capacity + customer['demand'] > max_capacity:
            # Start new route if current route is not empty
            if current_route:
                routes.append(current_route)
                current_route = []
                current_capacity = 0
        
        # Add customer to current route
        current_route.append(customer_id)
        current_capacity += customer['demand']
        
        # Check vehicle limit
        if len(routes) >= max_vehicles and current_route:
            break
    
    # Add the last route if not empty
    if current_route:
        routes.append(current_route)
    
    # Limit to max_vehicles
    routes = routes[:max_vehicles]
    
    return routes, visit_order


def calculate_vrp_fitness(routes: List[List[int]], 
                         depot_data: Dict, 
                         customers_data: List[Dict], 
                         show_details: bool = False) -> float:
    """
    Calculate VRP fitness (total distance).
    
    Args:
        routes: List of routes, each route is a list of customer IDs
        depot_data: Depot information dictionary
        customers_data: List of customer information dictionaries
        show_details: If True, shows detailed calculation
        
    Returns:
        Total distance of all routes
    """
    total_distance = 0
    route_details = []
    
    if show_details:
        logger.info("Detailed VRP Distance Calculation:")
        logger.info("="*60)
    
    for route_idx, route in enumerate(routes):
        route_distance = 0
        route_info = []
        
        if not route:  # Skip empty routes
            continue
        
        if show_details:
            logger.info(f"Route {route_idx + 1}: {route}")
        
        # Distance from depot to first customer
        first_customer = next(c for c in customers_data if c['id'] == route[0])
        dist_to_first = calculate_distance(depot_data, first_customer, show_details, f"Depot(0)", f"Customer({route[0]})")
        route_distance += dist_to_first
        
        if show_details:
            logger.info(f"  Depot(0) -> Customer({route[0]}): {dist_to_first:.3f}")
            route_info.append(f"0->{route[0]}: {dist_to_first:.3f}")
        
        # Distance between consecutive customers in route
        for i in range(len(route) - 1):
            current_customer = next(c for c in customers_data if c['id'] == route[i])
            next_customer = next(c for c in customers_data if c['id'] == route[i + 1])
            dist = calculate_distance(current_customer, next_customer, show_details, f"Customer({route[i]})", f"Customer({route[i + 1]})")
            route_distance += dist
            
            if show_details:
                logger.info(f"  Customer({route[i]}) -> Customer({route[i + 1]}): {dist:.3f}")
                route_info.append(f"{route[i]}->{route[i + 1]}: {dist:.3f}")
        
        # Distance from last customer back to depot
        last_customer = next(c for c in customers_data if c['id'] == route[-1])
        dist_to_depot = calculate_distance(last_customer, depot_data, show_details, f"Customer({route[-1]})", f"Depot(0)")
        route_distance += dist_to_depot
        
        if show_details:
            logger.info(f"  Customer({route[-1]}) -> Depot(0): {dist_to_depot:.3f}")
            route_info.append(f"{route[-1]}->0: {dist_to_depot:.3f}")
            logger.info(f"  Route {route_idx + 1} total distance: {route_distance:.3f}")
        
        route_details.append({
            'route': route,
            'distance': route_distance,
            'details': route_info
        })
        
        total_distance += route_distance
    
    if show_details:
        logger.info(f"Total Distance: {total_distance:.3f}")
        logger.info("="*60)
    
    return total_distance


def generate_random_values(n_individuals: int, problem_size: int) -> List[List[float]]:
    """
    Generate random values for each individual.
    
    Args:
        n_individuals: Number of individuals to generate
        problem_size: Size of each individual (number of customers)
        
    Returns:
        List of random value lists for each individual
    """
    random_values = []
    for i in range(n_individuals):
        individual_values = [round(random.random(), 3) for _ in range(problem_size)]
        random_values.append(individual_values)
    return random_values


def print_vrp_data(depot_data: Dict, customers_data: List[Dict], 
                  max_capacity: float, max_vehicles: int) -> None:
    """
    Print VRP problem data in a formatted way.
    
    Args:
        depot_data: Depot information dictionary
        customers_data: List of customer information dictionaries
        max_capacity: Maximum vehicle capacity
        max_vehicles: Maximum number of vehicles
    """
    logger.info("VRP Problem Data:")
    logger.info("="*50)
    logger.info(f"Max Capacity: {max_capacity}")
    logger.info(f"Max Vehicles: {max_vehicles}")
    logger.info(f"Number of Customers: {len(customers_data)}")
    logger.info("")
    
    logger.info(f"Depot: ID={depot_data['id']}, X={depot_data['x']}, Y={depot_data['y']}, Demand={depot_data['demand']}")
    logger.info("")
    
    logger.info("Customers:")
    logger.info(f"{'ID':<4} {'X':<6} {'Y':<6} {'Demand':<8}")
    logger.info("-" * 30)
    for customer in customers_data:
        logger.info(f"{customer['id']:<4} {customer['x']:<6} {customer['y']:<6} {customer['demand']:<8}")


def format_route_string(route: List[int]) -> str:
    """
    Format a route as a string with depot notation.
    
    Args:
        route: List of customer IDs in the route
        
    Returns:
        Formatted route string (e.g., "0-1-2-3-0")
    """
    if not route:
        return "0"
    return f"0-{'-'.join(map(str, route))}-0"


def calculate_route_demand(route: List[int], customers_data: List[Dict]) -> float:
    """
    Calculate total demand for a route.
    
    Args:
        route: List of customer IDs in the route
        customers_data: List of customer information dictionaries
        
    Returns:
        Total demand for the route
    """
    return sum(next(c['demand'] for c in customers_data if c['id'] == customer_id) 
              for customer_id in route)
