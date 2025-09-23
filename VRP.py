import csv
import pandas as pd
import numpy as np
import random
import sys
import os
import math
from datetime import datetime

class OutputLogger:
    """Class to handle dual output to both console and file"""
    def __init__(self, filename="vrp_output.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
        
        # Write header to log file
        self.log.write(f"Sailfish VRP Optimizer Output Log\n")
        self.log.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write(f"{'='*80}\n\n")
        self.log.flush()
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        if self.log:
            self.log.close()

def read_vrp_data_from_csv(depot_file="depot.csv", customer_file="customers.csv"):
    """
    Read VRP data from CSV files.
    Expected format for depot.csv: Nomor,X,Y,Permintaan
    Expected format for customers.csv: Nomor,X,Y,Permintaan
    """
    try:
        # Read depot data
        depot_df = pd.read_csv(depot_file)
        depot_data = {
            'id': int(depot_df.iloc[0]['Nomor']),
            'x': float(depot_df.iloc[0]['X']),
            'y': float(depot_df.iloc[0]['Y']),
            'demand': float(depot_df.iloc[0]['Permintaan'])
        }
        
        # Read customer data
        customers_df = pd.read_csv(customer_file)
        customers_data = []
        for _, row in customers_df.iterrows():
            customers_data.append({
                'id': int(row['Nomor']),
                'x': float(row['X']),
                'y': float(row['Y']),
                'demand': float(row['Permintaan'])
            })
        
        return depot_data, customers_data
    
    except FileNotFoundError:
        print("CSV files not found. Using default VRP data.")
        return get_default_vrp_data()

def get_default_vrp_data():
    """Return default VRP data from the image"""
    depot_data = {'id': 0, 'x': 30, 'y': 40, 'demand': 0}
    
    customers_data = [
        {'id': 1, 'x': 37, 'y': 52, 'demand': 19},
        {'id': 2, 'x': 49, 'y': 43, 'demand': 30},
        {'id': 3, 'x': 52, 'y': 64, 'demand': 16},
        {'id': 4, 'x': 31, 'y': 62, 'demand': 23},
        {'id': 5, 'x': 52, 'y': 33, 'demand': 11},
        {'id': 6, 'x': 42, 'y': 41, 'demand': 31},
        {'id': 7, 'x': 52, 'y': 41, 'demand': 15},
        {'id': 8, 'x': 57, 'y': 58, 'demand': 28},
        {'id': 9, 'x': 62, 'y': 42, 'demand': 14},
        {'id': 10, 'x': 42, 'y': 57, 'demand': 8},
        {'id': 11, 'x': 27, 'y': 68, 'demand': 7},
        {'id': 12, 'x': 43, 'y': 67, 'demand': 14},
        {'id': 13, 'x': 58, 'y': 27, 'demand': 19},
        {'id': 14, 'x': 37, 'y': 69, 'demand': 11},
        {'id': 15, 'x': 61, 'y': 33, 'demand': 26},
        {'id': 16, 'x': 62, 'y': 63, 'demand': 17},
        {'id': 17, 'x': 63, 'y': 69, 'demand': 6},
        {'id': 18, 'x': 45, 'y': 35, 'demand': 15}
    ]
    
    return depot_data, customers_data

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point1['x'] - point2['x'])**2 + (point1['y'] - point2['y'])**2)

def convert_random_to_route(random_values, depot_data, customers_data, max_capacity, max_vehicles):
    """
    Convert random values to VRP solution (routes with capacity constraints)
    
    Parameters:
    random_values: list of random values [0,1]
    depot_data: depot information
    customers_data: list of customer information
    max_capacity: maximum vehicle capacity
    max_vehicles: maximum number of vehicles
    
    Returns:
    routes: list of routes, each route is a list of customer IDs
    total_distance: total distance of all routes
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

def calculate_vrp_fitness(routes, depot_data, customers_data, show_details=False):
    """
    Calculate VRP fitness (total distance)
    
    Parameters:
    routes: list of routes, each route is a list of customer IDs
    depot_data: depot information
    customers_data: list of customer information
    show_details: bool, if True shows detailed calculation
    
    Returns:
    float: total distance of all routes
    """
    total_distance = 0
    route_details = []
    
    if show_details:
        print(f"\nDetailed VRP Distance Calculation:")
        print("="*60)
    
    for route_idx, route in enumerate(routes):
        route_distance = 0
        route_info = []
        
        if not route:  # Skip empty routes
            continue
        
        if show_details:
            print(f"\nRoute {route_idx + 1}: {route}")
        
        # Distance from depot to first customer
        first_customer = next(c for c in customers_data if c['id'] == route[0])
        dist_to_first = calculate_distance(depot_data, first_customer)
        route_distance += dist_to_first
        
        if show_details:
            print(f"  Depot(0) -> Customer({route[0]}): {dist_to_first:.3f}")
            route_info.append(f"0->{route[0]}: {dist_to_first:.3f}")
        
        # Distance between consecutive customers in route
        for i in range(len(route) - 1):
            current_customer = next(c for c in customers_data if c['id'] == route[i])
            next_customer = next(c for c in customers_data if c['id'] == route[i + 1])
            dist = calculate_distance(current_customer, next_customer)
            route_distance += dist
            
            if show_details:
                print(f"  Customer({route[i]}) -> Customer({route[i + 1]}): {dist:.3f}")
                route_info.append(f"{route[i]}->{route[i + 1]}: {dist:.3f}")
        
        # Distance from last customer back to depot
        last_customer = next(c for c in customers_data if c['id'] == route[-1])
        dist_to_depot = calculate_distance(last_customer, depot_data)
        route_distance += dist_to_depot
        
        if show_details:
            print(f"  Customer({route[-1]}) -> Depot(0): {dist_to_depot:.3f}")
            route_info.append(f"{route[-1]}->0: {dist_to_depot:.3f}")
            print(f"  Route {route_idx + 1} total distance: {route_distance:.3f}")
        
        route_details.append({
            'route': route,
            'distance': route_distance,
            'details': route_info
        })
        
        total_distance += route_distance
    
    if show_details:
        print(f"\nTotal Distance: {total_distance:.3f}")
        print("="*60)
    
    return total_distance

def print_vrp_data(depot_data, customers_data, max_capacity, max_vehicles):
    """Print VRP problem data"""
    print("\nVRP Problem Data:")
    print("="*50)
    print(f"Max Capacity: {max_capacity}")
    print(f"Max Vehicles: {max_vehicles}")
    print(f"Number of Customers: {len(customers_data)}")
    print()
    
    print(f"Depot: ID={depot_data['id']}, X={depot_data['x']}, Y={depot_data['y']}, Demand={depot_data['demand']}")
    print()
    
    print("Customers:")
    print(f"{'ID':<4} {'X':<6} {'Y':<6} {'Demand':<8}")
    print("-" * 30)
    for customer in customers_data:
        print(f"{customer['id']:<4} {customer['x']:<6} {customer['y']:<6} {customer['demand']:<8}")

class SailfishVRPOptimizer:
    def __init__(self, n_sailfish, n_sardines, depot_data, customers_data, 
                 max_capacity=170, max_vehicles=2, max_iter=100, A=4, epsilon=0.001, log_to_file=True):
        """
        Initialize Sailfish VRP Optimizer
        
        Parameters:
        n_sailfish: number of sailfish
        n_sardines: number of sardines
        depot_data: depot information dictionary
        customers_data: list of customer information dictionaries
        max_capacity: maximum vehicle capacity
        max_vehicles: maximum number of vehicles
        max_iter: maximum number of iterations
        A: sailfish optimizer parameter
        epsilon: convergence parameter
        log_to_file: whether to log output to file
        """
        if n_sardines <= n_sailfish:
            raise ValueError("Number of sardines must be greater than number of sailfish")
        
        self.original_n_sailfish = n_sailfish
        self.original_n_sardines = n_sardines
        self.n_sailfish = n_sailfish
        self.n_sardines = n_sardines
        self.depot_data = depot_data
        self.customers_data = customers_data
        self.max_capacity = max_capacity
        self.max_vehicles = max_vehicles
        self.max_iter = max_iter
        self.A = A
        self.epsilon = epsilon
        self.problem_size = len(customers_data)  # Number of customers
        
        # Set up logging
        self.log_to_file = log_to_file
        self.logger = None
        if log_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sailfish_vrp_SF{n_sailfish}_S{n_sardines}_{timestamp}.txt"
            self.logger = OutputLogger(filename)
            sys.stdout = self.logger
            print(f"VRP Output will be logged to: {filename}")
            print(f"Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Parameters: {n_sailfish} sailfish, {n_sardines} sardines, A={A}, epsilon={epsilon}")
            print("="*80 + "\n")
        
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
        
        # Best solution tracking
        self.best_routes = None
        self.best_fitness = float('inf')
        self.fitness_history = []
        
        # Elite tracking
        self.elite_sailfish_fitness = None
        self.elite_sailfish_position = None
        self.injured_sardine_fitness = None
        self.injured_sardine_position = None
        
        # Algorithm variables
        self.lambda_k_values = []
        self.PD = None
        self.AP = None
        self.current_iteration = 0
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if self.log_to_file and self.logger:
            print(f"\nRun completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
            sys.stdout = self.logger.terminal
            self.logger.close()
    
    def save_original_positions(self):
        """Save original positions before sorting and replacement"""
        self.original_sailfish_positions = [pos.copy() for pos in self.sailfish_random_values]
        self.original_sardine_positions = [pos.copy() for pos in self.sardine_random_values]
    
    def print_initial_parameters(self):
        """Print initial parameters and VRP data"""
        print("\n" + "="*80)
        print("1. INITIAL VARIABLES AND VRP DATA")
        print("="*80)
        
        print(f"Initial Parameters:")
        print(f"- Problem size: {self.problem_size} customers")
        print(f"- Sailfish population: {self.n_sailfish}")
        print(f"- Sardine population: {self.n_sardines}")
        print(f"- Maximum iterations: {self.max_iter}")
        print(f"- Parameter A: {self.A}")
        print(f"- Epsilon: {self.epsilon}")
        print()
        
        print_vrp_data(self.depot_data, self.customers_data, self.max_capacity, self.max_vehicles)
    
    def generate_random_values(self, n_individuals):
        """Generate random values for each individual"""
        random_values = []
        for i in range(n_individuals):
            individual_values = [round(random.random(), 3) for _ in range(self.problem_size)]
            random_values.append(individual_values)
        return random_values
    
    def print_random_populations(self):
        """Print random values for sailfish and sardines"""
        print("\n" + "="*80)
        print("2. RANDOM SAILFISH AND SARDINES")
        print("="*80)
        
        self.sailfish_random_values = self.generate_random_values(self.n_sailfish)
        self.sardine_random_values = self.generate_random_values(self.n_sardines)
        
        print("SAILFISH Random Values:")
        print(f"{'ID':<8}", end="")
        for i in range(self.problem_size):
            print(f"C{i+1:2}", end="   ")
        print()
        
        for i in range(self.n_sailfish):
            print(f"SF{i+1:<6}", end="")
            for val in self.sailfish_random_values[i]:
                print(f"{val:5.3f}", end=" ")
            print()
        
        print("\nSARDINE Random Values:")
        print(f"{'ID':<8}", end="")
        for i in range(self.problem_size):
            print(f"C{i+1:2}", end="   ")
        print()
        
        for i in range(self.n_sardines):
            print(f"S{i+1:<7}", end="")
            for val in self.sardine_random_values[i]:
                print(f"{val:5.3f}", end=" ")
            print()
    
    def print_routes_and_solutions(self):
        """Print routes for each sailfish and sardine"""
        print("\n" + "="*80)
        if self.current_iteration == 0:
            print("3. ROUTES FOR EACH SAILFISH AND SARDINE")
        else:
            print(f"ITERATION {self.current_iteration} - STEP 1: GENERATING NEW ROUTES")
        print("="*80)
        
        self.sailfish_routes = []
        self.sardine_routes = []
        
        print("SAILFISH Routes:")
        for i in range(self.n_sailfish):
            print(f"\n===== SF{i+1} ====================================================")
            
            random_vals = self.sailfish_random_values[i]
            routes, visit_order = convert_random_to_route(random_vals, self.depot_data, 
                                                        self.customers_data, self.max_capacity, self.max_vehicles)
            self.sailfish_routes.append(routes)
            
            print(f"Random values: {random_vals}")
            print(f"Visit order: {visit_order}")
            print(f"Routes (with depot returns): ", end="")
            for j, route in enumerate(routes):
                route_str = f"0-{'-'.join(map(str, route))}-0"
                print(f"Route{j+1}: {route_str}", end="  ")
            print()
            
            # Calculate route capacities
            for j, route in enumerate(routes):
                total_demand = sum(next(c['demand'] for c in self.customers_data if c['id'] == customer_id) 
                                 for customer_id in route)
                print(f"Route{j+1} demand: {total_demand}/{self.max_capacity}")
        
        print("\nSARDINE Routes:")
        for i in range(self.n_sardines):
            print(f"\n===== S{i+1} ====================================================")
            
            random_vals = self.sardine_random_values[i]
            routes, visit_order = convert_random_to_route(random_vals, self.depot_data, 
                                                        self.customers_data, self.max_capacity, self.max_vehicles)
            self.sardine_routes.append(routes)
            
            print(f"Random values: {random_vals}")
            print(f"Visit order: {visit_order}")
            print(f"Routes (with depot returns): ", end="")
            for j, route in enumerate(routes):
                route_str = f"0-{'-'.join(map(str, route))}-0"
                print(f"Route{j+1}: {route_str}", end="  ")
            print()
            
            # Calculate route capacities
            for j, route in enumerate(routes):
                total_demand = sum(next(c['demand'] for c in self.customers_data if c['id'] == customer_id) 
                                 for customer_id in route)
                print(f"Route{j+1} demand: {total_demand}/{self.max_capacity}")
    
    def calculate_detailed_fitness(self):
        """Calculate fitness for each individual with detailed calculations"""
        print("\n" + "="*80)
        if self.current_iteration == 0:
            print("4. DETAILED FITNESS CALCULATION FOR EACH INDIVIDUAL")
        else:
            print(f"ITERATION {self.current_iteration} - STEP 2: DETAILED FITNESS CALCULATION")
        print("="*80)
        
        self.sailfish_fitness = []
        self.sardine_fitness = []
        
        print("SAILFISH Fitness Calculations:")
        print("=" * 50)
        
        for i, routes in enumerate(self.sailfish_routes):
            print(f"\nCALCULATING FITNESS FOR SAILFISH SF{i+1}")
            fitness = calculate_vrp_fitness(routes, self.depot_data, self.customers_data, show_details=True)
            self.sailfish_fitness.append(fitness)
            
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_routes = routes
                print(f"     NEW BEST SOLUTION! Fitness: {fitness:.3f}")
        
        print("\n" + "=" * 50)
        print("SARDINE Fitness Calculations:")
        print("=" * 50)
        
        for i, routes in enumerate(self.sardine_routes):
            print(f"\nCALCULATING FITNESS FOR SARDINE S{i+1}")
            fitness = calculate_vrp_fitness(routes, self.depot_data, self.customers_data, show_details=True)
            self.sardine_fitness.append(fitness)
            
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_routes = routes
                print(f"     NEW BEST SOLUTION! Fitness: {fitness:.3f}")
    
    def print_fitness_summary(self):
        """Print summary of all fitness scores"""
        print(f"\n" + "="*80)
        if self.current_iteration == 0:
            print("FITNESS SUMMARY")
        else:
            print(f"ITERATION {self.current_iteration} - STEP 3: FITNESS SUMMARY")
        print("="*80)
        
        print("SAILFISH FITNESS SCORES:")
        print("-" * 30)
        for i, fitness in enumerate(self.sailfish_fitness):
            marker = " ⭐ BEST" if fitness == min(self.sailfish_fitness) else ""
            print(f"SF{i+1}: {fitness:.3f}{marker}")
        
        print("\nSARDINE FITNESS SCORES:")
        print("-" * 25)
        for i, fitness in enumerate(self.sardine_fitness):
            marker = " ⭐ BEST" if fitness == min(self.sardine_fitness) else ""
            print(f"S{i+1}: {fitness:.3f}{marker}")
        
        print(f"\nOVERALL SUMMARY:")
        print("-" * 20)
        print(f"Best Sailfish Fitness: {min(self.sailfish_fitness):.3f}")
        print(f"Best Sardine Fitness: {min(self.sardine_fitness):.3f}")
        print(f"Overall Best Fitness: {self.best_fitness:.3f}")
        print(f"Best Routes: {self.best_routes}")
        
        # Update elite positions
        best_sailfish_idx = self.sailfish_fitness.index(min(self.sailfish_fitness))
        self.elite_sailfish_fitness = min(self.sailfish_fitness)
        self.elite_sailfish_position = self.sailfish_random_values[best_sailfish_idx].copy()
        
        if self.sardine_fitness:
            best_sardine_idx = self.sardine_fitness.index(min(self.sardine_fitness))
            self.injured_sardine_fitness = min(self.sardine_fitness)
            self.injured_sardine_position = self.sardine_random_values[best_sardine_idx].copy()
    
    def perform_sailfish_sardine_replacement(self):
        """Replace sailfish with better sardines"""
        print(f"\n" + "="*80)
        print(f"ITERATION {self.current_iteration} - STEP 4: SAILFISH-SARDINE REPLACEMENT")
        print("="*80)
        
        worst_sailfish_fitness = max(self.sailfish_fitness)
        better_sardines = []
        
        for i, sardine_fitness in enumerate(self.sardine_fitness):
            if sardine_fitness < worst_sailfish_fitness:
                better_sardines.append((i, sardine_fitness))
        
        print(f"Analysis:")
        print(f"- Worst sailfish fitness: {worst_sailfish_fitness:.3f}")
        print(f"- Sardines better than worst sailfish: {len(better_sardines)}")
        
        if not better_sardines:
            print("- No replacement will occur")
            return
        
        better_sardines.sort(key=lambda x: x[1])
        sardines_to_remove = []
        replacements_made = []
        
        for sardine_idx, sardine_fitness in better_sardines:
            worst_sf_idx = self.sailfish_fitness.index(max(self.sailfish_fitness))
            worst_sf_fitness = self.sailfish_fitness[worst_sf_idx]
            
            if sardine_fitness < worst_sf_fitness:
                print(f"\nReplacement {len(replacements_made) + 1}:")
                print(f"- Sardine S{sardine_idx+1} (fitness: {sardine_fitness:.3f}) -> Sailfish SF{worst_sf_idx+1} (fitness: {worst_sf_fitness:.3f})")
                
                # Replace sailfish with sardine data
                self.sailfish_random_values[worst_sf_idx] = self.sardine_random_values[sardine_idx].copy()
                self.sailfish_routes[worst_sf_idx] = self.sardine_routes[sardine_idx]
                self.sailfish_fitness[worst_sf_idx] = self.sardine_fitness[sardine_idx]
                
                sardines_to_remove.append(sardine_idx)
                replacements_made.append({
                    'sardine_idx': sardine_idx,
                    'sailfish_idx': worst_sf_idx,
                    'new_fitness': sardine_fitness,
                    'old_fitness': worst_sf_fitness
                })
                
                if sardine_fitness < self.best_fitness:
                    self.best_fitness = sardine_fitness
                    self.best_routes = self.sardine_routes[sardine_idx]
                    print(f"  NEW OVERALL BEST SOLUTION! Fitness: {sardine_fitness:.3f}")
            else:
                break
        
        # Remove replaced sardines
        sardines_to_remove.sort(reverse=True)
        for sardine_idx in sardines_to_remove:
            del self.sardine_random_values[sardine_idx]
            del self.sardine_routes[sardine_idx]
            del self.sardine_fitness[sardine_idx]
            del self.original_sardine_positions[sardine_idx]
            self.n_sardines -= 1
        
        print(f"\nReplacement Summary: {len(replacements_made)} replacements made")
        
        # Update elite positions
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
    
    def calculate_pd_and_lambda_values(self):
        """Calculate PD and lambda values"""
        print(f"\n" + "="*80)
        if self.current_iteration == 0:
            print("5. CALCULATE PD AND LAMBDA VALUES")
        else:
            print(f"ITERATION {self.current_iteration} - STEP 5: CALCULATE PD AND LAMBDA VALUES")
        print("="*80)
        
        total_population = self.n_sailfish + self.n_sardines
        self.PD = 1 - (self.n_sailfish / total_population)
        
        print(f"Population Decline (PD) = 1 - ({self.n_sailfish} / {total_population}) = {self.PD:.6f}")
        print()
        
        self.lambda_k_values = []
        print("Lambda Calculations:")
        
        for k in range(self.n_sailfish):
            random_val = round(random.random(), 3)
            lambda_k = (2 * random_val * self.PD) - self.PD
            self.lambda_k_values.append(lambda_k)
            
            print(f"SF{k+1}: λ_{k+1} = (2 × {random_val} × {self.PD:.6f}) - {self.PD:.6f} = {lambda_k:.6f}")
    
    def update_sailfish_positions(self):
        """Update sailfish positions using CORRECTED formula with FITNESS VALUES"""
        print(f"\n" + "="*80)
        if self.current_iteration == 0:
            print("6. UPDATE SAILFISH POSITIONS")
        else:
            print(f"ITERATION {self.current_iteration} - STEP 6: UPDATE SAILFISH POSITIONS")
        print("="*80)
        
        print("CORRECTED FORMULA: SF[i] = elite_sailfish_fitness - lambda[k] × ((random × (elite_sailfish_fitness + injured_sardine_fitness)/2) - old_sailfish)")
        print(f"Elite Sailfish Fitness: {self.elite_sailfish_fitness:.3f}")
        print(f"Injured Sardine Fitness: {self.injured_sardine_fitness:.3f}")
        print("Using original positions for updates...")
        new_sailfish_positions = []
        
        for k in range(self.n_sailfish):
            print(f"\nUpdating SF{k+1}:")
            new_position = []
            
            for j in range(self.problem_size):
                rand = round(random.random(), 3)
                elite_sf_fitness = self.elite_sailfish_fitness
                injured_sardine_fitness = self.injured_sardine_fitness
                old_sailfish_j = self.original_sailfish_positions[k][j]
                
                # CORRECTED FORMULA: SF[i] = elite_sailfish_fitness - lambda[k] × ((random × (elite_sailfish_fitness + injured_sardine_fitness)/2) - old_sailfish)
                avg_elite_injured = (elite_sf_fitness + injured_sardine_fitness) / 2
                bracket_term = (rand * avg_elite_injured) - old_sailfish_j
                lambda_term = self.lambda_k_values[k] * bracket_term
                new_val = elite_sf_fitness - lambda_term
                
                # Normalize to [0,1] range
                # new_val = max(0, min(1, abs(new_val) % 1))
                # new_val = round(new_val, 3)
                new_position.append(new_val)
                
                print(f"  Pos[{j+1}]: {elite_sf_fitness:.3f} - {self.lambda_k_values[k]:.6f} × (({rand:.3f} × {avg_elite_injured:.3f}) - {old_sailfish_j:.3f}) = {new_val:.3f}")
            
            new_sailfish_positions.append(new_position)
            print(f"New position: {[f'{x:.3f}' for x in new_position]}")
        
        self.sailfish_random_values = new_sailfish_positions
        print("\nAll sailfish positions updated!")
    
    def calculate_ap_and_update_sardines(self):
        """Calculate AP and update sardine positions"""
        print(f"\n" + "="*80)
        if self.current_iteration == 0:
            print("7. CALCULATE AP AND UPDATE SARDINE POSITIONS")
        else:
            print(f"ITERATION {self.current_iteration} - STEP 7: CALCULATE AP AND UPDATE SARDINE POSITIONS")
        print("="*80)
        
        if self.n_sardines == 0:
            print("No sardines remaining. Skipping sardine update.")
            return
        
        self.AP = self.A * (1 - (2 * (self.current_iteration + 1) * self.epsilon))
        
        print(f"Attack Power (AP) = {self.A} × (1 - (2 × {self.current_iteration + 1} × {self.epsilon})) = {self.AP:.6f}")
        
        if self.AP >= 0.5:
            print(f"AP >= 0.5: Update ALL sardine positions")
            self.update_all_sardines()
        else:
            print(f"AP < 0.5: Partial sardine update")
            self.update_partial_sardines()
    
    def update_all_sardines(self):
        """Update all sardine positions when AP >= 0.5 using CORRECTED formula with FITNESS VALUES"""
        print("\nCORRECTED FORMULA: S[i] = random × (elite_sailfish_fitness - old_sardine + AP)")
        print(f"Elite Sailfish Fitness: {self.elite_sailfish_fitness:.3f}")
        print("Updating ALL sardines:")
        new_sardine_positions = []
        
        for i in range(self.n_sardines):
            print(f"\nUpdating S{i+1}:")
            new_position = []
            
            for j in range(self.problem_size):
                rand = round(random.random(), 3)
                elite_sf_fitness = self.elite_sailfish_fitness
                old_sardine_j = self.original_sardine_positions[i][j]
                
                # CORRECTED FORMULA: S[i] = random × (elite_sailfish_fitness - old_sardine + AP)
                bracket_term = elite_sf_fitness - old_sardine_j + self.AP
                new_val = rand * bracket_term
                
                # Normalize to [0,1] range
                # new_val = max(0, min(1, abs(new_val) % 1))
                # new_val = round(new_val, 3)
                new_position.append(new_val)
                
                print(f"  Pos[{j+1}]: {rand:.3f} × ({elite_sf_fitness:.3f} - {old_sardine_j:.3f} + {self.AP:.6f}) = {new_val:.3f}")
            
            new_sardine_positions.append(new_position)
            print(f"New position: {[f'{x:.3f}' for x in new_position]}")
        
        self.sardine_random_values = new_sardine_positions
        print("\nAll sardine positions updated!")
    
    def update_partial_sardines(self):
        """Update partial sardines when AP < 0.5 using CORRECTED formula with FITNESS VALUES"""
        print("\nCORRECTED FORMULA: S[i] = random × (elite_sailfish_fitness - old_sardine + AP)")
        print(f"Elite Sailfish Fitness: {self.elite_sailfish_fitness:.3f}")
        print("Partial sardine update:")
        
        alpha = int(self.n_sardines * self.AP)
        beta = int(self.problem_size * self.AP)
        
        print(f"alpha = {self.n_sardines} × {self.AP:.6f} = {alpha}")
        print(f"beta = {self.problem_size} × {self.AP:.6f} = {beta}")
        
        if alpha == 0 or beta == 0:
            print("Alpha or beta is 0, no sardines updated.")
            return
        
        sardines_to_update = random.sample(range(self.n_sardines), min(alpha, self.n_sardines))
        
        print(f"Selected sardines to update: {[f'S{i+1}' for i in sardines_to_update]}")
        
        for i in sardines_to_update:
            print(f"\nUpdating S{i+1} (partial):")
            positions_to_update = random.sample(range(self.problem_size), min(beta, self.problem_size))
            print(f"Updating positions: {[j+1 for j in positions_to_update]}")
            
            new_position = self.original_sardine_positions[i].copy()
            
            for j in positions_to_update:
                rand = round(random.random(), 3)
                elite_sf_fitness = self.elite_sailfish_fitness
                old_sardine_j = self.original_sardine_positions[i][j]
                
                # CORRECTED FORMULA: S[i] = random × (elite_sailfish_fitness - old_sardine + AP)
                bracket_term = elite_sf_fitness - old_sardine_j + self.AP
                new_val = rand * bracket_term
                
                # Normalize to [0,1] range
                # new_val = max(0, min(1, abs(new_val) % 1))
                # new_val = round(new_val, 3)
                new_position[j] = new_val
                
                print(f"  Pos[{j+1}]: {rand:.3f} × ({elite_sf_fitness:.3f} - {old_sardine_j:.3f} + {self.AP:.6f}) = {new_val:.3f}")
            
            self.sardine_random_values[i] = new_position
            print(f"New position: {[f'{x:.3f}' for x in new_position]}")
    
    def print_comprehensive_results_table(self):
        """Print comprehensive results table"""
        print(f"\n" + "="*120)
        if self.current_iteration == 0:
            print("COMPREHENSIVE RESULTS TABLE")
        else:
            print(f"ITERATION {self.current_iteration} - COMPREHENSIVE RESULTS TABLE")
        print("="*120)
        
        print(f"{'ID':<4} {'Random Values':<30} {'Routes':<40} {'Fitness':<10}")
        print("-" * 120)
        
        # Sailfish table
        for i in range(self.n_sailfish):
            random_str = str([f"{x:.3f}" for x in self.sailfish_random_values[i]])
            routes_str = str(self.sailfish_routes[i])
            fitness = self.sailfish_fitness[i]
            
            marker = " 🎯" if abs(fitness - self.best_fitness) < 1e-6 else ""
            print(f"SF{i+1:<2} {random_str:<30} {routes_str:<40} {fitness:<10.3f}{marker}")
        
        # Sardine table
        for i in range(self.n_sardines):
            random_str = str([f"{x:.3f}" for x in self.sardine_random_values[i]])
            routes_str = str(self.sardine_routes[i])
            fitness = self.sardine_fitness[i]
            
            marker = " 🎯" if abs(fitness - self.best_fitness) < 1e-6 else ""
            print(f"S{i+1:<3} {random_str:<30} {routes_str:<40} {fitness:<10.3f}{marker}")
        
        print(f"\nBest Solution: {self.best_routes} with fitness: {self.best_fitness:.3f}")
    
    def run_iteration_zero(self):
        """Run the initial iteration"""
        self.current_iteration = 0
        
        self.print_initial_parameters()
        self.print_random_populations()
        self.save_original_positions()
        self.print_routes_and_solutions()
        self.calculate_detailed_fitness()
        self.print_fitness_summary()
        self.print_comprehensive_results_table()
        self.calculate_pd_and_lambda_values()
        self.update_sailfish_positions()
        self.calculate_ap_and_update_sardines()
        
        self.fitness_history.append(self.best_fitness)
        
        print(f"\n" + "="*80)
        print("ITERATION 0 COMPLETED")
        print("="*80)
        print(f"Best fitness: {self.best_fitness:.3f}")
        print(f"Best routes: {self.best_routes}")
    
    def run_iteration(self, iteration_num):
        """Run a single iteration"""
        self.current_iteration = iteration_num
        
        print(f"\n" + "="*100)
        print(f"STARTING ITERATION {iteration_num}")
        print("="*100)
        
        self.save_original_positions()
        self.print_routes_and_solutions()
        self.calculate_detailed_fitness()
        self.print_fitness_summary()
        self.perform_sailfish_sardine_replacement()
        self.print_comprehensive_results_table()
        self.calculate_pd_and_lambda_values()
        self.update_sailfish_positions()
        self.calculate_ap_and_update_sardines()
        
        self.fitness_history.append(self.best_fitness)
        
        print(f"\n" + "="*80)
        print(f"ITERATION {iteration_num} COMPLETED")
        print("="*80)
        print(f"Best fitness: {self.best_fitness:.3f}")
        print(f"Best routes: {self.best_routes}")
    
    def check_convergence(self):
        """Check if algorithm has converged"""
        if len(self.fitness_history) < 2:
            return False
        
        improvement = abs(self.fitness_history[-2] - self.fitness_history[-1])
        return improvement < self.epsilon
    
    def run_optimization(self):
        """Run the complete optimization process"""
        print("STARTING SAILFISH VRP OPTIMIZATION ALGORITHM")
        print("="*80)
        
        self.run_iteration_zero()
        
        for iteration in range(1, self.max_iter + 1):
            if self.n_sardines == 0:
                print(f"\nNo sardines remaining after iteration {iteration-1}. Stopping.")
                break
            
            self.run_iteration(iteration)
            
            if self.check_convergence():
                print(f"\nConvergence achieved after iteration {iteration}")
                break
        
        self.print_final_results()
    
    def print_final_results(self):
        """Print final optimization results"""
        print(f"\n" + "="*100)
        print("FINAL VRP OPTIMIZATION RESULTS")
        print("="*100)
        
        print(f"Algorithm Parameters:")
        print(f"- Initial Sailfish: {self.original_n_sailfish}")
        print(f"- Initial Sardines: {self.original_n_sardines}")
        print(f"- Final Sailfish: {self.n_sailfish}")
        print(f"- Final Sardines: {self.n_sardines}")
        print(f"- Iterations: {len(self.fitness_history)}")
        print()
        
        print(f"VRP Parameters:")
        print(f"- Customers: {self.problem_size}")
        print(f"- Max Capacity: {self.max_capacity}")
        print(f"- Max Vehicles: {self.max_vehicles}")
        print()
        
        print(f"Best Solution:")
        print(f"- Routes: {self.best_routes}")
        print(f"- Total Distance: {self.best_fitness:.3f}")
        print()
        
        if self.best_routes:
            print("Detailed Best Solution Analysis:")
            calculate_vrp_fitness(self.best_routes, self.depot_data, self.customers_data, show_details=True)
            
            print("\nRoute Details:")
            total_customers = 0
            total_demand = 0
            
            for i, route in enumerate(self.best_routes):
                route_demand = sum(next(c['demand'] for c in self.customers_data if c['id'] == customer_id) 
                                 for customer_id in route)
                route_str = f"0-{'-'.join(map(str, route))}-0"
                print(f"Route {i+1}: {route_str} (Demand: {route_demand}/{self.max_capacity})")
                total_customers += len(route)
                total_demand += route_demand
            
            print(f"\nSummary:")
            print(f"- Total customers served: {total_customers}/{self.problem_size}")
            print(f"- Total demand served: {total_demand}")
            print(f"- Number of vehicles used: {len(self.best_routes)}/{self.max_vehicles}")
        
        print(f"\nFitness Evolution:")
        print(f"- Initial: {self.fitness_history[0]:.3f}")
        print(f"- Final: {self.fitness_history[-1]:.3f}")
        print(f"- Improvement: {self.fitness_history[0] - self.fitness_history[-1]:.3f}")
        print(f"- History: {[f'{f:.3f}' for f in self.fitness_history]}")
        
        print("\n" + "="*100)
        print("VRP OPTIMIZATION COMPLETED!")
        print("="*100)


def create_sample_csv_files():
    """Create sample CSV files for testing"""
    # Create depot.csv
    depot_data = [['Nomor', 'X', 'Y', 'Permintaan'], [0, 30, 40, 0]]
    with open('depot.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(depot_data)
    
    # Create customers.csv
    customers_data = [['Nomor', 'X', 'Y', 'Permintaan']]
    customers_info = [
        [1, 37, 52, 19], [2, 49, 43, 30], [3, 52, 64, 16], [4, 31, 62, 23],
        [5, 52, 33, 11], [6, 42, 41, 31], [7, 52, 41, 15], [8, 57, 58, 28],
        [9, 62, 42, 14], [10, 42, 57, 8], [11, 27, 68, 7], [12, 43, 67, 14],
        [13, 58, 27, 19], [14, 37, 69, 11], [15, 61, 33, 26], [16, 62, 63, 17],
        [17, 63, 69, 6], [18, 45, 35, 15]
    ]
    customers_data.extend(customers_info)
    
    with open('customers.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(customers_data)
    
    print("Sample CSV files created: depot.csv and customers.csv")


def main():
    """Main function to run the Sailfish VRP Optimizer"""
    
    # Option 1: Create sample CSV files
    create_sample_csv_files()
    
    # Option 2: Read from CSV files
    depot_data, customers_data = read_vrp_data_from_csv("depot.csv", "customers.csv")
    
    # Option 3: Use default data (uncomment to use)
    # depot_data, customers_data = get_default_vrp_data()
    
    # Create and run VRP optimizer
    optimizer = SailfishVRPOptimizer(
        n_sailfish=3,           # Number of sailfish
        n_sardines=5,           # Number of sardines
        depot_data=depot_data,
        customers_data=customers_data,
        max_capacity=170,       # Vehicle capacity
        max_vehicles=2,         # Maximum vehicles
        max_iter=10,            # Maximum iterations
        A=4,                    # Algorithm parameter
        epsilon=0.001,          # Convergence threshold
        log_to_file=True        # Enable file logging
    )
    
    # Run optimization
    optimizer.run_optimization()


if __name__ == "__main__":
    main()