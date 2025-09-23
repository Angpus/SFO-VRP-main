"""
Data loader module for VRP optimization.
Handles loading and validation of VRP data from CSV files.
"""

import csv
import logging
from typing import Dict, List, Tuple
import math

logger = logging.getLogger(__name__)


class VRPDataLoader:
    """Handles loading and validation of VRP data."""
    
    @staticmethod
    def read_vrp_data_from_csv(data_file: str = 'kecil.csv') -> Tuple[Dict, List[Dict]]:
        """
        Read VRP data from a single CSV file containing both depot and customers.
        
        Args:
            data_file: Path to the CSV file containing all data
            
        Returns:
            Tuple of (depot_data, customers_data) where:
            - depot_data: Dictionary with depot information
            - customers_data: List of customer information dictionaries
        """
        try:
            depot_data = None
            customers_data = []
            
            with open(data_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    # Convert string values to appropriate types
                    data = {
                        'id': int(row['Pelanggan']),
                        'x': float(row['X']),
                        'y': float(row['Y']),
                        'demand': float(row['Permintaan'])
                    }
                    
                    # First row (ID=0) is the depot
                    if data['id'] == 0:
                        depot_data = data
                    else:
                        customers_data.append(data)
            
            if depot_data is None:
                raise ValueError("No depot found in CSV file (missing row with ID=0)")
            
            logger.info(f"Successfully loaded data from {data_file}")
            logger.info(f"Depot: ID={depot_data['id']}, X={depot_data['x']}, Y={depot_data['y']}, Demand={depot_data['demand']}")
            logger.info(f"Customers: {len(customers_data)} customers loaded")
            
            return depot_data, customers_data
            
        except FileNotFoundError:
            logger.error(f"CSV file '{data_file}' not found")
            raise
        except KeyError as e:
            logger.error(f"Missing required column in CSV: {e}")
            raise
        except ValueError as e:
            logger.error(f"Invalid data in CSV: {e}")
            raise
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            raise
    
    @staticmethod
    def create_sample_data_csv(filename: str = 'kecil.csv', size: str = 'small') -> None:
        """
        Create a sample data CSV file with both depot and customer data.
        
        Args:
            filename: Name of the CSV file to create
            size: Size of data ('small', 'medium', 'big')
        """
        import random
        
        # Header
        data = [['Pelanggan', 'X', 'Y', 'Permintaan']]
        
        # Depot
        data.append([0, 30, 40, 0])
        
        # Generate customers based on size
        if size == 'small':
            num_customers = 18
            x_range = (20, 80)
            y_range = (20, 80)
            demand_range = (5, 35)
        elif size == 'medium':
            num_customers = 50
            x_range = (10, 100)
            y_range = (10, 100)
            demand_range = (5, 50)
        elif size == 'big':
            num_customers = 100
            x_range = (5, 120)
            y_range = (5, 120)
            demand_range = (5, 60)
        else:
            raise ValueError(f"Invalid size: {size}. Must be 'small', 'medium', or 'big'.")
        
        # Generate random customers
        for i in range(1, num_customers + 1):
            x = random.randint(x_range[0], x_range[1])
            y = random.randint(y_range[0], y_range[1])
            demand = random.randint(demand_range[0], demand_range[1])
            data.append([i, x, y, demand])
        
        with open(filename, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(data)
        
        logger.info(f"Sample {size} data file created: {filename} ({num_customers} customers)")
    
    @staticmethod
    def create_all_sample_files() -> None:
        """Create all sample data files (small, medium, big)."""
        VRPDataLoader.create_sample_data_csv('kecil.csv', 'small')
        VRPDataLoader.create_sample_data_csv('sedang.csv', 'medium')
        VRPDataLoader.create_sample_data_csv('besar.csv', 'big')
        logger.info("All sample data files created successfully!")
    
    @staticmethod
    def validate_vrp_data(depot_data: Dict, customers_data: List[Dict], 
                         max_capacity: float, max_vehicles: int) -> bool:
        """
        Validate VRP data for consistency and feasibility.
        
        Args:
            depot_data: Depot information dictionary
            customers_data: List of customer information dictionaries
            max_capacity: Maximum vehicle capacity
            max_vehicles: Maximum number of vehicles
            
        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Validate depot data
            if not depot_data:
                logger.error("Depot data is missing")
                return False
            
            required_depot_keys = ['id', 'x', 'y', 'demand']
            for key in required_depot_keys:
                if key not in depot_data:
                    logger.error(f"Missing required depot field: {key}")
                    return False
            
            if depot_data['id'] != 0:
                logger.error("Depot ID must be 0")
                return False
            
            if depot_data['demand'] != 0:
                logger.error("Depot demand must be 0")
                return False
            
            # Validate customers data
            if not customers_data:
                logger.error("No customers found")
                return False
            
            total_demand = 0
            customer_ids = set()
            
            for customer in customers_data:
                # Check required fields
                required_customer_keys = ['id', 'x', 'y', 'demand']
                for key in required_customer_keys:
                    if key not in customer:
                        logger.error(f"Missing required customer field: {key}")
                        return False
                
                # Check for duplicate IDs
                if customer['id'] in customer_ids:
                    logger.error(f"Duplicate customer ID: {customer['id']}")
                    return False
                customer_ids.add(customer['id'])
                
                # Check for valid demand
                if customer['demand'] <= 0:
                    logger.error(f"Invalid customer demand: {customer['demand']} (must be > 0)")
                    return False
                
                total_demand += customer['demand']
            
            # Check if all customers can be served
            if total_demand > max_capacity * max_vehicles:
                logger.error(f"Total demand ({total_demand}) exceeds total capacity ({max_capacity * max_vehicles})")
                return False
            
            # Check if any single customer exceeds vehicle capacity
            for customer in customers_data:
                if customer['demand'] > max_capacity:
                    logger.error(f"Customer {customer['id']} demand ({customer['demand']}) exceeds vehicle capacity ({max_capacity})")
                    return False
            
            logger.info("VRP data validation passed")
            logger.info(f"Total customers: {len(customers_data)}")
            logger.info(f"Total demand: {total_demand}")
            logger.info(f"Max capacity per vehicle: {max_capacity}")
            logger.info(f"Max vehicles: {max_vehicles}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during data validation: {e}")
            return False

    @staticmethod
    def recommend_capacity_and_vehicles(customers_data: List[Dict]) -> Tuple[float, int]:
        """
        Recommend a combination of (max_capacity, max_vehicles) that can feasibly
        serve all customers based on their demands.
        
        Strategy:
        - Choose the minimal number of vehicles V (starting from 1) such that
          capacity C = max(max_customer_demand, ceil(total_demand / V)) is feasible.
        - Among feasible options, prefer smaller V; break ties by smaller C.
        
        Returns:
            Tuple of (recommended_max_capacity, recommended_max_vehicles)
        """
        if not customers_data:
            return 0.0, 0
        total_demand = sum(c.get('demand', 0) for c in customers_data)
        max_customer_demand = max(c.get('demand', 0) for c in customers_data)
        num_customers = len(customers_data)

        best_pair: Tuple[float, int] = (float('inf'), float('inf'))
        for vehicles in range(1, num_customers + 1):
            # Minimum capacity needed with this many vehicles
            capacity = max(max_customer_demand, math.ceil(total_demand / vehicles))
            # Record candidate; prefer fewer vehicles, then smaller capacity
            candidate = (capacity, vehicles)
            if (candidate[1] < best_pair[1]) or (candidate[1] == best_pair[1] and candidate[0] < best_pair[0]):
                best_pair = candidate

        recommended_capacity, recommended_vehicles = best_pair
        return float(recommended_capacity), int(recommended_vehicles)
    
    @staticmethod
    def get_default_vrp_data() -> Tuple[Dict, List[Dict]]:
        """
        Get default VRP data for testing.
        
        Returns:
            Tuple of (depot_data, customers_data)
        """
        depot_data = {
            'id': 0,
            'x': 30,
            'y': 40,
            'demand': 0
        }
        
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
