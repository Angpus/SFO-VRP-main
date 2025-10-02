"""
Controller module for user interaction and input validation.
Handles terminal-based user interface for VRP optimization system.
"""

import os
import logging
from typing import Dict, Tuple
from .utils import print_system_header

logger = logging.getLogger(__name__)


class VRPController:
    """Handles user interaction and input validation for VRP system."""
    
    def __init__(self):
        """Initialize the controller."""
        self.available_data_files = {
            'small': 'kecil.csv',
            'medium': 'sedang.csv', 
            'big': 'besar.csv'
        }
    
    def display_welcome_message(self) -> None:
        """Display welcome message and system information."""
        print("\n")
        print_system_header("terminal")
        print("\n🚢 SAILFISH VRP OPTIMIZATION SYSTEM")
        print("="*80)
        print("Welcome to the Vehicle Routing Problem (VRP) optimization system!")
        print("This system uses the Sailfish Optimizer algorithm to solve VRP problems.")
        print("="*80)
    
    def get_user_preferences(self) -> Dict:
        """
        Get user preferences through interactive terminal interface.
        
        Returns:
            Dictionary containing user preferences
        """
        print("\n📋 SYSTEM CONFIGURATION")
        print("-" * 40)
        
        # Get data size preference
        data_size = self._get_data_size_preference()
        
        # Store current data file for validation
        self.current_data_file = self.available_data_files[data_size]
        
        # Get logging preference
        log_to_file = self._get_logging_preference()
        
        # Get optimization parameters
        optimization_params = self._get_optimization_parameters()
        
        # Combine all preferences
        config = {
            'data_file': self.available_data_files[data_size],
            'data_size': data_size,
            'log_to_file': log_to_file,
            'output_mode': getattr(self, 'output_mode', 'steps'),
            **optimization_params
        }
        
        return config
    
    def _get_data_size_preference(self) -> str:
        """
        Get user's data size preference with validation.
        
        Returns:
            Selected data size ('small', 'medium', or 'big')
        """
        while True:
            print("\n📊 DATA SIZE SELECTION")
            print("Choose the size of data to use for optimization:")
            print("1. Small data (kecil.csv) - 18 customers")
            print("2. Medium data (sedang.csv) - 75 customers") 
            print("3. Big data (besar.csv) - 100 customers")
            
            choice = input("\nEnter your choice (1/2/3): ").strip()
            
            if choice == '1':
                return 'small'
            elif choice == '2':
                return 'medium'
            elif choice == '3':
                return 'big'
            else:
                print("❌ Invalid choice! Please enter 1, 2, or 3.")
                continue
    
    def _get_logging_preference(self) -> bool:
        """
        Get user's logging preference with validation.
        
        Returns:
            True if user wants log files, False otherwise
        """
        while True:
            print("\n📝 LOGGING OPTIONS")
            print("Select output mode:")
            print("1. Steps to file, summary to terminal (recommended)")
            print("2. Summary only (terminal and file)")
            
            choice = input("\nEnter your choice (1/2): ").strip()
            
            if choice == '1':
                self.output_mode = 'steps'
                return True
            elif choice == '2':
                self.output_mode = 'summary'
                return True
            else:
                print("❌ Invalid choice! Please enter 1 or 2.")
                continue
    
    def _get_optimization_parameters(self) -> Dict:
        """
        Get optimization parameters with validation.
        
        Returns:
            Dictionary containing optimization parameters
        """
        print("\n⚙️ OPTIMIZATION PARAMETERS")
        print("You can use default parameters or customize them.")
        
        while True:
            choice = input("Use default parameters? (y/n): ").strip().lower()
            
            if choice in ['y', 'yes']:
                return self._get_default_parameters()
            elif choice in ['n', 'no']:
                return self._get_custom_parameters()
            else:
                print("❌ Invalid choice! Please enter 'y' or 'n'.")
                continue
    
    def _get_default_parameters(self) -> Dict:
        """
        Get default optimization parameters.
        
        Returns:
            Dictionary with default parameters
        """
        print("Using default parameters:")
        print("- Sailfish population: 3")
        print("- Sardine population: 5")
        print("- Max capacity: 170")
        print("- Max vehicles: 2")
        print("- Max iterations: 10")
        print("- Algorithm parameter A: 4 (constant)")
        print("- Convergence epsilon: 0.001 (constant)")
        
        return {
            'n_sailfish': 3,
            'n_sardines': 5,
            'max_capacity': 170,
            'max_vehicles': 2,
            'max_iter': 10,
            'A': 4,
            'epsilon': 0.001
        }
    
    def _get_custom_parameters(self) -> Dict:
        """
        Get custom optimization parameters from user.
        
        Returns:
            Dictionary with custom parameters
        """
        print("\n🔧 CUSTOM PARAMETERS")
        print("Enter custom values for optimization parameters:")
        
        params = {}
        
        # Sailfish population (positive integer)
        params['n_sailfish'] = self._get_positive_int_input(
            "Number of sailfish (positive integer): ", 3
        )
        
        # Sardine population (positive integer and > sailfish)
        while True:
            sardines = self._get_positive_int_input(
                f"Number of sardines (must be > {params['n_sailfish']}): ", params['n_sailfish'] + 2
            )
            if sardines > params['n_sailfish']:
                params['n_sardines'] = sardines
                break
            else:
                print(f"❌ Number of sardines must be greater than number of sailfish ({params['n_sailfish']}).")
        
        # Max capacity and vehicles with demand validation
        params['max_capacity'], params['max_vehicles'] = self._get_capacity_and_vehicles_with_validation()
        
        # Max iterations (positive integer)
        params['max_iter'] = self._get_positive_int_input(
            "Maximum iterations (positive integer): ", 10
        )
        
        # Algorithm parameter A (constant value)
        params['A'] = 4.0
        print(f"Algorithm parameter A: {params['A']} (constant)")
        
        # Convergence epsilon (constant value)
        params['epsilon'] = 0.001
        print(f"Convergence epsilon: {params['epsilon']} (constant)")
        
        return params
    
    def _get_positive_int_input(self, prompt: str, default: int) -> int:
        """
        Get positive integer input from user.
        
        Args:
            prompt: Input prompt message
            default: Default value if user skips
            
        Returns:
            Positive integer value
        """
        while True:
            try:
                value = input(prompt).strip()
                if value == "":
                    print(f"Using default value: {default}")
                    return int(default)
                int_value = int(value)
                if int_value > 0:
                    return int_value
                else:
                    print("❌ Please enter a positive integer.")
            except ValueError:
                print("❌ Please enter a valid integer.")
    
    def _get_positive_float_input(self, prompt: str, default: float) -> float:
        """
        Get positive float input from user.
        
        Args:
            prompt: Input prompt message
            default: Default value if user skips
            
        Returns:
            Positive float value
        """
        while True:
            try:
                value = input(prompt).strip()
                if value == "":
                    print(f"Using default value: {default}")
                    return float(default)
                float_value = float(value)
                if float_value > 0:
                    return float_value
                else:
                    print("❌ Please enter a positive number.")
            except ValueError:
                print("❌ Please enter a valid number.")
    
    def _get_capacity_and_vehicles_with_validation(self) -> Tuple[float, int]:
        """
        Get max capacity and max vehicles with demand validation.
        
        Returns:
            Tuple of (max_capacity, max_vehicles)
        """
        from src.data_loader import VRPDataLoader
        
        # Get data file from the current configuration
        data_file = getattr(self, 'current_data_file', 'kecil.csv')
        
        # Load data to get total demand
        try:
            depot_data, customers_data = VRPDataLoader.read_vrp_data_from_csv(data_file)
            total_demand = sum(c.get('demand', 0) for c in customers_data)
            max_customer_demand = max(c.get('demand', 0) for c in customers_data)
            
            print(f"\n📊 DEMAND INFORMATION")
            print(f"Total demand: {total_demand}, the value Maximum vehicle capacity * Maximum number of vehicles must be greater than or equal to the total demand")
            print(f"Maximum single customer demand: {max_customer_demand}")
            print(f"Number of customers: {len(customers_data)}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not load data for validation: {e}")
            total_demand = 0
            max_customer_demand = 0
        
        while True:
            # Get max capacity
            max_capacity = self._get_positive_float_input(
                "Maximum vehicle capacity (positive number): ", 170.0
            )
            
            # Get max vehicles
            max_vehicles = self._get_positive_int_input(
                "Maximum number of vehicles (positive integer): ", 2
            )
            
            # Validate against demand
            if total_demand > 0:
                total_capacity = max_capacity * max_vehicles
                if total_capacity < total_demand:
                    print(f"\n❌ INSUFFICIENT CAPACITY!")
                    print(f"Total capacity ({total_capacity}) is less than total demand ({total_demand})")
                    print(f"Please increase capacity or number of vehicles.")
                    print(f"Minimum required: {max_capacity} * {max_vehicles} >= {total_demand}")
                    continue
                
                if max_capacity < max_customer_demand:
                    print(f"\n❌ CAPACITY TOO SMALL!")
                    print(f"Vehicle capacity ({max_capacity}) is less than maximum customer demand ({max_customer_demand})")
                    print(f"Please increase capacity to at least {max_customer_demand}")
                    continue
            
            return max_capacity, max_vehicles
    
    def display_configuration_summary(self, config: Dict) -> None:
        """
        Display a summary of the selected configuration.
        
        Args:
            config: Configuration dictionary
        """
        print("\n" + "="*80)
        print("📋 CONFIGURATION SUMMARY")
        print("="*80)
        print(f"Data Size: {config['data_size'].title()} ({config['data_file']})")
        print(f"Logging: {'Enabled' if config['log_to_file'] else 'Disabled'}")
        print(f"Sailfish Population: {config['n_sailfish']}")
        print(f"Sardine Population: {config['n_sardines']}")
        print(f"Max Capacity: {config['max_capacity']}")
        print(f"Max Vehicles: {config['max_vehicles']}")
        print(f"Max Iterations: {config['max_iter']}")
        print(f"Algorithm Parameter A: {config['A']}")
        print(f"Convergence Epsilon: {config['epsilon']}")
        print("="*80)
    
    def confirm_start(self) -> bool:
        """
        Ask user to confirm starting the optimization.
        
        Returns:
            True if user confirms, False otherwise
        """
        while True:
            choice = input("\n🚀 Start optimization with these settings? (y/n): ").strip().lower()
            
            if choice in ['y', 'yes']:
                return True
            elif choice in ['n', 'no']:
                return False
            else:
                print("❌ Invalid choice! Please enter 'y' or 'n'.")
    
    def display_completion_message(self) -> None:
        """Display completion message."""
        print("\n" + "="*80)
        print("✅ OPTIMIZATION COMPLETED!")
        print("="*80)
        print("Thank you for using the Sailfish VRP Optimization System!")
        print("Check the generated log files for detailed results.")
        print("="*80)
    
    def check_data_files(self) -> bool:
        """
        Check if required data files exist.
        
        Returns:
            True if all files exist, False otherwise
        """
        missing_files = []
        
        for size, filename in self.available_data_files.items():
            if not os.path.exists(filename):
                missing_files.append(f"{size} ({filename})")
        
        if missing_files:
            print("\n⚠️ WARNING: Some data files are missing:")
            for file_info in missing_files:
                print(f"   - {file_info}")
            print("\nThe system will create sample data files if needed.")
            return False
        
        return True

