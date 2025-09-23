"""
Main application for Sailfish VRP Optimizer.
Demonstrates the modularized VRP optimization system with interactive user interface.
"""

import logging
from datetime import datetime
from typing import Dict

from src.controller import VRPController
from src.data_loader import VRPDataLoader
from src.sailfish_optimizer_v2 import SailfishVRPOptimizerV2
from src.utils import print_vrp_data, calculate_vrp_fitness, format_route_string, calculate_route_demand

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_vrp_optimization(config: Dict) -> Dict:
    """
    Run VRP optimization with given configuration.
    
    Args:
        config: Configuration dictionary containing optimization parameters
        
    Returns:
        Dictionary containing optimization results
    """
    logger.info("="*80)
    logger.info("SAILFISH VRP OPTIMIZATION SYSTEM")
    logger.info("="*80)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load VRP data
    logger.info("\nLoading VRP data...")
    depot_data, customers_data = VRPDataLoader.read_vrp_data_from_csv(
        config.get('data_file', 'kecil.csv')
    )
    
    # Validate data with user-provided parameters
    if not VRPDataLoader.validate_vrp_data(
        depot_data, customers_data,
        config.get('max_capacity'),
        config.get('max_vehicles')
    ):
        logger.error("VRP data validation failed!")
        return {}
    
    # Print problem data
    print_vrp_data(depot_data, customers_data, 
                   config.get('max_capacity'), 
                   config.get('max_vehicles'))
    
    # Create optimizer
    output_mode = config.get('output_mode', 'steps')
    optimizer = SailfishVRPOptimizerV2(
        n_sailfish=config.get('n_sailfish', 3),
        n_sardines=config.get('n_sardines', 5),
        depot_data=depot_data,
        customers_data=customers_data,
        max_capacity=config.get('max_capacity'),
        max_vehicles=config.get('max_vehicles'),
        max_iter=config.get('max_iter', 10),
        A=config.get('A', 4),
        epsilon=config.get('epsilon', 0.001),
        log_to_file=config.get('log_to_file', True),
        output_mode=config.get('output_mode', 'steps'),
        data_file=config.get('data_file', 'kecil.csv')
    )
    
    # Run optimization
    logger.info("\nStarting optimization...")
    results = optimizer.run_optimization()
    
    logger.info(f"\nOptimization completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return results


def print_results_summary(results: Dict) -> None:
    """
    Print a summary of optimization results.
    
    Args:
        results: Dictionary containing optimization results
    """
    if not results:
        logger.error("No results to display")
        return
    
    logger.info("\n" + "="*80)
    logger.info("OPTIMIZATION RESULTS SUMMARY")
    logger.info("="*80)
    
    # Algorithm parameters
    logger.info("Algorithm Parameters:")
    for key, value in results['algorithm_parameters'].items():
        logger.info(f"  {key.replace('_', ' ').title()}: {value}")
    
    # VRP parameters
    logger.info("\nVRP Parameters:")
    for key, value in results['vrp_parameters'].items():
        logger.info(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Best solution
    logger.info("\nBest Solution:")
    logger.info(f"  Total Distance: {results['best_solution']['total_distance']:.3f}")
    logger.info(f"  Routes: {results['best_solution']['routes']}")
    
    # Fitness evolution
    logger.info("\nFitness Evolution:")
    logger.info(f"  Initial Fitness: {results['fitness_evolution']['initial']:.3f}")
    logger.info(f"  Final Fitness: {results['fitness_evolution']['final']:.3f}")
    logger.info(f"  Improvement: {results['fitness_evolution']['improvement']:.3f}")
    
    logger.info("="*80)
    
    # Print detailed route analysis if available
    if 'best_solution' in results and 'routes' in results['best_solution']:
        from src.results_reporter import ResultsReporter
        from src.data_loader import VRPDataLoader
        
        # Load data for detailed analysis
        depot_data, customers_data = VRPDataLoader.read_vrp_data_from_csv(
            results['vrp_parameters'].get('data_file', 'kecil.csv')
        )
        
        # Create results reporter and print detailed analysis
        reporter = ResultsReporter()
        reporter._print_detailed_route_analysis(
            results['best_solution']['routes'],
            depot_data,
            customers_data,
            results['vrp_parameters']['max_capacity'],
            results['vrp_parameters']['max_vehicles']
        )


def main():
    """Main function to run the VRP optimization system."""
    
    # Initialize controller
    controller = VRPController()
    
    # Display welcome message
    controller.display_welcome_message()
    
    # Check data files and create if needed
    if not controller.check_data_files():
        print("\n📁 Creating sample data files...")
        VRPDataLoader.create_all_sample_files()
    
    # Get user preferences
    config = controller.get_user_preferences()
    
    # Display configuration summary
    controller.display_configuration_summary(config)
    
    # Confirm start
    if not controller.confirm_start():
        print("\n❌ Optimization cancelled by user.")
        return
    
    # Run optimization
    print("\n🚀 Starting optimization...")
    results = run_vrp_optimization(config)
    
    # Print summary
    print_results_summary(results)
    
    # Display completion message
    controller.display_completion_message()


if __name__ == "__main__":
    main()
