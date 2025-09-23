# Sailfish VRP Optimization System

A comprehensive Vehicle Routing Problem (VRP) optimization system that implements the Sailfish Optimizer algorithm. This system is designed with a modular architecture for better maintainability and extensibility.

## 🏗️ System Architecture

The system follows a modular design pattern with clear separation of concerns:

```
main.py (Entry Point)
    ↓
src/controller.py (User Interface)
    ↓
src/sailfish_optimizer_v2.py (Main Orchestrator)
    ↓
src/iteration_runner.py (Iteration Management)
    ↓
src/population_manager.py (Population Management)
src/position_updater.py (Position Updates)
src/replacement_manager.py (Population Replacement)
src/results_reporter.py (Results Display)
src/data_loader.py (Data Management)
src/utils.py (Utility Functions)
src/logger.py (Logging System)
```

## 📁 File Structure and Functionality

### 1. **main.py** - Application Entry Point

**Purpose**: Main application entry point that orchestrates the entire optimization process.

**Key Functions**:
- `run_vrp_optimization(config)`: Main optimization function that loads data, creates optimizer, and runs optimization
- `print_results_summary(results)`: Displays optimization results in a formatted way
- `main()`: Entry point that initializes controller and runs the complete workflow

**Data Flow**:
1. Loads VRP data from CSV files
2. Validates data integrity
3. Creates SailfishVRPOptimizerV2 instance
4. Runs optimization
5. Displays results

### 2. **src/controller.py** - User Interface Controller

**Purpose**: Handles all user interactions, input validation, and system configuration.

**Key Functions**:
- `display_welcome_message()`: Shows welcome screen and system information
- `get_user_preferences()`: Collects user configuration preferences
- `_get_data_size_preference()`: Handles data size selection (small/medium/big)
- `_get_optimization_parameters()`: Collects algorithm parameters (population sizes, iterations, etc.)
- `confirm_start()`: Gets user confirmation before starting optimization

**User Interaction Flow**:
1. Welcome message display
2. Data size selection (18, 75, or 100 customers)
3. Logging preference selection
4. Optimization parameter input
5. Configuration confirmation
6. Start confirmation

### 3. **src/sailfish_optimizer_v2.py** - Main Optimization Orchestrator

**Purpose**: Central orchestrator that coordinates all optimization modules and manages the main optimization loop.

**Key Functions**:
- `__init__()`: Initializes all component modules (PopulationManager, PositionUpdater, etc.)
- `run_optimization()`: Main optimization loop that runs iterations until stopping conditions are met
- `get_population_summary()`: Returns current population state
- `get_best_solution()`: Returns the best solution found

**Stopping Conditions**:
- **Sardine population empty**: `n_sardines == 0`
- **Maximum iterations reached**: `iteration >= max_iter`

**Optimization Flow**:
1. Run iteration zero (initial setup)
2. Run subsequent iterations until stopping conditions
3. Each iteration calls `iteration_runner.run_iteration()`
4. Collect and return final results

### 4. **src/iteration_runner.py** - Iteration Management

**Purpose**: Manages individual iterations of the optimization process, including both iteration zero and subsequent iterations.

**Key Functions**:
- `run_iteration_zero()`: Handles the special initial iteration (iteration 0)
- `run_iteration()`: Executes a single optimization iteration
- `get_final_results()`: Compiles final optimization statistics

**Iteration Zero Flow**:
1. Save original positions
2. Convert populations to routes
3. Calculate initial fitness
4. Update elite positions
5. Calculate PD and lambda values
6. Update sailfish positions
7. Calculate AP and update sardines

**Regular Iteration Flow**:
1. Save original positions and clear replacement tracking
2. Convert to routes and calculate fitness
3. Perform sailfish-sardine replacement
4. Print comprehensive results
5. Calculate PD and lambda values
6. Update sailfish positions (using replacement-specific positions)
7. Calculate AP and update sardines
8. Record fitness history

### 5. **src/population_manager.py** - Population Management

**Purpose**: Manages sailfish and sardine populations, their storage, fitness calculations, and population operations.

**Key Functions**:
- `initialize_populations()`: Creates initial random populations
- `save_original_positions()`: Stores original positions before updates
- `convert_populations_to_routes()`: Converts random values to actual VRP routes
- `calculate_population_fitness()`: Computes fitness for all individuals
- `update_elite_positions()`: Updates best sailfish and sardine positions
- `get_population_summary()`: Returns current population statistics

**Population Storage**:
- **Sailfish**: `sailfish_random_values`, `sailfish_routes`, `sailfish_fitness`
- **Sardines**: `sardine_random_values`, `sardine_routes`, `sardine_fitness`
- **Original positions**: `original_sailfish_positions`, `original_sardine_positions`
- **Replacement tracking**: `sailfish_replacement_map`, `sardine_positions_before_removal`

**Key Operations**:
1. **Initialization**: Creates random populations with proper constraints
2. **Route Conversion**: Transforms random values into valid VRP routes
3. **Fitness Calculation**: Computes total distance for each route
4. **Elite Tracking**: Maintains best solutions for position updates
5. **Replacement Support**: Tracks which sardines replaced which sailfish

### 6. **src/position_updater.py** - Position Update Engine

**Purpose**: Handles all position updates for sailfish and sardines using the Sailfish Optimizer formulas.

**Key Functions**:
- `update_sailfish_positions()`: Updates sailfish positions using lambda values
- `update_all_sardines()`: Updates all sardine positions when AP >= 0.5
- `update_partial_sardines()`: Updates partial sardine positions when AP < 0.5
- `calculate_pd_and_lambda_values()`: Computes PD (Population Decline) and lambda values
- `calculate_attack_power()`: Calculates Attack Power (AP) for sardine updates
- `_get_replacement_positions()`: Gets appropriate positions for sailfish updates after replacement

**Update Formulas**:

**Sailfish Update**:
```
SF[i] = original_position[i] + lambda[i] × (elite_sailfish_position - original_position[i])
```

**Sardine Update (AP ≥ 0.5)**:
```
S[i] = random × (elite_sailfish_fitness - old_sardine + AP)
```

**Sardine Update (AP < 0.5)**:
```
S[i] = random × (elite_sailfish_position - old_sardine + AP)
```

**Key Features**:
1. **Replacement-Aware Updates**: Uses sardine positions for replaced sailfish
2. **Adaptive Updates**: Different update strategies based on Attack Power
3. **Position Tracking**: Maintains original positions for proper updates

### 7. **src/replacement_manager.py** - Population Replacement Logic

**Purpose**: Manages the replacement of sailfish with better sardines during optimization.

**Key Functions**:
- `perform_sailfish_sardine_replacement()`: Main replacement logic
- `_update_elite_positions()`: Updates elite positions after replacement
- `get_replacement_statistics()`: Returns replacement statistics
- `reset_history()`: Clears replacement history

**Replacement Logic**:
1. **Analysis**: Identifies sardines better than worst sailfish
2. **Sorting**: Orders better sardines by fitness (ascending)
3. **Replacement**: Replaces worst sailfish with best sardines
4. **Position Storage**: Stores sardine positions before removal for later use
5. **Population Update**: Removes replaced sardines and updates counts
6. **Elite Update**: Updates best sailfish and sardine positions

**Key Features**:
- **Position Preservation**: Stores sardine positions before removal
- **Replacement Tracking**: Maps which sardines replaced which sailfish
- **Statistics**: Maintains detailed replacement history
- **Elite Management**: Updates elite positions after replacements

### 8. **src/results_reporter.py** - Results Display and Reporting

**Purpose**: Handles all output formatting, results display, and comprehensive reporting.

**Key Functions**:
- `print_routes_and_solutions()`: Displays detailed route information
- `print_fitness_summary()`: Shows fitness statistics
- `print_comprehensive_results_table()`: Comprehensive results display
- `print_final_results()`: Final optimization summary
- `print_algorithm_parameters()`: Algorithm parameter display

**Output Formats**:
1. **Route Display**: Shows customer sequences and distances
2. **Fitness Summary**: Population fitness statistics
3. **Comprehensive Table**: Detailed iteration results
4. **Final Results**: Complete optimization summary

### 9. **src/data_loader.py** - Data Management

**Purpose**: Handles VRP data loading, validation, and sample data creation.

**Key Functions**:
- `read_vrp_data_from_csv()`: Loads VRP data from CSV files
- `validate_vrp_data()`: Validates data integrity and constraints
- `create_all_sample_files()`: Creates sample data files for testing
- `create_sample_file()`: Creates individual sample data files

**Data Structure**:
- **Depot Data**: Location coordinates and capacity
- **Customer Data**: Location coordinates, demand, and service time
- **Validation**: Checks capacity constraints and data completeness

### 10. **src/utils.py** - Utility Functions

**Purpose**: Provides common utility functions used across the system.

**Key Functions**:
- `generate_random_values()`: Creates random position values
- `convert_random_to_route()`: Converts random values to VRP routes
- `calculate_vrp_fitness()`: Computes route fitness (total distance)
- `format_route_string()`: Formats routes for display
- `calculate_route_demand()`: Calculates total demand for a route

**Utility Operations**:
1. **Random Generation**: Creates random values within constraints
2. **Route Conversion**: Transforms random values to valid routes
3. **Fitness Calculation**: Computes optimization objective
4. **Formatting**: Provides consistent output formatting

### 11. **src/logger.py** - Logging System

**Purpose**: Manages logging configuration and output file management.

**Key Functions**:
- `setup_output_logger()`: Configures file logging
- `cleanup()`: Cleans up logging resources
- `get_logger()`: Returns configured logger instance

## 🔄 Algorithm Flow

### 1. **Initialization Phase**
```
User Input → Data Loading → Population Creation → Initial Fitness Calculation
```

### 2. **Iteration Zero (Special Case)**
```
Save Positions → Convert to Routes → Calculate Fitness → Update Elite Positions → 
Calculate PD/Lambda → Update Sailfish → Calculate AP → Update Sardines
```

### 3. **Regular Iterations**
```
Save Positions → Convert to Routes → Calculate Fitness → Perform Replacement → 
Calculate PD/Lambda → Update Sailfish (with replacement positions) → 
Calculate AP → Update Sardines → Record History
```

### 4. **Stopping Conditions**
- **Sardine Population Empty**: All sardines have been replaced
- **Maximum Iterations**: Reached the user-specified iteration limit

## 🎯 Key Features

1. **Modular Architecture**: Clean separation of concerns for maintainability
2. **Flexible Stopping**: Configurable stopping conditions
3. **Comprehensive Logging**: Detailed logging for analysis and debugging
4. **User-Friendly Interface**: Interactive terminal interface
5. **Data Validation**: Robust data validation and error handling
6. **Performance Tracking**: Detailed fitness evolution tracking
7. **Replacement Logic**: Sophisticated population replacement mechanism
8. **Position Management**: Intelligent position tracking for updates

## 🚀 Usage

1. **Run the system**: `python main.py`
2. **Select data size**: Choose from small (18), medium (75), or big (100) customer datasets
3. **Configure parameters**: Set population sizes, iterations, and algorithm parameters
4. **Start optimization**: Confirm and begin the optimization process
5. **Monitor progress**: Watch real-time optimization progress
6. **Review results**: Analyze final routes and optimization statistics

## 📊 Output

The system provides comprehensive output including:
- **Route Details**: Customer sequences and distances
- **Fitness Evolution**: Optimization progress over iterations
- **Population Statistics**: Current population state
- **Replacement History**: Detailed replacement tracking
- **Final Results**: Complete optimization summary
- **Log Files**: Detailed logs for analysis (optional)

This modular design ensures that each component has a single responsibility, making the system easy to understand, maintain, and extend.
