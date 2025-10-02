"""
Logging module for VRP optimization.
Handles dual output to both console and file with proper formatting.
"""

import sys
import logging
from datetime import datetime
from typing import Optional
import os


class OutputLogger:
    """Class to handle dual output to both console and file."""
    
    def __init__(self, filename: str = "vrp_output.txt"):
        """
        Initialize the output logger.
        
        Args:
            filename: Name of the log file
        """
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
        
        # Write header to log file
        self.log.write(f"Sailfish VRP Optimizer Output Log\n")
        self.log.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write(f"{'='*80}\n\n")
        self.log.flush()
    
    def write(self, message: str) -> None:
        """
        Write message to both terminal and log file.
        
        Args:
            message: Message to write
        """
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self) -> None:
        """Flush both terminal and log file."""
        self.terminal.flush()
        self.log.flush()
    
    def close(self) -> None:
        """Close the log file."""
        if self.log:
            self.log.close()


class VRPLogger:
    """Enhanced logger for VRP optimization with structured logging."""
    
    def __init__(self, log_to_file: bool = True, log_level: str = "INFO", output_mode: str = "steps", filename: Optional[str] = None):
        """
        Initialize VRP logger.
        
        Args:
            log_to_file: Whether to log to file
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.log_to_file = log_to_file
        self.output_mode = output_mode  # 'steps' or 'summary'
        self.output_logger = None
        self.original_stdout = sys.stdout
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(message)s',
            handlers=[]
        )
        
        # Add console handler (show only terminal-specific output)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.ERROR)  # Only show errors on terminal
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        
        # Add file handler if requested
        if log_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"vrp_optimization_{timestamp}.log"
            selected_filename = filename or default_filename
            file_handler = logging.FileHandler(selected_filename, encoding='utf-8')
            # Steps are logged at DEBUG; summary at INFO. File level depends on output_mode
            file_level = logging.DEBUG if (self.output_mode == "steps") else logging.INFO
            file_handler.setLevel(file_level)
            file_formatter = logging.Formatter('%(message)s')
            file_handler.setFormatter(file_formatter)
            
            # Configure root logger
            root_logger = logging.getLogger()
            root_logger.handlers.clear()  # Remove default handlers
            root_logger.addHandler(console_handler)
            root_logger.addHandler(file_handler)
            root_logger.setLevel(getattr(logging, log_level.upper()))
            
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"Logging to file: {selected_filename} ({self.output_mode})")
        else:
            # Configure root logger for console only
            root_logger = logging.getLogger()
            root_logger.handlers.clear()
            root_logger.addHandler(console_handler)
            root_logger.setLevel(getattr(logging, log_level.upper()))
            
            self.logger = logging.getLogger(__name__)
    
    def setup_output_logger(self, filename: str) -> None:
        """Deprecated: no longer redirect stdout; logging handlers manage output."""
        # Kept for backward compatibility; no-op now
        if self.log_to_file:
            self.logger.info(f"Output file set: {filename}")
    
    def log_optimization_start(self, parameters: dict) -> None:
        """
        Log optimization start with parameters.
        
        Args:
            parameters: Dictionary of optimization parameters
        """
        self.logger.info("="*80)
        self.logger.info("SAILFISH VRP OPTIMIZATION STARTED")
        self.logger.info("="*80)
        self.logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("Parameters:")
        for key, value in parameters.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("="*80)
    
    def log_iteration_start(self, iteration: int) -> None:
        """
        Log iteration start.
        
        Args:
            iteration: Iteration number
        """
        self.logger.info("")
        self.logger.info("="*100)
        self.logger.info(f"STARTING ITERATION {iteration}")
        self.logger.info("="*100)
    
    def log_iteration_end(self, iteration: int, best_fitness: float, best_routes: list) -> None:
        """
        Log iteration end with results.
        
        Args:
            iteration: Iteration number
            best_fitness: Best fitness value found
            best_routes: Best routes found
        """
        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info(f"ITERATION {iteration} COMPLETED")
        self.logger.info("="*80)
        self.logger.info(f"Best fitness: {best_fitness:.3f}")
        self.logger.info(f"Best routes: {best_routes}")
    
    def log_optimization_end(self, final_results: dict) -> None:
        """
        Log optimization end with final results.
        
        Args:
            final_results: Dictionary containing final optimization results
        """
        self.logger.info("")
        self.logger.info("="*100)
        self.logger.info("FINAL VRP OPTIMIZATION RESULTS")
        self.logger.info("="*100)
        
        for key, value in final_results.items():
            if isinstance(value, float):
                self.logger.info(f"{key}: {value:.3f}")
            else:
                self.logger.info(f"{key}: {value}")
        
        self.logger.info("")
        self.logger.info("="*100)
        self.logger.info("VRP OPTIMIZATION COMPLETED!")
        self.logger.info("="*100)
        self.logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def cleanup(self) -> None:
        """Cleanup logging resources."""
        if self.output_logger:
            sys.stdout = self.original_stdout
            self.output_logger.close()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
