"""
Itinerary Planning System - Main Module

This is the main entry point for the Itinerary Planning System.
It orchestrates the workflow for generating optimized itineraries.
"""

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
import os
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import components
# Data preprocessing
from data.attraction_processor import AttractionProcessor
from data.data_validator import DataValidator

# City allocation
from algorithms.city_allocation import CityAllocator

# Attraction grouping
from algorithms.attraction_grouper import AttractionGrouper

# Route optimization
from algorithms.route_optimizer import RouteOptimizer

# City transition
from algorithms.city_transition import CityTransitionHandler

# Itinerary building
from algorithms.itinerary_builder import ItineraryBuilder

# Transformer model
from models.transformer_handler import TransformerHandler

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Itinerary Planning System")
    
    parser.add_argument(
        "--input", "-i", 
        default="data/raw/fixed_cities.json",
        help="Path to input JSON file (default: data/raw/fixed_cities.json)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="itinerary.json",
        help="Path to output JSON file (default: itinerary.json)"
    )
    
    parser.add_argument(
        "--days", "-d",
        type=int,
        help="Override total days in itinerary"
    )
    
    parser.add_argument(
        "--preferences", "-p",
        nargs="+",
        help="User preferences (e.g., 'art history food')"
    )
    
    parser.add_argument(
        "--pace", 
        choices=["slow", "moderate", "fast"],
        default="moderate",
        help="Travel pace (default: moderate)"
    )
    
    parser.add_argument(
        "--embeddings", "-e",
        default="data/embeddings/attraction_embeddings.npz",
        help="Path to pre-generated attraction embeddings"
    )
    
    parser.add_argument(
        "--use-transformer",
        action="store_true",
        help="Use transformer model for semantic understanding"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate the input data, don't generate itinerary"
    )
    
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation of input data"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()

def main():
    """Main function to run the Itinerary Planning System."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting Itinerary Planning System")
    
    try:
        # Load data from specified input file (default: fixed_cities.json)
        input_path = args.input
        logger.info(f"Loading data from {input_path}")
        
        # Validate the input data if not skipped
        if not args.skip_validation:
            validator = DataValidator()
            valid, errors = validator.validate_file(input_path)
            
            if not valid:
                logger.error(f"Data validation failed with {len(errors)} errors:")
                for error in errors:
                    logger.error(f"  - {error}")
                return 1
            
            logger.info("Data validation passed")
            
            # If validate-only flag is set, exit here
            if args.validate_only:
                return 0
        
        # Load the data
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Process the data
        processor = AttractionProcessor()
        processed_data = processor.process_data(raw_data)
        
        # Extract parameters from processed data
        parameters = processed_data.get("parameters", {})
        
        # Override parameters with command-line arguments if provided
        if args.days:
            parameters["total_days"] = args.days
        if args.preferences:
            parameters["preferences"] = args.preferences
        if args.pace:
            parameters["travel_pace"] = args.pace
        
        if "total_days" not in parameters:
            parameters["total_days"] = 5  # Default to 5 days if not specified
        
        logger.info(f"Planning a {parameters['total_days']}-day itinerary")
        
        if "preferences" in parameters:
            logger.info(f"Preferences: {', '.join(parameters['preferences'])}")
        
        # IMPORTANT FIX: Limit the number of cities to match the requested total days
        if parameters["total_days"] < len(processed_data["cities"]):
            logger.info(f"Limiting to top {parameters['total_days']} cities (out of {len(processed_data['cities'])})")
            
            # Sort cities by number of attractions (simple importance metric)
            processed_data["cities"].sort(key=lambda c: len(c.get("attractions", [])), reverse=True)
            
            # Keep only the top N cities based on total_days
            processed_data["cities"] = processed_data["cities"][:parameters["total_days"]]
        
        # Initialize the transformer handler
        transformer = None
        if args.use_transformer:
            logger.info("Initializing transformer model")
            transformer = TransformerHandler()
            
            # Load pre-generated embeddings if available
            if Path(args.embeddings).exists():
                logger.info(f"Loading pre-generated embeddings from {args.embeddings}")
                success = transformer.load_embeddings(args.embeddings)
                if not success:
                    logger.warning("Failed to load embeddings, generating new ones")
                    transformer.generate_embeddings(processed_data["cities"])
                    
                    # Save embeddings for future use
                    embeddings_dir = Path(args.embeddings).parent
                    embeddings_dir.mkdir(parents=True, exist_ok=True)
                    transformer.save_embeddings(args.embeddings)
            else:
                logger.warning(f"Embeddings file not found: {args.embeddings}")
                logger.info("Generating new embeddings (this may take a while)")
                transformer.generate_embeddings(processed_data["cities"])
                
                # Save embeddings for future use
                embeddings_dir = Path(args.embeddings).parent
                embeddings_dir.mkdir(parents=True, exist_ok=True)
                transformer.save_embeddings(args.embeddings)
        
        # Step 1: Allocate days across cities
        logger.info("Allocating days across cities")
        allocator = CityAllocator()
        allocation = allocator.allocate_days(
            cities=processed_data["cities"],
            total_days=parameters["total_days"],
            preferences=parameters.get("preferences", []),
            travel_pace=parameters.get("travel_pace", "moderate")
        )
        
        logger.info(f"City sequence: {allocation['city_sequence']}")
        logger.info(f"Days per city: {allocation['days_per_city']}")
        
        # Step 2: Group attractions for each city
        logger.info("Grouping attractions")
        attraction_grouper = AttractionGrouper(transformer_handler=transformer)
        attraction_groups = attraction_grouper.group_attractions(
            cities=processed_data["cities"],
            days_per_city=allocation["days_per_city"],
            preferences=parameters.get("preferences", [])
        )
        
        # Step 3: Optimize routes for each day
        logger.info("Optimizing daily routes")
        route_optimizer = RouteOptimizer()
        optimized_routes = route_optimizer.optimize_daily_routes(attraction_groups)
        
        # Step 4: Handle city transitions
        logger.info("Planning city transitions")
        transition_handler = CityTransitionHandler()
        travel_plan = transition_handler.plan_city_transitions(
            city_sequence=allocation["city_sequence"],
            travel_days=allocation["travel_days"],
            optimized_routes=optimized_routes,
            cities_data=processed_data["cities"]
        )
        
        # Step 5: Build final itinerary
        logger.info("Building final itinerary")
        itinerary_builder = ItineraryBuilder()
        itinerary = itinerary_builder.build_itinerary(
            cities=processed_data["cities"],
            optimized_routes=optimized_routes,
            travel_plan=travel_plan,
            preferences=parameters.get("preferences", [])
        )
        
        # Save itinerary
        output_path = Path(args.output)
        itinerary_builder.save_itinerary(itinerary, output_path)
        
        logger.info(f"Itinerary saved to {output_path}")
        logger.info("Itinerary generation completed successfully")
        
        # Print summary
        print("\nItinerary Summary:")
        print(f"Total Days: {itinerary['meta']['total_days']}")
        print(f"Cities Visited: {len(allocation['city_sequence'])}")
        print(f"Total Attractions: {sum(len(day.get('attractions', [])) for day in itinerary['itinerary'])}")
        print(f"Output file: {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error generating itinerary: {e}")
        if args.debug:
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())