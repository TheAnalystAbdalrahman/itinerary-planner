"""
Data Validator Module

This module validates input data for the Itinerary Planning System,
ensuring that all required fields are present and properly formatted.
"""

import json
from typing import Dict, List, Any, Optional, Tuple, Set
import logging
from pathlib import Path
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataValidator:
    """
    Validates data inputs for the itinerary planning system.
    Ensures data completeness, consistency, and correctness.
    """
    
    def __init__(self):
        """Initialize the data validator."""
        # Required fields for each entity type
        self.required_city_fields = {"city_id", "country"}
        self.required_attraction_fields = {
            "attraction_id", "name", "category", "description", "visit_duration_hrs"
        }
        self.required_parameters = {"total_days"}
        
        # Valid ranges for numeric fields
        self.valid_ranges = {
            "visit_duration_hrs": (0.25, 8.0),  # Between 15 minutes and 8 hours
            "total_days": (1, 14),              # Between 1 and 14 days
            "latitude": (-90.0, 90.0),
            "longitude": (-180.0, 180.0)
        }
        
    def validate_data(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate the entire dataset for completeness and consistency.
        
        Args:
            data: The data dictionary to validate
            
        Returns:
            Tuple containing:
                - Boolean indicating if validation passed
                - List of validation error messages (empty if validation passed)
        """
        errors = []
        
        # Validate overall structure
        if "cities" not in data:
            errors.append("Missing 'cities' key in input data")
            return False, errors
            
        if "parameters" not in data:
            errors.append("Missing 'parameters' key in input data")
            return False, errors
            
        # Validate parameters
        param_valid, param_errors = self._validate_parameters(data.get("parameters", {}))
        errors.extend(param_errors)
        
        # Validate each city and its attractions
        for i, city in enumerate(data.get("cities", [])):
            city_valid, city_errors = self._validate_city(city, i)
            errors.extend(city_errors)
            
            # Validate attractions in this city
            for j, attraction in enumerate(city.get("attractions", [])):
                attr_valid, attr_errors = self._validate_attraction(attraction, i, j)
                errors.extend(attr_errors)
                
        # Cross-validate across cities
        cross_valid, cross_errors = self._cross_validate_cities(data.get("cities", []))
        errors.extend(cross_errors)
        
        # Validate that at least one city exists
        if not data.get("cities", []):
            errors.append("No cities found in input data")
            
        return len(errors) == 0, errors
    
    def _validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate the parameters section of the input data.
        
        Args:
            parameters: Parameters dictionary
            
        Returns:
            Tuple containing:
                - Boolean indicating if validation passed
                - List of validation error messages
        """
        errors = []
        
        # Check required parameters
        for field in self.required_parameters:
            if field not in parameters:
                errors.append(f"Required parameter '{field}' is missing")
                
        # Validate total_days
        if "total_days" in parameters:
            total_days = parameters["total_days"]
            try:
                total_days = int(total_days)
                min_val, max_val = self.valid_ranges["total_days"]
                if total_days < min_val or total_days > max_val:
                    errors.append(f"'total_days' must be between {min_val} and {max_val}")
            except (ValueError, TypeError):
                errors.append(f"'total_days' must be an integer, got {type(total_days)}")
                
        # Validate preferences if present
        if "preferences" in parameters:
            preferences = parameters["preferences"]
            if not isinstance(preferences, list):
                errors.append(f"'preferences' must be a list, got {type(preferences)}")
            elif not all(isinstance(p, str) for p in preferences):
                errors.append("All preferences must be strings")
                
        # Validate travel_pace if present
        if "travel_pace" in parameters:
            pace = parameters["travel_pace"]
            if not isinstance(pace, str):
                errors.append(f"'travel_pace' must be a string, got {type(pace)}")
            elif pace not in ["slow", "moderate", "fast"]:
                errors.append(f"'travel_pace' must be one of: slow, moderate, fast, got {pace}")
                
        return len(errors) == 0, errors
    
    def _validate_city(self, city: Dict[str, Any], city_index: int) -> Tuple[bool, List[str]]:
        """
        Validate a single city's data.
        
        Args:
            city: City data dictionary
            city_index: Index of the city in the cities list (for error reporting)
            
        Returns:
            Tuple containing:
                - Boolean indicating if validation passed
                - List of validation error messages
        """
        errors = []
        city_id = city.get("city_id", f"City #{city_index}")
        
        # Check required fields
        for field in self.required_city_fields:
            if field not in city:
                errors.append(f"City {city_id} is missing required field '{field}'")
                
        # Validate attractions array
        if "attractions" not in city:
            errors.append(f"City {city_id} is missing 'attractions' array")
        elif not isinstance(city["attractions"], list):
            errors.append(f"City {city_id}: 'attractions' must be a list")
        elif not city["attractions"]:
            errors.append(f"City {city_id} has no attractions")
            
        return len(errors) == 0, errors
    
    def _validate_attraction(self, attraction: Dict[str, Any], city_index: int, attr_index: int) -> Tuple[bool, List[str]]:
        """
        Validate a single attraction's data.
        
        Args:
            attraction: Attraction data dictionary
            city_index: Index of the parent city (for error reporting)
            attr_index: Index of the attraction in the city's attractions list
            
        Returns:
            Tuple containing:
                - Boolean indicating if validation passed
                - List of validation error messages
        """
        errors = []
        attr_id = attraction.get("attraction_id", f"Attraction #{attr_index}")
        city_ref = f"City #{city_index}"
        
        # Check required fields
        for field in self.required_attraction_fields:
            if field not in attraction:
                errors.append(f"{city_ref}, {attr_id} is missing required field '{field}'")
                
        # Validate numeric fields
        if "visit_duration_hrs" in attraction:
            dur = attraction["visit_duration_hrs"]
            try:
                dur_val = float(dur)
                min_val, max_val = self.valid_ranges["visit_duration_hrs"]
                if dur_val < min_val or dur_val > max_val:
                    errors.append(f"{city_ref}, {attr_id}: 'visit_duration_hrs' must be between {min_val} and {max_val}")
            except (ValueError, TypeError):
                errors.append(f"{city_ref}, {attr_id}: 'visit_duration_hrs' must be a number")
                
        # Validate coordinates if both are present
        has_lat = "latitude" in attraction
        has_lon = "longitude" in attraction
        
        if has_lat and has_lon:
            try:
                lat = float(attraction["latitude"])
                lon = float(attraction["longitude"])
                
                lat_min, lat_max = self.valid_ranges["latitude"]
                lon_min, lon_max = self.valid_ranges["longitude"]
                
                if lat < lat_min or lat > lat_max:
                    errors.append(f"{city_ref}, {attr_id}: 'latitude' must be between {lat_min} and {lat_max}")
                    
                if lon < lon_min or lon > lon_max:
                    errors.append(f"{city_ref}, {attr_id}: 'longitude' must be between {lon_min} and {lon_max}")
                    
            except (ValueError, TypeError):
                errors.append(f"{city_ref}, {attr_id}: Coordinates must be numeric values")
        elif has_lat or has_lon:
            # One coordinate is present but not the other
            errors.append(f"{city_ref}, {attr_id}: Both latitude and longitude must be provided together")
            
        # Validate description
        if "description" in attraction and len(attraction["description"].strip()) < 10:
            errors.append(f"{city_ref}, {attr_id}: Description is too short (less than 10 characters)")
            
        return len(errors) == 0, errors
    
    def _cross_validate_cities(self, cities: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate relationships between cities.
        
        Args:
            cities: List of city dictionaries
            
        Returns:
            Tuple containing:
                - Boolean indicating if validation passed
                - List of validation error messages
        """
        errors = []
        
        # Check for duplicate city IDs
        city_ids = set()
        for city in cities:
            city_id = city.get("city_id")
            if city_id:
                if city_id in city_ids:
                    errors.append(f"Duplicate city_id: '{city_id}'")
                else:
                    city_ids.add(city_id)
        
        # Check for duplicate attraction IDs across all cities
        attraction_ids = set()
        for city in cities:
            for attraction in city.get("attractions", []):
                attr_id = attraction.get("attraction_id")
                if attr_id:
                    if attr_id in attraction_ids:
                        errors.append(f"Duplicate attraction_id: '{attr_id}'")
                    else:
                        attraction_ids.add(attr_id)
                        
        return len(errors) == 0, errors
    
    def validate_file(self, filepath: str) -> Tuple[bool, List[str]]:
        """
        Validate data from a JSON file.
        
        Args:
            filepath: Path to the JSON file to validate
            
        Returns:
            Tuple containing:
                - Boolean indicating if validation passed
                - List of validation error messages
        """
        filepath = Path(filepath)
        logger.info(f"Validating data from {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            valid, errors = self.validate_data(data)
            
            if valid:
                logger.info(f"Validation successful for {filepath}")
            else:
                logger.warning(f"Validation failed for {filepath} with {len(errors)} errors")
                
            return valid, errors
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in {filepath}: {e}")
            return False, [f"JSON parsing error: {str(e)}"]
            
        except Exception as e:
            logger.error(f"Error validating {filepath}: {e}")
            return False, [f"Validation error: {str(e)}"]
    
    def generate_report(self, errors: List[str], filepath: Optional[str] = None) -> None:
        """
        Generate a validation report, optionally saving to a file.
        
        Args:
            errors: List of validation error messages
            filepath: Optional path to save the report
        """
        if not errors:
            report = "Validation successful. No errors found."
        else:
            report = f"Validation failed with {len(errors)} errors:\n\n"
            for i, error in enumerate(errors, 1):
                report += f"{i}. {error}\n"
        
        if filepath:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)
                
            logger.info(f"Validation report saved to {filepath}")
        
        return report
            
# Example usage
if __name__ == "__main__":
    validator = DataValidator()
    
    # Example of validating a file
    try:
        # Assuming you have a sample input file
        input_file = "data/raw/fixed_cities.json"
        valid, errors = validator.validate_file(input_file)
        
        if valid:
            print("Data validation passed!")
        else:
            print(f"Data validation failed with {len(errors)} errors:")
            for error in errors:
                print(f"- {error}")
                
            # Generate report
            validator.generate_report(errors, "data/validation/validation_report.txt")
            
    except Exception as e:
        print(f"Error during validation: {e}")