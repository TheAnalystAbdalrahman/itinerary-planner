"""
Attraction Processor Module

This module handles the preprocessing of attraction data for the Itinerary Planning System.
It cleans, normalizes, and structures attraction data for further processing.
"""

import json
import re
from typing import Dict, List, Any, Union
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AttractionProcessor:
    """
    Processes raw attraction data for use in the itinerary planning system.
    Handles data cleaning, normalization, and structuring.
    """
    
    def __init__(self):
        """Initialize the attraction processor."""
        self.known_categories = {
            "landmark", "museum", "park", "entertainment", "food", 
            "shopping", "cultural", "historical", "outdoor", "art"
        }
        
    def process_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the entire input data structure.
        
        Args:
            input_data: Raw JSON input data containing cities and attractions
            
        Returns:
            Processed data ready for itinerary planning
        """
        logger.info("Starting attraction data processing")
        
        processed_data = {
            "cities": [],
            "parameters": input_data.get("parameters", {})
        }
        
        # Process each city and its attractions
        for city in input_data.get("cities", []):
            processed_city = self._process_city(city)
            processed_data["cities"].append(processed_city)
        
        logger.info(f"Processed {len(processed_data['cities'])} cities")
        return processed_data
    
    def _process_city(self, city_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data for a single city.
        
        Args:
            city_data: Raw data for a city including its attractions
            
        Returns:
            Processed city data
        """
        processed_city = {
            "city_id": self._normalize_id(city_data.get("city_id", "")),
            "name": city_data.get("name", city_data.get("city_id", "Unknown City")),
            "country": city_data.get("country", "Unknown"),
            "attractions": []
        }
        
        # Process attractions for this city
        for attraction in city_data.get("attractions", []):
            processed_attraction = self._process_attraction(attraction)
            processed_city["attractions"].append(processed_attraction)
            
        return processed_city
    
    def _process_attraction(self, attraction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data for a single attraction.
        
        Args:
            attraction_data: Raw data for an attraction
            
        Returns:
            Processed attraction data
        """
        # Clean and normalize the attraction data
        processed_attraction = {
            "attraction_id": self._normalize_id(attraction_data.get("attraction_id", "")),
            "name": self._clean_text(attraction_data.get("name", "Unknown Attraction")),
            "category": self._normalize_category(attraction_data.get("category", "misc")),
            "description": self._clean_text(attraction_data.get("description", "")),
            "visit_duration_hrs": self._normalize_duration(attraction_data.get("visit_duration_hrs", 1.0)),
        }
        
        # Handle coordinates if available
        if "latitude" in attraction_data and "longitude" in attraction_data:
            processed_attraction["coordinates"] = {
                "latitude": float(attraction_data["latitude"]),
                "longitude": float(attraction_data["longitude"])
            }
        
        # Add any additional metadata that might be useful
        if "opening_hours" in attraction_data:
            processed_attraction["opening_hours"] = attraction_data["opening_hours"]
            
        if "popularity_score" in attraction_data:
            processed_attraction["popularity_score"] = float(attraction_data["popularity_score"])
            
        return processed_attraction
    
    def _normalize_id(self, id_str: str) -> str:
        """
        Normalize an ID string by removing special characters and converting to lowercase.
        
        Args:
            id_str: The ID string to normalize
            
        Returns:
            Normalized ID string
        """
        if not id_str:
            return f"id_{hash(str(id_str))}"
        
        # Remove special characters and convert to lowercase
        normalized = re.sub(r'[^\w]', '_', id_str.lower())
        return normalized
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing extra whitespace and standardizing formatting.
        
        Args:
            text: The text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace and standardize
        cleaned = re.sub(r'\s+', ' ', text).strip()
        return cleaned
    
    def _normalize_category(self, category: str) -> str:
        """
        Normalize attraction category to a standard format.
        
        Args:
            category: The category to normalize
            
        Returns:
            Normalized category string
        """
        normalized = category.lower().strip()
        
        # Map to known categories if possible
        for known in self.known_categories:
            if known in normalized:
                return known
                
        return "misc"
    
    def _normalize_duration(self, duration: Union[float, int, str]) -> float:
        """
        Normalize visit duration to hours as float.
        
        Args:
            duration: The duration to normalize
            
        Returns:
            Duration in hours as a float
        """
        try:
            if isinstance(duration, str):
                # Try to extract hours and minutes if in format like "2h 30m"
                if 'h' in duration and 'm' in duration:
                    hours_match = re.search(r'(\d+)h', duration)
                    mins_match = re.search(r'(\d+)m', duration)
                    
                    hours = float(hours_match.group(1)) if hours_match else 0
                    mins = float(mins_match.group(1)) / 60 if mins_match else 0
                    
                    return hours + mins
                else:
                    # Just convert to float
                    return float(duration)
            else:
                return float(duration)
        except (ValueError, AttributeError):
            logger.warning(f"Could not parse duration '{duration}', defaulting to 1.0 hours")
            return 1.0
    
    def load_from_file(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load and process data from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            Processed data dictionary
        """
        filepath = Path(filepath)
        logger.info(f"Loading data from {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            return self.process_data(raw_data)
            
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {e}")
            raise
    
    def save_to_file(self, data: Dict[str, Any], filepath: Union[str, Path]) -> None:
        """
        Save processed data to a JSON file.
        
        Args:
            data: Processed data dictionary
            filepath: Path where the JSON file should be saved
        """
        filepath = Path(filepath)
        logger.info(f"Saving processed data to {filepath}")
        
        # Ensure the directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving data to {filepath}: {e}")
            raise

# Example usage
if __name__ == "__main__":
    processor = AttractionProcessor()
    
    # Example of loading from a file
    try:
        # Assuming you have a sample input file
        input_file = Path("data/raw/fixed_cities.json")
        processed_data = processor.load_from_file(input_file)
        
        # Save processed data
        processor.save_to_file(processed_data, "data/processed/processed_attractions.json")
        
        print(f"Processed {sum(len(city['attractions']) for city in processed_data['cities'])} attractions")
    except Exception as e:
        print(f"Error in processing: {e}")