"""
City Allocation Algorithm

This module determines the optimal distribution of days across multiple cities 
for the Itinerary Planning System.
"""

import logging
from typing import Dict, List, Any, Tuple
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CityAllocator:
    """
    Determines how to allocate days across multiple cities based on 
    attraction importance, quantity, and travel logistics.
    """
    
    def __init__(self, travel_time_calculator=None):
        """
        Initialize the City Allocator.
        
        Args:
            travel_time_calculator: Optional function to calculate travel time between cities
        """
        self.travel_time_calculator = travel_time_calculator
    
    def allocate_days(self, 
                     cities: List[Dict[str, Any]], 
                     total_days: int, 
                     preferences: List[str] = None, 
                     travel_pace: str = "moderate") -> Dict[str, Any]:
        """
        Allocate days across multiple cities.
        
        Args:
            cities: List of city data dictionaries
            total_days: Total number of days available for the trip
            preferences: Optional list of user preferences
            travel_pace: Travel pace preference (slow, moderate, fast)
            
        Returns:
            A dict containing city sequence and days allocation
        """
        logger.info(f"Allocating {total_days} days across {len(cities)} cities")
        
        # If only one city, allocate all days to it
        if len(cities) == 1:
            return {
                "city_sequence": [cities[0]["city_id"]],
                "days_per_city": {cities[0]["city_id"]: total_days},
                "travel_days": {}
            }
        
        # Calculate importance scores for each city
        city_scores = self._calculate_city_scores(cities, preferences)
        
        # Calculate travel times between cities (if travel calculator is available)
        travel_times = self._calculate_travel_times(cities)
        
        # Determine how much time to set aside for travel between cities
        travel_days = self._calculate_travel_days(cities, travel_times, travel_pace)
        
        # Calculate available days for attractions (total days minus travel days)
        available_days = total_days - sum(travel_days.values())
        
        # Distribute available days based on city scores
        days_per_city = self._distribute_days(city_scores, available_days)
        
        # Determine optimal sequence of cities to visit
        city_sequence = self._determine_city_sequence(cities, days_per_city, travel_times)
        
        # Return allocation plan
        return {
            "city_sequence": city_sequence,
            "days_per_city": days_per_city,
            "travel_days": travel_days
        }
    
    def _calculate_city_scores(self, 
                              cities: List[Dict[str, Any]], 
                              preferences: List[str] = None) -> Dict[str, float]:
        """
        Calculate importance scores for each city.
        
        Args:
            cities: List of city data dictionaries
            preferences: Optional list of user preferences
            
        Returns:
            Dictionary of city_id -> importance score
        """
        # TODO: Implement sophisticated scoring based on attraction quality and quantity
        # For now, use a simple heuristic based on number of attractions
        
        city_scores = {}
        for city in cities:
            city_id = city["city_id"]
            num_attractions = len(city.get("attractions", []))
            
            # Basic score based on number of attractions
            score = num_attractions
            
            # Adjust score based on preferences if provided
            if preferences:
                preference_match_count = 0
                for attraction in city.get("attractions", []):
                    category = attraction.get("category", "")
                    if any(pref.lower() in category.lower() for pref in preferences):
                        preference_match_count += 1
                
                # Boost score based on preference matches
                if num_attractions > 0:
                    preference_factor = 1 + (preference_match_count / num_attractions)
                    score *= preference_factor
            
            city_scores[city_id] = score
        
        return city_scores
    
    def _calculate_travel_times(self, cities: List[Dict[str, Any]]) -> Dict[Tuple[str, str], float]:
        """
        Calculate estimated travel times between cities.
        
        Args:
            cities: List of city data dictionaries
            
        Returns:
            Dictionary mapping (from_city_id, to_city_id) -> travel_time_hrs
        """
        travel_times = {}
        
        # If a travel time calculator is provided, use it
        if self.travel_time_calculator:
            for i, city1 in enumerate(cities):
                for j, city2 in enumerate(cities):
                    if i != j:
                        city1_id = city1["city_id"]
                        city2_id = city2["city_id"]
                        travel_times[(city1_id, city2_id)] = self.travel_time_calculator(city1, city2)
        else:
            # Fallback: Rough estimates based on inter-city distances
            # TODO: Replace with actual distance-based calculations
            # For now, just use placeholder values
            for i, city1 in enumerate(cities):
                for j, city2 in enumerate(cities):
                    if i != j:
                        city1_id = city1["city_id"]
                        city2_id = city2["city_id"]
                        travel_times[(city1_id, city2_id)] = 4.0  # Placeholder: assume 4 hours average travel time
        
        return travel_times
    
    def _calculate_travel_days(self, 
                             cities: List[Dict[str, Any]], 
                             travel_times: Dict[Tuple[str, str], float],
                             travel_pace: str) -> Dict[Tuple[str, str], float]:
        """
        Calculate days needed for traveling between cities.
        
        Args:
            cities: List of city data dictionaries
            travel_times: Dictionary of travel times between cities
            travel_pace: Travel pace preference
            
        Returns:
            Dictionary mapping (from_city_id, to_city_id) -> travel_days
        """
        # Define thresholds for when travel consumes a full day
        pace_thresholds = {
            "slow": 3.0,      # > 3 hours = full day with slow pace
            "moderate": 5.0,  # > 5 hours = full day with moderate pace
            "fast": 7.0       # > 7 hours = full day with fast pace
        }
        
        threshold = pace_thresholds.get(travel_pace, 5.0)
        travel_days = {}
        
        for (from_city, to_city), time_hrs in travel_times.items():
            # If travel time exceeds threshold, allocate a full day
            if time_hrs > threshold:
                travel_days[(from_city, to_city)] = 1.0
            else:
                # For shorter travel times, allocate a portion of a day
                # This allows for some sightseeing on travel days
                travel_days[(from_city, to_city)] = time_hrs / 24.0
        
        return travel_days
    
    def _distribute_days(self, 
                        city_scores: Dict[str, float], 
                        available_days: float) -> Dict[str, int]:
        """
        Distribute available days among cities based on their scores.
        
        Args:
            city_scores: Dictionary of city_id -> importance score
            available_days: Number of days available for allocation
            
        Returns:
            Dictionary of city_id -> allocated days
        """
        total_score = sum(city_scores.values())
        days_per_city = {}
        remaining_days = available_days
        
        if total_score == 0:
            # If all scores are zero, distribute evenly
            days_per_city = {city_id: available_days // len(city_scores) for city_id in city_scores}
            return days_per_city
        
        # Initial allocation based on scores
        for city_id, score in city_scores.items():
            # Calculate proportional days and ensure at least 1 day per city
            days = max(1, (score / total_score) * available_days)
            days_per_city[city_id] = math.floor(days)  # Initial allocation is floor value
            remaining_days -= math.floor(days)
        
        # Distribute any remaining days to highest scoring cities
        sorted_cities = sorted(city_scores.items(), key=lambda x: x[1], reverse=True)
        for city_id, _ in sorted_cities:
            if remaining_days <= 0:
                break
            days_per_city[city_id] += 1
            remaining_days -= 1
        
        return days_per_city
    
    def _determine_city_sequence(self, 
                               cities: List[Dict[str, Any]], 
                               days_per_city: Dict[str, int],
                               travel_times: Dict[Tuple[str, str], float]) -> List[str]:
        """
        Determine the optimal sequence to visit cities.
        
        Args:
            cities: List of city data dictionaries
            days_per_city: Dictionary of city_id -> allocated days
            travel_times: Dictionary of travel times between cities
            
        Returns:
            Ordered list of city_ids representing the visit sequence
        """
        # TODO: Implement a more sophisticated sequencing algorithm
        # For now, use a greedy approach starting with the city that has most days
        
        # Get cities with allocated days
        cities_to_visit = [city["city_id"] for city in cities if city["city_id"] in days_per_city]
        
        # If no cities to visit, return empty sequence
        if not cities_to_visit:
            return []
        
        # Start with the city that has most days allocated
        current_city = max(days_per_city.items(), key=lambda x: x[1])[0]
        sequence = [current_city]
        remaining_cities = set(cities_to_visit) - {current_city}
        
        # Greedily add nearest city each time
        while remaining_cities:
            # Find nearest unvisited city
            next_city = min(
                remaining_cities,
                key=lambda city_id: travel_times.get((current_city, city_id), float('inf'))
            )
            
            sequence.append(next_city)
            remaining_cities.remove(next_city)
            current_city = next_city
        
        return sequence

# Example usage
if __name__ == "__main__":
    # Example data
    cities_data = [
        {
            "city_id": "paris",
            "name": "Paris",
            "country": "France",
            "attractions": [{"name": "Eiffel Tower"}, {"name": "Louvre Museum"}]
        },
        {
            "city_id": "amsterdam",
            "name": "Amsterdam", 
            "country": "Netherlands",
            "attractions": [{"name": "Van Gogh Museum"}]
        },
        {
            "city_id": "rome",
            "name": "Rome",
            "country": "Italy",
            "attractions": [{"name": "Colosseum"}, {"name": "Vatican"}, {"name": "Roman Forum"}]
        }
    ]
    
    # Initialize allocator
    allocator = CityAllocator()
    
    # Allocate days
    allocation = allocator.allocate_days(
        cities=cities_data,
        total_days=5,
        preferences=["art", "history"],
        travel_pace="moderate"
    )
    
    print("City Sequence:", allocation["city_sequence"])
    print("Days per City:", allocation["days_per_city"])
    print("Travel Days:", allocation["travel_days"])