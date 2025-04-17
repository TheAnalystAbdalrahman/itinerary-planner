"""
Route Optimizer Module

This module determines the optimal sequence to visit attractions
for the Itinerary Planning System.
"""

import logging
import math
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RouteOptimizer:
    """
    Determines the optimal sequence to visit attractions.
    
    Optimizes routes for each day to minimize travel time and
    create logical visit sequences while respecting time constraints.
    """
    
    def __init__(self, travel_speed_km_per_hour: float = 5.0):
        """
        Initialize the route optimizer.
        
        Args:
            travel_speed_km_per_hour: Average travel speed between attractions (walking/transport)
        """
        self.travel_speed = travel_speed_km_per_hour
        
        # Default opening hours (if not specified in attraction data)
        self.default_opening_hours = {
            "start": "09:00",
            "end": "18:00"
        }
        
        # Default time settings
        self.day_start_time = "09:00"  # Default day start time
        self.day_end_time = "21:00"    # Default day end time
        self.lunch_time = "13:00"      # Default lunch time
        self.lunch_duration_hrs = 1.0  # Default lunch duration
        
    def optimize_daily_routes(self, attraction_groups: Dict[str, List[List[Dict[str, Any]]]]) -> Dict[str, List[List[Dict[str, Any]]]]:
        """
        Optimize routes for each day in each city.
        
        Args:
            attraction_groups: Dictionary mapping city_id to list of daily attraction groups
            
        Returns:
            Dictionary mapping city_id to list of optimized daily routes
        """
        optimized_routes = {}
        
        for city_id, daily_groups in attraction_groups.items():
            logger.info(f"Optimizing routes for city: {city_id}")
            
            optimized_city_routes = []
            
            for day_idx, attractions in enumerate(daily_groups, 1):
                logger.info(f"Optimizing day {day_idx} with {len(attractions)} attractions")
                
                # Skip empty days
                if not attractions:
                    optimized_city_routes.append([])
                    continue
                
                # Optimize sequence for this day
                optimized_sequence = self._optimize_sequence(attractions)
                
                # Calculate timings for the optimized sequence
                timed_sequence = self._calculate_timings(optimized_sequence)
                
                optimized_city_routes.append(timed_sequence)
            
            optimized_routes[city_id] = optimized_city_routes
        
        return optimized_routes
    
    def _optimize_sequence(self, attractions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Determine the optimal sequence to visit attractions within a day.
        
        Args:
            attractions: List of attractions for a day
            
        Returns:
            Reordered list of attractions in optimal visit sequence
        """
        # If fewer than 2 attractions, no optimization needed
        if len(attractions) < 2:
            return attractions.copy()
        
        # Calculate distance matrix between all attractions
        distance_matrix = self._calculate_distance_matrix(attractions)
        
        # Try to find an optimal route using nearest neighbor algorithm
        # Start with the most central attraction (closest to all others)
        
        # Calculate centrality scores (sum of distances to all other attractions)
        centrality_scores = np.sum(distance_matrix, axis=1)
        
        # Start with the most central attraction (lowest total distance)
        start_idx = np.argmin(centrality_scores)
        
        # Build the route
        route = [start_idx]
        unvisited = set(range(len(attractions)))
        unvisited.remove(start_idx)
        
        # Nearest neighbor algorithm
        current = start_idx
        while unvisited:
            # Find nearest unvisited attraction
            nearest = min(unvisited, key=lambda i: distance_matrix[current][i])
            route.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        # Create the optimized sequence
        optimized = [attractions[i] for i in route]
        
        return optimized
    
    def _calculate_distance_matrix(self, attractions: List[Dict[str, Any]]) -> np.ndarray:
        """
        Calculate distance matrix between all attractions.
        
        Args:
            attractions: List of attractions
            
        Returns:
            2D numpy array of distances in kilometers
        """
        n = len(attractions)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                attr_i = attractions[i]
                attr_j = attractions[j]
                
                # Calculate distance if coordinates are available
                distance = self._calculate_distance(attr_i, attr_j)
                
                # Set symmetric distances
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance
        
        return distance_matrix
    
    def _calculate_distance(self, attr_i: Dict[str, Any], attr_j: Dict[str, Any]) -> float:
        """
        Calculate distance between two attractions in kilometers.
        
        Args:
            attr_i: First attraction
            attr_j: Second attraction
            
        Returns:
            Distance in kilometers
        """
        # Check if coordinates are available
        if "coordinates" in attr_i and "coordinates" in attr_j:
            # Extract coordinates
            lat1 = attr_i["coordinates"].get("latitude")
            lon1 = attr_i["coordinates"].get("longitude")
            lat2 = attr_j["coordinates"].get("latitude")
            lon2 = attr_j["coordinates"].get("longitude")
            
            # Calculate haversine distance if all coordinates are available
            if all(coord is not None for coord in [lat1, lon1, lat2, lon2]):
                return self._haversine_distance(lat1, lon1, lat2, lon2)
        
        # Fallback: return a default distance based on categories
        # Different categories often means different areas of the city
        cat_i = attr_i.get("category", "misc")
        cat_j = attr_j.get("category", "misc")
        
        if cat_i == cat_j:
            # Same category attractions might be closer
            return 2.0
        else:
            # Different category attractions might be further apart
            return 5.0
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great-circle distance between two points in kilometers.
        
        Args:
            lat1: Latitude of first point
            lon1: Longitude of first point
            lat2: Latitude of second point
            lon2: Longitude of second point
            
        Returns:
            Distance in kilometers
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        
        return c * r
    
    def _calculate_timings(self, attractions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate visit timings for an optimized sequence of attractions.
        
        Args:
            attractions: List of attractions in optimized sequence
            
        Returns:
            List of attractions with added timing information
        """
        if not attractions:
            return []
        
        # Create a deep copy of the attractions to avoid modifying the originals
        timed_attractions = []
        for attr in attractions:
            # Create a new dictionary with the original attraction data
            timed_attr = attr.copy()
            timed_attractions.append(timed_attr)
        
        # Convert day start time to hours (e.g., "09:00" -> 9.0)
        current_time = self._time_to_hours(self.day_start_time)
        
        # Calculate lunch time in hours
        lunch_time_hours = self._time_to_hours(self.lunch_time)
        lunch_scheduled = False
        
        # Calculate end time for each attraction and start time for the next
        for i in range(len(timed_attractions)):
            attr = timed_attractions[i]
            
            # Check if it's lunch time before this attraction
            if not lunch_scheduled and current_time >= lunch_time_hours:
                # Add lunch break
                current_time += self.lunch_duration_hrs
                lunch_scheduled = True
            
            # Get duration for this attraction
            duration = attr.get("visit_duration_hrs", 1.0)
            
            # Set start and end times
            attr["start_time"] = self._hours_to_time(current_time)
            attr["end_time"] = self._hours_to_time(current_time + duration)
            
            # Update current time
            current_time += duration
            
            # Add travel time to next attraction if not the last one
            if i < len(timed_attractions) - 1:
                next_attr = timed_attractions[i + 1]
                travel_time = self._calculate_travel_time(attr, next_attr)
                current_time += travel_time
        
        return timed_attractions
    
    def _calculate_travel_time(self, from_attr: Dict[str, Any], to_attr: Dict[str, Any]) -> float:
        """
        Calculate travel time between two attractions in hours.
        
        Args:
            from_attr: Origin attraction
            to_attr: Destination attraction
            
        Returns:
            Travel time in hours
        """
        # Calculate distance
        distance_km = self._calculate_distance(from_attr, to_attr)
        
        # Calculate travel time (distance / speed)
        travel_time_hrs = distance_km / self.travel_speed
        
        # Add a small buffer time for transitions (at least 15 minutes)
        buffer_time = max(0.25, travel_time_hrs * 0.1)
        
        return travel_time_hrs + buffer_time
    
    def _time_to_hours(self, time_str: str) -> float:
        """
        Convert time string (e.g., "09:30") to hours (e.g., 9.5).
        
        Args:
            time_str: Time string in format "HH:MM"
            
        Returns:
            Time in hours as float
        """
        try:
            hours, minutes = map(int, time_str.split(":"))
            return hours + minutes / 60
        except (ValueError, AttributeError):
            # Default to 9:00 AM if time string is invalid
            return 9.0
    
    def _hours_to_time(self, hours: float) -> str:
        """
        Convert hours (e.g., 9.5) to time string (e.g., "09:30").
        
        Args:
            hours: Time in hours as float
            
        Returns:
            Time string in format "HH:MM"
        """
        # Handle edge cases
        if hours < 0:
            hours = 0
        elif hours >= 24:
            hours = hours % 24
        
        # Split into hours and minutes
        h = int(hours)
        m = int((hours - h) * 60)
        
        # Format as string
        return f"{h:02d}:{m:02d}"

# Example usage
if __name__ == "__main__":
    # Example attraction groups
    attraction_groups = {
        "paris": [
            # Day 1
            [
                {
                    "name": "Eiffel Tower",
                    "category": "landmark",
                    "visit_duration_hrs": 2.5,
                    "coordinates": {"latitude": 48.8584, "longitude": 2.2945}
                },
                {
                    "name": "Louvre Museum",
                    "category": "museum",
                    "visit_duration_hrs": 3.0,
                    "coordinates": {"latitude": 48.8606, "longitude": 2.3376}
                },
                {
                    "name": "Notre-Dame Cathedral",
                    "category": "historical",
                    "visit_duration_hrs": 1.5,
                    "coordinates": {"latitude": 48.8530, "longitude": 2.3499}
                }
            ],
            # Day 2
            [
                {
                    "name": "Montmartre",
                    "category": "cultural",
                    "visit_duration_hrs": 2.0,
                    "coordinates": {"latitude": 48.8867, "longitude": 2.3431}
                },
                {
                    "name": "Arc de Triomphe",
                    "category": "landmark",
                    "visit_duration_hrs": 1.0,
                    "coordinates": {"latitude": 48.8738, "longitude": 2.2950}
                }
            ]
        ]
    }
    
    # Initialize optimizer
    optimizer = RouteOptimizer()
    
    # Optimize routes
    optimized_routes = optimizer.optimize_daily_routes(attraction_groups)
    
    # Print results
    for city_id, daily_routes in optimized_routes.items():
        print(f"\nOptimized routes for {city_id}:")
        
        for day, attractions in enumerate(daily_routes, 1):
            print(f"\nDay {day}:")
            
            for i, attr in enumerate(attractions, 1):
                name = attr.get("name", f"Attraction #{i}")
                category = attr.get("category", "misc")
                start_time = attr.get("start_time", "N/A")
                end_time = attr.get("end_time", "N/A")
                
                print(f"  {start_time} - {end_time}: {name} ({category})")