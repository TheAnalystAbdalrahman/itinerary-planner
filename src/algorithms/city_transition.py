"""
City Transition Handler Module

This module handles transitions between cities in the itinerary
for the Itinerary Planning System.
"""

import logging
import math
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CityTransitionHandler:
    """
    Handles transitions between cities in the itinerary.
    
    Calculates travel times, determines optimal departure/arrival times,
    and adjusts daily plans around travel days.
    """
    
    def __init__(self):
        """Initialize the city transition handler."""
        # Default transit speeds in km/h
        self.transit_speeds = {
            "train": 120.0,  # High-speed train
            "bus": 80.0,     # Intercity bus
            "flight": 800.0, # Commercial flight
            "car": 90.0,     # Car travel
            "ferry": 40.0    # Ferry
        }
        
        # Default transit options based on distance
        self.distance_thresholds = {
            # Up to 300km: prefer train
            300: "train",
            # 300-800km: train or flight
            800: "train",
            # Above 800km: prefer flight
            float('inf'): "flight"
        }
        
        # Ideal departure time ranges
        self.ideal_departure_times = {
            "train": ("08:00", "10:00"),
            "bus": ("08:00", "09:00"),
            "flight": ("10:00", "14:00"), 
            "car": ("08:00", "09:00"),
            "ferry": ("09:00", "11:00")
        }
        
        # Buffer times (hours before departure)
        self.buffer_times = {
            "train": 0.5,   # 30 minutes
            "bus": 0.5,      # 30 minutes
            "flight": 2.0,   # 2 hours
            "car": 0.5,      # 30 minutes
            "ferry": 1.0     # 1 hour
        }
    
    def plan_city_transitions(self, 
                            city_sequence: List[str],
                            travel_days: Dict[Tuple[str, str], float],
                            optimized_routes: Dict[str, List[List[Dict[str, Any]]]],
                            cities_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Plan transitions between cities in the itinerary.
        
        Args:
            city_sequence: Ordered list of city IDs to visit
            travel_days: Dictionary mapping (from_city, to_city) to travel time in days
            optimized_routes: Dictionary mapping city_id to list of optimized daily routes
            cities_data: Optional list of city data with coordinates
            
        Returns:
            Dictionary containing travel segments and adjusted routes
        """
        if len(city_sequence) <= 1:
            return {
                "travel_segments": [],
                "adjusted_routes": optimized_routes
            }
        
        logger.info(f"Planning transitions between {len(city_sequence)} cities")
        
        # Plan all travel segments
        travel_segments = self._plan_travel_segments(
            city_sequence=city_sequence,
            travel_days=travel_days,
            cities_data=cities_data
        )
        
        # Adjust daily routes to account for travel time
        adjusted_routes = self._adjust_routes_for_travel(
            city_sequence=city_sequence,
            optimized_routes=optimized_routes,
            travel_segments=travel_segments
        )
        
        # Return travel plan
        return {
            "travel_segments": travel_segments,
            "adjusted_routes": adjusted_routes
        }
    
    def _plan_travel_segments(self, 
                             city_sequence: List[str],
                             travel_days: Dict[Tuple[str, str], float],
                             cities_data: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Plan travel segments between consecutive cities.
        
        Args:
            city_sequence: Ordered list of city IDs to visit
            travel_days: Dictionary mapping (from_city, to_city) to travel time in days
            cities_data: Optional list of city data with coordinates
            
        Returns:
            List of travel segment dictionaries
        """
        travel_segments = []
        
        # Create a city lookup dictionary for faster access
        city_lookup = {}
        if cities_data:
            for city in cities_data:
                city_id = city.get("city_id")
                if city_id:
                    city_lookup[city_id] = city
        
        # For each consecutive city pair, create a travel segment
        for i in range(len(city_sequence) - 1):
            from_city = city_sequence[i]
            to_city = city_sequence[i + 1]
            
            # Get city names if available
            from_city_name = city_lookup.get(from_city, {}).get("name", from_city)
            to_city_name = city_lookup.get(to_city, {}).get("name", to_city)
            
            # Calculate distance between cities
            distance_km = self._calculate_distance_between_cities(
                from_city, to_city, city_lookup
            )
            
            # Determine transportation mode
            transport_mode = self._determine_transport_mode(distance_km)
            
            # Calculate travel time
            travel_time_hrs = self._calculate_travel_time(distance_km, transport_mode)
            
            # Determine optimal departure and arrival times
            departure_time, arrival_time = self._determine_travel_times(travel_time_hrs, transport_mode)
            
            # Create travel segment
            segment = {
                "from_city_id": from_city,
                "from_city": from_city_name,
                "to_city_id": to_city,
                "to_city": to_city_name,
                "transport_mode": transport_mode,
                "distance_km": round(distance_km, 1),
                "travel_time_hrs": round(travel_time_hrs, 1),
                "departure_time": departure_time,
                "arrival_time": arrival_time,
                "travel_day_index": i  # Index of travel day relative to first day
            }
            
            travel_segments.append(segment)
        
        return travel_segments
    
    def _calculate_distance_between_cities(self, 
                                         from_city: str, 
                                         to_city: str,
                                         city_lookup: Dict[str, Dict[str, Any]]) -> float:
        """
        Calculate distance between two cities in kilometers.
        
        Args:
            from_city: Origin city ID
            to_city: Destination city ID
            city_lookup: Dictionary mapping city_id to city data
            
        Returns:
            Distance in kilometers
        """
        # If coordinates are available, calculate haversine distance
        from_city_data = city_lookup.get(from_city, {})
        to_city_data = city_lookup.get(to_city, {})
        
        from_coords = from_city_data.get("coordinates", {})
        to_coords = to_city_data.get("coordinates", {})
        
        if from_coords and to_coords:
            from_lat = from_coords.get("latitude")
            from_lon = from_coords.get("longitude")
            to_lat = to_coords.get("latitude")
            to_lon = to_coords.get("longitude")
            
            if all(coord is not None for coord in [from_lat, from_lon, to_lat, to_lon]):
                return self._haversine_distance(from_lat, from_lon, to_lat, to_lon)
        
        # If coordinates not available, use a rough estimate based on city names
        # This is a very rough approximation and should be replaced with actual distances
        # from a distance API or database if available
        
        # Average distance between major European cities is around 500-800km
        return 600.0
    
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
    
    def _determine_transport_mode(self, distance_km: float) -> str:
        """
        Determine the best transport mode based on distance.
        
        Args:
            distance_km: Distance in kilometers
            
        Returns:
            Transport mode (train, bus, flight, car, ferry)
        """
        for threshold, mode in sorted(self.distance_thresholds.items()):
            if distance_km <= threshold:
                return mode
        
        return "flight"  # Default to flight for very long distances
    
    def _calculate_travel_time(self, distance_km: float, transport_mode: str) -> float:
        """
        Calculate travel time based on distance and transport mode.
        
        Args:
            distance_km: Distance in kilometers
            transport_mode: Transport mode
            
        Returns:
            Travel time in hours
        """
        # Get speed for this transport mode
        speed = self.transit_speeds.get(transport_mode, 100.0)  # Default to 100 km/h
        
        # Calculate basic travel time
        base_time = distance_km / speed
        
        # Add buffer for terminals, stations, etc.
        buffer = 0.0
        if transport_mode == "flight":
            # Add 2 hours for airport procedures
            buffer = 2.0
        elif transport_mode in ["train", "bus"]:
            # Add 30 minutes for station procedures
            buffer = 0.5
        
        # Add buffer for check-in, boarding, etc.
        return base_time + buffer
    
    def _determine_travel_times(self, travel_time_hrs: float, transport_mode: str) -> Tuple[str, str]:
        """
        Determine optimal departure and arrival times.
        
        Args:
            travel_time_hrs: Travel time in hours
            transport_mode: Transport mode
            
        Returns:
            Tuple of (departure_time, arrival_time) as strings in format "HH:MM"
        """
        # Get ideal departure time range for this transport mode
        ideal_range = self.ideal_departure_times.get(transport_mode, ("09:00", "10:00"))
        departure_time_str = ideal_range[0]  # Use earliest ideal time
        
        # Convert departure time to hours
        departure_time_hrs = self._time_to_hours(departure_time_str)
        
        # Calculate arrival time
        arrival_time_hrs = departure_time_hrs + travel_time_hrs
        
        # Format times as strings
        departure_time = self._hours_to_time(departure_time_hrs)
        arrival_time = self._hours_to_time(arrival_time_hrs)
        
        return departure_time, arrival_time
    
    def _adjust_routes_for_travel(self, 
                                 city_sequence: List[str],
                                 optimized_routes: Dict[str, List[List[Dict[str, Any]]]],
                                 travel_segments: List[Dict[str, Any]]) -> Dict[str, List[List[Dict[str, Any]]]]:
        """
        Adjust daily routes to account for travel time.
        
        Args:
            city_sequence: Ordered list of city IDs to visit
            optimized_routes: Dictionary mapping city_id to list of optimized daily routes
            travel_segments: List of travel segment dictionaries
            
        Returns:
            Adjusted routes dictionary
        """
        # Create a deep copy to avoid modifying the original
        adjusted_routes = {}
        for city_id, routes in optimized_routes.items():
            adjusted_routes[city_id] = []
            for day_route in routes:
                adjusted_routes[city_id].append(day_route.copy())
        
        # For each travel segment, adjust the routes
        for segment in travel_segments:
            from_city = segment["from_city_id"]
            to_city = segment["to_city_id"]
            departure_time = segment["departure_time"]
            
            # Convert departure time to hours
            departure_hrs = self._time_to_hours(departure_time)
            
            # Get buffer time for this transport mode
            transport_mode = segment["transport_mode"]
            buffer_hrs = self.buffer_times.get(transport_mode, 0.5)
            
            # Calculate the latest time you should be done with activities
            latest_activity_end = departure_hrs - buffer_hrs
            
            # Find the last day in the origin city
            if from_city in adjusted_routes and adjusted_routes[from_city]:
                last_day_routes = adjusted_routes[from_city][-1]
                
                # Adjust activities on the departure day
                adjusted_day = []
                for attr in last_day_routes:
                    # Check if this activity ends after the latest allowable time
                    end_time = attr.get("end_time", "18:00")
                    end_hrs = self._time_to_hours(end_time)
                    
                    if end_hrs <= latest_activity_end:
                        # This activity can be completed before departure
                        adjusted_day.append(attr)
                    else:
                        # This activity conflicts with departure
                        # You could either remove it or adjust its end time
                        # Here we'll adjust the end time if it starts before the latest time
                        start_time = attr.get("start_time", "09:00")
                        start_hrs = self._time_to_hours(start_time)
                        
                        if start_hrs < latest_activity_end:
                            # Activity can be partially completed
                            adjusted_attr = attr.copy()
                            adjusted_attr["end_time"] = self._hours_to_time(latest_activity_end)
                            adjusted_attr["notes"] = f"Shortened due to {transport_mode} to {segment['to_city']} at {departure_time}"
                            adjusted_day.append(adjusted_attr)
                
                # Update the last day with adjusted activities
                adjusted_routes[from_city][-1] = adjusted_day
                
                # Add a note about the departure
                if adjusted_day:
                    departure_note = {
                        "name": f"Travel to {segment['to_city']}",
                        "category": "travel",
                        "start_time": departure_time,
                        "end_time": segment["arrival_time"],
                        "transport_mode": transport_mode,
                        "is_travel_segment": True
                    }
                    adjusted_routes[from_city][-1].append(departure_note)
        
        return adjusted_routes
    
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
    # Example data
    city_sequence = ["paris", "amsterdam", "berlin"]
    
    # Travel days (simplified for example)
    travel_days = {
        ("paris", "amsterdam"): 0.5,  # Half a day
        ("amsterdam", "berlin"): 0.5  # Half a day
    }
    
    # Example optimized routes
    optimized_routes = {
        "paris": [
            # Day 1
            [
                {
                    "name": "Eiffel Tower",
                    "category": "landmark",
                    "start_time": "09:00",
                    "end_time": "11:30"
                },
                {
                    "name": "Louvre Museum",
                    "category": "museum",
                    "start_time": "12:30",
                    "end_time": "15:30"
                }
            ]
        ],
        "amsterdam": [
            # Day 2
            [
                {
                    "name": "Van Gogh Museum",
                    "category": "museum",
                    "start_time": "10:00",
                    "end_time": "13:00"
                },
                {
                    "name": "Canal Cruise",
                    "category": "entertainment",
                    "start_time": "14:00",
                    "end_time": "16:00"
                }
            ]
        ],
        "berlin": [
            # Day 3
            [
                {
                    "name": "Brandenburg Gate",
                    "category": "landmark",
                    "start_time": "09:00",
                    "end_time": "10:30"
                },
                {
                    "name": "Berlin Wall Memorial",
                    "category": "historical",
                    "start_time": "11:00",
                    "end_time": "13:00"
                }
            ]
        ]
    }
    
    # Example city data
    cities_data = [
        {
            "city_id": "paris",
            "name": "Paris",
            "country": "France",
            "coordinates": {"latitude": 48.8566, "longitude": 2.3522}
        },
        {
            "city_id": "amsterdam",
            "name": "Amsterdam",
            "country": "Netherlands",
            "coordinates": {"latitude": 52.3676, "longitude": 4.9041}
        },
        {
            "city_id": "berlin",
            "name": "Berlin",
            "country": "Germany",
            "coordinates": {"latitude": 52.5200, "longitude": 13.4050}
        }
    ]
    
    # Initialize transition handler
    handler = CityTransitionHandler()
    
    # Plan city transitions
    travel_plan = handler.plan_city_transitions(
        city_sequence=city_sequence,
        travel_days=travel_days,
        optimized_routes=optimized_routes,
        cities_data=cities_data
    )
    
    # Print travel segments
    print("Travel Segments:")
    for i, segment in enumerate(travel_plan["travel_segments"], 1):
        print(f"\nTravel Segment {i}:")
        print(f"  From: {segment['from_city']} ({segment['from_city_id']})")
        print(f"  To: {segment['to_city']} ({segment['to_city_id']})")
        print(f"  Mode: {segment['transport_mode']}")
        print(f"  Distance: {segment['distance_km']} km")
        print(f"  Travel Time: {segment['travel_time_hrs']} hours")
        print(f"  Departure: {segment['departure_time']}")
        print(f"  Arrival: {segment['arrival_time']}")
    
    # Print adjusted routes
    print("\nAdjusted Routes:")
    for city_id, days in travel_plan["adjusted_routes"].items():
        print(f"\n{city_id.capitalize()} Schedule:")
        
        for day_idx, attractions in enumerate(days, 1):
            print(f"\n  Day {day_idx}:")
            
            for attr in attractions:
                name = attr.get("name", "Unknown")
                start = attr.get("start_time", "N/A")
                end = attr.get("end_time", "N/A")
                category = attr.get("category", "misc")
                
                print(f"    {start} - {end}: {name} ({category})")
                
                if attr.get("notes"):
                    print(f"      Note: {attr['notes']}")