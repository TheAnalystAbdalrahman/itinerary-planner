"""
Itinerary Builder Module

This module generates the final itinerary by combining all optimized components
for the Itinerary Planning System.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ItineraryBuilder:
    """
    Builds the final itinerary by combining all optimized components.
    
    Creates a detailed day-by-day itinerary with practical details,
    travel instructions, and formatted output.
    """
    
    def __init__(self):
        """Initialize the itinerary builder."""
        # Placeholder for city information
        self.city_info = {}
        
        # Default meal times
        self.meal_times = {
            "breakfast": ("08:00", "09:00"),
            "lunch": ("13:00", "14:00"),
            "dinner": ("19:00", "20:30")
        }
    
    def build_itinerary(self, 
                       cities: List[Dict[str, Any]],
                       optimized_routes: Dict[str, List[List[Dict[str, Any]]]],
                       travel_plan: Dict[str, Any],
                       preferences: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Build the final itinerary.
        
        Args:
            cities: List of city data dictionaries
            optimized_routes: Dictionary mapping city_id to list of optimized daily routes
            travel_plan: Dictionary containing travel segments and adjusted routes
            preferences: Optional list of user preferences
            
        Returns:
            Complete itinerary dictionary
        """
        logger.info("Building final itinerary")
        
        # Build a lookup for city information
        self._build_city_lookup(cities)
        
        # Extract travel segments
        travel_segments = travel_plan.get("travel_segments", [])
        
        # Use adjusted routes if available, otherwise use optimized routes
        routes = travel_plan.get("adjusted_routes", optimized_routes)
        
        # Build itinerary days by combining city routes and travel segments
        itinerary_days = self._build_itinerary_days(routes, travel_segments)
        
        # Add practical details and recommendations
        enhanced_days = self._add_practical_details(itinerary_days, preferences)
        
        # Build the final itinerary object
        itinerary = {
            "itinerary": enhanced_days,
            "travel_segments": travel_segments,
            "meta": {
                "cities_visited": len(routes),
                "total_days": len(enhanced_days),
                "preferences": preferences if preferences else []
            }
        }
        
        logger.info(f"Itinerary built with {len(enhanced_days)} days across {len(routes)} cities")
        
        return itinerary
    
    def _build_city_lookup(self, cities: List[Dict[str, Any]]) -> None:
        """
        Build a lookup dictionary for city information.
        
        Args:
            cities: List of city data dictionaries
        """
        self.city_info = {}
        
        for city in cities:
            city_id = city.get("city_id")
            if city_id:
                self.city_info[city_id] = city
    
    def _build_itinerary_days(self, 
                            routes: Dict[str, List[List[Dict[str, Any]]]],
                            travel_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build itinerary days by combining city routes and travel segments.
        
        Args:
            routes: Dictionary mapping city_id to list of daily routes
            travel_segments: List of travel segment dictionaries
            
        Returns:
            List of itinerary day dictionaries
        """
        # Create a flattened sequence of days
        days = []
        day_counter = 1
        
        # Mapping of day to travel segments
        travel_day_mapping = {}
        for segment in travel_segments:
            day_idx = segment.get("travel_day_index", 0)
            travel_day_mapping[day_idx] = segment
        
        # Iterate through cities and their days
        current_travel_day = 0
        
        for city_id, city_days in routes.items():
            city_name = self._get_city_name(city_id)
            
            for day_idx, attractions in enumerate(city_days):
                # Check if this is a travel day
                is_travel_day = False
                travel_segment = None
                
                if current_travel_day in travel_day_mapping:
                    travel_segment = travel_day_mapping[current_travel_day]
                    
                    # Check if this travel segment departs from this city
                    if travel_segment["from_city_id"] == city_id:
                        is_travel_day = True
                
                # Create day entry
                day_entry = {
                    "day": day_counter,
                    "city": city_name,
                    "city_id": city_id,
                    "attractions": attractions,
                    "is_travel_day": is_travel_day
                }
                
                # Add travel information if applicable
                if is_travel_day:
                    day_entry["travel"] = {
                        "to_city": travel_segment["to_city"],
                        "to_city_id": travel_segment["to_city_id"],
                        "transport_mode": travel_segment["transport_mode"],
                        "departure_time": travel_segment["departure_time"],
                        "arrival_time": travel_segment["arrival_time"]
                    }
                    
                    # Increment travel day counter
                    current_travel_day += 1
                
                # Add day to itinerary
                days.append(day_entry)
                day_counter += 1
        
        return days
    
    def _add_practical_details(self, 
                             itinerary_days: List[Dict[str, Any]],
                             preferences: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Add practical details and recommendations to itinerary days.
        
        Args:
            itinerary_days: List of itinerary day dictionaries
            preferences: Optional list of user preferences
            
        Returns:
            Enhanced itinerary days with practical details
        """
        enhanced_days = []
        
        for day in itinerary_days:
            enhanced_day = day.copy()
            
            # Add city information
            city_id = day.get("city_id")
            if city_id in self.city_info:
                city_data = self.city_info[city_id]
                enhanced_day["city_info"] = {
                    "country": city_data.get("country", ""),
                    "language": city_data.get("language", ""),
                    "currency": city_data.get("currency", ""),
                    "timezone": city_data.get("timezone", "")
                }
            
            # Add meal suggestions
            enhanced_day["meals"] = self._suggest_meals(
                day.get("attractions", []),
                city_id,
                preferences
            )
            
            # Add practical tips
            enhanced_day["practical_tips"] = self._generate_practical_tips(
                enhanced_day,
                is_travel_day=day.get("is_travel_day", False)
            )
            
            # Add weather information if available
            if "weather" in self.city_info.get(city_id, {}):
                enhanced_day["weather"] = self.city_info[city_id]["weather"]
            
            enhanced_days.append(enhanced_day)
        
        return enhanced_days
    
    def _suggest_meals(self, 
                     attractions: List[Dict[str, Any]],
                     city_id: str,
                     preferences: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Suggest meals based on attractions schedule.
        
        Args:
            attractions: List of attraction dictionaries for the day
            city_id: City ID
            preferences: Optional list of user preferences
            
        Returns:
            Dictionary of meal suggestions
        """
        meals = {}
        
        # Get attraction time spans
        time_spans = []
        for attraction in attractions:
            # Skip if no timing information
            if "start_time" not in attraction or "end_time" not in attraction:
                continue
                
            # Skip travel segments
            if attraction.get("is_travel_segment", False):
                continue
                
            start_hrs = self._time_to_hours(attraction["start_time"])
            end_hrs = self._time_to_hours(attraction["end_time"])
            
            time_spans.append((start_hrs, end_hrs))
        
        # Suggest breakfast
        breakfast_start, breakfast_end = self._time_to_hours(self.meal_times["breakfast"][0]), self._time_to_hours(self.meal_times["breakfast"][1])
        breakfast_conflict = any(start <= breakfast_end and end >= breakfast_start for start, end in time_spans)
        
        if not breakfast_conflict:
            meals["breakfast"] = {
                "time": self.meal_times["breakfast"][0],
                "suggestion": self._get_meal_suggestion(city_id, "breakfast", preferences)
            }
        
        # Suggest lunch
        lunch_start, lunch_end = self._time_to_hours(self.meal_times["lunch"][0]), self._time_to_hours(self.meal_times["lunch"][1])
        lunch_conflict = any(start <= lunch_end and end >= lunch_start for start, end in time_spans)
        
        if not lunch_conflict:
            meals["lunch"] = {
                "time": self.meal_times["lunch"][0],
                "suggestion": self._get_meal_suggestion(city_id, "lunch", preferences)
            }
        
        # Suggest dinner
        dinner_start, dinner_end = self._time_to_hours(self.meal_times["dinner"][0]), self._time_to_hours(self.meal_times["dinner"][1])
        dinner_conflict = any(start <= dinner_end and end >= dinner_start for start, end in time_spans)
        
        if not dinner_conflict:
            meals["dinner"] = {
                "time": self.meal_times["dinner"][0],
                "suggestion": self._get_meal_suggestion(city_id, "dinner", preferences)
            }
        
        return meals
    
    def _get_meal_suggestion(self, 
                           city_id: str, 
                           meal_type: str,
                           preferences: Optional[List[str]] = None) -> str:
        """
        Get a meal suggestion based on city and preferences.
        
        Args:
            city_id: City ID
            meal_type: Type of meal (breakfast, lunch, dinner)
            preferences: Optional list of user preferences
            
        Returns:
            Meal suggestion string
        """
        # Check if we have specific city cuisine information
        city_data = self.city_info.get(city_id, {})
        cuisine = city_data.get("cuisine", [])
        
        # Default suggestions
        default_suggestions = {
            "breakfast": "Local café or hotel breakfast",
            "lunch": "Try a casual local restaurant",
            "dinner": "Enjoy dinner at a recommended restaurant"
        }
        
        # If we have cuisine information, make more specific suggestions
        if cuisine and isinstance(cuisine, list) and len(cuisine) > 0:
            if meal_type == "breakfast":
                return f"Enjoy a local breakfast with {cuisine[0]} specialties"
            elif meal_type == "lunch":
                return f"Try {cuisine[0]} cuisine at a local restaurant"
            elif meal_type == "dinner":
                if len(cuisine) > 1:
                    return f"Experience authentic {cuisine[1]} cuisine for dinner"
                else:
                    return f"Savor {cuisine[0]} specialties for dinner"
        
        # Check preferences for food-related preferences
        if preferences:
            food_prefs = [p for p in preferences if p.lower() in ["food", "cuisine", "local", "authentic", "dining"]]
            if food_prefs:
                return f"Explore local cuisine based on your {', '.join(food_prefs)} preferences"
        
        # Return default suggestion if no specific information
        return default_suggestions.get(meal_type, "Enjoy a meal at your preferred time")
    
    def _generate_practical_tips(self, 
                               day_info: Dict[str, Any],
                               is_travel_day: bool = False) -> List[str]:
        """
        Generate practical tips for the day.
        
        Args:
            day_info: Day information dictionary
            is_travel_day: Whether this is a travel day
            
        Returns:
            List of practical tips
        """
        tips = []
        
        # Basic tips
        tips.append("Carry a map or use offline maps on your phone")
        tips.append("Keep important documents secure")
        
        # City-specific tips
        city_id = day_info.get("city_id")
        if city_id in self.city_info:
            city_data = self.city_info[city_id]
            
            # Add language tip if available
            if "language" in city_data:
                tips.append(f"Basic phrases in {city_data['language']} can be helpful")
            
            # Add currency tip if available
            if "currency" in city_data:
                tips.append(f"Local currency is {city_data['currency']}")
            
            # Add transportation tip
            tips.append(f"Consider getting a day pass for public transport in {day_info['city']}")
        
        # Travel day specific tips
        if is_travel_day and "travel" in day_info:
            transport_mode = day_info["travel"].get("transport_mode")
            departure_time = day_info["travel"].get("departure_time")
            
            if transport_mode and departure_time:
                # Different tips based on transport mode
                if transport_mode == "flight":
                    tips.append(f"Arrive at the airport at least 2 hours before your {departure_time} flight")
                    tips.append("Check baggage allowance and restrictions")
                elif transport_mode == "train":
                    tips.append(f"Arrive at the train station 30 minutes before your {departure_time} departure")
                elif transport_mode == "bus":
                    tips.append(f"Arrive at the bus station 30 minutes before your {departure_time} departure")
                elif transport_mode == "ferry":
                    tips.append(f"Arrive at the ferry terminal 1 hour before your {departure_time} departure")
                
                tips.append(f"Pack essentials in your hand luggage for the journey to {day_info['travel']['to_city']}")
        
        # Weather-related tips
        if "weather" in day_info:
            weather = day_info["weather"]
            if "temperature" in weather:
                temp = weather["temperature"]
                if temp > 25:  # Warm weather
                    tips.append("Dress for warm weather and stay hydrated")
                elif temp < 10:  # Cold weather
                    tips.append("Dress warmly with layers for cold weather")
            
            if "conditions" in weather:
                conditions = weather["conditions"].lower()
                if "rain" in conditions:
                    tips.append("Pack an umbrella or raincoat")
        
        # Attraction-specific tips
        attractions = day_info.get("attractions", [])
        for attraction in attractions:
            category = attraction.get("category", "").lower()
            
            # Museum tips
            if "museum" in category:
                tips.append("Check for special exhibitions or tours at museums")
                break  # Only add this tip once
        
        # Limit to maximum 5 tips to avoid overwhelming the user
        return tips[:5]
    
    def _get_city_name(self, city_id: str) -> str:
        """
        Get the city name from city ID.
        
        Args:
            city_id: City ID
            
        Returns:
            City name
        """
        if city_id in self.city_info:
            return self.city_info[city_id].get("name", city_id)
        return city_id.capitalize()
    
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
    
    def save_itinerary(self, itinerary: Dict[str, Any], filepath: str) -> bool:
        """
        Save the itinerary to a JSON file.
        
        Args:
            itinerary: Itinerary dictionary
            filepath: Path to save the JSON file
            
        Returns:
            Boolean indicating success
        """
        try:
            filepath = Path(filepath)
            
            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(itinerary, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Itinerary saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving itinerary: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Example data
    cities = [
        {
            "city_id": "paris",
            "name": "Paris",
            "country": "France",
            "language": "French",
            "currency": "Euro (€)",
            "timezone": "CET",
            "cuisine": ["French", "Pastries", "Wine"]
        },
        {
            "city_id": "amsterdam",
            "name": "Amsterdam",
            "country": "Netherlands",
            "language": "Dutch",
            "currency": "Euro (€)",
            "timezone": "CET",
            "cuisine": ["Dutch", "Cheese", "Seafood"]
        }
    ]
    
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
                    "start_time": "13:00",
                    "end_time": "16:00"
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
        ]
    }
    
    # Example travel plan
    travel_plan = {
        "travel_segments": [
            {
                "from_city_id": "paris",
                "from_city": "Paris",
                "to_city_id": "amsterdam",
                "to_city": "Amsterdam",
                "transport_mode": "train",
                "distance_km": 430.0,
                "travel_time_hrs": 3.5,
                "departure_time": "08:00",
                "arrival_time": "11:30",
                "travel_day_index": 0
            }
        ],
        "adjusted_routes": optimized_routes
    }
    
    # Preferences
    preferences = ["art", "history", "food"]
    
    # Initialize itinerary builder
    builder = ItineraryBuilder()
    
    # Build itinerary
    itinerary = builder.build_itinerary(
        cities=cities,
        optimized_routes=optimized_routes,
        travel_plan=travel_plan,
        preferences=preferences
    )
    
    # Save itinerary
    builder.save_itinerary(itinerary, "example_itinerary.json")
    
    # Print itinerary summary
    print("Itinerary Summary:")
    print(f"Total Days: {itinerary['meta']['total_days']}")
    print(f"Cities Visited: {itinerary['meta']['cities_visited']}")
    
    # Print day-by-day outline
    print("\nDay-by-Day Outline:")
    for day in itinerary["itinerary"]:
        print(f"\nDay {day['day']}: {day['city']}")
        
        if day.get("is_travel_day"):
            travel = day["travel"]
            print(f"  Travel to {travel['to_city']} by {travel['transport_mode']}")
            print(f"  Departure: {travel['departure_time']}, Arrival: {travel['arrival_time']}")
        
        print("  Attractions:")
        for attr in day["attractions"]:
            name = attr.get("name", "Unknown")
            start = attr.get("start_time", "N/A")
            end = attr.get("end_time", "N/A")
            
            print(f"    {start} - {end}: {name}")
        
        if "meals" in day:
            print("  Meals:")
            for meal_type, meal_info in day["meals"].items():
                print(f"    {meal_type.capitalize()} ({meal_info['time']}): {meal_info['suggestion']}")
        
        if "practical_tips" in day:
            print("  Practical Tips:")
            for tip in day["practical_tips"]:
                print(f"    - {tip}")