"""
Attraction Grouper Module

This module groups attractions that work well together in daily plans for
the Itinerary Planning System.
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
import math
import random
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AttractionGrouper:
    """
    Groups attractions that work well together in daily plans.
    
    Uses transformer embeddings to find compatible attractions,
    balances different types of attractions, and respects time constraints.
    """
    
    def __init__(self, transformer_handler=None, use_semantic_grouping=True):
        """
        Initialize the attraction grouper.
        
        Args:
            transformer_handler: Optional transformer handler for semantic grouping
            use_semantic_grouping: Whether to use semantic grouping or fallback to basic grouping
        """
        self.transformer_handler = transformer_handler
        self.use_semantic_grouping = use_semantic_grouping and transformer_handler is not None
        
        # Default visit durations by category (hours)
        self.default_durations = {
            "museum": 3.0,
            "art": 2.5,
            "landmark": 2.0,
            "historical": 2.0,
            "cultural": 2.0,
            "entertainment": 3.0,
            "outdoor": 2.0,
            "park": 1.5,
            "food": 1.5,
            "shopping": 1.5,
            "misc": 1.0
        }
        
        # Category balance weights
        self.category_weights = {
            "museum": 0.8,       # Museums can be tiring if too many in one day
            "art": 0.8,          # Art galleries can be tiring if too many in one day
            "landmark": 1.0,      
            "historical": 1.0,
            "cultural": 1.0,
            "entertainment": 1.2, # Entertainment venues add variety
            "outdoor": 1.2,       # Outdoor activities add variety
            "park": 1.0,
            "food": 1.5,          # Food stops are good to include
            "shopping": 0.7,      # Shopping is lower priority
            "misc": 0.5
        }
    
    def group_attractions(self, 
                         cities: List[Dict[str, Any]], 
                         days_per_city: Dict[str, int],
                         preferences: List[str] = None,
                         max_hours_per_day: float = 8.0) -> Dict[str, List[List[Dict[str, Any]]]]:
        """
        Group attractions into daily plans for each city.
        
        Args:
            cities: List of city data dictionaries
            days_per_city: Dictionary mapping city_id to number of days allocated
            preferences: Optional list of user preferences
            max_hours_per_day: Maximum hours of activities per day
            
        Returns:
            Dictionary mapping city_id to list of daily attraction groups
        """
        logger.info("Grouping attractions for daily plans")
        
        all_groups = {}
        
        # Process each city
        for city in cities:
            city_id = city.get("city_id")
            
            # Skip if city not in allocation plan
            if city_id not in days_per_city:
                continue
            
            # Get days allocated to this city
            num_days = days_per_city[city_id]
            
            if num_days <= 0:
                continue
            
            # Get attractions for this city
            attractions = city.get("attractions", [])
            
            if not attractions:
                all_groups[city_id] = [[] for _ in range(num_days)]
                continue
            
            # Group attractions for this city
            city_groups = self._group_city_attractions(
                city_id=city_id, 
                attractions=attractions, 
                num_days=num_days, 
                preferences=preferences,
                max_hours_per_day=max_hours_per_day
            )
            
            all_groups[city_id] = city_groups
        
        return all_groups
    
    def _group_city_attractions(self, 
                               city_id: str, 
                               attractions: List[Dict[str, Any]], 
                               num_days: int,
                               preferences: List[str] = None,
                               max_hours_per_day: float = 8.0) -> List[List[Dict[str, Any]]]:
        """
        Group attractions for a single city.
        
        Args:
            city_id: City ID
            attractions: List of attractions in the city
            num_days: Number of days allocated to this city
            preferences: Optional list of user preferences
            max_hours_per_day: Maximum hours of activities per day
            
        Returns:
            List of daily attraction groups
        """
        # If semantic grouping is enabled and transformer handler is available
        if self.use_semantic_grouping and self.transformer_handler is not None:
            return self._semantic_grouping(
                city_id=city_id,
                attractions=attractions,
                num_days=num_days,
                preferences=preferences,
                max_hours_per_day=max_hours_per_day
            )
        else:
            # Fallback to basic grouping
            return self._basic_grouping(
                attractions=attractions,
                num_days=num_days,
                preferences=preferences,
                max_hours_per_day=max_hours_per_day
            )
    
    def _semantic_grouping(self, 
                          city_id: str, 
                          attractions: List[Dict[str, Any]], 
                          num_days: int,
                          preferences: List[str] = None,
                          max_hours_per_day: float = 8.0) -> List[List[Dict[str, Any]]]:
        """
        Group attractions using semantic understanding with transformer embeddings.
        
        Args:
            city_id: City ID
            attractions: List of attractions in the city
            num_days: Number of days allocated to this city
            preferences: Optional list of user preferences
            max_hours_per_day: Maximum hours of activities per day
            
        Returns:
            List of daily attraction groups
        """
        logger.info(f"Using semantic grouping for {city_id}")
        
        # Initialize daily groups
        daily_groups = [[] for _ in range(num_days)]
        daily_hours = [0.0 for _ in range(num_days)]
        
        # Calculate similarity matrix
        similarity_matrix = self.transformer_handler.calculate_similarity_matrix(
            city_id=city_id,
            attractions=attractions
        )
        
        # Find complementary attractions
        complementary = self.transformer_handler.find_complementary_attractions(
            city_id=city_id,
            attractions=attractions
        )
        
        # Score attractions by preference matches
        attraction_scores = self._score_attractions_by_preferences(
            city_id=city_id,
            attractions=attractions,
            preferences=preferences
        )
        
        # Sort attractions by preference score (highest first)
        sorted_indices = sorted(
            range(len(attractions)), 
            key=lambda i: attraction_scores[i], 
            reverse=True
        )
        
        # First, allocate top attractions (one per day)
        for day in range(min(num_days, len(sorted_indices))):
            idx = sorted_indices[day]
            attraction = attractions[idx]
            
            # Add to daily group
            daily_groups[day].append(attraction)
            
            # Update hours
            duration = self._get_attraction_duration(attraction)
            daily_hours[day] += duration
        
        # Keep track of assigned attractions
        assigned = set(sorted_indices[:num_days])
        
        # For each day, find complementary attractions
        for day in range(num_days):
            if not daily_groups[day]:
                continue
                
            # Get seed attractions in this day
            seed_attractions = [a.get("attraction_id", "") for a in daily_groups[day]]
            
            # Potential complementary attractions for this day
            potential_attractions = set()
            for seed_id in seed_attractions:
                if seed_id in complementary:
                    potential_attractions.update(complementary[seed_id])
            
            # Find indices of potential attractions
            potential_indices = [
                i for i in range(len(attractions)) 
                if attractions[i].get("attraction_id", "") in potential_attractions
                and i not in assigned
            ]
            
            # Sort potential attractions by similarity to seed attractions
            if potential_indices:
                # Calculate average similarity to already assigned attractions
                similarities = []
                for idx in potential_indices:
                    avg_similarity = 0.0
                    for seed_idx in [i for i in range(len(attractions)) if i in assigned]:
                        avg_similarity += similarity_matrix[idx, seed_idx]
                    avg_similarity /= len(assigned)
                    similarities.append((idx, avg_similarity))
                
                # Sort by similarity (highest first)
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                # Add complementary attractions until day is full
                for idx, _ in similarities:
                    attraction = attractions[idx]
                    duration = self._get_attraction_duration(attraction)
                    
                    # Check if adding this attraction exceeds max hours
                    if daily_hours[day] + duration <= max_hours_per_day:
                        daily_groups[day].append(attraction)
                        daily_hours[day] += duration
                        assigned.add(idx)
        
        # Allocate remaining attractions
        remaining_indices = [i for i in range(len(attractions)) if i not in assigned]
        
        for idx in remaining_indices:
            attraction = attractions[idx]
            duration = self._get_attraction_duration(attraction)
            
            # Find the day with the least hours
            best_day = min(range(num_days), key=lambda d: daily_hours[d])
            
            # Check if adding this attraction exceeds max hours
            if daily_hours[best_day] + duration <= max_hours_per_day:
                daily_groups[best_day].append(attraction)
                daily_hours[best_day] += duration
            else:
                # Try to find any day that can fit this attraction
                for day in range(num_days):
                    if daily_hours[day] + duration <= max_hours_per_day:
                        daily_groups[day].append(attraction)
                        daily_hours[day] += duration
                        break
        
        # Balance categories within each day
        for day in range(num_days):
            daily_groups[day] = self._balance_categories(daily_groups[day], max_hours_per_day)
        
        return daily_groups
    
    def _basic_grouping(self, 
                       attractions: List[Dict[str, Any]], 
                       num_days: int,
                       preferences: List[str] = None,
                       max_hours_per_day: float = 8.0) -> List[List[Dict[str, Any]]]:
        """
        Basic grouping of attractions without semantic understanding.
        
        Args:
            attractions: List of attractions in the city
            num_days: Number of days allocated to this city
            preferences: Optional list of user preferences
            max_hours_per_day: Maximum hours of activities per day
            
        Returns:
            List of daily attraction groups
        """
        logger.info("Using basic grouping (no transformer model)")
        
        # Initialize daily groups
        daily_groups = [[] for _ in range(num_days)]
        daily_hours = [0.0 for _ in range(num_days)]
        
        # Group attractions by category
        attractions_by_category = defaultdict(list)
        for attraction in attractions:
            category = attraction.get("category", "misc")
            attractions_by_category[category].append(attraction)
        
        # Prioritize categories based on preferences
        category_priority = self._prioritize_categories(attractions_by_category.keys(), preferences)
        
        # Distribute attractions across days
        day_index = 0
        
        # First, ensure each day has a mix of categories
        for category in category_priority:
            category_attractions = attractions_by_category[category]
            
            # Skip if no attractions in this category
            if not category_attractions:
                continue
                
            # Take one attraction from each priority category for each day
            for i in range(min(num_days, len(category_attractions))):
                attraction = category_attractions[i]
                duration = self._get_attraction_duration(attraction)
                
                # If adding this attraction would exceed max hours, skip to next day
                if daily_hours[day_index] + duration > max_hours_per_day:
                    day_index = (day_index + 1) % num_days
                
                # Add attraction to current day
                daily_groups[day_index].append(attraction)
                daily_hours[day_index] += duration
                
                # Move to next day
                day_index = (day_index + 1) % num_days
        
        # Mark attractions that have been assigned
        assigned_attractions = set()
        for group in daily_groups:
            for attraction in group:
                assigned_attractions.add(attraction.get("attraction_id", ""))
        
        # Assign remaining attractions
        remaining = [a for a in attractions if a.get("attraction_id", "") not in assigned_attractions]
        
        for attraction in remaining:
            duration = self._get_attraction_duration(attraction)
            
            # Find the day with the most available time
            best_day = min(range(num_days), key=lambda d: daily_hours[d])
            
            # Check if adding this attraction exceeds max hours
            if daily_hours[best_day] + duration <= max_hours_per_day:
                daily_groups[best_day].append(attraction)
                daily_hours[best_day] += duration
        
        # Balance categories within each day
        for day in range(num_days):
            daily_groups[day] = self._balance_categories(daily_groups[day], max_hours_per_day)
        
        return daily_groups
    
    def _prioritize_categories(self, categories: List[str], preferences: List[str] = None) -> List[str]:
        """
        Prioritize attraction categories based on preferences.
        
        Args:
            categories: List of category names
            preferences: Optional list of user preferences
            
        Returns:
            Sorted list of categories by priority
        """
        # Default priority weights
        priority = {
            "landmark": 10,      # Famous landmarks are must-sees
            "historical": 9,     # Historical sites are typically important
            "museum": 8,         # Museums can be major attractions
            "cultural": 7,
            "art": 6,
            "entertainment": 5,
            "outdoor": 4,
            "park": 3,
            "food": 2,
            "shopping": 1,
            "misc": 0
        }
        
        # Adjust priority based on preferences
        if preferences:
            for category in categories:
                # Boost priority if category matches any preference
                if any(pref.lower() in category.lower() for pref in preferences):
                    priority[category] = priority.get(category, 0) + 10
        
        # Sort categories by priority
        return sorted(categories, key=lambda c: priority.get(c, 0), reverse=True)
    
    def _score_attractions_by_preferences(self, 
                                         city_id: str,
                                         attractions: List[Dict[str, Any]], 
                                         preferences: List[str] = None) -> List[float]:
        """
        Score attractions based on how well they match user preferences.
        
        Args:
            city_id: City ID
            attractions: List of attractions
            preferences: Optional list of user preferences
            
        Returns:
            List of scores for each attraction
        """
        scores = [1.0] * len(attractions)  # Default score
        
        # If no preferences provided, score based on category weights
        if not preferences:
            for i, attraction in enumerate(attractions):
                category = attraction.get("category", "misc")
                scores[i] = self.category_weights.get(category, 0.5)
            return scores
            
        # If transformer handler is available, use it for preference matching
        if self.transformer_handler is not None:
            try:
                # Find attractions by preference
                matches = self.transformer_handler.find_attractions_by_preference(
                    city_id=city_id,
                    attractions=attractions,
                    preferences=preferences,
                    top_k=len(attractions)  # Get scores for all attractions
                )
                
                # Convert matches to scores
                for i, attraction in enumerate(attractions):
                    attr_id = attraction.get("attraction_id", "")
                    
                    # Check if this attraction appears in preference matches
                    for pref, matched_ids in matches.items():
                        if attr_id in matched_ids:
                            # Higher score for closer matches (earlier in the list)
                            position = matched_ids.index(attr_id)
                            match_score = 1.0 + (len(matched_ids) - position) / len(matched_ids)
                            scores[i] = max(scores[i], match_score)
            except Exception as e:
                logger.warning(f"Error scoring attractions by transformer: {e}")
        
        # Fallback to simple category matching
        for i, attraction in enumerate(attractions):
            category = attraction.get("category", "misc")
            
            # Check if category matches any preference
            if any(pref.lower() in category.lower() for pref in preferences):
                scores[i] += 1.0
                
            # Apply category weight
            scores[i] *= self.category_weights.get(category, 0.5)
        
        return scores
    
    def _get_attraction_duration(self, attraction: Dict[str, Any]) -> float:
        """
        Get the duration of an attraction in hours.
        
        Args:
            attraction: Attraction data dictionary
            
        Returns:
            Duration in hours
        """
        # Try to get duration from attraction data
        if "visit_duration_hrs" in attraction:
            try:
                return float(attraction["visit_duration_hrs"])
            except (ValueError, TypeError):
                pass
        
        # Fallback to default duration based on category
        category = attraction.get("category", "misc")
        return self.default_durations.get(category, 1.0)
    
    def _balance_categories(self, 
                           attractions: List[Dict[str, Any]], 
                           max_hours_per_day: float) -> List[Dict[str, Any]]:
        """
        Balance categories within a daily group to ensure variety.
        
        Args:
            attractions: List of attractions for a day
            max_hours_per_day: Maximum hours of activities per day
            
        Returns:
            Balanced list of attractions
        """
        if len(attractions) <= 1:
            return attractions
        
        # Count categories
        category_counts = defaultdict(int)
        for attraction in attractions:
            category = attraction.get("category", "misc")
            category_counts[category] += 1
        
        # If no category has more than 2 attractions, no need to balance
        if all(count <= 2 for count in category_counts.values()):
            return attractions
        
        # Find overrepresented categories
        overrepresented = [cat for cat, count in category_counts.items() if count > 2]
        
        if not overrepresented:
            return attractions
        
        # Sort attractions by category and then by duration
        sorted_attractions = sorted(
            attractions,
            key=lambda a: (
                a.get("category", "misc") in overrepresented,
                -self._get_attraction_duration(a)
            ),
            reverse=True
        )
        
        # Keep track of total hours and categories used
        total_hours = 0.0
        used_categories = set()
        balanced_attractions = []
        
        # First pass: Add one attraction from each category
        for attraction in sorted_attractions:
            category = attraction.get("category", "misc")
            
            if category not in used_categories:
                duration = self._get_attraction_duration(attraction)
                
                if total_hours + duration <= max_hours_per_day:
                    balanced_attractions.append(attraction)
                    total_hours += duration
                    used_categories.add(category)
        
        # Second pass: Fill remaining time with attractions from different categories
        for attraction in sorted_attractions:
            if attraction not in balanced_attractions:
                duration = self._get_attraction_duration(attraction)
                
                if total_hours + duration <= max_hours_per_day:
                    balanced_attractions.append(attraction)
                    total_hours += duration
        
        return balanced_attractions

# Example usage
if __name__ == "__main__":
    # Example data
    cities_data = [
        {
            "city_id": "paris",
            "name": "Paris",
            "country": "France",
            "attractions": [
                {
                    "attraction_id": "eiffel_tower",
                    "name": "Eiffel Tower",
                    "category": "landmark",
                    "description": "Iconic iron tower offering city views from various observation levels.",
                    "visit_duration_hrs": 2.5
                },
                {
                    "attraction_id": "louvre_museum",
                    "name": "Louvre Museum",
                    "category": "museum",
                    "description": "World's largest art museum and historic monument housing the Mona Lisa.",
                    "visit_duration_hrs": 4.0
                },
                {
                    "attraction_id": "notre_dame",
                    "name": "Notre-Dame Cathedral",
                    "category": "historical",
                    "description": "Medieval Catholic cathedral known for its French Gothic architecture.",
                    "visit_duration_hrs": 1.5
                }
            ]
        }
    ]
    
    # City allocation (days per city)
    days_per_city = {"paris": 2}
    
    # Initialize attraction grouper (without transformer)
    grouper = AttractionGrouper(use_semantic_grouping=False)
    
    # Group attractions
    groups = grouper.group_attractions(
        cities=cities_data,
        days_per_city=days_per_city,
        preferences=["art", "history"]
    )
    
    # Print results
    for city_id, daily_groups in groups.items():
        print(f"\nCity: {city_id}")
        
        for day, attractions in enumerate(daily_groups, 1):
            print(f"  Day {day}:")
            
            total_duration = 0.0
            for attraction in attractions:
                duration = attraction.get("visit_duration_hrs", 1.0)
                total_duration += duration
                
                print(f"    - {attraction['name']} ({attraction['category']}, {duration} hrs)")
                
            print(f"    Total: {total_duration} hours")