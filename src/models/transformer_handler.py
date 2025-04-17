"""
Transformer Handler Module

This module manages transformer model operations for the Itinerary Planning System.
It handles embedding generation and similarity calculations for attraction descriptions.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Union, Optional
import numpy as np
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check if sentence-transformers is available, otherwise show a helpful message
try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    logger.warning(
        "sentence-transformers package not found. "
        "Install it with: pip install sentence-transformers"
    )

class TransformerHandler:
    """
    Handles transformer model operations for semantic understanding of attractions.
    
    This class loads a pre-trained sentence transformer model and uses it to:
    1. Generate embeddings for attraction descriptions
    2. Calculate similarity scores between attractions
    3. Identify attractions that complement each other
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[str] = None):
        """
        Initialize the transformer handler.
        
        Args:
            model_name: Name of the pre-trained sentence transformer model
            cache_dir: Directory to cache model files
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.embeddings = {}
        
        if TRANSFORMER_AVAILABLE:
            self._load_model()
        else:
            logger.warning("Transformer functionality will be limited without the sentence-transformers package")
    
    def _load_model(self) -> None:
        """Load the pre-trained sentence transformer model."""
        try:
            logger.info(f"Loading transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)
            logger.info(f"Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading transformer model: {e}")
            self.model = None
    
    def generate_embeddings(self, cities: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for all attractions in the provided cities.
        
        Args:
            cities: List of city data dictionaries
            
        Returns:
            Dictionary mapping attraction_id to its embedding vector
        """
        if not TRANSFORMER_AVAILABLE or self.model is None:
            logger.warning("Cannot generate embeddings: model not available")
            # Return empty embeddings
            return {}
        
        logger.info("Generating attraction embeddings")
        
        # Collect all attraction descriptions and IDs
        attractions = []
        attraction_ids = []
        
        for city in cities:
            city_id = city.get("city_id", "")
            for attraction in city.get("attractions", []):
                attraction_id = attraction.get("attraction_id", "")
                
                # Skip if no ID
                if not attraction_id:
                    continue
                
                # Create a full ID including city to ensure uniqueness
                full_id = f"{city_id}_{attraction_id}"
                
                # Get the description and name
                description = attraction.get("description", "")
                name = attraction.get("name", "")
                category = attraction.get("category", "")
                
                # Create a rich text for embedding that includes all relevant information
                rich_text = f"{name}. {description} Category: {category}"
                
                attractions.append(rich_text)
                attraction_ids.append(full_id)
        
        # Generate embeddings if there are attractions
        if attractions:
            try:
                # Generate embeddings in batches to save memory
                batch_size = 32
                all_embeddings = []
                
                for i in range(0, len(attractions), batch_size):
                    batch = attractions[i:i+batch_size]
                    embeddings = self.model.encode(batch, show_progress_bar=False)
                    all_embeddings.extend(embeddings)
                
                # Map attraction IDs to their embeddings
                self.embeddings = {
                    attraction_ids[i]: all_embeddings[i] 
                    for i in range(len(attraction_ids))
                }
                
                logger.info(f"Generated embeddings for {len(self.embeddings)} attractions")
                
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                self.embeddings = {}
        
        return self.embeddings
    
    def calculate_similarity_matrix(self, city_id: str, attractions: List[Dict[str, Any]]) -> np.ndarray:
        """
        Calculate similarity matrix for attractions within a city.
        
        Args:
            city_id: ID of the city
            attractions: List of attraction data dictionaries
            
        Returns:
            2D numpy array containing similarity scores between attractions
        """
        if not self.embeddings:
            logger.warning("No embeddings available for similarity calculation")
            # Return identity matrix (each attraction only similar to itself)
            return np.eye(len(attractions))
        
        n = len(attractions)
        similarity_matrix = np.zeros((n, n))
        
        # Get attraction IDs for this city
        attraction_ids = [
            f"{city_id}_{attr.get('attraction_id', '')}" 
            for attr in attractions
        ]
        
        # Calculate cosine similarity between all pairs
        for i in range(n):
            for j in range(n):
                # Get embeddings for both attractions
                emb_i = self.embeddings.get(attraction_ids[i])
                emb_j = self.embeddings.get(attraction_ids[j])
                
                if emb_i is not None and emb_j is not None:
                    # Calculate cosine similarity
                    similarity = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j))
                    similarity_matrix[i, j] = similarity
                else:
                    # If embeddings not available, use identity matrix behavior
                    similarity_matrix[i, j] = 1.0 if i == j else 0.0
        
        return similarity_matrix
    
    def find_complementary_attractions(self, 
                                       city_id: str, 
                                       attractions: List[Dict[str, Any]], 
                                       threshold: float = 0.6) -> Dict[str, List[str]]:
        """
        Find complementary attractions based on similarity scores.
        
        Args:
            city_id: ID of the city
            attractions: List of attraction data dictionaries
            threshold: Similarity threshold for considering attractions as complementary
            
        Returns:
            Dictionary mapping attraction_id to list of complementary attraction_ids
        """
        complementary_map = defaultdict(list)
        
        # Calculate similarity matrix
        similarity_matrix = self.calculate_similarity_matrix(city_id, attractions)
        
        # Find complementary attractions based on similarity
        for i in range(len(attractions)):
            attr_i = attractions[i]
            attr_i_id = attr_i.get("attraction_id", "")
            
            if not attr_i_id:
                continue
                
            for j in range(len(attractions)):
                if i == j:
                    continue
                    
                attr_j = attractions[j]
                attr_j_id = attr_j.get("attraction_id", "")
                
                if not attr_j_id:
                    continue
                
                # Check if the similarity exceeds the threshold
                if similarity_matrix[i, j] >= threshold:
                    complementary_map[attr_i_id].append(attr_j_id)
        
        return dict(complementary_map)
    
    def get_category_embeddings(self, categories: List[str]) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for attraction categories.
        
        Args:
            categories: List of category names
            
        Returns:
            Dictionary mapping category name to its embedding vector
        """
        if not TRANSFORMER_AVAILABLE or self.model is None:
            logger.warning("Cannot generate category embeddings: model not available")
            return {}
        
        category_embeddings = {}
        
        try:
            # Generate enriched category descriptions
            category_descriptions = [
                f"Attractions related to {category}, including {category} sites and {category} activities."
                for category in categories
            ]
            
            # Generate embeddings
            embeddings = self.model.encode(category_descriptions, show_progress_bar=False)
            
            # Map categories to embeddings
            category_embeddings = {
                categories[i]: embeddings[i] 
                for i in range(len(categories))
            }
            
        except Exception as e:
            logger.error(f"Error generating category embeddings: {e}")
        
        return category_embeddings
    
    def find_attractions_by_preference(self, 
                                      city_id: str, 
                                      attractions: List[Dict[str, Any]], 
                                      preferences: List[str],
                                      top_k: int = 5) -> Dict[str, List[str]]:
        """
        Find attractions that match user preferences.
        
        Args:
            city_id: ID of the city
            attractions: List of attraction data dictionaries
            preferences: List of user preferences
            top_k: Number of top attractions to return per preference
            
        Returns:
            Dictionary mapping preference to list of matching attraction_ids
        """
        if not self.embeddings or not TRANSFORMER_AVAILABLE or self.model is None:
            logger.warning("Cannot match preferences: embeddings or model not available")
            return {}
        
        preference_matches = {}
        
        try:
            # Generate embeddings for preferences
            preference_embeddings = self.model.encode(preferences, show_progress_bar=False)
            
            # Get attraction IDs and embeddings for this city
            attraction_ids = [
                attr.get("attraction_id", "") 
                for attr in attractions
            ]
            
            attraction_embeddings = [
                self.embeddings.get(f"{city_id}_{attr_id}")
                for attr_id in attraction_ids
            ]
            
            # Filter out None embeddings
            valid_indices = [
                i for i, emb in enumerate(attraction_embeddings) 
                if emb is not None
            ]
            
            filtered_attr_ids = [attraction_ids[i] for i in valid_indices]
            filtered_embeddings = [attraction_embeddings[i] for i in valid_indices]
            
            # For each preference, find matching attractions
            for i, preference in enumerate(preferences):
                pref_embedding = preference_embeddings[i]
                
                # Calculate similarity scores
                scores = [
                    np.dot(pref_embedding, attr_emb) / (np.linalg.norm(pref_embedding) * np.linalg.norm(attr_emb))
                    for attr_emb in filtered_embeddings
                ]
                
                # Get top-k attractions
                if scores:
                    top_indices = np.argsort(scores)[-top_k:][::-1]
                    preference_matches[preference] = [filtered_attr_ids[j] for j in top_indices]
                else:
                    preference_matches[preference] = []
                
        except Exception as e:
            logger.error(f"Error matching preferences: {e}")
        
        return preference_matches
    
    def save_embeddings(self, filepath: Union[str, Path]) -> bool:
        """
        Save embeddings to a file.
        
        Args:
            filepath: Path where to save the embeddings
            
        Returns:
            Boolean indicating if saving was successful
        """
        filepath = Path(filepath)
        
        try:
            # Create directory if it doesn't exist
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Save embeddings
            np.savez(
                filepath, 
                **{key: np.array(val) for key, val in self.embeddings.items()}
            )
            
            logger.info(f"Embeddings saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            return False
    
    def load_embeddings(self, filepath: Union[str, Path]) -> bool:
        """
        Load embeddings from a file.
        
        Args:
            filepath: Path to the embeddings file
            
        Returns:
            Boolean indicating if loading was successful
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.error(f"Embeddings file not found: {filepath}")
            return False
        
        try:
            # Load embeddings
            loaded = np.load(filepath)
            
            # Convert to dictionary
            self.embeddings = {key: loaded[key] for key in loaded.files}
            
            logger.info(f"Loaded embeddings for {len(self.embeddings)} attractions from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return False

# Example usage
# Example usage
if __name__ == "__main__":
    import json
    from pathlib import Path
    import time
    from tqdm import tqdm  # For progress bars, install with pip install tqdm
    
    # Load real data from the fixed_cities.json file
    data_path = Path("data/raw/fixed_cities.json")
    
    try:
        print("Loading data...")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            cities_data = data.get("cities", [])
        
        print(f"Loaded data for {len(cities_data)} cities")
        
        # Calculate total attractions to process
        total_attractions = sum(len(city.get("attractions", [])) for city in cities_data)
        print(f"Total attractions to process: {total_attractions}")
        
        # Initialize transformer handler
        handler = TransformerHandler(model_name="all-MiniLM-L6-v2")
        
        if TRANSFORMER_AVAILABLE and handler.model is not None:
            # Generate embeddings for ALL cities (this will take time)
            start_time = time.time()
            print(f"Generating embeddings for all {len(cities_data)} cities (this may take several minutes)...")
            
            # Process in batches to avoid memory issues
            batch_size = 10  # Process 10 cities at a time
            all_embeddings = {}
            
            for i in range(0, len(cities_data), batch_size):
                batch = cities_data[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(cities_data) + batch_size - 1)//batch_size}...")
                
                batch_embeddings = handler.generate_embeddings(batch)
                all_embeddings.update(batch_embeddings)
                
                # Optional: Save intermediate results
                handler.embeddings = all_embeddings
                handler.save_embeddings("data/embeddings/attraction_embeddings_partial.npz")
            
            # Update handler with all embeddings
            handler.embeddings = all_embeddings
            
            # Calculate processing time
            elapsed_time = time.time() - start_time
            print(f"Embedding generation completed in {elapsed_time:.2f} seconds")
            print(f"Generated embeddings for {len(all_embeddings)} attractions")
            
            # Create embeddings directory if it doesn't exist
            embeddings_dir = Path("data/embeddings")
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            
            # Save all embeddings
            handler.save_embeddings("data/embeddings/attraction_embeddings_full.npz")
            
            # Perform comprehensive analysis
            print("\n======= Attraction Analysis =======")
            
            # Find cities with the most attractions
            city_attraction_counts = [(city["city_id"], city.get("name", city["city_id"]), 
                                     len(city.get("attractions", []))) 
                                     for city in cities_data]
            
            top_cities = sorted(city_attraction_counts, key=lambda x: x[2], reverse=True)[:10]
            
            print("\nTop 10 cities by number of attractions:")
            for i, (city_id, city_name, count) in enumerate(top_cities, 1):
                print(f"{i}. {city_name}: {count} attractions")
            
            # Select top cities for detailed analysis
            analysis_cities = [city for city in cities_data 
                             if city["city_id"] in [c[0] for c in top_cities[:5]]]
            
            # Analyze complementary attractions for top cities
            print("\nAnalyzing complementary attractions for top cities...")
            
            for city in analysis_cities:
                city_id = city["city_id"]
                city_name = city.get("name", city_id)
                attractions = city.get("attractions", [])
                
                if len(attractions) < 2:
                    continue
                
                print(f"\n--- {city_name} ({len(attractions)} attractions) ---")
                
                # Calculate similarity matrix
                similarity_matrix = handler.calculate_similarity_matrix(
                    city_id=city_id,
                    attractions=attractions
                )
                
                # Find complementary attractions with multiple thresholds
                for threshold in [0.8, 0.7, 0.6]:
                    complementary = handler.find_complementary_attractions(
                        city_id=city_id,
                        attractions=attractions,
                        threshold=threshold
                    )
                    
                    if complementary:
                        print(f"\nFound {len(complementary)} attractions with complementary matches (threshold {threshold})")
                        
                        # Find the most connected attractions (with most complementary matches)
                        most_connected = sorted(complementary.items(), key=lambda x: len(x[1]), reverse=True)[:3]
                        
                        print("\nMost connected attractions:")
                        for attr_id, compl_list in most_connected:
                            # Find the attraction name
                            attr_name = next((a.get("name", attr_id) for a in attractions 
                                           if a.get("attraction_id") == attr_id), attr_id)
                            
                            print(f"• {attr_name} complements with {len(compl_list)} other attractions")
                        
                        break  # Stop after finding a threshold with results
            
            # Analyze preferences matching
            print("\n======= Preference Matching Analysis =======")
            
            # Common travel preferences
            preferences = ["art", "history", "food", "nature", "architecture", 
                         "relaxation", "adventure", "shopping", "local culture"]
            
            # Pick one of the top cities for preference analysis
            analysis_city = analysis_cities[0]
            city_id = analysis_city["city_id"]
            city_name = analysis_city.get("name", city_id)
            attractions = analysis_city.get("attractions", [])
            
            print(f"\nAnalyzing preference matches for {city_name}...")
            
            # Find attractions by preference
            preference_matches = handler.find_attractions_by_preference(
                city_id=city_id,
                attractions=attractions,
                preferences=preferences,
                top_k=3  # Top 3 attractions per preference
            )
            
            # Display results
            for preference, attr_ids in preference_matches.items():
                if attr_ids:
                    print(f"\nTop attractions for '{preference}' preference:")
                    
                    for attr_id in attr_ids:
                        # Find the attraction details
                        attraction = next((a for a in attractions if a.get("attraction_id") == attr_id), None)
                        
                        if attraction:
                            print(f"• {attraction.get('name', attr_id)} ({attraction.get('category', 'misc')})")
            
            # Generate category embeddings and find relationships
            categories = set()
            for city in cities_data:
                for attraction in city.get("attractions", []):
                    if "category" in attraction:
                        categories.add(attraction["category"])
            
            print(f"\nFound {len(categories)} unique attraction categories")
            
            # Get category embeddings
            category_embeddings = handler.get_category_embeddings(list(categories))
            
            # Find similar categories
            if category_embeddings and len(category_embeddings) > 1:
                print("\nSimilar category pairs:")
                
                category_list = list(category_embeddings.keys())
                
                for i in range(len(category_list)):
                    for j in range(i+1, len(category_list)):
                        cat1 = category_list[i]
                        cat2 = category_list[j]
                        
                        emb1 = category_embeddings[cat1]
                        emb2 = category_embeddings[cat2]
                        
                        # Calculate cosine similarity
                        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                        
                        if similarity > 0.7:  # Only show highly similar categories
                            print(f"• {cat1} ↔ {cat2}: {similarity:.2f}")
            
            print("\nAnalysis complete! Embeddings saved to data/embeddings/attraction_embeddings_full.npz")
            
        else:
            print("Transformer model not available. Install sentence-transformers package.")
    
    except FileNotFoundError:
        print(f"Data file not found: {data_path}")
        print("Make sure the file exists at the specified location.")
    
    except json.JSONDecodeError:
        print(f"Error parsing JSON from: {data_path}")
        print("Make sure the file contains valid JSON.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()