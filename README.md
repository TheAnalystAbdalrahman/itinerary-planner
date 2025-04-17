# Itinerary Planner

An intelligent multi-city travel itinerary planning system that generates optimized travel plans using transformer-based semantic understanding and constraint optimization algorithms.

## 🌟 Features

- **Transformer-Powered Semantic Understanding**: Groups attractions that are thematically related
- **Multi-City Support**: Plan trips across multiple cities with optimized transitions
- **Custom Preferences**: Tailor itineraries to specific interests like art, history, food, etc.
- **Adjustable Travel Pace**: Choose between slow, moderate, and fast travel styles
- **Route Optimization**: Efficiently sequence attractions to minimize travel time
- **Smart Allocation**: Distributes days across cities based on attraction importance
- **Practical Details**: Includes meal suggestions, travel tips, and logistics information

## 📋 Table of Contents

- [Installation](#-installation)
- [Usage](#-usage)
- [Command-line Parameters](#-command-line-parameters)
- [Example Prompts](#-example-prompts)
- [Data Format](#-data-format)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Contributing](#-contributing)

## 🔧 Installation

### Prerequisites

- Python 3.8+
- Pip or Conda for package management

### Steps

1. **Clone the repository**

```bash
git clone https://github.com/TheAnalystAbdalrahman/itinerary-planner.git
cd itinerary-planner
```

2. **Create a virtual environment**

Using venv:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Using conda:
```bash
conda create -n TravelForg python=3.10
conda activate TravelForg
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download pre-trained transformer model**

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

5. **Generate embeddings (one-time setup)**

```bash
python src/main.py --use-transformer
```

This will generate attraction embeddings and save them to `data/embeddings/`.

## 🚀 Usage

The basic command to generate an itinerary:

```bash
python src/main.py --use-transformer --days 7 --preferences art history food --output my_itinerary.json
```

To visualize the system running with full logging:

```bash
python src/main.py --use-transformer --days 5 --preferences landmark museum --debug
```

## 🎛️ Command-line Parameters

| Parameter | Short | Description | Default |
|-----------|-------|-------------|---------|
| `--input` | `-i` | Path to input JSON file | data/raw/fixed_cities.json |
| `--output` | `-o` | Path to output JSON file | itinerary.json |
| `--days` | `-d` | Number of days for the trip | 5 |
| `--preferences` | `-p` | Space-separated list of preferences | None |
| `--pace` | | Travel pace (slow, moderate, fast) | moderate |
| `--use-transformer` | | Enable transformer model for semantic understanding | False |
| `--embeddings` | `-e` | Path to pre-generated attraction embeddings | data/embeddings/attraction_embeddings.npz |
| `--validate-only` | | Only validate the input data | False |
| `--skip-validation` | | Skip validation of input data | False |
| `--debug` | | Enable debug logging | False |

## 💡 Example Prompts

### By Interest

#### Cultural Exploration
```bash
# Art & Museum Tour
python src/main.py --use-transformer --days 7 --preferences art museum cultural --output art_museum_tour.json

# Historical Journey
python src/main.py --use-transformer --days 10 --preferences historical landmark ancient --output historical_journey.json

# Architectural Wonders
python src/main.py --use-transformer --days 6 --preferences architecture landmark historical --output architectural_wonders.json
```

#### Nature & Outdoor
```bash
# Outdoor Adventure
python src/main.py --use-transformer --days 8 --preferences outdoor adventure hiking --output outdoor_adventure.json

# Nature & Wildlife
python src/main.py --use-transformer --days 7 --preferences nature wildlife park --output nature_wildlife.json

# Beach & Coastal Tour
python src/main.py --use-transformer --days 6 --preferences beach coastal relaxation --output beach_vacation.json
```

#### Food & Lifestyle
```bash
# Culinary Journey
python src/main.py --use-transformer --days 5 --preferences food culinary local --output culinary_journey.json

# Shopping & Entertainment
python src/main.py --use-transformer --days 4 --preferences shopping entertainment nightlife --output shopping_entertainment.json

# Urban Exploration
python src/main.py --use-transformer --days 7 --preferences urban modern city --output urban_exploration.json
```

### By Trip Duration

```bash
# Weekend Getaway (3 days)
python src/main.py --use-transformer --days 3 --preferences landmark food entertainment --pace fast --output weekend_getaway.json

# One-Week Vacation (7 days)
python src/main.py --use-transformer --days 7 --preferences balanced landmark cultural food --output one_week_vacation.json

# Two-Week Adventure (14 days)
python src/main.py --use-transformer --days 14 --preferences mixed variety cultural --output two_week_adventure.json
```

### By Travel Pace

```bash
# Relaxed Exploration (slow pace)
python src/main.py --use-transformer --days 7 --preferences cultural relaxation --pace slow --output relaxed_exploration.json

# Balanced Itinerary (moderate pace)
python src/main.py --use-transformer --days 7 --preferences variety balanced --pace moderate --output balanced_itinerary.json

# Fast-Paced Tour (fast pace)
python src/main.py --use-transformer --days 7 --preferences landmark highlight --pace fast --output fast_paced_tour.json
```

### Using Pre-generated Embeddings

```bash
# Use full embeddings
python src/main.py --use-transformer --embeddings data/embeddings/attraction_embeddings_full.npz --days 7 --output full_embeddings_itinerary.json

# Use partial embeddings
python src/main.py --use-transformer --embeddings data/embeddings/attraction_embeddings_partial.npz --days 5 --output partial_embeddings_itinerary.json
```

## 📄 Data Format

### Input Format

The system expects a JSON file with cities and attractions:

```json
{
  "cities": [
    {
      "city_id": "paris",
      "name": "Paris",
      "country": "France",
      "attractions": [
        {
          "attraction_id": "eiffel_tower",
          "name": "Eiffel Tower",
          "category": "landmark",
          "description": "Iconic iron tower offering city views...",
          "visit_duration_hrs": 2.5,
          "latitude": 48.8584,
          "longitude": 2.2945
        },
        // More attractions...
      ]
    },
    // More cities...
  ],
  "parameters": {
    "total_days": 5,
    "travel_pace": "moderate",
    "preferences": ["art", "history", "food"]
  }
}
```

### Output Format

The system generates a comprehensive itinerary in JSON format:

```json
{
  "itinerary": [
    {
      "day": 1,
      "city": "Paris",
      "attractions": [
        {
          "name": "Eiffel Tower",
          "start_time": "09:00",
          "end_time": "11:30",
          "notes": "Best views in morning"
        },
        // More attractions...
      ],
      "meals": { /* meal suggestions */ },
      "practical_tips": [ /* useful tips */ ]
    },
    // More days...
  ],
  "travel_segments": [
    {
      "day": 3,
      "from_city": "Paris",
      "to_city": "Amsterdam",
      "departure_time": "08:00",
      "arrival_time": "11:30",
      "transport_mode": "train"
    }
  ],
  "meta": {
    "cities_visited": 2,
    "total_days": 5,
    "preferences": ["art", "history", "food"]
  }
}
```

## 📁 Project Structure

```
itinerary-planner/
├── data/                       # Data directory
│   ├── raw/                    # Raw input data (fixed_cities.json)
│   ├── processed/              # Processed attraction data
│   └── embeddings/             # Stored embeddings for attractions
│
├── src/                        # Source code
│   ├── algorithms/             # Core algorithms
│   │   ├── attraction_grouper.py   # Group compatible attractions
│   │   ├── city_allocation.py      # Allocate days across cities
│   │   ├── city_transition.py      # Handle city transitions
│   │   ├── itinerary_builder.py    # Build final itinerary
│   │   └── route_optimizer.py      # Optimize attraction sequence
│   ├── data/                   # Data processing modules
│   │   ├── attraction_processor.py # Process attraction data
│   │   └── data_validator.py       # Validate input data
│   ├── models/                 # Model directory
│   │   └── transformer_handler.py  # Handle transformer model operations
│   └── main.py                 # Main entry point
│
├── .gitignore                  # Git ignore file
├── environment.yml             # Conda environment file
├── itinerary.json              # Sample output
├── README.md                   # This file
└── requirements.txt            # Project dependencies
```

## 🔍 How It Works

The system processes data through several stages:

1. **Data Preprocessing**: Validates and processes attraction data
2. **Transformer Model**: Provides semantic understanding of attractions
3. **City Allocation**: Distributes days across multiple cities based on importance
4. **Attraction Grouping**: Creates coherent daily plans using semantic understanding
5. **Route Optimization**: Sequences attractions efficiently to minimize travel time
6. **City Transition**: Handles movements between cities
7. **Itinerary Building**: Assembles the final itinerary with all details

## 👨‍💻 Contributing

Contributions to improve the itinerary planner are welcome! Please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🧠 Team

- JANAT ZAMAN
- MOHAMMED ALFATEH
- ABDALRAHMAN BASHIR