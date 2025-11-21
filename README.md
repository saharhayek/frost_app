# frost_app
A Python toolkit for processing Meteo-France historical weather data, evaluating station completeness, and matching French cities with their closest weather stations using KDTree or Haversine distance.

This project is used for large-scale frost-day analysis, climate trends, and spatial weather research.

## Features
**✔ Weather Data Processing**
**Load data from Meteo-France (local or remote).
Clean & format raw columns.
Restrict to a configurable time range.
Remove stations below a configurable completion threshold.
Department-level filtering (e.g., "13", "04"…).

✔ Data Quality Analysis
Compute missing data percentages per year.
Plot completion-rate distribution across stations.

✔ City Processing
Load French communes data.
Standardize columns (INSEE code, name, latitude, longitude).
Fix missing coordinates for known communes.
Department-level filtering.

✔ Nearest-Station Matching
Two methods supported:
KDTree (fast)
Very fast spatial lookup (Euclidean approximation).
Recommended for large datasets.
Haversine (accurate)
Computes great-circle distances on Earth.
Ideal for final validation.

**Both return:**
Closest station ID
Name
Altitude
Distance (km)

**config.py defines all default settings:**
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
START_DATE = 2014
END_DATE = 2023
COMPLETION_RATE_THRESHOLD = 0.65
DEFAULT_WEATHER_URL = "https://object.files.data.gouv.fr/..."
