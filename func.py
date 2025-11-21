import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial import cKDTree
from haversine import haversine, Unit

from config import *

def compute_missing_values_over_time(df):
    missing_values_df = (df.groupby(df['date'].dt.year)
                      .agg({'tmin': lambda x: x.isnull().mean() * 100,
                           'station_id': 'nunique'})).rename(columns={'tmin': 'missing_values',
                                                                 'station_id': 'stations_in_activity'})
    return missing_values_df

def plot_missing_values_and_stations(df):
    plt.figure(figsize=(12, 6))
    sns.barplot(x=df.index, y=df['missing_values'])
    plt.xticks(rotation=90)
    plt.title('Percentage of Missing Values by Year')
    plt.xlabel('Year')
    plt.ylabel('Percentage of Missing Values')
    plt.show()
    
def plot_completion_rate_distribution(df: pd.DataFrame):
    completion_rate = df.groupby('station_id')['tmin'].apply(lambda x: x.notnull().mean() * 100)
    plt.figure(figsize=(12, 6))
    sns.histplot(completion_rate, bins=100, kde=True)
    plt.title('Distribution of Completion Rate per Station')
    plt.xlabel('Completion Rate (%)')
    plt.ylabel('Number of Stations')
    plt.show()

def process_weather_data(
                         dept: str,
                         local_file: bool=False,
                         start_date: str=START_DATE,
                         end_date: str=END_DATE,
                         completion_rate_threshold: float=COMPLETION_RATE_THRESHOLD,
                         remove_stations_below_threshold: bool=True,
                         raw_data_path: str=RAW_DATA_PATH,
                         default_url: str=DEFAULT_WEATHER_URL,
                         ):

    filename = f"Q_{dept}_previous-1950-2023_RR-T-Vent.csv.gz"
    if local_file:
        weather_filename = os.path.join(raw_data_path, filename)
    else:
        weather_filename = f"{default_url}{filename}"

    d = {
        'NUM_POSTE': (str, 'station_id'),
        'NOM_USUEL': (str, 'station_name'),
        'LAT': (float, 'latitude'),
        'LON': (float, 'longitude'),
        'ALTI': (float, 'alti'),
        'AAAAMMJJ': (str, 'date'),
        'TN': (float, 'tmin'),
    }

    weather_df = pd.read_csv(weather_filename,
                            compression="gzip",
                            sep=';',
                            usecols=d.keys(),
                            dtype={k: v[0] for k, v in d.items()},
                            ).rename(columns={k: v[1] for k, v in d.items()})

    weather_df['date'] = pd.to_datetime(weather_df['date'], format='%Y%m%d')
    
    # Slice the dataframe to keep only the data from start_date to end_date
    weather_df = weather_df.loc[weather_df['date'].dt.year.between(start_date, end_date, inclusive='both')]

    # Remove all observations from stations that have a completion rate lower than the threshold
    if remove_stations_below_threshold:
        completion_rate = weather_df.groupby('station_id')['tmin'].apply(lambda x: x.notnull().mean())
        valid_stations = completion_rate[completion_rate >= completion_rate_threshold].index
        weather_df = weather_df[weather_df['station_id'].isin(valid_stations)]

    return weather_df


def find_closest_stations_kdtree(city_df: pd.DataFrame, stations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Find the closest weather station for each city using KDTree for efficient spatial search.
    
    Parameters:
    -----------
    city_df : pd.DataFrame
        DataFrame containing cities with 'lat' and 'lon' columns
    stations_df : pd.DataFrame  
        DataFrame containing stations with 'latitude', 'longitude', 'station_name', 
        'station_id', and 'alti' columns
        
    Returns:
    --------
    pd.DataFrame
        Input city_df with additional columns for closest station information
    """
    # Make a copy to avoid modifying the original dataframe
    result_df = city_df.copy()
    
    # Prepare city and station coordinates as arrays
    city_coords = city_df[['lat', 'lon']].to_numpy()
    station_coords = stations_df[['latitude', 'longitude']].to_numpy()

    # Build a KDTree for station coordinates
    tree = cKDTree(station_coords)

    # For each city, find the index of the closest station
    distances, indices = tree.query(city_coords)

    # Add the closest station index and distance to result_df
    result_df['closest_station_idx_with_kdtree'] = indices
    result_df['closest_station_distance_km_with_kdtree'] = distances * 100

    # Add station info (station name, ID, and altitude)
    result_df['closest_station_name_with_kdtree'] = stations_df.iloc[indices]['station_name'].values
    result_df['closest_station_NUM_POSTE_with_kdtree'] = stations_df.iloc[indices]['station_id'].values
    result_df['closest_station_alti_with_kdtree'] = stations_df.iloc[indices]['alti'].values

    return result_df


def process_cities_data(raw_data_path: str = RAW_DATA_PATH,
                        filename: str = 'communes-france-2025.csv.gz',
                        dept_list: list | None = None) -> pd.DataFrame:
    """
    Load and process cities data from CSV file.
    
    Parameters:
    -----------
    raw_data_path : str
        Path to the raw data directory
    filename : str
        Name of the cities CSV file
    dept_list : list or None
        List of department codes to filter cities. If None, all cities are returned.
        
    Returns:
    --------
    pd.DataFrame
        Processed cities dataframe
    """
    d = {
        "code_insee": ["string", "insee_code"],
        "nom_standard": ["string", "name"],
        "dep_code": ["string", "dep_code"],
        "dep_nom": ["string", "dep_name"],
        "latitude_centre": ["float32", "lat"],
        "longitude_centre": ["float32", "lon"],
    }

    city_df = pd.read_csv(
        os.path.join(raw_data_path, filename),
        compression="gzip",
        usecols=d.keys(),
        dtype={k: v[0] for k, v in d.items()},
    ).rename(columns={k: v[1] for k, v in d.items()})

    # Fill missing coordinates for known cities
    missing_cities_lat_lon = {
        "Marseille": [43.295, 5.372],
        "Paris": [48.866, 2.333],
        "Culey": [48.755, 5.266],
        "Les Hauts-Talican": [49.3436, 2.0193],
        "Lyon": [45.75, 4.85],
        "Bihorel": [49.4542, 1.1162],
        "Saint-Lucien": [48.6480, 1.6229],
        "L'Oie": [46.7982, -1.1302],
        "Sainte-Florence": [46.7965, -1.1520],
    }
    
    for city, (lat, lon) in missing_cities_lat_lon.items():
        city_df.loc[city_df["name"] == city, "lat"] = lat
        city_df.loc[city_df["name"] == city, "lon"] = lon
        
    if dept_list is not None:
        # Filter cities by department codes
        city_df = city_df[city_df['insee_code'].str[:2].isin(dept_list)]

    return city_df


def get_all_good_stations(dept_list: list) -> pd.DataFrame:
    """
    Get all good weather stations (with sufficient data) for a list of departments.
    
    Parameters:
    -----------
    dept_list : list
        List of department codes (as strings, e.g., ['04', '13'])
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing good stations with their metadata
    """
    dfs = []
    for dept in dept_list:
        df = process_weather_data(dept=dept, remove_stations_below_threshold=True)
        station_info = df[['station_id', 'station_name', 'latitude', 'longitude', 'alti']].drop_duplicates()
        dfs.append(station_info)
        print(f"Done with dept N° {dept}")
    
    return pd.concat(dfs, ignore_index=True)


def find_closest_stations_haversine(city_df: pd.DataFrame, stations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Find the closest weather station for each city using Haversine distance calculation.
    
    This method calculates the actual great-circle distance between coordinates on Earth's surface,
    making it more accurate than KDTree for geographic coordinates, but slower for large datasets.
    
    Parameters:
    -----------
    city_df : pd.DataFrame
        DataFrame containing cities with 'lat' and 'lon' columns
    stations_df : pd.DataFrame  
        DataFrame containing stations with 'latitude', 'longitude', 'station_name', 
        'station_id', and 'alti' columns
        
    Returns:
    --------
    pd.DataFrame
        Input city_df with additional columns for closest station information
    """
    # Make a copy to avoid modifying the original dataframe
    result_df = city_df.copy()
    
    # Prepare city and station coordinates as arrays
    city_coords = city_df[['lat', 'lon']].to_numpy()
    station_coords = stations_df[['latitude', 'longitude']].to_numpy()

    closest_station_idx = []
    closest_station_distance = []

    for city in city_coords:
        # Compute all distances from this city to all stations
        distances = [haversine(city, station, unit=Unit.KILOMETERS) for station in station_coords]
        min_idx = np.argmin(distances)
        closest_station_idx.append(min_idx)
        closest_station_distance.append(distances[min_idx])

    # Add the closest station index and distance to result_df
    result_df['closest_station_idx_with_haversine'] = closest_station_idx
    result_df['closest_station_distance_km_with_haversine'] = closest_station_distance

    # Add station info (station name, ID, altitude)
    result_df['closest_station_name_with_haversine'] = stations_df.iloc[closest_station_idx]['station_name'].values
    result_df['closest_station_NUM_POSTE_with_haversine'] = stations_df.iloc[closest_station_idx]['station_id'].values
    result_df['closest_station_alti_with_haversine'] = stations_df.iloc[closest_station_idx]['alti'].values

    return result_df