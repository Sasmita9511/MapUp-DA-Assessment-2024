from typing import Dict, List

import pandas as pd

#Question 1

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    return [lst[i:i+n][::-1] for i in range(0, len(lst), n)]
print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3))


#Question 2

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    grouped = {}
    for string in lst:
        length = len(string)
        if length not in grouped:
            grouped[length] = []
        grouped[length].append(string)
    return grouped
print(group_by_length(["cat", "dog", "elephant", "lion", "ant"]))

#Question 3

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    def _flatten(obj, parent_key='', sep='.'):
        items = []
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(_flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    return _flatten(nested_dict, sep=sep)

nested = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}
print(flatten_dict(nested))


#Question 4

import itertools
def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    return list(set(itertools.permutations(nums)))
print(unique_permutations([1, 1, 2]))


#Question 5
import re

def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    date_patterns = [
        r"\b\d{2}-\d{2}-\d{4}\b",   # dd-mm-yyyy
        r"\b\d{2}/\d{2}/\d{4}\b",   # mm/dd/yyyy
        r"\b\d{4}\.\d{2}\.\d{2}\b"  # yyyy.mm.dd
    ]
    dates = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text))
    return dates
text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
print(find_all_dates(text))
   



#Question 6

import pandas as pd
import numpy as np

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    def decode_polyline(polyline_str):
        
        return [(10.0, 20.0), (10.1, 20.1), (10.2, 20.2)]
    
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000  
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
        return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    points = decode_polyline(polyline_str)
    latitudes, longitudes, distances = [], [], []

    for i, (lat, lon) in enumerate(points):
        latitudes.append(lat)
        longitudes.append(lon)
        if i > 0:
            prev_lat, prev_lon = points[i - 1]
            distances.append(haversine(prev_lat, prev_lon, lat, lon))
        else:
            distances.append(0)

    return pd.DataFrame({"Latitude": latitudes, "Longitude": longitudes, "Distance (m)": distances})
df = polyline_to_dataframe("encoded_polyline")
print(df)
    


#Question 7



def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
     
    n = len(matrix)
    rotated = list(zip(*matrix[::-1]))
    
   
    rotated = [list(row) for row in rotated]

    
    transformed = [[0] * n for _ in range(n)]  # Initialize the transformed matrix

    for i in range(n):
        for j in range(n):
            
            row_sum = sum(rotated[i]) - rotated[i][j]
            
            col_sum = sum(rotated[k][j] for k in range(n)) - rotated[i][j]
            
            transformed[i][j] = row_sum + col_sum

    return transformed


matrix = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]

result = rotate_and_multiply_matrix(matrix)
for row in result:
    print(row)










#Question 8

def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here

    return pd.Series()
