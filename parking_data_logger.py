import pandas as pd
from datetime import datetime


# Function to log parking data
def log_parking_data(timestamp, available_spaces, log_file="parking_space_log.csv"):
    """
    Log parking space availability data to a CSV file.

    :param timestamp: Current timestamp when data is logged.
    :param available_spaces: Number of available parking spaces.
    :param log_file: Path to the CSV file where data will be stored.
    """
    # Create or append to the parking space log file
    with open(log_file, "a") as file:
        if file.tell() == 0:  # Add headers if the file is empty
            file.write("Timestamp,AvailableSpaces\n")
        file.write(f"{timestamp},{available_spaces}\n")
