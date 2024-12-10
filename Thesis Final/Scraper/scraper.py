import pandas as pd
import logging
from pybaseball import statcast
from datetime import datetime
import sys
import os

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

def fetch_and_save_statcast_data(start_date, end_date, output_file='statcast_data.csv'):
    """
    Fetches Statcast data for the given date range and saves it to a CSV file,
    filtering out spring training games.
    
    Parameters:
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.
        output_file (str): The name of the output CSV file.
    """
    try:
        # Fetch data using pybaseball
        logging.info(f"Fetching Statcast data from {start_date} to {end_date}...")
        data = statcast(start_dt=start_date, end_dt=end_date)
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        sys.exit(1)

    if data.empty:
        logging.error("No data found for the given date range.")
        sys.exit(1)

    # Filter out spring training games (game_type == 'S')
    logging.info("Filtering out spring training games...")
    data_filtered = data[data['game_type'] != 'S']

    if data_filtered.empty:
        logging.error("No data left after filtering out spring training games.")
        sys.exit(1)

    # Save data to CSV
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        data_filtered.to_csv(output_file, index=False)
        logging.info(f"Filtered data successfully saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving data to CSV: {e}")
        sys.exit(1)

if __name__ == '__main__':
    import argparse

    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Fetch and save Statcast data to CSV.')
    parser.add_argument('--start_date', type=str, required=True, help='Start date in YYYY-MM-DD format.')
    parser.add_argument('--end_date', type=str, required=True, help='End date in YYYY-MM-DD format.')
    parser.add_argument('--output_file', type=str, default='statcast_data.csv', help='Output CSV file name.')

    args = parser.parse_args()

    # Validate date format
    try:
        datetime.strptime(args.start_date, '%Y-%m-%d')
        datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError as e:
        logging.error("Invalid date format. Please use 'YYYY-MM-DD'.")
        sys.exit(1)

    # Fetch data and save to CSV
    fetch_and_save_statcast_data(args.start_date, args.end_date, args.output_file)

# python scraper.py --start_date 2023-05-01 --end_date 2024-10-16 --output_file data/statcast.csv
# python scraper.py --start_date 2021-04-01 --end_date 2023-11-01 --output_file data/statcast_2021_2023.csv
# python scraper.py --start_date 2023-05-01 --end_date 2024-10-16 --output_file data/statcast.csv
# EX: python scraper.py --start_date 2024-08-01 --end_date 2024-08-30 --output_file data/statcast_august_2024.csv

#python scraper.py --start_date 2021-04-01 --end_date 2021-11-01 --output_file data/statcast_2021.csv
#python scraper.py --start_date 2022-04-07 --end_date 2022-11-03 --output_file data/statcast_2022.csv
#python scraper.py --start_date 2023-03-30 --end_date 2023-11-03 --output_file data/statcast_2023.csv

#python scraper.py --start_date 2024-03-28 --end_date 2024-06-26 --output_file data/statcast_2024_1st_half.csv