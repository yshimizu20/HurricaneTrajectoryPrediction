# Author: Mike Zhang
# CPSC 452 Hurricane Trajectory Prediction
#
# Purpose: This program downloads and cleans the HURDAT2 dataset.
# First, the user must download the HURDAT .txt file off the web
# and convert it to .csv format before running this code.

import csv

# Function to clean and rewrite HURDAT
def download_write_HURDAT(input_file, output_file):
    
    # Open the input CSV file for reading and the output CSV file for writing
    with open(input_file, newline='') as csvfile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(csvfile)
        writer = csv.writer(outfile)
        
        # Write the header row with the new column labels
        writer.writerow(['ID', 'hurricane_name', 'year', 'full_date', 'time_index', 'lat_center', 'long_center', 'wind_speed'])
        
        # Process each row in the input file
        for row in reader:
            # Extract the ID and hurricane name and assign to each row
            if row[5].strip() == '':  # Check if the fifth column is empty
                ID = row[0]
                hurr_name = row[1]
                continue
            else:
                # Process the rows under this label and name
                year = str(row[0])[:4]
                full_time = row[0]
                time_index = row[1]
                lat_center = row[4]
                long_center = row[5]
                wind_speed = row[6]
                
                # Write the processed row to the output file
                writer.writerow([ID, hurr_name, year, full_time, time_index, lat_center, long_center, wind_speed])

# Uncleaned HURDAT2 data
uncleaned_HURDAT2 = "HURDAT2_ucdata.csv"

# Final cleaned HURDAT2 data
cleaned_HURDAT2 = "HURDAT2_final.csv"

# Function call to download and clean HURDAT2
download_write_HURDAT(uncleaned_HURDAT2, cleaned_HURDAT2)


