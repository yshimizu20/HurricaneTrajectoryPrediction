# Author: Mike Zhang
# CPSC 452 Hurricane Trajectory Prediction
#
# Purpose: This program downloads and cleans the HURDAT2 dataset.

import csv


# Function to clean and rewrite HURDAT
def download_write_HURDAT(input_file, output_file):

    # Open the input CSV file for reading and the output CSV file for writing
    with open(input_file, newline="") as csvfile, open(
        output_file, "w", newline=""
    ) as outfile:
        reader = csv.reader(csvfile)
        writer = csv.writer(outfile)

        # Process each row in the input file
        for row in reader:
            name = row[1]
            year = row[2]
            month = row[3]
            day = row[4]
            hour = row[5]
            lat = row[6]
            long = row[7]
            wind = row[10]
            pressure = row[11]

            # Write the processed row to the output file
            writer.writerow([name, year, month, day, hour, lat, long, wind, pressure])


# Uncleaned HURDAT2 data
uncleaned_HURDAT2 = "storms.csv"

# Final cleaned HURDAT2 data
cleaned_HURDAT2 = "HURDAT2_final.csv"

# Function call to download and clean HURDAT2
download_write_HURDAT(uncleaned_HURDAT2, cleaned_HURDAT2)

print("HURDAT2 data has been cleaned and written to HURDAT2_final.csv")
