'''
CPSC 452 Hurricane Trajectory Prediction
Assembled by Mike Zhang

Purpose: This program downloads the HURSAT dataset.
This function is inspired by code from the following github.
However, I have added components necessary to the functioning of our specific data.
https://github.com/23ccozad/hurricane-wind-speed-cnn/blob/master/download.py
'''

import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import tarfile


def download_hursat(years):
    # Reads in the tracking dataset from HURDAT2
    best_track_data = pd.read_csv('HURDAT2_final.csv')

    for year in years:
        # Scrapes a webpage to get list of all .tar.gz files
        year_directory_url = 'https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06/' + year
        year_directory_page = requests.get(year_directory_url).text
        year_directory_soup = BeautifulSoup(year_directory_page, 'html.parser')
        year_directory_file_urls = [year_directory_url + '/' + node.get('href') for node in
                                    year_directory_soup.find_all('a') if node.get('href').endswith('tar.gz')]
        print('\n' + year + ' file loaded.')

        files_processed = 0
        for storm_file_url in year_directory_file_urls:
            # Extract the name of the hurricane and the year from the file name
            name = storm_file_url.split('_')[-2]
            
            # Make all letters of name lowercase except for the first letter
            name = name[0].upper() + name[1:].lower()

            year = int(storm_file_url.split('_')[3][:4])
            
            file_has_match_in_best_track = not best_track_data.loc[
                (best_track_data['year'] == year) & (best_track_data['name'] == name)
            ].empty

            if file_has_match_in_best_track:
                # Build a string, which will be file path where we save the .tar.gz when downloaded
                file_name = storm_file_url.split('/')[-1]
                storm_file_path = 'HURSAT_Imagery/' + file_name

                # Create the Satellite Imagery folder if it doesn't already exist
                if not os.path.exists('HURSAT_Imagery'):
                    os.makedirs('HURSAT_Imagery')

                # Open the .tar.gz and copy it's contents to a local file
                request = requests.get(storm_file_url, allow_redirects=True)
                open(storm_file_path, 'wb').write(request.content)
                request.close()

                # Open the .tar.gz file and loop through each file inside
                tar = tarfile.open(storm_file_path)
                file_prefixes_in_directory = []
                for file_name in tar.getnames():
                    # Get the date and time of the satellite image, and the name of the satellite that took the image
                    full_time = file_name.split(".")[2] + file_name.split(".")[3] + file_name.split(".")[4]
                    hour = file_name.split(".")[5]
                    satellite = file_name.split(".")[7][:3]

                    # If best_track_data[month] is only one digit, add a 0 to the front of it
                    if len(str(best_track_data['month'][0])) == 1:
                        track_month = best_track_data['month'].apply(lambda x: '0' + str(x))
                    else:
                        track_month = best_track_data['month'].apply(lambda x: str(x))

                    # if best_track_data[day] is only one digit, add a 0 to the front of it
                    if len(str(best_track_data['day'][0])) == 1:
                        track_day = best_track_data['day'].apply(lambda x: '0' + str(x))
                    else:
                        track_day = best_track_data['day'].apply(lambda x: str(x))
                    
                    # if best_track_data[hour] is only one digit, add a 0 to the front of it and 00. Otherwise, only add 00 to the end of it
                    if len(str(best_track_data['hour'][0])) == 1:
                        track_hour = best_track_data['hour'].apply(lambda x: '0' + str(x) + '00')
                    else:
                        track_hour = best_track_data['hour'].apply(lambda x: str(x) + '00')

                    # Combine the year, month, day columns into a single column
                    track_full_time = best_track_data['year'].astype(str) + track_month + track_day
                        
                    # Determine whether the best track dataset has a record for the date and time of this storm.
                    file_has_match_in_best_track = not best_track_data.loc[
                        (track_full_time == full_time) & (track_hour == hour)].empty

                    # Determine whether another image of this hurricane at this exact time has already been extracted
                    # from the .tar.gz
                    is_redundant = '.'.join(file_name.split('.')[:6]) in file_prefixes_in_directory

                    # If the requirements are met, extract the netcdf file from this .tar.gz and save it locally
                    if file_has_match_in_best_track and not is_redundant and satellite == "GOE":
                        f = tar.extractfile(file_name)
                        open('HURSAT_Imagery/' + file_name, 'wb').write(f.read())
                        file_prefixes_in_directory.append('.'.join(file_name.split('.')[:6]))

                tar.close()
                os.remove(storm_file_path)

            files_processed += 1
            print_progress('Processing Files for ' + str(year), files_processed, len(year_directory_file_urls))


def print_progress(action, progress, total):
    percent_progress = round((progress / total) * 100, 1)
    print('\r' + action + '... ' + str(percent_progress) + '% (' + str(progress) + ' of ' + str(total) + ')', end='')


if __name__ == "__main__":
    # List of years to download
    #YEARS_TO_DOWNLOAD = ['2016']

    # Create a list from 1978 to 2016
    # YEARS_TO_DOWNLOAD = [str(year) for year in range(1978, 2016)]
    YEARS_TO_DOWNLOAD = [str(year) for year in range(2002, 2016)]

    download_hursat(YEARS_TO_DOWNLOAD)
    
    