# NEUTRON BACKGROUND SURVEY
- This is the main project work for my industrial placement

A background neutron survey was performed in the experimental halls of ISIS and focused along the neutron beamlines of several instruments. 
The survey was used to map out the relative background levels and to compare neutron energy distributions across the facility.

This repository reads in every diamon folder found in the 'data\measurements' folder. Each file has a corresponding reference
with coordinates and a beamname attached to it. The reference csv is found in 'data\measurement_location.csv' and the target station shutter reference is found in 'data\target_station_data.csv'.

Existing data can either be loaded manually by loading the folder and channel information can be extracted by pulling using the influx query tool from the InfluxDB. To save time there is the option to pickle existing data.

Flow:
Run through src/main.py.
1. Load diamon data from 'data\measurements' folder.
2. Load shutter information from INFLUXDB / existing pickle
3. Filter shutter information to diamon data

Once loaded the various scripts can be used to manipulate the data from the detector

### Beamline.py

Beamline class object. used to attach instrument information and configuration to data. Sets neighbours and position 
within the facility

### diamon_analysis.py

Set of functions to aid with analysis.
-filter channel shutters for varying conditions
- convert series to df
- filter location
- repeat data
Key function: filter_shutter_analysis - must be ran to produce a df of all data

### diamon_read_data.py

Script to load data using diamon.py. initialises diamon object and stores into dictionary

### influx_data_query.py

Creates a query api to fetch data from the InfluxDB from accel controls at ISIS. Requires a valid auth key to work.
Query can be called using two dates or a beam cycle from ISIS.

### main.py
Main program. Loads diamon and shutter data into a dictionary. Must use to do further analysis.
### meny.py
-UNFINISHED-
additional feature to interactively run program from command line

### neutronics_analysis.py

set of useful tools for neutronics data analysis, string cleaning, loading files. WIP.

### plotting.py
all plotting scripts for this project using matplotlib found here
spectra, dose maps, dose time, unc_time, energy distirbution, counts etc

### shutter_analysis.py
A query is created from the ISIS accelerator controls Influx Database containing all shutter and beam information. For every measurement a flag for the corresponding
beamline shutter and current measurement is recorded at each point in time.

Once read in through the main function, the jupyter notebook acts as an interface to query data or plot graphs using the script 'plotting'.
