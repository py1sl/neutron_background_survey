# neutron_background_survey

A background neutron survey was performed in the experimental halls of ISIS and focused along the neutron beamlines of several instruments. 
The survey was used to map out the relative background levels and to compare neutron energy distributions across the facility.

This repository reads in every diamon folder found in either 'TS1 Measurements' or 'TS2 Measurements' folder under the data folder in the rep. Each file has a corresponding reference
with coordinates and a beamname attached to it. 
A query is created from the ISIS accelerator controls Influx Database containing all shutter and beam information. For every measurement a flag for the corresponding
beamline shutter and current measurement is recorded at each point in time.

Once read in through the main function, the jupyter notebook acts as an interface to query data or plot graphs using the script 'plotting'.
