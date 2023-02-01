import DIAMON_analysis as da
import diamon_read_data as dia
import numpy as np
import pandas as pd

folder_path = r"C:\Users\sfs81547\Documents\DIAMON project\TS1 measurements\*"


#reads in the folder contianing data for: unfold, rate and out data separated into columns of the measurement
all_data = dia.read_folder(folder_path)
all_data = np.array(all_data)
#all_data = all_data[1:,:]


energy_peaks, flux_peaks = da.find_spect_peaks(all_data[:,0])
df = pd.DataFrame({'energy peaks' :energy_peaks, 'flux peaks' : flux_peaks}, columns=['energy peaks', 'flux peaks'])
df1 = df.explode('energy peaks', "flux peaks")
df1.to_csv("energy_flux_peaks1.csv")
print(energy_peaks)
print(flux_peaks)
avg_flux = da.average_daily_data(all_data[:,0])
da.plot_avg_spect(all_data[0,0].energy_bins, avg_flux)

#convert to data series
unfold_datalist = []
for data in all_data[:,0]:
    unfold_data = dia.convert_to_ds(data)
    unfold_datalist.append(unfold_data)

da.plot_combined_spect(all_data[:,0])
   
unfold_dataframe = pd.DataFrame(unfold_datalist)
da.direction_bar_plot(unfold_dataframe)
da.stack_bar_plot(unfold_dataframe)
da.plot_dose_rate(unfold_dataframe)
names = [data[0].name for data in all_data]
da.plot_detector_counts(all_data[:,2], names)
da.direction_stack_bar_plot(unfold_dataframe)


#take measurements from ts2
new_df = da.filter_df_by_date("11.10|12.10", unfold_dataframe)
da.plot_dose_rate(new_df)
