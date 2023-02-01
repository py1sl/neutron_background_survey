import matplotlib.pyplot as plt
import numpy as np
import diamon_analysis as da

def plot_spect(data):
    energy, flux = da.extract_spectrum(data)
    plt.xscale("log")
    plt.step(energy, flux, label=data.name)
    plt.xlabel("Energy (MeV)")
    plt.ylabel("Flux (cm$^{-2}$ s$^{-1}$)")
    plt.legend()
    plt.savefig("plots/single_energy_spectrum.png")
    plt.show()
    
def plot_combined_spect(data_array, key=None):
    
    for index, data in data_array.iterrows():

        energy, flux = da.extract_spectrum(data)
        plt.xscale("log")
        plt.step(energy, flux, label=data["reference"])
        plt.xlabel("Energy (MeV)")
        plt.ylabel("Flux (cm$^{-2}$s$^{-1}$)")
        plt.legend(fontsize=12, loc=1)
    if key:
        plt.title("Spectra for " + key + " beamline")
        plt.savefig("plots/combined_energy_spectra " + key +" .png")
    plt.show()
    
def plot_detector_counts(rate_data, labels):
    for i, rate in enumerate(rate_data):
        # add cumulative time
        rate['time'] = rate['Dt(s)'].cumsum()
        rate['counts'] = rate["Det1"].cumsum()
        # plot counts over time
        plt.step(rate["time"], rate["counts"], label=labels[i], marker='x')
        plt.xlabel("Time (s)")
        plt.ylabel("Counts")
        plt.legend()
    plt.savefig("plots/detector_counts.png")
    plt.show()

def plot_dose_rate(df, key=None):
    plt.rcParams["figure.figsize"] = [2,2]
    ax = df["norm_dose_rate"].plot(kind='bar', yerr=(df["dose_rate_uncert"])/100 * df['norm_dose_rate'], capsize=4, color='purple', figsize=(5,5))
    ax.set_xlabel('Distance from monolith (m)')
    ax.set_ylabel(r"normalised dose rate ( $\frac{\mu Sv}{\mu A})$")
    ax.set_xticklabels(round(df["distance"], 3))
    if key:
        plt.title("dose rate for " + key + " beamline")
        #plt.savefig("plots/dose_rate_" + key + ".png")
    plt.show()
    
def stack_bar_plot(data_frame, xlabel=None, ylabel=None, key =None):
    #stack_df = (data_frame.filter(cols)).astype(float)

    ax = data_frame[["thermal", "epi", "fast"]].plot(kind='bar', stacked=True,figsize=(5,5))
    ax.set_xlabel("distance from monolith (m)")
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(data_frame["reference"], rotation = 45)
    ax.set_xticklabels(round(data_frame["distance"], 3))
    if key:
        plt.title("energy distribution bar plot for " + key + " beamline")
       # plt.savefig("plots/energy_ranges_bar_plot" + key + ".png")
    plt.show()
 
def direction_bar_plot(dataframe, key):
    
    ax = dataframe[["F", "FL", "FR", "R", "RR", "RL"]].plot(kind='bar',figsize = (5,5))
    ax.set_ylabel("Counts")
    ax.set_xticklabels(dataframe["reference"], rotation=60)
    if key:
        plt.title("direction bar plot for " + key + " beamline")
        plt.savefig("plots/direction_plot_" + key + ".png")
    plt.show()

def direction_stack_bar_plot(df):
    
    df['sum_dir'] = df [["F", "FL", "FR", "R", "RR", "RL"]].sum(axis=1)
    df["F_norm"] = df["F"]/df["sum_dir"]
    df["FL_norm"] = df["FL"]/df["sum_dir"]
    df["FR_norm"] = df["FR"]/df["sum_dir"]
    df["R_norm"] = df["R"]/df["sum_dir"]
    df["RL_norm"] = df["RL"]/df["sum_dir"]
    df["RR_norm"] = df["RR"]/df["sum_dir"]

    axis = df[["F_norm", "FR_norm", "FL_norm", "R_norm", "RR_norm", "RL_norm"]].plot(kind="bar", stacked=True)
    fig = axis.get_figure()
    
def plot_avg_spect(energy, flux):
    
    plt.step(energy, flux)
    plt.xscale("log")
    plt.xlabel("Energy (MeV)")
    plt.ylabel("Flux (cm$^{-2}$s$^{-1}$)")
    plt.legend()
    plt.savefig("plots/average_energy_spectra.png")
    plt.show()

def plot_dose_distance(df, key=None):
    plt.rcParams["figure.figsize"] = [5,5]
    ax = plt.subplot()
    ax.errorbar(df['distance'], df['dose_rate'], yerr=df["dose_rate_uncert"]/100, marker='x', ls='None')
    ax.set_xlabel('distance (m)')
    ax.set_ylabel('dose rate ($\mu$ Sv\h)')
    #ax.set_xticklabels(df["reference"], rotation=20)
    if key:
        plt.title("dose rate vs distance for " + key + " beamline")
        plt.savefig("plots/dose_rate_" + key + ".png")
    plt.show()
def create_meshgrid(x,y):
    return np.meshgrid(x,y)