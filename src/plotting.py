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
    ax = df["norm_dose"].plot(kind='bar', yerr=(df["dose_rate_uncert"])/100 * df['norm_dose'], capsize=4, color='purple', figsize=(5,5))
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

def find_change_time(data):
    df = data["out"]["shutter-open"]
    filter = (df.ne(df.shift()))
    change_times = data["out"][filter]
    return change_times

def plot_dose_time(data):

    #get starting status then record change in difference
    #todo: generalise for all measurements?
    change_times = find_change_time(data)
    x = data['out']['t(s)']
    y = data['out']['norm_dose']
    y2 = data['out']['ts2_current']
    fig = plt.figure(figsize=(8,8))

    ax = fig.add_subplot(111)
    ax2 = fig.add_subplot(111, frame_on=False)
    ax.plot(x, y, color='b', marker=None)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r" normalised dose rate ( $\frac{\mu Sv}{\mu A -hour})$", color='b')
    colors = ['black', 'orange', 'green']
    for time, status in zip(change_times["t(s)"], change_times["shutter-open"]):
        if status:
            color = colors[2]
            label = "open"
        else:
            color = colors[0]
            label= "closed"
        plt.vlines(time, 0, max(y2), ls='dashdot', color=color, label=label)
    ax2.plot(x, y2, color='r', marker=None, alpha=0.5)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('current $\mu$ A', color='r') 
    ax2.yaxis.set_label_position('right')
    plt.legend(loc="lower right")
    name = da.get_names(data["reference"])[1]
    distance = da.get_distance(data["reference"])["distance"].values[0]
    plt.title("Comparison between dose rate and current with the shutter status at " + name + "\n at a distance : {:.2f} m away".format(distance))
    
def plot_energy_time(data, key, beamline):
    distance = da.get_distance(data["reference"])["distance"].values[0]
    fast = data["out"]["Fast%"].astype(float)
    epi = data["out"]["Epit%"].astype(float)
    therm = data["out"]["Ther%"].astype(float)
    times = data["out"]["t(s)"]
    ts2_current = data["out"]["ts2_current"]
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.plot(times, fast, color='g', label='fast', marker=None)
    ax.plot(times, epi, color='orange', label='epithermal', marker=None)
    ax.plot(times, therm, color='r', label='thermal', marker=None)
    colors = ['black', 'orange', 'green']
    ax2 = fig.add_subplot(111, frame_on=False)
    if beamline != "epb":
        change_times = find_change_time(data)
        for time, status in zip(change_times["t(s)"], change_times["shutter-open"]):
            if status:
                color = colors[2]
                label = "open"
            else:
                color = colors[0]
                label= "closed"
            ax2.vlines(time, 0, np.max(ts2_current), ls='dashdot', color=color, label=label)
    
    ax.set_yticks(np.arange(0, 110, 10))
    ax2.plot(times, ts2_current, color='r', marker=None, alpha=0.5)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('current $\mu$ A', color='r') 
    ax2.yaxis.set_label_position('right')
    name = da.get_names(data["reference"])[1]
    ax.set_xlabel("Time t(s)")
    
    ax.set_ylabel("percentage (%)")
    ax.legend(loc="upper right")
    ax2.legend(loc="upper center")
    plt.title("Fast, thermal and epithermal energy distribution over time for " + name + "\n at a distance : {:.2f} m away".format(distance))
    plt.show()
    