import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import numpy as np
import src.diamon_analysis as da
import src.neutronics_analysis as na
from scipy import interpolate
import os
import pandas as pd

def plot_spectra(data, fname=None, title="", save_table=False):
    """Plot of spectrum - flux vs energy for a measurement.
    split across thermal, epithermal and fast neutrons

    Args:
        data (df): df of data
    """

    colors = cm.rainbow(np.linspace(0,1, len(data)))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Energy (MeV)")
    plt.ylabel("Flux (cm$^{-2}$s$^{-1}$ MeV$^{-1}$)")
    plt.title("Neutron typical energy spectra " + title)
    dic = {}
    for color, result in zip(colors, data):
        energy, flux = da.extract_spectrum(result)
        bin_widths = na.calc_bin_widths(energy)
        norm_flux = flux / bin_widths
        plt.step(energy, norm_flux, label=result.beamlines.name, color=color)
        # create table of values
        if save_table is True:
            df = pd.DataFrame({"energy (MeV)": energy, "flux (n/cm2/s/MeV)": norm_flux})
            df.to_csv("spectra_df/" + result.beamlines.name + ".csv", index=False)
    plt.legend(fontsize=12, loc=1)

    if fname is not None:
        save_name = save_fig("spectra_plots", fname)
        plt.savefig(save_name)

    plt.show()
    return dic
    
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

def plot_dose_rate(dfs, key=None, beam=None):
    colors = ["g", "r"]
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(25,8))
    for df, ax, color in zip(dfs, axs, colors):
        plt.rcParams["figure.figsize"] = [20,20]
        df = df.sort_values(by="distance")
        ax.bar(x=df.index, height=df["norm_dose"], yerr=(df["Hr_un%"])/100 * df['norm_dose'], capsize=4, color=color, width=0.6, align="edge")
        ax.set_xlabel('Distance from monolith (m)')
        ax.set_ylabel(r"normalised dose rate ( $\frac{\mu Sv}{35 \mu A})$")
        ax.set_xticks(df.index, df["distance"], rotation="vertical")
        ax.set_ylim(0, 5)
        ax.set_xticklabels(np.round((df["distance"]), 2))
        if color == "g":
            status = "open"
        else:
            status = "closed"
        if key:
            ax.set_title("dose rate for " + key + " beamline with " + beam  + " " + status)
    savename = save_fig("dose_rates", "imat_chipir")
    plt.savefig(savename)
    plt.show()
    
def stack_bar_plot(data_frame, save_name, xlabel=None, ylabel=None, key =None):
    #stack_df = (data_frame.filter(cols)).astype(float)

    ax = data_frame[["thermal", "epi", "fast"]].plot(kind='bar', stacked=True,figsize=(5,5))
    ax.set_xlabel("distance from monolith (m)")
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(data_frame["reference"], rotation = 45)
    ax.set_xticklabels(round(data_frame["distance"], 3))
    if key:
        plt.title("energy distribution bar plot for " + key + " beamline")
    savename = save_fig("energy_bar_plot", save_name)
    plt.savefig(savename+ ".png")
    plt.show()
 
def direction_bar_plot(dataframe, key):
    
    ax = dataframe[["F", "FL", "FR", "R", "RR", "RL"]].plot(kind='bar',figsize = (5,5))
    ax.set_ylabel("Counts")
    ax.set_xticklabels(dataframe["reference"], rotation=60)
    if key:
        plt.title("direction bar plot for " + key + " beamline")
        #plt.savefig("plots/direction_plot_" + key + ".png")
    plt.show()

def direction_stack_bar_plot(df, save_name):
    
    df['sum_dir'] = df[["F", "FL", "FR", "R", "RR", "RL"]].sum(axis=1)
    df["F_norm"] = df["F"]/df["sum_dir"]
    df["FL_norm"] = df["FL"]/df["sum_dir"]
    df["FR_norm"] = df["FR"]/df["sum_dir"]
    df["R_norm"] = df["R"]/df["sum_dir"]
    df["RL_norm"] = df["RL"]/df["sum_dir"]
    df["RR_norm"] = df["RR"]/df["sum_dir"]

    axis = df[["F_norm", "FR_norm", "FL_norm", "R_norm", "RR_norm", "RL_norm"]].plot(kind="bar", stacked=True)
    savefig = save_fig("direction_bar_plot", save_name)
    plt.savefig(savefig + ".png")
    fig = axis.get_figure()

def plot_dose_distance(dfs, key=None):
    fig, ax= plt.subplots(nrows=1, ncols=1, figsize=(20,10))
    colors = ["g", "r"]
    for df, color in zip(dfs, colors): 
        plt.rcParams["figure.figsize"] = [5,5]
        ax.errorbar(df['distance'], df["norm_dose"], yerr=df["norm_dose"]*df["Hr_un%"]/100, marker='x', ls='None', color=color)
        #ax.set_yscale("log")
        ax.set_xlabel('distance (m)')
        ax.set_ylim(0, 5)
        ax.set_ylabel('dose rate ($\mu$ Sv\h)')
        #ax.set_xticklabels(df["reference"], rotation=20)
        if key:
            plt.title("dose rate vs distance for " + key + " beamline")
            # plt.savefig("plots/dose_rate_" + key + ".png")
        path = save_fig("df_plots", "dose_distance")
        plt.savefig(path + ".png")
    plt.show()
def create_meshgrid(x,y):
    return np.meshgrid(x,y)

def find_change_time(data, shutter):
    df = data.out_data[shutter]
    filter = (df.ne(df.shift()))
    change_times = data.out_data[filter].iloc[1:]
    return change_times

def plot_energy_time(data, beamline, shutter):
    if shutter == "own":
        shutter = "shutter-open"
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
        change_times = find_change_time(data, shutter)
        for time, status in zip(change_times["t(s)"], change_times[shutter]):
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
    ax2.set_ylim(0, 40)
    name = da.get_names(data["reference"])[1]
    ax.set_xlabel("Time t(s)")
    
    ax.set_ylabel("percentage (%)")
    ax.legend(loc="upper right")
    ax2.legend(loc="upper center")
    #plt.title("Fast, thermal and epithermal energy distribution over time for " + name + "\n at a distance : {:.2f} m away".format(distance))
    plt.show()

def convert_float(data):
    return np.array(data.astype(float))

def plot_energy_map(data):
    x = data['x']
    y = data['y']
    ther = convert_float(data['Ther%'])
    epi = convert_float(data['Epit%'])
    fast = convert_float(data['Fast%'])
    values = {"thermal": ther, "epithermal":epi, "fast": fast}
    fig, ax = plt.subplots(1, 3, figsize = (22,7))
    for i, (key, value) in enumerate(values.items()):
        
        #scat = ax[i].plot(x,y)
        scat = ax[i].scatter(x, y , c=value, s=4, cmap='jet')
        if data["shutter-open"].all():
            status = "open"
        else:
            status = "closed"
        ax[i].set_title("Percentage of " + key +" neutrons with the beamline shutter " + status)
        plt.colorbar(scat)

def func(x, y):

    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2

def plot_df(dfs, z_labels, levels, cticks, labels=["", ""]):
    for z_label in z_labels:
        df_list = {label: da.split_df_axis(data, z_label) for label, data in zip(labels,dfs)}
        p.plot_dose_map(df_list, z_label,  levels, cticks)
def plot_dose_map(df_dict, z_label, labels, levels, cticks, save_name=["", ""]):

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(17,8))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    for i, (df, ax) in enumerate(zip(df_dict.values(), axs.ravel())):
        scat = contour_plot(df, ax, labels[i], levels, norm=colors.LogNorm())

    cbar = fig.colorbar(scat, ax=axs.ravel().tolist(), ticks=(cticks), format='%1.2f')
    cbar.ax.set_title(r" $\frac{\mu Sv}{35\mu A -hour}$")

    plt.suptitle("Heat map of " + z_label +" distribution in TS2", horizontalalignment='center', x=0.4, y=0.99, fontsize=15)

    if save_name != ["", ""]:
        folder_name = save_name[0]
        file_name = save_name[1]
        savename = save_fig(folder_name, file_name)
        plt.savefig(savename + ".png")

    plt.show()

def contour_plot(df, ax, status, levels, norm):
    # target grid to interpolate to
    xi = np.arange(-40, 40, 0.1)
    yi = np.arange(-60, 60, 0.1)
    xi, yi = np.meshgrid(xi, yi)

    # interpolate
    zi_pos = interpolate.griddata((df[0]["x"],df[0]["y"]), df[0]["z"] ,(xi,yi), method='linear', rescale=True)
    scat = ax.contourf(xi,yi,zi_pos, levels=levels, cmap='viridis', norm=norm)
    zi_neg = interpolate.griddata((df[1]["x"],df[1]["y"]), df[1]["z"],(xi,yi ), method='linear', rescale=True)
    scat = ax.contourf(xi,yi,zi_neg, levels=levels, cmap='viridis', norm=norm)
    ax.scatter(df[0]["x"],df[0]["y"], alpha=0.7, color='black')
    ax.scatter(df[1]["x"], df[1]["y"], alpha=0.7, color='black')
    # ax.plot(x,y,'k.' )
    ax.grid(alpha=0.4, color='black')
    ax.set_title("shutter " + status, fontsize=18)
    ax.set_xlabel('x (m)',fontsize=13)
    ax.set_ylabel('y (m)',fontsize=13)
    return scat
    
def plot_unc_time(data):
    
    change_times = find_change_time(data)
    x = data['out']['t(s)']
    y = data['out']['Hr_un%']
    y2 = data['out']['ts2_current']
    fig = plt.figure(figsize=(8,8))

    ax = fig.add_subplot(111)
    ax2 = fig.add_subplot(111, frame_on=False)
    ax.plot(x, y, color='b', marker=None)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"dose rate % Uncertainty", color='b')
    colors = ['black', 'orange', 'green']

    #plt.axvspan(time, change_times["t(s)"][i+1])
    i = 0
    for time, status in zip(change_times["t(s)"], change_times["shutter-open"]):
        if status is True:
            color = colors[2]
            label = "open"
            i +=1
        else:
            color = colors[0]
            label= "closed"
            i +=1
        plt.vlines(time, 0, max(y2), ls='dashdot', color=color, label=label)
    ax2.plot(x, y2, color='r', marker=None, alpha=0.5)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('current $\mu$ A', color='r') 
    ax2.yaxis.set_label_position('right')
    plt.legend(loc="lower right")
    name = da.get_names(data["reference"])[1]
    distance = da.get_distance(data["reference"])["distance"].values[0]
    plt.title("Comparison between dose rate uncertainty and current over time with the shutter status at " + name + "\n at a distance : {:.2f} m away".format(distance))

def plot_shutter_change(df, name, ax, max_y, colors):
    times = df["t(s)"].to_numpy()
    for time in times:
        status =  df[df["t(s)"] == time][name].values[0]
        if status == True:
            color = colors[0]
        else:
            color = colors[1]
        ax.vlines(time, 0, max_y, ls='dashdot', label =name + " is " + str(status), color=color)
    return

def plot_dose_time(data, save_name=["", ""], selected_shutters="all"):
    folder_name = save_name[0]
    file_name = save_name[1]
    if selected_shutters == "all":
        names = data.beamlines.all_neighbours.index.to_numpy()
        
        sel_names = [name for name in names if name in data.out_data.columns]
    elif selected_shutters == "closest":
        names = data.beamlines.closest_neighbours.index
        names = np.append(names, data.beamlines.name)
        sel_names = [name for name in names if name in data.out_data.columns]
    else:
        # check the selected shutter is valid
        for shutter in selected_shutters:
            if shutter not in data.beamlines.all_neighbours.index:
                raise ValueError("wrong value selected")
        names = selected_shutters
    start = {sel_names[i]: data.out_data[shutter].iloc[0] for i, shutter in enumerate(sel_names)}
    change_times = {sel_names[i]:find_change_time(data, shutter) for i, shutter in enumerate(sel_names)}

    open_start = [name for name, value in start.items() if value]
    closed_start = [name for name, value in start.items() if  not value]
    if open_start == []:
        open_start = ["None"]
    if closed_start == []:
        closed_start = ["None"]
    text = ["Shutter status at start: \n" + r" $\bf{OPEN}$: ", ", ".join(open_start), "\n", r"$\bf{CLOSED}$: ", ", ".join(closed_start)]
    #get starting status then record change in difference
    #todo: generalise for all measurements?
    x = data.out_data['t(s)']
    y = data.out_data['H*(10)r']
    y2 = data.out_data[data.beamlines.current_info]

    # plot parameters
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax2 = fig.add_subplot(111, frame_on=False)
    ax.plot(x, y, color='b', marker=None)
    #if max(y) > 5:
    #ax.set_ylim(0, 2)
    ax.set_xlabel("time (s)")
    #ax.set_xlim(x.iloc[240], max(x))
    ax.set_ylabel(r" normalised dose rate ( $\frac{\mu Sv}{35\mu A -hour})$", color='b')
    colors = {"chipir":['deepskyblue', 'midnightblue'], "imat":['fuchsia', 'purple'], "let":["goldenrod", "darkred"], "nimrod":["lime", "darkolivegreen"],
              "wish":['magenta', 'crimson'], "larmor":['aqua', 'navy'], "offspec":['greenyellow', 'darkgreen'], "inter":['olive', 'olivedrab'], "polref":['lightcoral', 'darkred'], "sans2d":['silver', 'black'], "zoom":['bisque', 'darkorange'],
              "sandals":['deepskyblue', 'midnightblue'], "alf":['fuchsia', 'purple'], "surf":["goldenrod", "darkred"], "crisp":["lime", "darkolivegreen"], "loq":['aqua', 'navy'], "osiris":['lightcoral', 'darkred'], "iris":[], "polaris":['bisque', 'darkorange'], 
              "tosca":['magenta', 'crimson'], "ines":['silver', 'black'], "emma":['olive', 'olivedrab'], "muon_n":["aquamarine", "darkslategrey"], "muon_s":["aquamarine", "darkslategrey"],
              "pearl":['magenta', 'crimson'], "hrpd":['silver', 'black'], "enginx":['fuchsia', 'purple'], "gem":['deepskyblue', 'midnightblue'], "mari":['lightcoral', 'darkred'], 
              "merlin":['aqua', 'navy'], "sxd":['bisque', 'darkorange'], "vesuvio":["goldenrod", "darkred"], "maps":['greenyellow', 'darkgreen']}

    for name, df in change_times.items():
        for time, status in zip(df["t(s)"], df[name]):
            if status is True:
                color = colors[name][0]
                label = name + " shutter open"
            else:
                color = colors[name][1]
                label= name + " shutter closed"
            ax.vlines(time, 0, max(y), ls='dashdot', color=color, label=label, linewidths=1.5)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.scatter(x, y2, color='r', marker="x", alpha=0.5)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('current $\mu$ A', color='r') 
    ax2.yaxis.set_label_position('right')
    ax2.set_ylim(0,160)
    #ax2.set_xlim(x.iloc[240], max(x))
    plt.legend(by_label.values(), by_label.keys(), loc="upper right")
    name = data.beamlines.name
    data.find_distance()
    distance = data.distance
    #plt.xlim(x.iloc[240], max(x))
    plt.figtext(0.5, -0.03, " ".join(text), ha="center", fontsize=12)
    plt.title("Comparison between dose rate and current with the shutter status at " + name  + " (" + data.reference["Measurement Reference"][0] + ")\n at a distance : {:.2f} m away".format(distance))
    if  save_name== ["", ""]:
        path = save_fig(str(data.file_name), "dose_time")
    else:
        path = save_fig(folder_name, file_name)
    plt.savefig(path+ ".png", bbox_inches="tight")
    plt.show()

def save_fig(folder_name, file_type):
    dir_path = os.path.dirname(__file__)
    results_dir = os.path.join(dir_path, "plots\\" + str(folder_name) + "\\")
    name = file_type
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, name)