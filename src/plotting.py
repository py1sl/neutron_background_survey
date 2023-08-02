import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import colors
from matplotlib import cm
import numpy as np
import src.diamon_analysis as da
import src.neutronics_analysis as na
from scipy import interpolate
import os
from diamon import diamon
import pandas as pd

class shutter_colors:
    def __init__(self):
        self.colors = {"chipir":['deepskyblue', 'midnightblue'], "imat":['fuchsia', 'purple'], 
                       "let":["goldenrod", "darkred"], "nimrod":["lime", "darkolivegreen"], "wish":['magenta', 'crimson'], 
                       "larmor":['aqua', 'navy'], "offspec":['greenyellow', 'darkgreen'], "inter":['olive', 'olivedrab'], 
                       "polref":['lightcoral', 'darkred'], "sans2d":['silver', 'black'], "zoom":['bisque', 'darkorange'],
                        "sandals":['deepskyblue', 'midnightblue'], "alf":['fuchsia', 'purple'], 
                        "surf":["goldenrod", "darkred"], "crisp":["lime", "darkolivegreen"], "loq":['aqua', 'navy'], 
                        "osiris":['lightcoral', 'darkred'], "iris":["thistle", "indigo"], "polaris":['bisque', 'darkorange'], 
                        "tosca":['magenta', 'crimson'], "ines":['silver', 'black'], "emma":['olive', 'olivedrab'],
                        "muon_n":["aquamarine", "darkslategrey"], "muon_s":["aquamarine", "darkslategrey"],
                        "pearl":['magenta', 'crimson'], "hrpd":['silver', 'black'], "enginx":['fuchsia', 'purple'], 
                        "gem":['deepskyblue', 'midnightblue'], "mari":['lightcoral', 'darkred'], "merlin":['aqua', 'navy'],
                        "sxd":['bisque', 'darkorange'], "vesuvio":["goldenrod", "darkred"], "maps":['greenyellow', 'darkgreen']}
    def get_color(self, shutters : list[str]):
        colors = {shutter : self.colors[shutter] for shutter in shutters}
        return colors

def plot_heat_map(df_list : list[list[list[pd.DataFrame]]], z_labels : list[str], bound : list[float],
                  titles:list[str], levels: np.array, cticks: np.array, norm : bool =False, save_name=["", ""]):
    """This function plots a dose map for diamon data using a 3d list of df

    Args:
        df_list (list[list[pd.DataFrame]]):  3d list of df. each 1d list of df is first pos and negative y, 
        then shutter comparison, then extra rows of plots if necessary
        z_labels (list[str]): list of labels of the heat map
        levels (np.array): 
        cticks (np.array): _description_
    """
    nrows = len(df_list)
    ncols = len(df_list[0])
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16,8))
    # pot split df by axis
    for title, dfs in zip(titles, df_list):
        for i, (ax, df) in enumerate(zip(axs.ravel(), dfs)):
            dfs = da.split_df_axis(df, title)
            scat = contour_plot(ax, dfs, z_labels[i], levels, norm, bound)
        cbar = fig.colorbar(scat, ax=axs.ravel().tolist(), ticks=(cticks), format='%1.3f')
        cbar.ax.set_title(r" $\frac{\mu Sv}{35\mu A -hour}$")

        plt.suptitle("Heat map of " + title +" distribution in TS2", horizontalalignment='center', x=0.4, y=0.99, fontsize=15)

    if save_name != ["", ""]:
        savename = save_fig(save_name[0], save_name[1])
        plt.savefig(savename + ".png")
    plt.show()

def meshgrid(xmin, xmax, ymin, ymax, step):
    """
    creates numpy meshgrid from rectangular boundaries
    Args:
        xmin (float): 
        xmax (float): 
        ymin (float): 
        ymax (float): 
        step (float): 

    Returns:
        floats: _description_
    """
    xi = np.arange(xmin, xmax, step)
    yi = np.arange(ymin, ymax, step)
    xi, yi = np.meshgrid(xi, yi)
    return xi, yi

def interpolate_data(ax, x, y, z, mesh, levels, norm : bool, method="linear", cmap="jet"):
    """_summary_

    Args:
        ax (_type_): 
        x (_type_): 
        y (_type_): 
        z (_type_): 
        mesh (_type_): 
        levels (_type_): 
        norm (bool): 
        method (str, optional): . Defaults to "linear".
        cmap (str, optional): . Defaults to "jet".

    Returns:
        _type_: _description_
    """
    zi = interpolate.griddata((x,y), z ,(mesh[0],mesh[1]), method=method, rescale=True)
    if norm == True:
        contourf = ax.contourf(mesh[0], mesh[1], zi, levels=levels, cmap=cmap, norm=colors.LogNorm())
    else:
        contourf = ax.contourf(mesh[0], mesh[1], zi, levels=levels, cmap=cmap)
    return contourf
    
def contour_plot(ax, dfs, title, levels, norm, bound : list[float]):
    """_summary_

    Args:
        ax (_type_): _description_
        dfs (_type_): _description_
        title (_type_): _description_
        levels (_type_): _description_
        norm (_type_): _description_
        bound (list[float]): _description_

    Returns:
        _type_: _description_
    """
    # target grid to interpolate to
    

    # interpolate - check whether df is pos and neg or just one axis
    xi, yi = meshgrid(bound[0], bound[1], bound[2], bound[3], 0.1)
    # df is pos and negative y to separate from monolith
    for df in dfs:
        scat = interpolate_data(ax, df["x"], df["y"], df["z"] ,[xi,yi], levels, norm)
        ax.scatter(df["x"],df["y"], alpha=0.6, color='black', marker="x")
    ax.grid(alpha=0.4, color='black')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('x (m)',fontsize=13)
    ax.set_ylabel('y (m)',fontsize=13)
    ax.set_xlim(bound[0], bound[1])
    ax.set_ylim(bound[2], bound[3])
    return scat

def plot_spectra(data : list[diamon], fname: str =None, title : str="", save_table : bool=False):
    """
    Plot of spectrum - flux vs energy for a measurement.
    split across thermal, epithermal and fast neutrons

    Args:
        data (list[diamon]): list of diamon class instance
        fname (str, optional): name of file to save. Defaults to None.
        title (str, optional): title of spectra plot. Defaults to "".
        save_table (bool, optional): option to save table of spectra to csv. Defaults to False.

    Returns:
        dic: dictionary of
    """
    colors = cm.rainbow(np.linspace(0,1, len(data)))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Energy (MeV)")
    plt.ylabel("Flux (cm$^{-2}$s$^{-1}$ MeV$^{-1}$)")
    plt.title("Neutron typical energy spectra " + title)
    plt.ylim(10e-4, 20e8)
    for color, result in zip(colors, data):
        energy, flux = da.extract_spectrum(result)
        bin_widths = na.calc_bin_widths(energy)
        norm_flux = flux / bin_widths
        plt.step(energy, norm_flux, label=(result.beamlines.name + " - " + str(result.reference["Measurement Reference"].iloc[0])), color=color)
        # create table of values
        if save_table is True:
            df = pd.DataFrame({"energy (MeV)": energy, "flux (n/cm2/s/MeV)": norm_flux})
            df.to_csv("spectra_df/" + result.beamlines.name + ".csv", index=False)
    plt.legend(fontsize=12, loc=1)

    if fname is not None:
        save_name = save_fig("spectra_plots", fname)
        plt.savefig(save_name)

    plt.show()

def plot_channel_time(channel_dic : dict[pd.DataFrame], dates : list[str] = None):
    """plots a channel from influx db over a period of time

    Args:
        channel_dic (dict[pd.dataframe]): dictr of channel df
        cycles (list[str], optional): _description_. Defaults to None.
    """
    fig, axs = plt.subplots(nrows=1, ncols=len(channel_dic), figsize=(18,7))
    fig.suptitle("Raw channel data over a continuous stretch between " + str(dates[0]) + " and " + str(dates[1]), fontsize = 18)
    for name, ax in zip(channel_dic.keys(), axs):
        ax.set_title(name)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
        ax.plot(channel_dic[name]["_value"])
        ax.set_ylabel("channel value")
        ax.set_xlabel("date")
    plt.show()

def find_change_time(data, shutter):
    """
    find point in time a shutter/parameter changes
    Args:
        data (_type_): _description_
        shutter (_type_): _description_

    Returns:
        _type_: _description_
    """
    df = data.out_data[shutter]
    filter = (df.ne(df.shift()))
    change_times = data.out_data[filter].iloc[1:]
    return change_times

def plot_dose_time(data : diamon, label : str, selected_shutters : str = "all", shutters : list[str] = None,
                    save_name : list[str] = ["", ""]):
    """
    plots dose vs time with current as background in plot showing changes in shuttert
    Args:
        df (pd.DataFrame): df of diamon information
        label (str): string paramater to extract eg dose
        selected_shutters (str, optional): which shutters to analyse change - either a select one, neighbours or all. Defaults to "all".
        save_name (list[str], optional): folder and file name to save plot to. Defaults to ["", ""].
    """
    sel_names = da.select_shutters(data, selected_shutters, shutters)
    change_times = {sel_names[i]:find_change_time(data, shutter) for i, shutter in enumerate(sel_names)}
    #---------------------------------------------------------------------
    #fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=1, figsize=((8,8)))
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(111)
    ax2 = fig.add_subplot(111, frame_on=False)
    plot_shutter_change(change_times, ax1)

    ax1.set_xlabel("time (s)")
    ax1.set_ylabel(r" normalised dose rate ( $\frac{\mu Sv}{35\mu A -hour})$", color='b')
    ax2.yaxis.tick_right()
    ax2.set_ylabel('current $\mu$ A', color='r') 
    ax2.yaxis.set_label_position('right')

    ax1.plot(data.out_data["t(s)"], data.out_data[label], color='b', marker=None)
    ax2.scatter(data.out_data["t(s)"], data.out_data[data.beamlines.current_info], color='r', alpha=0.5, label="current", marker="*")    
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper right")
    plt.title("Comparison between dose rate and current with the shutter status at " + data.beamlines.name  + " (" + data.reference["Measurement Reference"][0] + ")\n at a distance : {:.2f} m away".format(data.distance))
    initial_shutter = get_initial_shutter_value(data.out_data, sel_names)
    text = ["Shutter status at start: \n" + r" $\bf{OPEN}$: ", ", ".join(initial_shutter[0]), "\n", r"$\bf{CLOSED}$: ", ", ".join(initial_shutter[1])]
    plt.figtext(0.5, -0.03, " ".join(text), ha="center", fontsize=12)

    if save_name != ["", ""]:
        path = p.save_fig(save_name[0], save_name[1])
        plt.savefig(path+ ".png", bbox_inches="tight")
    plt.show()

def get_initial_shutter_value(df : pd.DataFrame, names :list[str]):
    """
    extracts which shutters are open or closed at start
    Args:
        df (pd.DataFrame): 
        names (list[str]): 
    Returns:
        list of str descripting which shutters are open/closed at the start
    """
    start = {names[i]: df[shutter].iloc[0] for i, shutter in enumerate(names)}
    open_start = [name for name, value in start.items() if value]
    closed_start = [name for name, value in start.items() if  not value]
    open_start = fill_empty_list(open_start)
    closed_start = fill_empty_list(closed_start)
    return [open_start, closed_start]

def fill_empty_list(x, replace = "None"):
    """
    x : list of unknown size
    replace : what to replace empty list with
    """
    if x == []:
        return [replace]
    else:
        return x

def plot_shutter_change(change_times : dict[pd.DataFrame], ax : plt.axes):
    """ 
    This function plots a vline at a point in time where the value of the shutter changes

    Args:
        change_times (dict[pd.DataFrame]) : 
        ax (plt.axes): _description_
        colors (_type_): _description_
    """
    shutters = change_times.keys()
    colors = p.shutter_colors().get_color(shutters)
    for name, df in change_times.items():             
        for time in df["t(s)"].to_numpy():
            status =  df[df["t(s)"] == time][name].values[0]
            if status == True:
                color = colors[name][0]
            else:
                color = colors[name][1]
            ax.vlines(time, ymin=0, ymax=1, ls='dashdot', label =name + " is " + str(status), color=color, transform=ax.get_xaxis_transform())

def plot_energy_time(data, beamline, shutter):
    if shutter == "own":
        shutter = "shutter-open"
    fast = data.out_data["Fast%"].astype(float)
    epi = data.out_data["Epit%"].astype(float)
    therm = data.out_data["Ther%"].astype(float)
    times = data.out_data["t(s)"]
    ts2_current = data.out_data["ts2_current"]
    
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
    plt.title("Fast, thermal and epithermal energy distribution over time for " + name + "\n at a distance : {:.2f} m away".format(data.distance))
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

def save_fig(folder_name : str, file_name : str):
    """
    creates folder path to save figures to using folder name and file name
    Args:
        folder_name (str): _description_
        file_name (str): _description_

    Returns:
        os.path: os path directory to saved folder
    """
    dir_path = os.path.dirname(__file__)
    results_dir = os.path.join(dir_path, "plots\\" + str(folder_name) + "\\")
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, file_name)