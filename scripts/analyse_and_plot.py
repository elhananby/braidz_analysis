import braidz_analysis as bz
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import sys


def main(args):
    # first check all files exist
    file_list = args.files
    base_folder = args.base_folder

    if base_folder is not None:
        file_list = [os.path.join(base_folder, f) for f in file_list]

    for f in file_list:
        if not os.path.exists(f):
            raise FileNotFoundError(f"File {f} not found")

    # read all data
    data = bz.braidz.read_multiple_braidz(file_list)

    # create output folder (if it doesn't exist)
    output_folder = args.output
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # run the analysis
    analysis = args.analysis

    # Check which analysis to run
    if analysis == "opto":
        output_data = bz.processing.get_opto_data(data["df"], data["opto"])
    elif analysis == "saccade":
        output_data = bz.processing.get_all_saccades(data["df"])
    elif analysis == "stim":
        output_data = bz.processing.get_stim_data(data["df"], data["stim"])
    else:
        raise ValueError(f"Analysis type {analysis} not recognised")

    # create figure
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(ncols=4, nrows=2, width_ratios=[0.1, 1, 1, 1], figure=fig)

    ## STIM/OPTO-CENTERED PLOTS
    # first left plot is data label
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis("off")
    ax1.text(0.5, 0.5, "Stim/Opto-centered", fontsize=16, ha="center")

    # plot angular velocity
    ax1 = fig.add_subplot(gs[0, 1])
    bz.plotting.plot_mean_and_std(
        np.abs(np.rad2deg(output_data["angular_velocity"])), ax1
    )
    ax1.set_xlabel("Frames")
    ax1.set_ylabel("Angular velocity (deg/s)")

    # plot linear velocity
    ax2 = fig.add_subplot(gs[0, 2])
    bz.plotting.plot_mean_and_std(output_data["linear_velocity"], ax2)
    ax2.set_xlabel("Frames")
    ax2.set_ylabel("Linear velocity (m/s)")

    # plot heading change
    ax3 = fig.add_subplot(gs[0, 3], projection="polar")
    bz.plotting.plot_histogram(output_data["heading_difference"], ax3, density=True)
    ax3.set_theta_zero_location("N")

    ## SACCADE-CENTERED PLOTS
    # first left plot is data label
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis("off")
    ax4.text(0.5, 0.5, "Saccade-centered", fontsize=16, ha="center")

    # plot angular velocity
    ax5 = fig.add_subplot(gs[1, 1])
    bz.plotting.plot_mean_and_std(
        np.abs(np.rad2deg(output_data["angular_velocity_peak_centered"])), ax5
    )
    ax5.set_xlabel("Frames")
    ax5.set_ylabel("Angular velocity (deg/s)")

    # plot linear velocity
    ax6 = fig.add_subplot(gs[1, 2])
    bz.plotting.plot_mean_and_std(output_data["linear_velocity_peak_centered"], ax6)
    ax6.set_xlabel("Frames")
    ax6.set_ylabel("Linear velocity (m/s)")

    # plot heading change
    ax7 = fig.add_subplot(gs[1, 3], projection="polar")
    bz.plotting.plot_histogram(
        output_data["heading_difference_peak_centered"], ax7, density=True
    )
    ax7.set_theta_zero_location("N")

    # save the figure
    fig.savefig(os.path.join(output_folder, f"{analysis}_data.png"))
    plt.close(fig)

    # exit
    print(f"Analysis complete. Data saved to {output_folder}")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse and plot braidz data")

    # parse a list of files
    parser.add_argument(
        "files", metavar="F", type=str, nargs="+", help="list of files to analyse"
    )

    # add optional "base folder" argument
    parser.add_argument("--base_folder", type=str, help="base folder for data files")

    # add optional "label" argument
    parser.add_argument("--label", type=str, help="label for the data")

    # parse the output folder
    parser.add_argument(
        "--output", type=str, default="output", help="output folder for plots"
    )

    # parse the analysis type
    parser.add_argument(
        "--analysis",
        type=str,
        default="all",
        help="analysis type: saccade, opto, or stim",
        choices=["saccade", "opto", "stim"],
    )

    args = parser.parse_args()
    main(args)
