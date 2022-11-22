"""Visualization tools for OpArt Library."""


import json
import math
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from numpy import ma
from mpl_toolkits.axes_grid1 import make_axes_locatable

import geopandas
import contextily as ctx


import src as sc


COLOR_MAP = {
    "light_orange": "#E69F00",
    "light_blue": "#56B4E9",
    "teal": "#009E73",
    "yellow": "#F0E442",
    "dark_blue": "#0072B2",
    "dark_orange": "#D55E00",
    "pink": "#CC79A7",
    "purple": "#9370DB",
    "black": "#000000",
    "silver": "#DCDCDC",
}


def assignment_heatmat(assignment_df):
    """Returns heatmap of art/building assignment probabilities."""
    fig, ax = plt.subplots(figsize=(10, 12))  # pylint: disable=unused-variable
    assignment = assignment_df.values

    # Create masked array to make 0 values transparent
    data = np.array(assignment).transpose()
    data = np.array(data, dtype=float)
    data = ma.masked_invalid(np.where(data == 0, np.nan, data))

    heatmap = ax.pcolor(data, cmap="Purples", vmin=np.nanmin(data), vmax=np.nanmax(data))

    # add reference colorbar
    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size="1%", pad=0.7, pack_start=True)
    fig.add_axes(cax)
    fig.colorbar(heatmap, cax=cax, orientation="horizontal")

    # want a more natural, table-like display
    ax.invert_yaxis()

    # move x labels to the top of the chart
    ax.xaxis.tick_top()

    ax.set_xticks(0.5 + np.arange(assignment_df.shape[0]))
    ax.set_xticklabels(assignment_df.index)

    # rotate x labels 45 degrees
    plt.setp(ax.get_xticklabels(), rotation=90, ha="left", va="center", rotation_mode="anchor")

    # Get ytick strings
    ax.set_yticks(0.5 + np.arange(assignment_df.shape[1]))
    ax.set_yticklabels(assignment_df.columns)

    plt.show()


def campus_building_map():
    """Returns labeled campus map."""
    names = sc.hall_url_name_dict

    # Opening JSON file
    with open("../data/hall_dict.json", "r", encoding="utf-8") as file:
        hall_dict = json.load(file)

    # Load campus building detials.
    loc_df = pd.DataFrame(
        index=list(hall_dict.keys()), columns=["name", "hall_type", "longitude", "latitude"]
    )
    for key, val in hall_dict.items():
        loc_df.loc[key, "name"] = names[key]
        loc_df.loc[key, "hall_type"] = val["hall_type"]
        loc_df.loc[key, "longitude"] = val["longitude"]
        loc_df.loc[key, "latitude"] = val["latitude"]

    color_map = {
        "Academic Building": COLOR_MAP["light_blue"],
        "Residence Halls": COLOR_MAP["light_orange"],
        "Other": COLOR_MAP["pink"],
        "Unknown": COLOR_MAP["teal"],
    }

    medford_df = loc_df[loc_df["latitude"] >= 42.4]

    fig, ax = plt.subplots(figsize=(10, 10))  # pylint: disable=unused-variable

    # Create geopandas dataframe
    geo_df = geopandas.GeoDataFrame(
        medford_df, geometry=geopandas.points_from_xy(medford_df.longitude, medford_df.latitude)
    )

    # Add spherical coordinate reference system (crs) to lat/long pairs.
    geo_df.crs = "EPSG:4326"

    # Project onto a flat crs for mapping.
    geo_df = geo_df.to_crs(epsg=3857)

    # Add color codings
    color = [color_map[h] for h in geo_df["hall_type"]]

    # Plot listings as points.
    geo_df.plot(ax=ax, marker="o", markersize=50, c=color)

    # Add basemap behind geopandas plot.
    ctx.add_basemap(ax, zoom=17, alpha=0.4)

    # Annotate buildings of type "Other".
    x_coord = geo_df[geo_df["name"] == "Goddard Chapel"].geometry.x
    y_coord = geo_df[geo_df["name"] == "Goddard Chapel"].geometry.y

    ax.annotate(
        "Goddard Chapel", xy=(x_coord, y_coord), xytext=(-40, 7), textcoords="offset points"
    )

    x_coord = geo_df[geo_df["name"] == "Gifford House"].geometry.x
    y_coord = geo_df[geo_df["name"] == "Gifford House"].geometry.y

    ax.annotate("Gifford House", xy=(x_coord, y_coord), xytext=(-40, 7), textcoords="offset points")

    for key, val in color_map.items():
        plt.scatter([], [], color=val, label=key)

    # Turn off axes
    ax.set_axis_off()
    ax.legend()
    plt.show()

    boston_df = loc_df[loc_df["latitude"] < 42.4]

    fig, ax = plt.subplots(figsize=(10, 10))  # pylint: disable=unused-variable

    # Create geopandas dataframe
    geo_df = geopandas.GeoDataFrame(
        boston_df, geometry=geopandas.points_from_xy(boston_df.longitude, boston_df.latitude)
    )

    # Add spherical coordinate reference system (crs) to lat/long pairs.
    geo_df.crs = "EPSG:4326"

    # Project onto a flat crs for mapping.
    geo_df = geo_df.to_crs(epsg=3857)

    # Add color codings
    color = [color_map[h] for h in geo_df["hall_type"]]

    # Plot listings as points.
    geo_df.plot(ax=ax, marker="o", markersize=50, c=color)

    # Add Buffer
    pts = geopandas.GeoSeries(geo_df["geometry"])
    circles = pts.buffer(100)
    circles.plot(ax=ax, marker="o", markersize=50, alpha=0)

    x_coord = geo_df[geo_df["name"] == "School at the Museum of Fine Arts"].geometry.x
    y_coord = geo_df[geo_df["name"] == "School at the Museum of Fine Arts"].geometry.y

    ax.annotate("SMFA", xy=(x_coord, y_coord), xytext=(-13, 7), textcoords="offset points")

    # Add basemap behind geopandas plot.
    ctx.add_basemap(ax, zoom=17, alpha=0.4)

    # Turn off axes
    ax.set_axis_off()
    plt.show()


def beeswarm_building_gender(
    demo_cat, building_list=[]
):  # pylint: disable=dangerous-default-value, too-many-locals
    """Returns beeswarm plot of counts by race and gender.

    Input:
        demo_cat: (string) "race" or "region"
        building_list: (list) string building names.

    Returns:
        Beeswarm style horizontal bar chart of total counts.
    """
    my_path = "../data/filled_buildings/"
    student_df = sc.get_student_enrollment_data()
    hall_df = sc.get_hall_by_school_table()
    if len(building_list) == 0:
        building_list = list(hall_df.index)

    color_dict = {"Woman": "#c85194", "Man": "#f1db54", "Transgender": COLOR_MAP["teal"]}

    n_rows = int(math.ceil(len(building_list) ** (1 / 2)))
    n_cols = math.ceil(len(building_list) / n_rows)

    fig, ax = plt.subplots(  # pylint: disable=unused-variable
        n_rows, n_cols, figsize=(14, 18), sharex=True, sharey=True
    )

    y_labels = list(student_df[demo_cat].value_counts().keys())

    for idx, name in enumerate(building_list):

        if n_cols == 1:
            axs = ax[idx]
        else:
            axs = ax[idx // n_rows, idx % n_cols]

        my_file = my_path + name + "_students.csv"
        name = sc.hall_short_name_dict.get(name, name)

        building_df = pd.read_csv(my_file.lower())
        grouped_df = building_df.groupby(demo_cat)

        count_df = pd.DataFrame(0, index=y_labels, columns=["Transgender", "Woman", "Man"])

        for group in building_df[demo_cat].unique():
            gender_dict = grouped_df.get_group(group)["gender"].value_counts().to_dict()

            for key, val in gender_dict.items():
                count_df.loc[group, key] = val

        y_tick_val = 0
        y_ticks = []
        for count_idx in count_df.index:
            for col in count_df.columns:

                x_values = np.arange(count_df.loc[count_idx, col]) + np.random.normal(
                    0, 10, count_df.loc[count_idx, col]
                )
                y_values = np.full(len(x_values), float(y_tick_val))
                y_values = y_values + np.random.normal(
                    0, 0.5, len(y_values)
                )  # add gaussian jitter.

                s_values = np.full(len(x_values), 5)  # set marker size.

                axs.scatter(x_values, y_values, s=s_values, color=color_dict[col])
                y_tick_val = y_tick_val + 1

            y_ticks.append(y_tick_val - 1.5)
            y_tick_val = y_tick_val + 0.75
        axs.set_title(name, fontsize=25)

        axs.set_yticks(y_ticks)
        axs.set_yticklabels(count_df.index, fontsize=15)

        axs.spines["right"].set_visible(False)
        axs.spines["top"].set_visible(False)

    for j in range(len(building_list), n_rows**2):
        if n_cols > 1:
            ax[j // n_rows, j % n_cols].axis("off")

    # add legend
    if n_cols == 1:
        axs = ax[0]
    else:
        axs = ax[0, 0]
    axs.scatter([], [], color="#f1db54", s=[12], label="Man")
    axs.scatter([], [], color="#c85194", s=[12], label="Woman")
    axs.scatter([], [], color=COLOR_MAP["teal"], s=[12], label="Transgender or Non-Binary")
    # handles, labels = axs.get_legend_handles_labels()
    plt.show()


def beeswarm_gender(
    demo_df, demo_cat, title="Total Count by Race and Gender"
):  # pylint disable=too-many-locals
    """Returns beeswarm plot of counts by race and gender.

    Input:
        demo_df: (dataframe) student_df or art_df
        demo_cat: (string) "race" or "region"
        title: (string) plot title.

    Returns:
        Beeswarm style horizontal bar chart of total counts.
    """
    student_df = sc.get_student_enrollment_data()
    y_labels = list(student_df[demo_cat].value_counts().to_dict().keys())

    color_dict = {"Woman": "#c85194", "Man": "#f1db54", "Transgender": COLOR_MAP["teal"]}

    grouped_df = demo_df.groupby(demo_cat)
    count_df = pd.DataFrame(0, index=y_labels, columns=["Transgender", "Woman", "Man"])

    for group in demo_df[demo_cat].unique():
        gender_dict = grouped_df.get_group(group)["gender"].value_counts().to_dict()

        for key, val in gender_dict.items():
            count_df.loc[group, key] = val

    fig, ax = plt.subplots(figsize=(10, 7))  # pylint: disable=unused-variable
    y_tick_val = 0
    y_ticks = []
    for idx in count_df.index:
        for col in ["Transgender", "Woman", "Man"]:

            x_values = np.arange(count_df.loc[idx, col]) + np.random.normal(
                0, 10, count_df.loc[idx, col]
            )
            y_values = np.full(len(x_values), float(y_tick_val))
            y_values = y_values + np.random.normal(0, 0.5, len(y_values))  # add gaussian jitter.

            s_values = np.full(len(x_values), 5)  # set marker size.

            ax.scatter(x_values, y_values, s=s_values, color=color_dict[col])
            y_tick_val = y_tick_val + 1

        y_ticks.append(y_tick_val - 1.5)
        y_tick_val = y_tick_val + 0.75

    ax.scatter([], [], color="#f1db54", s=[25], label="Man")
    ax.scatter([], [], color="#c85194", s=[25], label="Woman")
    ax.scatter([], [], color=COLOR_MAP["teal"], s=[25], label="Transgender or Non-Binary")

    ax.set_xlabel("Count")
    ax.set_title(title, fontsize=18)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(count_df.index, fontsize=15)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.ylim(-0.5, y_tick_val)
    plt.legend()
    plt.show()
