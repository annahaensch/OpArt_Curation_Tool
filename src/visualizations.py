import pandas as pd
import numpy as np

import os
import sys

import matplotlib.pyplot as plt
import numpy.ma as ma
from mpl_toolkits.axes_grid1 import make_axes_locatable


ROOT = os.popen("git rev-parse --show-toplevel").read().split("\n")[0]
sys.path.append(ROOT)

import src as sc
import geopandas
import contextily as ctx  
import json

from os import listdir
from os.path import isfile, join
import math

COLOR_MAP = {"light_orange":"#E69F00",
             "light_blue":"#56B4E9",
             "teal":"#009E73",
             "yellow":"#F0E442",
             "dark_blue":"#0072B2",
             "dark_orange":"#D55E00",
             "pink":"#CC79A7",
             "purple":"#9370DB",
             "black":"#000000",
             "silver":"#DCDCDC"}

def assignment_heatmat(assignment_df):
    """ Returns heatmap of art/building assignment probabilities.
    """
    fig, ax = plt.subplots(figsize = (10,12))
    P = assignment_df.values

    # Create masked array to make 0 values transparent
    data = np.array(P).transpose()
    data = np.array(data, dtype = float)
    data = ma.masked_invalid(np.where(data ==0, np.nan, data))

    heatmap = ax.pcolor(data, cmap="Purples", 
                        vmin=np.nanmin(data), vmax=np.nanmax(data))

    # add reference colorbar
    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size="1%", pad=0.7, pack_start=True)
    fig.add_axes(cax)
    fig.colorbar(heatmap, cax=cax, orientation="horizontal")

    # want a more natural, table-like display
    ax.invert_yaxis()

    # move x labels to the top of the chart 
    ax.xaxis.tick_top()

    ax.set_xticks(.5 + np.arange(assignment_df.shape[0]))
    ax.set_xticklabels(assignment_df.index)

    # rotate x labels 45 degrees
    plt.setp(ax.get_xticklabels(), 
             rotation=90, 
             ha="left",
             va="center",
             rotation_mode="anchor")

    # Get ytick strings
    ax.set_yticks(.5 + np.arange(assignment_df.shape[1]))
    ax.set_yticklabels(assignment_df.columns)

    plt.show()
    

def campus_building_map():
    """ Returns labeled campus map.
    """
    names = sc.hall_url_name_dict
    
    # Opening JSON file
    f = open('../data/hall_dict.json')
    hall_dict = json.load(f)

    # Load campus building detials.
    loc_df = pd.DataFrame(index = list(hall_dict.keys()), 
        columns = ["name","hall_type","longitude","latitude"])
    for k,v in hall_dict.items():
        loc_df.loc[k,"name"] = names[k]
        loc_df.loc[k,"hall_type"] = v["hall_type"]
        loc_df.loc[k,"longitude"] = v["longitude"]
        loc_df.loc[k,"latitude"] = v["latitude"]
            
    color_map = {"Academic Building":COLOR_MAP["light_blue"],
             "Residence Halls":COLOR_MAP["light_orange"],
             "Other":COLOR_MAP["pink"], 
             "Unknown":COLOR_MAP["teal"]}


    medford_df = loc_df[loc_df["latitude"] >= 42.4]

    fig, ax = plt.subplots(figsize = (10,10))

    # Create geopandas dataframe
    geo_df = geopandas.GeoDataFrame(medford_df, geometry=geopandas.points_from_xy(medford_df.longitude, medford_df.latitude))

    # Add spherical coordinate reference system (crs) to lat/long pairs.
    geo_df.crs = "EPSG:4326" 

    # Project onto a flat crs for mapping.
    geo_df = geo_df.to_crs(epsg=3857) 

    # Add color codings 
    c = [color_map[h] for h in geo_df["hall_type"]]

    # Plot listings as points.
    geo_df.plot(ax = ax, marker = "o", markersize = 50, c = c)

    # Add basemap behind geopandas plot.
    ctx.add_basemap(ax, zoom = 17, alpha = 0.4)

    # Annotate buildings of type "Other".
    x = geo_df[geo_df['name'] == "Goddard Chapel"].geometry.x
    y = geo_df[geo_df['name'] == "Goddard Chapel"].geometry.y

    ax.annotate("Goddard Chapel", xy=(x, y), xytext=(-40, 7), textcoords="offset points")

    x = geo_df[geo_df['name'] == "Gifford House"].geometry.x
    y = geo_df[geo_df['name'] == "Gifford House"].geometry.y

    ax.annotate("Gifford House", xy=(x, y), xytext=(-40, 7), textcoords="offset points")

    for k,v in color_map.items():
        plt.scatter([],[],color = v, label = k)

    # Turn off axes
    ax.set_axis_off()
    ax.legend()
    plt.show()

    boston_df = loc_df[loc_df["latitude"] < 42.4]

    fig, ax = plt.subplots(figsize = (10,10))

    # Create geopandas dataframe
    geo_df = geopandas.GeoDataFrame(boston_df, geometry=geopandas.points_from_xy(boston_df.longitude, boston_df.latitude))

    # Add spherical coordinate reference system (crs) to lat/long pairs.
    geo_df.crs = "EPSG:4326" 

    # Project onto a flat crs for mapping.
    geo_df = geo_df.to_crs(epsg=3857) 

    # Add color codings 
    c = [color_map[h] for h in geo_df["hall_type"]]

    # Plot listings as points.
    geo_df.plot(ax = ax, marker = "o", markersize = 50, c = c)

    # Add Buffer
    pts = geopandas.GeoSeries(geo_df["geometry"])
    circles = pts.buffer(100)
    circles.plot(ax = ax, marker = "o", markersize = 50, alpha = 0)

    x = geo_df[geo_df['name'] == 'School at the Museum of Fine Arts'].geometry.x
    y = geo_df[geo_df['name'] == 'School at the Museum of Fine Arts'].geometry.y

    ax.annotate("SMFA", xy=(x, y), xytext=(-13, 7), textcoords="offset points")

    # Add basemap behind geopandas plot.
    ctx.add_basemap(ax, zoom = 17, alpha = 0.4)

    # Turn off axes
    ax.set_axis_off()
    plt.show()


def beeswarm_building_gender(demo_cat, title, building_list = list(sc.hall_short_name_dict.keys())):
    """ Returns beeswarm plot of counts by race and gender. 

    Input:
        demo_cat: (string) "race" or "region" 
        title: (string)
        building_list: (list) string building names.

    Returns: 
        Beeswarm style horizontal bar chart of total counts.
    """
    my_path = "../data/filled_buildings/"
    hall_df, student_df, art_df = sc.load_data()

    color_dict = {"Woman":"#c85194",
                  "Man":"#f1db54",
                  "Transgender":COLOR_MAP["teal"]}

    n = int(math.ceil(len(building_list) **(1/2)))
    m = math.ceil(len(building_list) / n)

    fig, ax = plt.subplots(n,m, figsize = (14,18), sharex = True, sharey = True)

    y_labels = list(student_df[demo_cat].value_counts().keys())

    for i in range(len(building_list)):

        if m == 1:
            axs = ax[i]
        else:
            axs = ax[i//n, i%m]

        my_file = join(my_path, building_list[i]+"_students.csv")
        name = sc.hall_short_name_dict.get(building_list[i],building_list[i])
        
        df = pd.read_csv(my_file)
        grouped_df = df.groupby(demo_cat)

        count_df = pd.DataFrame(0, index = y_labels, 
            columns = ["Transgender","Woman","Man"])
        
        for y in df[demo_cat].unique():
            gender_dict = grouped_df.get_group(y)["gender"].value_counts().to_dict()

            for k,v in gender_dict.items():
                count_df.loc[y,k] = v

        y = 0
        y_ticks = []
        for idx in count_df.index:  
            for col in count_df.columns:

                x_values = np.arange(count_df.loc[idx,col]) + np.random.normal(0,10,count_df.loc[idx,col])
                y_values = np.full(len(x_values),float(y))
                y_values = y_values + np.random.normal(0,.5,len(y_values)) # add gaussian jitter.

                s_values = np.full(len(x_values),5) # set marker size.

                axs.scatter(x_values, y_values, s = s_values, color = color_dict[col])
                y = y+1

            y_ticks.append(y-1.5)
            y = y+.75
        axs.set_title(name, fontsize = 25)

        axs.set_yticks(y_ticks)
        axs.set_yticklabels(count_df.index, fontsize = 15)

        
        axs.spines['right'].set_visible(False)
        axs.spines['top'].set_visible(False)
        
    for j in range(i+1,n ** 2):
        if m > 1:
            ax[j//n,j%m].axis("off")    

    # add legend
    if m == 1:
        axs = ax[0]
    else:
        axs = ax[0,0]
    axs.scatter([],[], color = "#f1db54", s = [12], label = "Man")
    axs.scatter([],[], color = "#c85194", s = [12], label = "Woman")
    axs.scatter([],[], color = COLOR_MAP["teal"], s = [12], label = "Transgender or Non-Binary")
    handles, labels = axs.get_legend_handles_labels()
    plt.show()


def beeswarm_gender(demo_df, demo_cat, title = "Total Count by Race and Gender"):
    """ Returns beeswarm plot of counts by race and gender. 

    Input:
        demo_df: (dataframe) student_df or art_df
        demo_cat: (string) "race" or "region" 

    Returns: 
        Beeswarm style horizontal bar chart of total counts.
    """
    hall_df, student_df, art_df = sc.load_data()
    y_labels = list(student_df[demo_cat].value_counts().to_dict().keys())

    color_dict = {"Woman":"#c85194",
                  "Man":"#f1db54",
                  "Transgender":COLOR_MAP["teal"]}

    grouped_df = demo_df.groupby(demo_cat)
    count_df = pd.DataFrame(0, index = y_labels, 
        columns = ["Transgender","Woman","Man"])
    
    for y in demo_df[demo_cat].unique():
        gender_dict = grouped_df.get_group(y)["gender"].value_counts().to_dict()

        for k,v in gender_dict.items():
            count_df.loc[y,k] = v

    fig, ax = plt.subplots(figsize = (10,7))
    y = 0
    y_ticks = []
    for idx in count_df.index:        
        for col in ["Transgender","Woman","Man"]:

            x_values = np.arange(count_df.loc[idx,col]) + np.random.normal(0,10,count_df.loc[idx,col])
            y_values = np.full(len(x_values),float(y))
            y_values = y_values + np.random.normal(0,.5,len(y_values)) # add gaussian jitter.

            s_values = np.full(len(x_values),5) # set marker size.

            ax.scatter(x_values, y_values, s = s_values, color = color_dict[col])
            y = y+1

        y_ticks.append(y-1.5)
        y = y+.75

    ax.scatter([],[], color = "#f1db54", s = [25], label = "Man")
    ax.scatter([],[], color = "#c85194", s = [25], label = "Woman")
    ax.scatter([],[], color = COLOR_MAP["teal"], s = [25], label = "Transgender or Non-Binary")

    ax.set_xlabel("Count")
    ax.set_title(title, fontsize = 18)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(count_df.index, fontsize = 15)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.ylim(-0.5,y)
    plt.legend()
    plt.show()