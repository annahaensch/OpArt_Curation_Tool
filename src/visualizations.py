import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import numpy.ma as ma
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
sys.path.append('..')

import src as sc

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
    gender_map, race_map, region_map = sc.get_mapping_dicts()
    g_map = {str(v):k for k,v in gender_map.items()}
    r_map = {str(v):k for k,v in race_map.items()}
    e_map = {str(v):k for k,v in region_map.items()}

    y_labels = ["{},{},{}".format(g_map[str(c[0])],r_map[str(c[1])],e_map[
        str(c[2])]) for c in assignment_df.columns]


    ax.set_yticks(.5 + np.arange(assignment_df.shape[1]))
    ax.set_yticklabels(y_labels)

    plt.show()