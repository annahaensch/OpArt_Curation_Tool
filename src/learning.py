import pandas as pd
import numpy as np

import logging
import os
import sys
import json
import itertools
import numpy.ma as ma

from scipy.special import logsumexp

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

logging.basicConfig(level=logging.INFO)

def get_mapping_dicts():
    """
    Get enum to name mappings for gender, race, and region.
    """
    with open("../data/mappings.json","r") as file:
        mapping_dict = json.load(file)

        gender_map = mapping_dict["gender_mapping"]
        race_map = mapping_dict["race_mapping"]
        region_map = mapping_dict["region_mapping"]

    return gender_map, race_map, region_map


def load_data():
    """
    Load student, art, and building data.
    """
    # Gather building data
    hall_df = pd.read_csv("../data/hall_df.csv", index_col = 0)
    hall_df.sort_index(inplace = True)

    # Drop Capen House. 
    #hall_df = hall_df[hall_df.index != "capen_house"]

    # Gather student data
    student_df = pd.read_csv("../data/student_df.csv", 
                                index_col = 0)
    
    # Gather artwork data
    art_df = pd.read_csv("../data/2021_10_04_art_data_cleaned.csv", 
        index_col = 0)

    return hall_df, student_df, art_df



def get_quantized_student_data(student_df,gender_map, race_map, region_map):
    """
    Return quantized student categories by attribute.
    """
    S = student_df.shape[0]
    
    gender_quant = {g:student_df["gender_enum"].value_counts().to_dict().get(g,
                                            0)/S for g in gender_map.values()}
    race_quant = {g:student_df["race_enum"].value_counts().to_dict().get(g,
                                            0)/S for g in race_map.values()}
    region_quant = {g:student_df["region_enum"].value_counts().to_dict().get(g,                             
                                            0)/S for g in region_map.values()}

    return gender_quant, race_quant, region_quant


def get_quantized_art_data(art_df,gender_map, race_map, region_map):
    """
    Return quantized art categories by attribute.
    """
    A = art_df.shape[0]

    gender_quant = {g:art_df["gender_enum"].value_counts().to_dict().get(g,0)/A 
                                            for g in gender_map.values()}
    race_quant = {g:art_df["race_enum"].value_counts().to_dict().get(g,0)/A 
                                            for g in race_map.values()}
    region_quant = {g:art_df["region_enum"].value_counts().to_dict().get(g,0)/A 
                                            for g in region_map.values()}

    return gender_quant, race_quant, region_quant


def get_building_capacity_df():
    """
    Return dataframe of art capacity by building.
    """
    art_df = pd.read_csv(
        "../data/2021_05_10_Artist_Subject_Donor_Data_Cleaned.csv", 
        index_col = 0)
    art_df = art_df[~art_df["homeloc"].isna()]
    art_df = art_df[art_df["homeloc"] != "Crozier Fine Arts"]

    # Clean strings
    locs = [t.split("Medford, ")[-1] for t in art_df['homeloc']]
    locs = [t.split("Boston, ")[-1] for t in locs]
    locs = [t.replace(" ","_").replace(".","").strip("_").lower() for t in locs]
    locs = [t.replace("eaton_hal","eaton_hall") for t in locs]

    art_df["loc"] = locs
    building_dict = art_df["loc"].value_counts().to_dict()

    building_df = pd.DataFrame([v for v in building_dict.values()
        ], index = building_dict.keys(), columns = ["capacity"])
    building_df.sort_index(inplace = True)
    
    return building_df


def get_art_capacity_with_downsampling(art_df, categories = ["gender","race","region"]):
    """
    Return dataframe with columns "tuple","original_index","capacity".

    Input:
        art_df: (dataframe) art works with attributes
        categories: (list of strings) list containing "gender","race" or "region".

    Returns:
        Datraframe with downsampled category tuples, original artwork indices,
        and artwork type capacity (i.e. how many works of the given type exist in
        the collection).
    """
    cat_enum = ["{}_enum".format(c) for c in categories]
    
    df = art_df.copy()
    for c in cat_enum:
        df = df[df[c] != 0]
        
    art_tuple_series = pd.Series([tuple(v) for v in df[cat_enum].values], 
                            index = df.index)
    
    art_tuple_dict = art_tuple_series.value_counts().to_dict()

    # Sort values by tuple.
    art_tuple_series.sort_values(inplace = True)
    
    # Initialize empty dataframe
    art_capacity_df = pd.DataFrame()
    art_capacity_df["tuples"] = art_tuple_series.values
    art_capacity_df["original_index"] = list(art_tuple_series.index)
    
    # Add capacity from art_tuple_dict.
    for i in art_capacity_df.index:
        art_capacity_df.loc[i,"capacity"] = art_tuple_dict[
                                    art_capacity_df.loc[i,"tuples"]]    

    # Drop duplicate values.
    art_capacity_df.drop_duplicates(subset = ["tuples"], keep = "first", 
        inplace = True)
    art_capacity_df.reset_index(drop = True, inplace = True)

    # Assert that all art pieces are being counted.
    assert art_capacity_df.loc[:,"capacity"].sum() == df.shape[0]

    # Clip capacity at 100 to prevent overuse.
    art_capacity_df["capacity"] = art_capacity_df["capacity"].clip(upper = 100)

    return art_capacity_df


def compute_cost_matrix(art_df, 
                hall_df, 
                alpha):

    folder_files = os.listdir('../data/filled_buildings')
    hall_files = [f.split("_students.csv")[0].lower() for f in folder_files]
    hall_files = [f for f in hall_files if f in list(hall_df.index)]
    hall_files.sort()

    hall_index = list(hall_df.index)
    hall_index.sort()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    assert set(hall_files) == set(hall_index)

    # Initialize empty dataframe
    cost_df = pd.DataFrame(index = hall_files,
                          columns = [int(i) for i in art_df.index])
    
    # Load mappings
    gender_map, race_map, region_map = get_mapping_dicts()

    gender_quant_a, race_quant_a, region_quant_a = get_quantized_art_data(    
                                                        art_df,
                                                        gender_map, 
                                                        race_map, 
                                                        region_map)

    for i in range(len(hall_files)):

        df = pd.read_csv("../data/filled_buildings/{}_students.csv".format(
            hall_files[i]), index_col = 0)
        df.reset_index(drop = True, inplace = True)
        
        # Get quant dicts
        gender_quant_s, race_quant_s, region_quant_s = get_quantized_student_data(    
                                                            df,
                                                            gender_map, 
                                                            race_map, 
                                                            region_map)


        # Compute quantized student vectors.
        stu = df[["gender_enum","race_enum","region_enum"]].values.copy()
        q_stu = [[gender_quant_s.get(stu[i,0],0),
                  race_quant_s.get(stu[i,1],0),
                  region_quant_s.get(stu[i,2],0)] for i in df.index]
        df_quant_s = pd.DataFrame(q_stu,
                    columns = ["gender_quant","race_quant","region_quant"],
                    index = df.index)

        # Compute quantized building mode.
        mode = df.mode()[["gender_enum","race_enum","region_enum"]].values.copy()
        q_mode = [[gender_quant_s.get(mode[0,0],0),
                race_quant_s.get(mode[0,1],0),
                region_quant_s.get(mode[0,2],0)]]
        df_quant_mode = pd.DataFrame(q_mode,
                    columns = ["gender_quant","race_quant","region_quant"],
                    index = [0])

        # Compute quantized art vectors.
        art = art_df[["gender_enum","race_enum","region_enum"]].values.copy()
        q_art = [[gender_quant_a.get(art[i,0],0),
                  race_quant_a.get(art[i,1],0),
                  region_quant_a.get(art[i,2],0)] for i in art_df.index]
        df_quant_a = pd.DataFrame(q_art,
                    columns = ["gender_quant","race_quant","region_quant"],
                    index = art_df.index)

        # Reshape dataframes to compute diff. 
        a = df_quant_a.values.reshape(-1,df_quant_a.shape[0],df_quant_a.shape[1])
        s = df_quant_s.values.reshape(df_quant_s.shape[0],-1,df_quant_s.shape[1])

        # Compute student building diff and student art diff.
        stu_build_diff = pd.DataFrame(
            np.linalg.norm(df_quant_s.values - df_quant_mode.values, axis = 1), 
                        index = df_quant_s.index)
        stu_art_diff = pd.DataFrame(
                        np.linalg.norm(a - s, axis = 2), 
                        index = df_quant_s.index, 
                        columns = df_quant_a.index)

        art_prob = np.sum(alpha * stu_build_diff.values / stu_art_diff.values, axis = 0)

        art_prob = np.exp(art_prob - logsumexp(art_prob))

        assert np.abs(np.sum(art_prob) - 1) < 1e-08
       
        cost_df.loc[hall_files[i],:] = list(art_prob / np.sum(art_prob))

    return cost_df


def learn_optimal_assignment(cost_df, building_capacity, art_capacity, lam):
    """ Return n_buildings x n_artworkks assignment array
    """
    C = cost_df.values
    num_buildings = cost_df.shape[0]
    num_arts = cost_df.shape[1]

    dt = 0.001 #step size
    t = np.arange(1,num_arts + 1)

    ones_vector= np.ones((num_buildings,1))
    lam = 1

    # Initialize assignment array.
    P = np.eye(num_buildings,num_arts)

    for _ in range(1000):

        # Gradient descent.
        term2b = np.matmul(ones_vector,np.matmul(np.transpose(ones_vector),P
                          )-np.transpose(art_capacity))

        P = P - dt*C - lam *dt*(term2b)

        # Projection
        for i in range(num_buildings):
            v = P[i,:]
            if np.all(v>0):
                v = v/np.sum(v)
                P[i]=v * building_capacity[i]
            else:
                mu = v[np.argsort(-v)]
                tmp = np.divide(np.cumsum(mu)-building_capacity[i],t)
                idx_negative = np.argwhere(mu-tmp<=0)
                try:
                    idx_neg = idx_negative[0]
                    idx_neg = idx_neg.item()
                except:
                    idx_neg = -1
                theta = (np.sum(mu[0:idx_neg])-building_capacity[i])/(idx_neg)
                P[i,:] =  np.maximum(P[i,:]-theta,0)

    return P


def run_art_assignment(method, alpha, lam):
    
    # Load mappings
    gender_map, race_map, region_map = get_mapping_dicts()

    # Load data
    hall_df, student_df, art_df = load_data()

    # Get quantized student data.
    gender_quant_s, race_quant_s, region_quant_s = get_quantized_student_data(    
                                                        student_df,
                                                        gender_map, 
                                                        race_map, 
                                                        region_map)

    # Get quantized  art data.
    gender_quant_a, race_quant_a, region_quant_a = get_quantized_art_data(    
                                                        art_df,
                                                        gender_map, 
                                                        race_map, 
                                                        region_map)

    # Get building capacity column vector.
    building_capacity = get_building_capacity_df().values

    logging.info("\n Computing cost matrix...")

    # Compute full n_buildings x n_artworks cost matrix.
    cost_df = compute_cost_matrix(art_df = art_df, 
                                        hall_df = hall_df, 
                                        gender_quant_s = gender_quant_s, 
                                        race_quant_s = race_quant_s, 
                                        region_quant_s = region_quant_s, 
                                        gender_quant_a = gender_quant_a, 
                                        race_quant_a = race_quant_a, 
                                        region_quant_a = region_quant_a, 
                                        alpha = -1)

    if method == "assign_with_downsampling":
        art_capacity_df = get_art_capacity_with_downsampling(art_df)

        # Reduce cost df to remove duplicate columns.
        art_capacity = art_capacity_df["capacity"].values.reshape(-1,1)

        cost_df = cost_df.loc[:,art_capacity_df["original_index"].values]
        
        logging.info("\n Learning optimal assignment...")

        P = learn_optimal_assignment(cost_df, 
                                    building_capacity, 
                                    art_capacity, 
                                    lam)
        
        # Check that assignment numbers are sufficiently close to building capacity.
        assert np.all(np.sum(P, axis = 1) - building_capacity.reshape(
                                                                1,-1) < 1e-10)

        # Convert the assignment array to a dataframe for readability.
        assignment_df = pd.DataFrame(P, index = cost_df.index,
                          columns = art_capacity_df["tuples"].values)

        return assignment_df

    else:
        raise NotImplementedError("This is in progress...")

if __name__ == "__main__":

    method = sys.argv[1]
    alpha = sys.argv[2]
    lam = sys.argv[3]
    run_art_assignment(method, alpha, lam)