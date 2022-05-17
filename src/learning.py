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
from datetime import datetime

logging.basicConfig(level=logging.INFO)

import os
ROOT = os.popen("git rev-parse --show-toplevel").read().split("\n")[0]

import sys
sys.path.append('..')

import src as sc

def get_mapping_dicts():
    """
    Get enum to name mappings for gender, race, and region.
    """
    with open(ROOT + "/data/mappings.json","r") as file:
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
    hall_df = pd.read_csv(ROOT + "/data/hall_df.csv", index_col = 0)
    hall_df.sort_index(inplace = True)

    # Drop Capen House. 
    #hall_df = hall_df[hall_df.index != "capen_house"]

    # Gather student data
    student_df = sc.get_student_enrollment_data()
    
    # Gather artwork data.
    art_df = sc.process_art_dataframe()

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

    return gender_quant, race_quant


def get_quantized_art_data(art_df,gender_map, race_map, region_map):
    """
    Return quantized art categories by attribute.
    """
    A = art_df.shape[0]

    gender_quant = {g:art_df["gender_enum"].value_counts().to_dict().get(g,0)/A 
                                            for g in gender_map.values()}
    race_quant = {g:art_df["race_enum"].value_counts().to_dict().get(g,0)/A 
                                            for g in race_map.values()}

    return gender_quant, race_quant


def get_building_capacity_df():
    """
    Return dataframe of art capacity by building.
    """
    try:
        art_df = pd.read_csv(ROOT + "/data/2022_03_04_art_data_cleaned.csv", 
        index_col = 0)
    except:
        art_df = process_art_dataframe()
        

    # Remove art in storage.
    art_df = art_df[art_df["loc"] != "crozier_fine_arts"].copy()

    building_dict = art_df["loc"].value_counts().to_dict()

    building_df = pd.DataFrame([v for v in building_dict.values()
        ], index = building_dict.keys(), columns = ["capacity"])
    building_df.sort_index(inplace = True)
    
    return building_df


def get_art_capacity_with_downsampling(art_df, categories = ["gender","race"]):
    """
    Return dataframe with columns "tuple","original_index","capacity".

    Input:
        art_df: (dataframe) art works with attributes
        categories: (list of strings) list containing "gender" or "race".

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

    art_string_series = pd.Series([", ".join(v) for v in df[categories].values], 
                            index = df.index)
    
    art_tuple_dict = art_tuple_series.value_counts().to_dict()
    
    # Initialize empty dataframe
    art_capacity_df = pd.DataFrame()
    art_capacity_df["tuple"] = art_tuple_series.values
    art_capacity_df["string"] = art_string_series.values
    art_capacity_df["original_index"] = list(art_tuple_series.index)

    art_capacity_df.sort_values(by = "tuple", ascending = True, inplace = True)
    art_capacity_df.reset_index(drop = True, inplace = True)
    
    # Add capacity from art_tuple_dict.
    for i in art_capacity_df.index:
        art_capacity_df.loc[i,"capacity"] = art_tuple_dict[
                                    art_capacity_df.loc[i,"tuple"]]    

    # Drop duplicate values.
    art_capacity_df.drop_duplicates(subset = ["tuple"], keep = "first", 
        inplace = True)
    art_capacity_df.reset_index(drop = True, inplace = True)

    # Assert that all art pieces are being counted.
    assert art_capacity_df.loc[:,"capacity"].sum() == df.shape[0]

    # Clip capacity at 100 to prevent overuse.
    art_capacity_df["capacity"] = art_capacity_df["capacity"].clip(upper = 100)

    return art_capacity_df


def compute_cost_matrix(art_df, 
                hall_df,
                categories = ["gender","race"],
                alpha = -1,
                beta = 100):
    """
    Return dataframe with columns "tuple","original_index","capacity".

    Input:
        art_df: (dataframe) art works with attributes
        hall_df: (dataframe) one-hot dataframe with halls as index, schools as columns
        categories: (list of strings) list containing "gender" or "race".
        alpha: (float) model parameter determining outlier importance.
        beta: (float) model parameter determining art rep. importance.
        
    Returns:
        Datraframe where the entry in row m and column n is the 
        cost to hang artwork m in building n.
    """    
    cat_enum = ["{}_enum".format(c) for c in categories]
    cat_quant = ["{}_quant".format(c) for c in categories]

    # Drop rows from art_df where category value is unreported.
    new_art_df = art_df.copy()
    for c in cat_enum:
        new_art_df = new_art_df[new_art_df[c] != 0]
    new_art_df.reset_index(drop = True, inplace = True)
    
    # Check that the filled building data is available.
    my_path = ROOT + "/data/filled_buildings/"
    hall_files = ["{}_students.csv".format(f) for f in hall_df.index]
    hall_files.sort()
                  
    folder_files = set(os.listdir(my_path))
    assert set(hall_files).issubset(folder_files)
    
    # Load mappings
    mappings = get_mapping_dicts()
    mapping_dict = {"gender":mappings[0],
                   "race":mappings[1],
                   "region":mappings[2]}

    # Get quantized artwork dictionaries and reduced dataframe.
    quant_a = get_quantized_art_data(new_art_df,
                        gender_map = mapping_dict["gender"], 
                        race_map = mapping_dict["race"], 
                        region_map = mapping_dict["region"])

    quant_a_dict = {"gender":quant_a[0],
                   "race":quant_a[1]
                   }

    for c in categories:
        new_art_df["{}_quant".format(c)] = new_art_df["{}_enum".format(c)].map(quant_a_dict[c])
        
    new_art_df["tuple"] = [tuple(v) for v in new_art_df[cat_enum].values]
    new_art_df.sort_values(by = "tuple", inplace = True)
    new_art_df = new_art_df.drop_duplicates(subset = ["tuple"])
    new_art_df.set_index("tuple", drop = True, inplace = True)
    df_quant_a = new_art_df[cat_quant]
    
    # Initialize empty dataframe
    cost_df = pd.DataFrame(index = hall_df.index,
                          columns = list(new_art_df.index))

    for i in range(len(hall_files)):
        name = hall_files[i].split("_students.csv")[0]
        df = pd.read_csv(ROOT + "/data/filled_buildings/{}".format(
            hall_files[i]), index_col = 0)
        df.reset_index(drop = True, inplace = True)

        # Initialize empty dataframes
        df_quant_s = pd.DataFrame(index = df.index, 
                    columns = cat_quant)
        df_quant_mode = pd.DataFrame(index = [0],
                    columns = cat_quant)
        
        # Get quantized student dictionaries
        quant_s = get_quantized_student_data(df,
                        gender_map = mapping_dict["gender"], 
                        race_map = mapping_dict["race"], 
                        region_map = mapping_dict["region"])

        quant_s_dict = {"gender":quant_s[0],
                       "race":quant_s[1]}

        
        for c in categories:
            # Compute quantized student vectors.
            df_quant_s["{}_quant".format(c)
                      ] = df["{}_enum".format(c)].map(quant_s_dict[c])
            
            # Compute quantized building mode.
            m = df["{}_enum".format(c)].mode()[0]
            quant_m = quant_s_dict[c][m]
            df_quant_mode.loc[0,"{}_quant".format(c)] = quant_m


        # One-hot array with dimension #students x #categories where entry
        # [i,k] = 1 if student i shares attribute k with the building mode.
        bld_mode = df[cat_enum].mode().values[0,:].reshape(-1,len(cat_enum))
        stu_build_delta = np.where(df[cat_enum].values - bld_mode == 0,0,1)
        
        # Array of differences between student quantile and mode quantile.
        diff = (df_quant_s.values - df_quant_mode.values).astype(float)

        # Array with dimension #students x 1 x 1
        stu_build_norm = np.linalg.norm(diff * stu_build_delta, axis = 1)
        stu_build_norm = stu_build_norm.reshape(stu_build_norm.shape[0],1,-1)

        # Array with dimension #students x 1 x 1
        numerator = stu_build_norm
        
        # Compute denominator
        student_enum = df[cat_enum].values.reshape(df.shape[0],-1,len(categories))
        art_enum = new_art_df[cat_enum].values.reshape(-1,new_art_df.shape[0],len(categories))

        # One-hot array with dimenion #students x #artworks x #categories
        # where entry [i,j,k] = 1 if student i and artwork j share attribute k.
        stu_art_delta = np.where((student_enum - art_enum) == 0, 0, 1).astype(float)

        # Array with dimension #students x #artworks
        stu_art_norm = np.linalg.norm(stu_art_delta.astype(float), axis = 2)
        stu_art_norm = np.where(stu_art_norm == 0, 1e-08,stu_art_norm)

        # Array with dimension 1 X #artworks where entry [0,m] is the 
        # probability of an artwork sharing all attributes with artwork m.
        art_likelihood = np.prod(df_quant_a, axis = 1).values.reshape(-1, new_art_df.shape[0])
        
        # Array with dimension #students x 1 x #artworks
        denominator = stu_art_norm * art_likelihood
        denominator = denominator.reshape(denominator.shape[0],-1,denominator.shape[1])

        art_prob = (alpha / beta) * np.sum((numerator / denominator), axis = 0)
        art_prob = np.exp(art_prob - logsumexp(art_prob))

        assert np.abs(np.sum(art_prob) - 1) < 1e-08
       
        cost_df.loc[name,:] = art_prob

    return cost_df

def get_normalizing_constants():
    """ Return normalizing contstants for lambda and tau

    Returns: 
        tuple with (norm_lam_factor, norm_tau_factor) which is used to scale
        lambda and tau for grid search.
    """
    hall_df, student_df, art_df = sc.load_data()
    
    # Compute building capacity
    building_capacity_df = sc.get_building_capacity_df()
    building_capacity =  building_capacity_df.values
    num_buildings =  building_capacity_df.shape[0]
    
    # Compute art capacity
    art_capacity_df = sc.get_art_capacity_with_downsampling(art_df,
                    categories = ["gender","race"])
    art_capacity = art_capacity_df["capacity"].values

    # Get current assignment
    current_assignment_df = pd.read_csv(ROOT + "/data/current_assignment_df.csv", index_col = 0)
    current_assignment = current_assignment_df.values

    beta_values = np.array([.01,1,10])
    term1 = np.zeros((beta_values.shape[0],10))
    term2 = np.zeros((beta_values.shape[0],10))
    term3 = np.zeros((beta_values.shape[0],10))
    for m in range(beta_values.shape[0]):
        # Compute full n_buildings x n_artworks cost matrix.
        cost_df = sc.compute_cost_matrix(art_df = art_df, 
                                        hall_df = hall_df,
                                        categories = ["gender","race"],
                                        alpha = -1,
                                        beta = beta_values[m])
        num_arts = cost_df.shape[1]
        t = np.arange(1,num_arts + 1)
        for k in range(10):
            # Pick a random permutation in S
            P = np.random.randn(num_buildings,num_arts)  
            # Projection
            for ii in range(num_buildings):
                v = P[ii,:]
                if np.all(v>0):
                    v = v/np.sum(v)
                    P[ii]=v * building_capacity[ii]
                else:
                    mu = v[np.argsort(-v)]
                    tmp = np.divide(np.cumsum(mu)-building_capacity[ii],t)
                    idx_negative = np.argwhere(mu-tmp<=0)
                    try:
                        idx_neg = idx_negative[0]
                        idx_neg = idx_neg.item()
                    except:
                        idx_neg = -1
                    theta = (np.sum(mu[0:idx_neg])-building_capacity[ii])/(idx_neg)
                    P[ii,:] =  np.maximum(P[ii,:]-theta,0)  
            term1[m,k] = np.trace(np.matmul(np.transpose(cost_df.values),P))
            term2[m,k] = np.linalg.norm(np.sum(P,axis=0)-art_capacity)**2
            term3[m,k] = np.linalg.norm(P-current_assignment)**2

    term1_avg = np.mean(term1)
    term2_avg = np.mean(term2)
    term3_avg = np.mean(term3)

    norm_lam_factor = term1_avg/term2_avg
    norm_tau_factor = term1_avg/term3_avg
    
    return norm_lam_factor, norm_tau_factor

def learn_optimal_assignment(cost_df, lam, tau,init):
    """ Return n_buildings x n_artworkks assignment array

    Input: 
        cost_df: (dataframe) cost dataframe typically computed with 
            `compute_cost_matrix`.
        lam: (float) lambda factor determines weight of artwork capacity 
            constraints in optimization.
        tau: (float) tau factor determines weight of preference for current 
            assignment in optimization.
        init: (int) one of the following: 
                1 - identity matrix initialization
                2 - uniform initialization
                3 - current assignment initialization
                4 - random permutation initialization
    Returns:
        num_buildings x num_arts assignment dataframe where the entry in row n 
        and column m is the copies of artwork m to by hung in building n.
    """
    C = cost_df.values

    # Load data
    hall_df, student_df, art_df = sc.load_data()
    
    # Compute building capacity
    building_capacity_df = sc.get_building_capacity_df()
    building_capacity =  building_capacity_df.values
    num_buildings =  building_capacity_df.shape[0]
    
    # Compute art capacity
    art_capacity_df = sc.get_art_capacity_with_downsampling(art_df,
                    categories = ["gender","race"])
    art_capacity = art_capacity_df["capacity"].values
    num_arts = art_capacity_df.shape[0]

    # Get current assignment
    current_assignment_df = pd.read_csv(ROOT + "/data/current_assignment_df.csv", index_col = 0)
    current_assignment = current_assignment_df.values

    dt = 0.5 * (1/((lam * num_buildings) + tau)) #step size
    t = np.arange(1,num_arts + 1)

    ones_vector= np.ones((num_buildings,1))

    # Initialize assignment array.
    if init==1:
        P = np.eye(num_buildings,num_arts)
    elif init==2:
        P = 1/(num_arts)*np.divide(np.ones((num_buildings,num_arts)),building_capacity)
    elif init==3:
        P = current_assignment
    else:
        P = sample_general_simplex(num_buildings,num_arts,building_capacity)
    energy = np.zeros((10,1))
    for k in range(10):
        # Gradient descent.
        term2b = np.matmul(ones_vector,np.matmul(np.transpose(ones_vector),P
                          )-np.transpose(art_capacity))

        P = P - dt*C - lam *dt*(term2b)- tau*dt*(P-current_assignment)
        # Projection
        for i in range(num_buildings):
            v = P[i,:]
            if np.all(v>0):
                v = v/np.sum(v)
                P[i]=v * building_capacity[i]
            else:
                mu = v[np.argsort(-v)]
                tmp = (np.cumsum(mu) - building_capacity[i])/t
                K = np.argwhere(tmp < mu)[-1][0] + 1
                theta = (np.sum(mu[:K]) - building_capacity[i]) / (K)
                P[i,:] =  np.maximum(P[i,:]-theta,0)
        # Compute objective value
        energy1 = np.trace(np.matmul(np.transpose(C),P))
        energy2 = 0.5*lam*np.linalg.norm(np.sum(P,axis=0)-art_capacity)**2
        energy3 = 0.5*tau*np.linalg.norm(P-current_assignment)**2
        energy[k] = energy1+energy2+energy3

    return pd.DataFrame(P, index = cost_df.index,
                   columns = art_capacity_df["string"].values)


def validate_assignment(assignment_df):
    """ Print validation dataframe

    Input: 
        assignment_df: (dataframe) output from `learn_optimal_assignment`

    Output: 
        Dataframe indexed by race and gender with probability of seeing 
        identity on campus in optimized approach and baseline approach.
    """

    # Load validation data.
    hall_df, student_df, art_df = load_data()

    race_index = [r+","+r for r in list(student_df["race"].value_counts().keys()
                                        ) if r!= "Unreported"]
    race_index = ",".join(race_index).split(",")

    gender_index = [list(student_df["gender"].value_counts().keys()
                                        )for r in range(int(len(race_index)/2))]
    gender_index = [",".join(g) for g in gender_index]
    gender_index = ",".join(gender_index).split(",")

    val_df = pd.DataFrame(np.zeros((len(race_index),3)), index = [
        race_index, gender_index], columns = ["Optimized","Baseline","Total"])

    # Get current art locations and process dataframe.
    art_loc_df = pd.read_csv(ROOT + "/data/2021_05_07_Artist_Subject_Donor_Data_v3.csv"
                                            )[["OBJECTID","HOMELOC"]]
    art_df_with_loc = art_df.merge(art_loc_df, 
                                    left_on = "objectid",
                                    right_on = "OBJECTID")
    art_df_with_loc = art_df_with_loc[art_df_with_loc["HOMELOC"
                                        ] != "Crozier Fine Arts"]
    art_df_with_loc["loc"] = [n.split("Medford, ")[-1].split(
        "Boston, ")[-1].strip(" ").lower().replace(".","").replace(
                        " ","_") for n in art_df_with_loc["HOMELOC"]]
    art_df_with_loc["loc"] = art_df_with_loc["loc"].replace(
                                    "eaton_hal","eaton_hall")
    
    # Iterative over buildings in question.
    for h in assignment_df.index:
        building_df = pd.read_csv(ROOT + "/data/filled_buildings/{}_students.csv".format(h))
        
        art_slice_df = art_df_with_loc[art_df_with_loc["loc"] == h]
        
        for c in val_df.index:
            race = c[0]
            gender = c[1]
            
            # Optimized probability of art in building with attribute race, gender
            A = assignment_df.loc[[h],:]
            P = A/np.sum(A.values)
            try:
                O_PP = P["{}, {}".format(gender, race)].iloc[0]
            except: 
                O_PP = 0
            N = building_df[(building_df["race"] == race)&(
                        building_df["gender"] == gender)].shape[0]
            val_df.loc[(race, gender),"Optimized"] += O_PP * N
            
            
            # Baseline probability of art in building with attribute c
            B_PP = art_slice_df[(art_slice_df["race"] == race)&(
                            art_slice_df["gender"] == gender)].shape[0]/art_slice_df.shape[0]
            val_df.loc[(race, gender),"Baseline"] += B_PP * N
            
            # Total number of students with attribute c
            val_df.loc[(race, gender),"Total"] += N
    
    val_df = pd.DataFrame(val_df.values/val_df["Total"].values.reshape(
                -1,1), index = val_df.index, columns = val_df.columns)
    
    return val_df[["Optimized","Baseline"]]

# This script samples a matrix P of size N by M where P>=0 and P1 = h
# In other words, each row of P, denoted by P_i lies in the general simplex
# where (P_i)_j>=0 and sum_{j=1}^{M} (P_i) = h_i
def sample_general_simplex(N,M,h):
    P = np.zeros((N,M))
    for i in range(N):
        init_sum = 0
        for j in range(M-1):
            phi = np.random.uniform(0,1)
            P[i,j] =(1-init_sum)*(1-np.power(phi,1/(M-j-1))) 
            init_sum = init_sum+ P[i,j]
        P[i,M-1]  = 1- init_sum
        P[i,:] = np.multiply(P[i,:],h[i])
    return P

def baseline_average_value(category = "gender", in_group = "Man"):
    # Load data
    hall_df, student_df, art_df = load_data()
    df = pd.DataFrame(np.nan,index = student_df["student_id"].values,
                     columns = hall_df.index)
    in_index = []
    out_index = []
    for h in hall_df.index:
        df_h = pd.read_csv(ROOT + "/data/filled_buildings/{}_students.csv".format(h), index_col = 0)
        # set value to 1 if student goes through building.
        df.loc[df_h["student_id"].values,h] = 1
    
    # Divide by total buildings inhabited by student.
    df = df/np.sum(df, axis = 1).values.reshape(-1,1)
    df.fillna(0, inplace = True)
    
    # Count total number of art by by building and category. 
    in_dict = art_df[art_df[category] == in_group]["loc"].value_counts().to_dict()
    out_dict = art_df[art_df[category] != in_group]["loc"].value_counts().to_dict()
  
    # Computed expected values for "in group"
    in_idx = student_df[student_df[category] == in_group].index
    df_in = df.loc[in_idx]
    df_in = df_in * np.array([in_dict.get(c,0) for c in df.columns]).reshape(1,-1)
    in_expected = np.sum(df_in.values)/len(in_idx)
    
    # Compute expected value for non "in group"
    out_idx = student_df[student_df[category] != in_group].index
    df_out = df.loc[out_idx]
    df_out = df_out * np.array(
                        [out_dict.get(c,0) for c in df.columns]
                        ).reshape(1,-1)

    out_expected = np.sum(df_out.values)/len(out_idx)
    
    return in_expected, out_expected

def optimized_average_value(assignment_df, category = "gender", in_group = "Man"):
    # Load data
    hall_df, student_df, art_df = load_data()
    df = pd.DataFrame(np.nan,index = student_df["student_id"].values,
                     columns = hall_df.index)
    in_index = []
    out_index = []
    for h in hall_df.index:
        df_h = pd.read_csv(ROOT + "/data/filled_buildings/{}_students.csv".format(h), index_col = 0)
        # set value to 1 if student goes through building.
        df.loc[df_h["student_id"].values,h] = 1
    
    # Divide by total buildings inhabited by student.
    df = df/np.sum(df, axis = 1).values.reshape(-1,1)
    df.fillna(0, inplace = True)
    
    # Count total number of art by by building and category. 
    in_dict = art_df[art_df[category] == in_group]["loc"].value_counts().to_dict()
    out_dict = art_df[art_df[category] != in_group]["loc"].value_counts().to_dict()
  
    # Computed expected values for "in group"
    in_idx = student_df[student_df[category] == in_group].index
    df_in = df.loc[in_idx]
    df_in = df_in * np.sum(assignment_df[
        [c for c in assignment_df.columns if in_group in c]], axis = 1).values.reshape(1,-1)
    in_expected = np.sum(df_in.values)/len(in_idx)
    
    # Compute expected value for non "in group"
    out_idx = student_df[student_df[category] != in_group].index
    df_out = df.loc[out_idx]
    df_out = df_out * np.sum(assignment_df[
        [c for c in assignment_df.columns if not in_group in c]], axis = 1).values.reshape(1,-1)

    out_expected = np.sum(df_out.values)/len(out_idx)
    
    return in_expected, out_expected


def run_art_assignment(beta, lam, tau, init):
    """
    Input: 
        beta: (float) beta factor determines weight of the diversity 
            objectives in the optimization (i.e. "term 1").
        lam: (float) lambda factor determines weight of artwork capacity 
            constraints in optimization (i.e. "term 2").
        tau: (float) tau factor determines weight of preference for current 
            assignment in optimization (i.e. "term 3").
        init: (int) one of the following: 
                1 - identity matrix initialization
                2 - uniform initialization
                3 - current assignment initialization
                4 - random permutation initialization
    Returns:
        num_buildings x num_arts assignment dataframe where the entry in row n 
        and column m is the copies of artwork m to by hung in building n.
    """

    # Load mappings
    gender_map, race_map, region_map = get_mapping_dicts()

    # Load data
    hall_df, student_df, art_df = load_data()

    # Compute building capacity
    building_capacity_df = sc.get_building_capacity_df()
    building_capacity =  building_capacity_df.values

    # Compute art capacity
    art_capacity_df = sc.get_art_capacity_with_downsampling(art_df,
                    categories = ["gender","race"])
    art_capacity = art_capacity_df["capacity"].values

    logging.info("Computing cost matrix...")
    logging.info(f"beta = {beta}")

    # Compute full n_buildings x n_artworks cost matrix.
    cost_df = compute_cost_matrix(art_df = art_df, 
                                        hall_df = hall_df,
                                        categories = ["gender","race"],
                                        alpha = -1,
                                        beta = beta)

    # Reduce cost df to remove duplicate columns.
    cost_df = cost_df.rename(columns = {art_capacity_df.loc[i,"tuple"
        ]:art_capacity_df.loc[i,"string"] for i in art_capacity_df.index})

    # Compute normalizing constants for lambda and tau
    norm_lam_factor, norm_tau_factor = get_normalizing_constants()

    logging.info("Computing assignment matrix...")
    logging.info(f"lambda = {lam}, tau = {tau}, init = {init}")

    # Compute assignment matrix
    assignment_df = learn_optimal_assignment(cost_df, 
                             lam = norm_lam_factor*lam, 
                             tau=norm_tau_factor*tau,
                             init = init
                             ) 

    # Check that assignment numbers are sufficiently close to building capacity.
    assert np.all(assignment_df.sum(axis = 1).values.reshape(-1,1) - 
                                building_capacity_df.values.reshape(-1,1) < 1e-08)

    d = datetime.now().strftime("%Y%m%d%H%M%S")
    suffix = f"{beta}_{lam}_{tau}_{init}_{d}.csv"
    assignment_df.to_csv(f"{ROOT}/output/assignment_df_{suffix}")
    cost_df.to_csv(f"{ROOT}/output/cost_df_{suffix}")

    logging.info(f"Cost matrix printed to: {ROOT}/output/cost_df_{suffix}")
    logging.info(f"Assignment matrix printed to: {ROOT}/output/assignment_df_{suffix}")

    return assignment_df


if __name__ == "__main__":

    beta = int(sys.argv[1])
    lam = int(sys.argv[2])
    tau = int(sys.argv[3])
    init = int(sys.argv[4])

    run_art_assignment(beta, lam, tau, init)