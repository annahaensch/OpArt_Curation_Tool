# Preprocessing of Data
import pandas as pd

import sys
sys.path.append('..')

import src as sc

# This is a dictionary for the hall names used by the TUAG system.
hall_tuag_name_dict = {
		'Crozier Fine Arts' : "crozier_fine_arts", 
		'Medford, Aidekman': "aidekman", 
		'Medford, Anderson  ':"anderson",
		'Medford, Tisch Library ':"tisch_library", 
		'Medford, Ballou  ':"ballou",
		'Medford, Granoff Music Bld':"granoff_music_bld", 
		'Medford, CLIC':"clic",
		'Medford, Univ. Advancement':"univ_advancement", 
		'Medford, Gifford House':"gifford_house",
		'Boston, Jean Mayer HNRA Center':"jean_mayer_hnra_center", 
		'Boston, SMFA':"smfa",
		'Medford, Tisch Library':"tisch_library", 
		'Medford, Fine Arts House':"fine_arts_house",
		'Medford, Goddard Chapel':"goddard_chapel", 
		'Medford, Granoff Fam. Hillel Center':"granoff_fam_hillel_center",
		'Medford, Sophia Gordon Hall':"sophia_gordon_hall", 
		'Medford, Eaton Hal':"eaton_hall",
		'Medford, Packard Hall':"packard_hall", 
		'Boston, Dental School':"dental_school",
		'Medford, Granoff Music':"granoff_music_bld", 
		'The Fletcher School':"goddard_hall",
		'Medford, Miner Hall':"miner_hall", 
		'Medford, Lane Hall':"lane_hall",
		'Medford, Houston Hall':"houston_hall", 
		'Medford, Capen House ':"capen_house",
		'Medford, Granoff Music ':"granoff_music_bld", 
		'Crozier ':"crozier_fine_arts",
		'Unknown':"crozier_fine_arts"
}

tuag_race_dict = {
		 'White':"White", 
		 'hispanic or latinx':"Hispanics of any race", 
		 'Hispanic or Latinx':"Hispanics of any race", 
		 'Black':"Black or African American",
		 'Asian':"Asian", 
		 'Unknown':"Unreported", 
		 'Not Inferred':"Unreported", 
		 'Other/white':"White", 
		 'Other':"Two or more races",
		 'Group':"Two or more races", 
		 'Other/Native American':"American Indian or Alaska Native", 
		 'Other/Natvie American':"American Indian or Alaska Native",
		 'Other/Inuit':"American Indian or Alaska Native"
}

tuag_gender_dict = {
	'Man':"Man", 
	'Woman':"Woman", 
	'Not Inferred':"Unreported", 
	'not inferred':"Unreported", 
	'Unknown':"Unreported", 
	'Group':"Unreported"
}

tuag_region_dict = {
	'United States':"North America", 
	'Great Britain':"Europe",
	'Russia':"Asia", 
	'Unknown':"Unreported", 
	'Mexico':"Central America",
	'Ivory Coast':"Africa", 
	'South Africa':"Africa", 
	'Republic of Korea':"Asia", 
	'France':"Europe",
	'Spain':"Europe", 
	'Greece':"Europe", 
	'Netherlands':"Europe", 
	'Germany':"Europe", 
	'Sweden':"Europe", 
	'Italy':"Europe",
	'Colombia':"South America", 
	'Canada':"North America", 
	'Ghana':"Africa", 
	'Egypt':"Africa", 
	'Benin':"Africa", 
	'Japan':"Asia", 
	'China':"Asia",
	'Australia':"Oceania", 
	'India':"Asia", 
	'Hong Kong':"Asia",
	'Korea':"Asia", 
	'Bulgaria':"Europe",
	'Czech Republic':"Europe", 
	'Israel':"Asia", 
	'Denmark':"Europe", 
	'Austria':"Europe", 
	'Hungary':"Europe",
	'Ireland':"Europe", 
	'Ukraine':"Europe",
	'Belgium':"Europe", 
	'Switzerland':"Europe",
	 'Albania':"Europe",
	'Gibraltar':"Europe", 
	'Portugal':"Europe", 
	'Guatemala':"Central America", 
	'Brazil':"South America",
	'Jamaica':"Caribbean",
	'Argentina':"South America", 
	'Chile':"South America", 
	'Cuba':"Caribbean", 
	'Uraguay':"South America", 
	'Iran':"Asia", 
	'Lebanon':"Asia"
}


def process_art_dataframe():

	mappings = sc.get_mapping_dicts()
	mapping_dict = {"gender":mappings[0],
               "race":mappings[1],
               "region":mappings[2]}

	art_df = pd.read_excel("../data/PC_Artist_Subject_Donor_Data.xlsx")
	art_df.fillna("Unknown", inplace = True)
	art_df["loc"] = [hall_tuag_name_dict[a] for a in art_df["Current Location (Campus and Building)"]]
	art_df.columns = [c.lower().strip(" ").replace(" ","_").replace("__","_") for c in art_df.columns]
	art_df = art_df[[c for c in art_df.columns if not "unnamed" in c]].copy()
	art_df["artist"] = [a for a in art_df["creator_1"]]

	art_df["gender"] = [tuag_gender_dict[a] for a in art_df['creator_1_gender']]
	art_df["race"] = [tuag_race_dict[a] for a in art_df['creator_1_ethnicity']]
	art_df["region"] = [tuag_region_dict[a] for a in art_df["creator_1_country"]]

	art_df["gender_enum"] = [mapping_dict["gender"][i] for i in art_df["gender"]]
	art_df["race_enum"] = [mapping_dict["race"][i] for i in art_df["race"]]
	art_df["region_enum"] = [mapping_dict["region"][i] for i in art_df["region"]]

	col_of_interest = [
					"objectid",
					"artist",
					"loc",
					"gender",
					"race",
					"region",
					"gender_enum",
					"race_enum",
					"region_enum"
					]
	art_df[col_of_interest].to_csv("../data/2022_03_04_art_data_cleaned.csv")
	return art_df[col_of_interest]