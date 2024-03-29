# Preprocessing of Data
import pandas as pd
import numpy as np
import pdfplumber
import json
import os
import sys
import itertools

import logging

logging.basicConfig(level=logging.INFO)


ROOT = os.popen("git rev-parse --show-toplevel").read().split("\n")[0]
sys.path.append(ROOT)

import src as sc

# This is a dictionary for the hall names used by the TUAG system.
hall_tuag_name_dict = {
    "Crozier Fine Arts": "crozier_fine_arts",
    "Medford, Aidekman": "aidekman",
    "Medford, Anderson  ": "anderson",
    "Medford, Tisch Library ": "tisch_library",
    "Medford, Ballou  ": "ballou",
    "Medford, Granoff Music Bld": "granoff_music_bld",
    "Medford, CLIC": "clic",
    "Medford, Univ. Advancement": "univ_advancement",
    "Medford, Gifford House": "gifford_house",
    "Boston, Jean Mayer HNRA Center": "jean_mayer_hnra_center",
    "Boston, SMFA": "smfa",
    "Medford, Tisch Library": "tisch_library",
    "Medford, Fine Arts House": "fine_arts_house",
    "Medford, Goddard Chapel": "goddard_chapel",
    "Medford, Granoff Fam. Hillel Center": "granoff_fam_hillel_center",
    "Medford, Sophia Gordon Hall": "sophia_gordon_hall",
    "Medford, Eaton Hal": "eaton_hall",
    "Medford, Packard Hall": "packard_hall",
    "Boston, Dental School": "dental_school",
    "Medford, Granoff Music": "granoff_music_bld",
    "The Fletcher School": "goddard_hall",
    "Medford, Miner Hall": "miner_hall",
    "Medford, Lane Hall": "lane_hall",
    "Medford, Houston Hall": "houston_hall",
    "Medford, Capen House ": "capen_house",
    "Medford, Granoff Music ": "granoff_music_bld",
    "Crozier ": "crozier_fine_arts",
    "Unknown": "crozier_fine_arts",
}

tuag_race_dict = {
    "White": "White",
    "hispanic or latinx": "Hispanics of any race",
    "Hispanic or Latinx": "Hispanics of any race",
    "Black": "Black or African American",
    "Asian": "Asian",
    "Unknown": "Unreported",
    "Not Inferred": "Unreported",
    "Other/white": "White",
    "Other": "Two or more races",
    "Group": "Two or more races",
    "Other/Native American": "American Indian or Alaska Native",
    "Other/Natvie American": "American Indian or Alaska Native",
    "Other/Inuit": "American Indian or Alaska Native",
}

tuag_gender_dict = {
    "Man": "Man",
    "Woman": "Woman",
    "Not Inferred": "Unreported",
    "not inferred": "Unreported",
    "Unknown": "Unreported",
    "Group": "Unreported",
}

tuag_region_dict = {
    "United States": "North America",
    "Great Britain": "Europe",
    "Russia": "Asia",
    "Unknown": "Unreported",
    "Mexico": "Central America",
    "Ivory Coast": "Africa",
    "South Africa": "Africa",
    "Republic of Korea": "Asia",
    "France": "Europe",
    "Spain": "Europe",
    "Greece": "Europe",
    "Netherlands": "Europe",
    "Germany": "Europe",
    "Sweden": "Europe",
    "Italy": "Europe",
    "Colombia": "South America",
    "Canada": "North America",
    "Ghana": "Africa",
    "Egypt": "Africa",
    "Benin": "Africa",
    "Japan": "Asia",
    "China": "Asia",
    "Australia": "Oceania",
    "India": "Asia",
    "Hong Kong": "Asia",
    "Korea": "Asia",
    "Bulgaria": "Europe",
    "Czech Republic": "Europe",
    "Israel": "Asia",
    "Denmark": "Europe",
    "Austria": "Europe",
    "Hungary": "Europe",
    "Ireland": "Europe",
    "Ukraine": "Europe",
    "Belgium": "Europe",
    "Switzerland": "Europe",
    "Albania": "Europe",
    "Gibraltar": "Europe",
    "Portugal": "Europe",
    "Guatemala": "Central America",
    "Brazil": "South America",
    "Jamaica": "Caribbean",
    "Argentina": "South America",
    "Chile": "South America",
    "Cuba": "Caribbean",
    "Uraguay": "South America",
    "Iran": "Asia",
    "Lebanon": "Asia",
}

student_demo_dict = {
    "School of Arts & Sciences - Liberal Arts": {
        "Woman": [
            "American Indian or Alaska Native",
            "Asian",
            "Black or African American",
            "Hispanics of any race",
            "Nat. Hawaiian or Other Pac Island",
            "White",
            "Two or more races",
            "Unreported",
            "Unreported",
        ],
        "Man": [
            "American Indian or Alaska Native",
            "Asian",
            "Black or African American",
            "Hispanics of any race",
            "Nat. Hawaiian or Other Pac Island",
            "White",
            "Two or more races",
            "Unreported",
            "Unreported",
        ],
    },
    "School of Arts & Sciences - Grad": {
        "Woman": [
            "Asian",
            "Black or African American",
            "Hispanics of any race",
            "White",
            "Two or more races",
            "Unreported",
            "Unreported",
        ],
        "Man": [
            "American Indian or Alaska Native",
            "Asian",
            "Black or African American",
            "Hispanics of any race",
            "White",
            "Two or more races",
            "Unreported",
            "Unreported",
        ],
    },
    "School at the Museum of Fine Arts": {
        "Woman": [
            "American Indian or Alaska Native",
            "Asian",
            "Black or African American",
            "Hispanics of any race",
            "White",
            "Two or more races",
            "Unreported",
            "Unreported",
        ],
        "Man": [
            "Asian",
            "Black or African American",
            "Hispanics of any race",
            "White",
            "Two or more races",
            "Unreported",
            "Unreported",
        ],
    },
    "School of Engineering - Undergrad": {
        "Woman": [
            "Asian",
            "Black or African American",
            "Hispanics of any race",
            "Nat. Hawaiian or Other Pac Island",
            "White",
            "Two or more races",
            "Unreported",
            "Unreported",
        ],
        "Man": [
            "Asian",
            "Black or African American",
            "Hispanics of any race",
            "Nat. Hawaiian or Other Pac Island",
            "White",
            "Two or more races",
            "Unreported",
            "Unreported",
        ],
    },
    "School of Engineering - Grad": {
        "Woman": [
            "American Indian or Alaska Native",
            "Asian",
            "Black or African American",
            "Hispanics of any race",
            "Nat. Hawaiian or Other Pac Island",
            "White",
            "Two or more races",
            "Unreported",
            "Unreported",
        ],
        "Man": [
            "Asian",
            "Black or African American",
            "Hispanics of any race",
            "White",
            "Two or more races",
            "Unreported",
            "Unreported",
        ],
    },
    "School of Veterinary Medicine": {
        "Woman": [
            "Asian",
            "Black or African American",
            "Hispanics of any race",
            "White",
            "Two or more races",
            "Unreported",
            "Unreported",
        ],
        "Man": [
            "Asian",
            "Hispanics of any race",
            "White",
            "Two or more races",
            "Unreported",
        ],
    },
    "School of Law and Diplomacy": {
        "Woman": [
            "Asian",
            "Black or African American",
            "Hispanics of any race",
            "White",
            "Two or more races",
            "Unreported",
            "Unreported",
        ],
        "Man": [
            "Asian",
            "Black or African American",
            "Hispanics of any race",
            "White",
            "Two or more races",
            "Unreported",
            "Unreported",
        ],
    },
    "School of Nutrition and Science Policy": {
        "Woman": [
            "Asian",
            "Black or African American",
            "Hispanics of any race",
            "White",
            "Two or more races",
            "Unreported",
            "Unreported",
        ],
        "Man": [
            "Asian",
            "Black or African American",
            "Hispanics of any race",
            "White",
            "Two or more races",
            "Unreported",
        ],
    },
    "Graduate School of Biomedical Sciences": {
        "Woman": [
            "Asian",
            "Black or African American",
            "Hispanics of any race",
            "White",
            "Two or more races",
            "Unreported",
            "Unreported",
        ],
        "Man": [
            "Asian",
            "Black or African American",
            "Hispanics of any race",
            "White",
            "Two or more races",
            "Unreported",
            "Unreported",
        ],
    },
    "School of Dental Medicine": {
        "Woman": [
            "Asian",
            "Black or African American",
            "Hispanics of any race",
            "White",
            "Two or more races",
            "Unreported",
            "Unreported",
        ],
        "Man": [
            "American Indian or Alaska Native",
            "Asian",
            "Black or African American",
            "Hispanics of any race",
            "White",
            "Two or more races",
            "Unreported",
            "Unreported",
        ],
    },
    "School of Medicine - MD": {
        "Woman": [
            "Asian",
            "Black or African American",
            "Hispanics of any race",
            "White",
            "Two or more races",
            "Unreported",
            "Unreported",
        ],
        "Man": [
            "American Indian or Alaska Native",
            "Asian",
            "Black or African American",
            "Hispanics of any race",
            "White",
            "Two or more races",
            "Unreported",
        ],
    },
    "School of Medicine - PHPD": {
        "Woman": [
            "Asian",
            "Black or African American",
            "Hispanics of any race",
            "Nat. Hawaiian or Other Pac Island",
            "White",
            "Two or more races",
            "Unreported",
            "Unreported",
        ],
        "Man": [
            "Asian",
            "Black or African American",
            "Hispanics of any race",
            "White",
            "Two or more races",
            "Unreported",
            "Unreported",
        ],
    },
    "University College": {
        "Woman": [
            "Asian",
            "Black or African American",
            "Hispanics of any race",
            "White",
            "Two or more races",
            "Unreported",
            "Unreported",
        ],
        "Man": [
            "Asian",
            "Black or African American",
            "Hispanics of any race",
            "White",
            "Unreported",
        ],
    },
    "College of Special Studies": {
        "Woman": [
            "Black or African American",
            "Hispanics of any race",
            "White",
            "Unreported",
            "Unreported",
        ],
        "Man": ["Black or African American", "Two or more races", "Unreported"],
    },
}


def process_art_dataframe():

    with open(ROOT + "/data/mappings.json", "r") as file:
        mapping_dict = json.load(file)

    mapping_dict = {
        "gender": mapping_dict["gender_mapping"],
        "race": mapping_dict["race_mapping"],
        "region": mapping_dict["region_mapping"],
    }

    art_df = pd.read_excel(ROOT + "/data/TUAG_Artist_Subject_Donor_Data.xlsx")
    art_df.fillna("Unknown", inplace=True)
    art_df["loc"] = [
        hall_tuag_name_dict[a] for a in art_df["Current Location (Campus and Building)"]
    ]
    art_df.columns = [
        c.lower().strip(" ").replace(" ", "_").replace("__", "_") for c in art_df.columns
    ]
    art_df = art_df[[c for c in art_df.columns if not "unnamed" in c]].copy()
    art_df["artist"] = [a for a in art_df["creator_1"]]

    art_df["gender"] = [tuag_gender_dict[a] for a in art_df["creator_1_gender"]]
    art_df["race"] = [tuag_race_dict[a] for a in art_df["creator_1_ethnicity"]]
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
        "region_enum",
    ]
    art_df[col_of_interest].to_csv(ROOT + "/data/art_data_cleaned.csv")
    return art_df[col_of_interest]


def process_student_dataframe():
    df_totals = pd.DataFrame(
        columns=["school", "race", "gender", "full_time", "part_time", "total"]
    )

    # Data downloaded from Tufts 2021 Fall Enrollment Calculator.
    with pdfplumber.open(ROOT + r"/data/Tufts_2021_Fall_Enrollment_Calculator_Data.pdf") as pdf:
        for p in range(len(pdf.pages)):
            page = pdf.pages[p]

            table = page.extract_table()
            totals = [
                [int(i.replace(",", "")) for i in t.split(" ")] for t in np.array(table)[2:-1, -1]
            ]
            df = pd.DataFrame(totals, columns=["full_time", "part_time", "total"])
            df["race"] = "Unreported"
            df["gender"] = "Unreported"
            df["school"] = "Unreported"

            df.fillna("Unreported", inplace=True)
            df_totals = pd.concat([df_totals, df])

    df_totals.fillna("Unreported", inplace=True)
    df_totals.reset_index(drop=True, inplace=True)

    i = 0
    for k, v in student_demo_dict.items():
        for kk, vv in student_demo_dict[k].items():
            df_totals.loc[i : i + len(vv) - 1, "race"] = [r for r in vv]
            df_totals.loc[i : i + len(vv) - 1, "gender"] = kk
            df_totals.loc[i : i + len(vv) - 1, "school"] = k.split(" - ")[0]
            i += len(vv)

    df_totals = df_totals[["school", "gender", "race", "full_time", "part_time", "total"]]

    df_totals = df_totals.iloc[:197, :]

    assert df_totals["total"].sum() == 13293  # Known total enrollment Fall 2021

    df_totals.to_csv(ROOT + "/data/Tufts_2021_Fall_Enrollment_Calculator_Data.csv")
    return df_totals


def get_categorical_indices(student_df, categories=["gender", "race"]):
    """Returns list indices for categorical values

    Input:
        student_df: (dataframe) dataframe of students
        categories: (list of strings) list containing "gender" or "race".

    Output:
        Tuple of (enumeration index, string index)
    """
    # Load mappings
    mappings = sc.get_mapping_dicts()
    mapping_dict = {"gender": mappings[0], "race": mappings[1], "region": mappings[2]}

    cat_enum = [f"{c}_enum" for c in categories]
    string_index = []
    enum_index = []

    list_of_str_lists = []
    list_of_enum_lists = []
    for c in categories:
        list_of_str_lists.append(list(mapping_dict[c].keys()))
        list_of_enum_lists.append(list(mapping_dict[c].values()))

    str_idx = list(itertools.product(*list_of_str_lists))
    stu_idx = [tuple(s) for s in student_df.drop_duplicates(categories)[categories].values]
    str_idx = [s for s in str_idx if s in stu_idx and s[1] != "Unreported"]

    enum_idx = list(itertools.product(*list_of_enum_lists))
    stu_idx = [tuple(s) for s in student_df.drop_duplicates(categories)[cat_enum].values]
    enum_idx = [s for s in enum_idx if s in stu_idx and s[1] != 0]

    return enum_idx, str_idx


if __name__ == "__main__":

    logging.info("\n Processing art dataframe...")
    art_data = process_art_dataframe()
    art_data = art_data[art_data["loc"] != "crozier_fine_arts"]
    halls = art_data["loc"].unique()
    halls.sort()

    logging.info("\n Processing student dataframe...")
    process_student_dataframe()
    student_df = sc.get_student_enrollment_data()

    # Check if current_assignment_df exists, and if not, create it.
    exists = os.path.exists(ROOT + "/data/current_assignment_df.csv")
    if exists == False:
        enum_idx, str_idx = sc.get_categorical_indices(
            student_df=student_df, categories=["gender", "race"]
        )
        reduced_art_data = art_data[
            (art_data["gender"] != "Unreported") & (art_data["race"] != "Unreported")
        ]
        mappings = sc.get_mapping_dicts()
        mapping_dict = {"gender": mappings[0], "race": mappings[1], "region": mappings[2]}
        col_index = [", ".join(c) for c in str_idx]
        current_assignment_df = pd.DataFrame(0, index=halls, columns=col_index)
        loc_group = reduced_art_data.groupby("loc")
        for hall, hall_df in loc_group:
            gender_group = hall_df.groupby("gender")
            for gender, g_df in gender_group:
                for k, v in g_df["race"].value_counts().to_dict().items():
                    current_assignment_df.loc[hall, f"{gender}, {k}"] = v
        current_assignment_df.to_csv(ROOT + "/data/current_assignment_df.csv")

    # Check if hall_dict.json exists, and if not, create it.
    exists = os.path.exists(ROOT + "/data/hall_dict.json")
    if exists == False:
        logging.info(
            "\n Scaping hall data, this requires a network connection and takes a few moments..."
        )

        hall_dict = {}
        for hall in halls:
            hall_dict[hall] = sc.get_hall_dict(hall)
            with open(ROOT + "/data/hall_dict.json", "w") as fp:
                json.dump(hall_dict, fp)
    logging.info("\n Processing hall dataframe...")
    sc.get_hall_by_school_table()
