"""
The underlying student gender and race data was obtained from the public
Fall 2020 Enrollment Calculator (https://provost.tufts.edu/institutionalresearch/enrollment/).
Relevant data was downloaded, manually entered into an Excel spreadsheet and saved as csv:
* `../data/2021_10_19_EC_school_gender_race.pdf`
* `../data/2021_10_19_EC_school_gender_race.ods`
* `../data/2021_10_19_EC_school_gender_race.csv`


The underlying student region data was obtained from the
Fall 2020 Enrollment Calculator With Region
(https://tableau.uit.tufts.edu/#/site/IR/workbooks/6160/views)
(provided by Christina Butler, Director of the Tufts Office of Institutional Research
-- link might be password protected, but AH has access).Relevant data was
downloaded, manually entered into an Excel spreadsheet and saved as csv:

* `../data/2021_10_04_EC_school_gender_region.pdf`
* `../data/2021_10_04_EC_school_gender_region.ods`
* `../data/2021_10_04_EC_school_gender_region.csv`
"""
import os
import sys
import json
import logging
import requests

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup

ROOT = os.popen("git rev-parse --show-toplevel").read().split("\n")[0]
sys.path.append(ROOT)

import src as sc  # pylint: disable=wrong-import-position

# These are latitude and longitude values copied by hand from google.
hall_loc_dict_google = {
    "aidekman": "42.404577445357646, -71.11816612130153",
    "anderson": "42.406500240258325, -71.11601155569957",
    "ballou": "42.40730139450371, -71.11653626395496",
    "capen_house": "42.40560729720486, -71.11723085199174",
    "clic": "42.40355455977418, -71.11373119988099",
    "dental_school": "42.35077898498154, -71.0617586752671",
    "eaton_hall": "42.40703659215104, -71.11827244777004",
    "fine_arts_house": "42.40518492161451, -71.11740062871739",
    "gifford_house": "42.40729126002778, -71.12063602686325",
    "goddard_chapel": "42.40732529112991, -71.11884573242514",
    "goddard_hall": "42.407887497397965, -71.12131452630499",
    "granoff_fam_hillel_center": "42.40960137735758, -71.12035618824363",
    "granoff_music_bld": "42.40470027821146, -71.11731877712019",
    "houston_hall": "42.4085281434443, -71.1216589612613",
    "jean_mayer_hnra_center": "42.35104764796711, -71.06285834220928",
    "lane_hall": "42.40930273282628, -71.11914035384562",
    "miner_hall": "42.40682588046766, -71.11533463429204",
    "packard_hall": "42.408078870775434, -71.11852608082816",
    "smfa": "42.33880817740776, -71.09622110729772",
    "sophia_gordon_hall": "42.405234409998506, -71.11804457660654",
    "tisch_library": "42.40638608452481, -71.11828651522619",
    "univ_advancement": "42.41198146190384, -71.11452789802702",
}

# This dictionary maps the art data hall names to the Tufts url hall names.
hall_url_name_dict = {
    "aidekman": "Aidekman Arts Center",
    "anderson": "SEC-Anderson Hall",
    "ballou": "Ballou Hall",
    "capen_house": "Capen House",
    "clic": "Collaborative Learning and Innovation Complex %28CLIC%29&",
    "dental_school": "Tufts University School of Dental Medicine",
    "eaton_hall": "Eaton Hall",
    "fine_arts_house": "Fine Arts House",
    "gifford_house": "Gifford House",
    "goddard_chapel": "Goddard Chapel",
    "goddard_hall": "Goddard Hall",
    "granoff_fam_hillel_center": "Granoff Family Hillel Center",
    "granoff_music_bld": "Perry and Marty Granoff Music Center",
    "houston_hall": "Houston Hall",
    "jean_mayer_hnra_center": "Jean Mayer USDA Human Nutrition Research Center on Aging",
    "lane_hall": "Lane Hall",
    "miner_hall": "Miner Hall",
    "packard_hall": "Packard Hall",
    "smfa": "School at the Museum of Fine Arts",
    "sophia_gordon_hall": "Sophia Gordon Hall",
    "tisch_library": "Tisch Library",
    "univ_advancement": "80 George Street",
}

hall_short_name_dict = {
    "aidekman": "Aidekman Arts Center",
    "anderson": "SEC-Anderson",
    "ballou": "Ballou Hall",
    "capen_house": "Capen House",
    "clic": "CLIC",
    "dental_school": "Dental School",
    "eaton_hall": "Eaton Hall",
    "fine_arts_house": "Fine Arts House",
    "gifford_house": "Gifford House",
    "goddard_chapel": "Goddard Chapel",
    "goddard_hall": "Goddard Hall",
    "granoff_fam_hillel_center": "Granoff Hillel Center",
    "granoff_music_bld": "Granoff Music Center",
    "houston_hall": "Houston Hall",
    "jean_mayer_hnra_center": "Jean Mayer HNRC",
    "lane_hall": "Lane Hall",
    "miner_hall": "Miner Hall",
    "packard_hall": "Packard Hall",
    "smfa": "SMFA",
    "sophia_gordon_hall": "Sophia Gordon Hall",
    "tisch_library": "Tisch Library",
    "univ_advancement": "80 George Street",
}

hall_school_dict = {
    "aidekman": ["School of Arts & Sciences", "Public Use"],
    "anderson": ["School of Arts & Sciences", "School of Engineering"],
    "ballou": ["School of Arts & Sciences", "General Administration", "Public Use"],
    "capen_house": ["Residence Hall"],
    "clic": ["School of Arts & Sciences", "School of Engineering"],
    "dental_school": ["School of Dental Medicine"],
    "eaton_hall": ["School of Arts & Sciences"],
    "fine_arts_house": [
        "School of Arts & Sciences",
        "School at the Museum of Fine Arts",
        "Residence Hall",
    ],
    "gifford_house": ["General Administration", "Public Use"],
    "goddard_chapel": ["General Administration", "Public Use"],
    "goddard_hall": ["School of Law and Diplomacy"],
    "granoff_fam_hillel_center": ["General Administration"],
    "granoff_music_bld": ["School of Arts & Sciences", "General Administration", "Public Use"],
    "houston_hall": ["School of Arts & Sciences", "Residence Hall"],
    "jean_mayer_hnra_center": [
        "School of Nutrition and Science Policy",
        "Graduate School of Biomedical Sciences",
        "School of Medicine",
    ],
    "lane_hall": ["School of Arts & Sciences"],
    "miner_hall": ["School of Arts & Sciences"],
    "packard_hall": ["School of Arts & Sciences"],
    "smfa": ["School at the Museum of Fine Arts"],
    "sophia_gordon_hall": ["Residence Hall"],
    "tisch_library": ["Public Use"],
    "univ_advancement": [
        "School of Arts & Sciences",
        "School of Nutrition and Science Policy",
        "General Administration",
    ],
}


def get_hall_dict(hall):
    """
    Generates dictionary for hall giving url, lat, lon and departments.
    """
    hall_dict = {}
    hall_name = hall_url_name_dict[hall]
    dept = []
    for hall_type in ["Academic Building", "Residence Halls", "Other"]:
        if len(dept) == 0:
            url = "https://m.tufts.edu/tufts_mobile/map_all/detail?feed=maps_all&id=maps_all%2F{}%2F{}&parentId=maps_all%2F{}".format(  # pylint: disable=line-too-long, consider-using-f-string
                hall_type.replace(" ", "%20"), hall_name, hall_type.replace(" ", "%20")
            )
            # Request specified url
            request = requests.get(url)
            html_content = request.text
            soup = BeautifulSoup(html_content, "lxml")

            # Check that the url directs to a meaninful page.
            if "No results found" in [
                s.text for s in soup.find_all("span", {"class": "kgoui_list_item_title"})
            ]:
                pass
            else:
                hall_dict["hall_type"] = hall_type

                # Try to get latitude and longitude from Tufts page.
                for par in [s.text for s in soup.find_all("p")]:
                    if "GPS Coordinates: " in par:
                        coord = par.strip("GPS Coordinates: ")

                        # Latitude and longitude are listed backwards on the Tufts website.
                        hall_dict["latitude"] = float(coord.split(", ")[1])
                        hall_dict["longitude"] = float(coord.split(", ")[0])

                # Get the list of departments from the Tufts page.
                for l_item in soup.find_all("li"):
                    dept += [str(s.text) for s in l_item.find_all("strong")]

                # If no departments were listed, and "Departments Unknown" to depts.
                if len(dept) == 0:
                    dept = ["Departments Unknown"]

                hall_dict["url"] = url
                hall_dict["departments"] = dept

    # If latitude and longitude were not obtained from the page, get them from google.
    hall_dict["latitude"] = hall_dict.get(
        "latitude", float(hall_loc_dict_google[hall].split(", ")[0])
    )
    hall_dict["longitude"] = hall_dict.get(
        "longitude", float(hall_loc_dict_google[hall].split(", ")[1])
    )

    # If no url was found, fill in as "Unknown"
    hall_dict["hall_type"] = hall_dict.get("hall_type", "Unknown")
    hall_dict["url"] = hall_dict.get("url", "Unknown")
    hall_dict["departments"] = hall_dict.get("departments", ["Departments Unknown"])

    return hall_dict


def print_hall_dictionary_to_json():
    """
    Print dictionary of hall dictionaries to json file.
    """
    building_df = sc.get_building_capacity_df()

    hall_dict = {}
    for hall in building_df.index:
        hall_dict[hall] = get_hall_dict(hall)

    # Fix Lane Hall
    hall_dict["lane_hall"]["hall_type"] = "Academic Building"

    with open(f"{ROOT}/data/hall_dict.json", "w", encoding="utf-8") as outfile:
        json.dump(hall_dict, outfile)


def get_hall_by_school_table(student_df):
    """Return num_halls x num_schools one-hot table."""
    schools = list(student_df["school"].unique())

    with open(f"{ROOT}/data/hall_dict.json", encoding="utf-8") as json_file:
        hall_dict = json.load(json_file)

    loc_df = pd.DataFrame(columns=["name", "type"])
    loc_df["name"] = list(hall_dict.keys())
    loc_df["type"] = [v["hall_type"] for v in hall_dict.values()]
    loc_df.sort_values(by=["name"], inplace=True)

    locs = list(loc_df["name"])
    create_hall_df = pd.DataFrame(
        0, index=locs, columns=schools + ["General Administration", "Residence Hall", "Public Use"]
    )

    for hall in loc_df[loc_df["type"] == "Residence Halls"]["name"]:
        create_hall_df.loc[hall, "Residence Hall"] = 1

    for h_idx in create_hall_df.index:
        depts = hall_school_dict[h_idx]

        create_hall_df.loc[h_idx, depts] = 1

    create_hall_df.to_csv(f"{ROOT}/data/hall_df.csv")

    return create_hall_df


def get_student_enrollment_data():
    """Read student enrollment data."""
    df_total = pd.read_csv(
        ROOT + "/data/Tufts_2021_Fall_Enrollment_Calculator_Data.csv", index_col=0
    )

    df_students = pd.DataFrame()
    for i in df_total.index:
        df = pd.DataFrame(
            index=range(df_total.loc[i, "total"]), columns=["school", "gender", "race"]
        )
        df["school"] = df_total.loc[i, "school"]
        df["gender"] = df_total.loc[i, "gender"]
        df["race"] = df_total.loc[i, "race"]

        df_students = pd.concat([df_students, df])
    assert df_students.shape[0] == 13293  # Known total enrollment Fall 2021

    df_students.reset_index(drop=True, inplace=True)
    df_students.reset_index(inplace=True)
    df_students.rename(columns={"index": "student_id"}, inplace=True)

    gender_map, race_map, region_map = sc.get_mapping_dicts()  # pylint: disable=unused-variable

    df_students["gender_enum"] = [gender_map[g] for g in df_students["gender"]]
    df_students["race_enum"] = [race_map[g] for g in df_students["race"]]

    df_students.to_csv(ROOT + "/data/student_df.csv")

    return df_students


def fill_academic_building(building, student_df, hall_df):
    """Fill academic buildings with students with randomness."""
    total_enrollment = student_df.shape[0]
    depts = [c for c in hall_df.columns if hall_df.loc[building, c] == 1]
    df = student_df[student_df["school"].isin(depts)]

    other_students = student_df[~student_df["school"].isin(depts)]
    extra_users = 0
    if "General Administration" in depts:
        extra_users += int(total_enrollment * (0.01))

    if "Public Use" in depts:
        extra_users += int(total_enrollment * (0.02))

    idx = np.random.choice(other_students.index, extra_users, replace=False)
    df = pd.concat([df, other_students.loc[idx, :]])

    df.reset_index(drop=True, inplace=True)

    return df


def fill_residence_halls(student_df, hall_df):
    """Fill residence halls with randomness."""
    res_hall_dict = {
        "houston_hall": 256,
        "sophia_gordon_hall": 256,
        "fine_arts_house": 14,
        "capen_house": 26,
    }

    assigned = []
    remainder_df = student_df.copy()
    for key, val in res_hall_dict.items():

        if key == "capen_house":
            idx1 = np.random.choice(
                remainder_df[
                    remainder_df["race"].isin(
                        ["Black or African American", "Two or more races", "Hispanics of any races"]
                    )
                ].index,
                5,
            )
            idx2 = np.random.choice(
                remainder_df[remainder_df["race"].isin(["Black or African American"])].index,
                val - 5,
            )
            idx = list(idx1) + list(idx2)

        else:
            idx = np.random.choice(remainder_df.index, val)

        df = remainder_df.loc[idx, :]

        assigned += list(idx)
        remainder_df = remainder_df[~remainder_df.index.isin(assigned)]

        # Add in additional traffic if building holds academic, or admin departments.
        df_academic = fill_academic_building(
            building=key, student_df=student_df[~student_df.index.isin(idx)], hall_df=hall_df
        )

        df = pd.concat([df, df_academic])
        df.to_csv(f"{ROOT}/data/filled_buildings/{key}_students.csv")

    return df


def fill_buildings(student_df, hall_df):
    """Fill buildings with randomness."""
    exists = os.path.exists(f"{ROOT}/data/filled_buildings")
    if exists is False:
        os.mkdir(f"{ROOT}/data/filled_buildings")

    logging.info("\n Filling residence halls...")
    fill_residence_halls(student_df=student_df, hall_df=hall_df)

    logging.info("\n Filling remaining buildings...")
    for building in hall_df.index:
        if hall_df.loc[building, "Residence Hall"] == 0:
            df = fill_academic_building(building, student_df=student_df, hall_df=hall_df)

            df.to_csv(f"{ROOT}/data/filled_buildings/{building}_students.csv")


if __name__ == "__main__":

    student_df = get_student_enrollment_data()
    hall_df = get_hall_by_school_table(student_df=student_df)

    fill_buildings(student_df=student_df, hall_df=hall_df)
