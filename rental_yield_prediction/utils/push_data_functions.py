import pandas as pd
import numpy as np
import re

def extract_area(desc) -> pd.DataFrame:
    extracted = re.findall("[0-9]+ sqft", desc, flags=re.IGNORECASE)
    if len(extracted)==0:
        extracted=np.nan
    return extracted

def mean_area(area):
    if area == np.nan:
        pass
    else:
        if isinstance(area, list):
            int_arr = []
            for i in area:
                i = float(i.split(" ")[0])
                int_arr.append(i)
            area = sum(int_arr)/len(int_arr)
        else:
            pass
    return area

def extract_balconies(desc):
    extracted = re.findall("[a-z0-9/:/-/./*/+]+ balcon[a-z]+", desc, re.IGNORECASE)
    if len(extracted)==0:
        extracted=np.nan
    return extracted

def num_balcony(bal_str):
    if bal_str==np.nan:
        pass
    else:
        if isinstance(bal_str, list):
            bal_str = bal_str[0].split(" ")[0]
        else:
            pass
    return bal_str

def extract_age(desc):
    extracted = re.findall("age [a-z0-9/:/./-/*]+ [a-z0-9/:/-/.]+ [a-z0-9/:/-/.]+", desc, re.IGNORECASE)
    if len(extracted)==0:
        extracted=np.nan
    else:
        for i in extracted:
            if "age of construction" in i.lower():
                extracted=i
            else:
                extracted=np.nan
    return extracted

def age_num(age):
    if isinstance(age, str):
        age = age.split(":")
        if len(age)>1:
            age=age[1].strip()
        else:
            age=age[0]
    return age

def num_amenities(amenities):
    num_amenity = len(amenities.split(","))
    return num_amenity

def transform_bhk(bhk):
    num_bhk = bhk.split(" ")[0]
    if '+' in num_bhk:
        num_bhk = int(num_bhk[:-1])
        num_bhk += 1
    elif num_bhk=="Apartments/Flats":
        num_bhk=1
    else:
        num_bhk=int(num_bhk)
    return num_bhk

def clean_floor(floor):
    if floor=="Ground":
        floor=0
    else:
        floor = floor[:len(floor)-2]
    return floor

def transform_rent(rent):
    if 'lakh' in rent.lower():
        rent = float(rent.split(" ")[0])*100000
    elif "Contact for Price" in rent:
        rent = np.nan
    else:
        rent = float(rent.split(",")[0])*1000
    return rent
