from bs4 import BeautifulSoup
import time
import pandas as pd
import numpy as np
import psycopg2
import re
from sqlalchemy import create_engine, URL
from rental_yield_prediction.utils.data_cleaning_utils.data_cleaning_func import extract_area, mean_area, extract_balconies, num_balcony, extract_age, age_num, num_amenities, transform_bhk, clean_floor, transform_rent
from rental_yield_prediction.constants.push_data_constants import COLS_ORDER, HOSTNAME, DATABASE, USERNAME, PASSWORD, PORT_ID
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


with open("webpage/443 11 Property for Rent in Mumbai, Rental Properties _ Sulekha.html", "r", encoding='utf-8') as f:
    doc = BeautifulSoup(f, 'html.parser')

print(doc.title)
listings = doc.find_all(class_="sk-card listing-card sk-shadow")
print(len(listings))

c=1
title = []
address = []
price = []
size = []
floor = []
parking = []
preference = []
furnishing  = []
veg = []
more = []
amenities_list = []
posted_date = []
for listing in listings:
    # print("-"*40)
    # print("Listing# - ", c)
    title1 = listing.find(class_="sk-h6").string.strip()
    title.append(title1)
    location1 = listing.find(class_="sk-caption-text truncate").string.strip()
    address.append(location1)
    price1 = listing.find(class_="sk-h5 rupee")
    if price1==None:
        price1 = listing.find(class_="price-info").get_text(strip=True)
        price.append(price1)

    else:
        price1 = price1.string.strip()
        price.append(price1)

    config = listing.find(class_ ="prime-highlights scroll-wrap mobile-hide")
    bhk = config.find("strong").string.strip()
    size.append(bhk)
    more_info = listing.find_all(class_="sk-chip sk-base-fill")

    text_1 = listing.find(class_="sk-body-text-1 truncate").string.strip()
    more.append(text_1)
    amenities = listing.find(class_="sk-lead-text-small")
    if amenities==None:
        amenities = listing.find(class_="sk-body-text-1 sk-line-clamp").string.strip()
        amenities_list.append(amenities)
    else:
        amenities = amenities.string.strip()
        amenities_list.append(amenities)
    posted = listing.find(class_="posted")
    # print(title1)
    # print(location1)
    # print(price1)
    # print(bhk)
    if len(more_info)==5:
        floor.append(more_info[0].string)
        parking.append(more_info[1].string)
        preference.append(more_info[2].string)
        furnishing.append(more_info[3].string)
        veg.append(more_info[4].string)
    elif len(more_info)==4:
        floor.append(more_info[0].string)
        parking.append(more_info[1].string)
        preference.append(more_info[2].string)
        furnishing.append(more_info[3].string)
        veg.append(np.nan)
    elif len(more_info)==3:
        floor.append(more_info[0].string)
        parking.append(more_info[1].string)
        preference.append(more_info[2].string)
        furnishing.append(np.nan)
        veg.append(np.nan)
    elif len(more_info)==2:
        floor.append(more_info[0].string)
        parking.append(more_info[1].string)
        preference.append(np.nan)
        furnishing.append(np.nan)
        veg.append(np.nan)
    elif len(more_info)==1:
        floor.append(more_info[0].string)
        parking.append(np.nan)
        preference.append(np.nan)
        furnishing.append(np.nan)
        veg.append(np.nan)
    else:
        floor.append(np.nan)
        parking.append(np.nan)
        preference.append(np.nan)
        furnishing.append(np.nan)
        veg.append(np.nan)

    posted_date.append(posted.get_text(strip=True).replace(posted.strong.get_text(strip=True), ''))
    # print(text_1)
    # print(amenities)
    # print(posted.get_text(strip=True).replace(posted.strong.get_text(strip=True), ''))
    # print("-"*40)
    c=c+1


housing_data_dict = {"Title":title, "Address":address, "BHK":size, "Floor":floor, "Parking": parking, "Preference":preference, "Furnishing":furnishing, "Veg/Non-veg":veg, "description": more, "Amenities": amenities_list, 
                     "Posted_date":posted_date, "Rent":price}
housing_data = pd.DataFrame(housing_data_dict)

# Extracting data from the description column
housing_data.drop(columns=["Title", "Posted_date"], inplace=True)
housing_data["area"] = housing_data["description"].apply(extract_area)
housing_data["area"] = housing_data["area"].apply(mean_area)
housing_data["age"] = housing_data["description"].apply(extract_age)
housing_data["age"] = housing_data["age"].apply(age_num)
housing_data["balconies"] = housing_data["description"].apply(extract_balconies)
housing_data["balconies"] = housing_data["balconies"].apply(num_balcony)
housing_data["age"].replace(['New', 'age of construction. The'], [0, np.nan], inplace=True)
housing_data["balconies"].replace(['No', '3+', 'the', 'a'], [0, 4, 1, 1], inplace=True)
housing_data["age"] = housing_data["age"].astype("float")
housing_data["balconies"] = housing_data["balconies"].fillna(0)
housing_data["balconies"] = housing_data["balconies"].astype("int")

# Dropping row with wrong data
idx = housing_data[housing_data["Floor"].apply(lambda x: len(x.split("(of")))==1].index[0]
housing_data.drop(index=idx, inplace=True)
housing_data.reset_index(drop=True, inplace=True)

# Cleaning the data in the columns
housing_data["num_amenities"] = housing_data["Amenities"].apply(num_amenities)
housing_data["num_BHK"] = housing_data["BHK"].apply(transform_bhk)
housing_data["Rent"] = housing_data["Rent"].apply(transform_rent)
housing_data["Total_floor"] = housing_data["Floor"].apply(lambda x: x.split("(of")[1].strip().split(" ")[0])
housing_data["Floor"] = housing_data["Floor"].apply(lambda x: x.split(" ")[0])
housing_data["Floor"] = housing_data["Floor"].apply(clean_floor)
housing_data["Total_floor"].replace("50+", 50, inplace=True)
housing_data["Floor"] = housing_data["Floor"].astype("int")
housing_data["Total_floor"] = housing_data["Total_floor"].astype("int")
housing_data["Parking"].replace(['No Car Parking', '1 Car Parking', '2 Car Parking', '3 Car Parking'], [0,1,2,3], inplace=True)
housing_data["Suburb"] = housing_data["Address"].str.split(",").str[0]
housing_data["Rent"] = housing_data["Rent"].fillna(housing_data.Rent.mean())
housing_data["Veg/Non-veg"] = housing_data["Veg/Non-veg"].fillna("Veg/Non Veg")

# Dropping non-essential columns and reordering dataframe
housing_data.drop(columns=["Address", "BHK", "Amenities"], inplace=True)
housing_data = housing_data[COLS_ORDER]
print(housing_data.head())
print(housing_data.isnull().sum())
print(housing_data.info())
print(housing_data.shape)


# Creating table in sql database and insert dataframe records
url_object = URL.create(
    "postgresql",
    username=USERNAME,
    password=PASSWORD,
    host=HOSTNAME,
    database=DATABASE,
)
engine = create_engine(url_object)
housing_data.to_sql(name="rental_data", con=engine, if_exists="replace", index_label="id")


# Connecting to the postgresql server
# conn = None
# cur = None
# try:
#     conn = psycopg2.connect(
#         host=HOSTNAME,
#         dbname=DATABASE,
#         user=USERNAME,
#         password=PASSWORD,
#         port=PORT_ID
#     )
#     cur = conn.cursor()

#     # Creating table in postgresql
#     create_table = ''' 
#     CREATE TABLE IF NOT EXISTS rental_data(
#     id INT PRIMARY KEY,
#     Suburb VARCHAR(30),
#     Floor INT,
#     Total_floor INT,
#     Parking INT,
#     num_BHK INT,
#     Preference VARCHAR(30),
#     Furnishing VARCHAR(30),
#     Food_pref VARCHAR(30),
#     num_amenities INT,
#     age DECIMAL,
#     balconies INT,
#     area DECIMAL,
#     Rent DECIMAL
#     );
#     '''
#     cur.execute(create_table)

#     insert_data = 

#     conn.commit()
# except Exception as e:
#     print(e)

# finally:
#     if cur is not None:
#         cur.close()
#     if conn is not None:
#         conn.close()





