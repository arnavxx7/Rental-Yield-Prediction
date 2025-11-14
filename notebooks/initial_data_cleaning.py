import requests
from bs4 import BeautifulSoup
import time
import numpy as np
import pandas as pd
import pickle
url = "https://property.sulekha.com/residential-property-for-rent/mumbai?ref=propertyhomepage"
# response = requests.get(url)
# print(response.status_code)

with open("443 11 Property for Rent in Mumbai, Rental Properties _ Sulekha.html", "r", encoding='utf-8') as f:
    doc = BeautifulSoup(f, 'html.parser')

print(doc.title)
listings = doc.find_all(class_="sk-card listing-card sk-shadow")
print(len(listings))
time.sleep(5)
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
    print("-"*40)
    print("Listing# - ", c)
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
    print(title1)
    print(location1)
    print(price1)
    print(bhk)
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
    print(text_1)
    print(amenities)
    print(posted.get_text(strip=True).replace(posted.strong.get_text(strip=True), ''))
    print("-"*40)
    c=c+1


housing_data_dict = {"Title":title, "Address":address, "BHK":size, "Floor":floor, "Parking": parking, "Preference":preference, "Furnishing":furnishing, "Veg/Non-veg":veg, "description": more, "Amenities": amenities_list, 
                     "Posted_date":posted_date, "Rent":price}
housing_data = pd.DataFrame(housing_data_dict)

print(housing_data.head())
print(housing_data.shape)

housing_data.to_csv("new_housing_data.csv", index=False)


