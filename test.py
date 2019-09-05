import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import csv
filename="lolw2.csv"

slno=[]
Pick_Time=[]
Start_Time_of_Picking=[]
SKU=[]
Actual_Quantity=[]
User=[]
manning=[]
last_station_served_by_user=[]
number_of_items_container=[]
total_quantity_of_items_in_container=[]
volume_of_items_present_in_container=[]
Total_Weight_of_items_present_in_container=[]
volume_of_items=[]
Height_of_item=[]
Length_of_item=[]
Width_of_item=[]
Cube_of_item=[]
weight_of_each_item=[]
total_time_spent_by_user_in_minutes=[]
total_quantity_picked_by_user=[]
Location=[]
number_of_container_conveyor=[]
station=[]
day=[]
datalist=[]

with open(filename, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)
    for line in csvreader:
        slno=line[0]
        Pick_Time =line[1]
        Start_Time_of_Picking =line [2]
        SKU=line[3]
        Actual_Quantity =line [4]
        User = line[5]
        manning=line[6]
        last_station_served_by_user=line[7]
        number_of_items_container = line[8]
        total_quantity_of_items_in_container =line [9]
        volume_of_items_present_in_container = line[10]
        Total_Weight_of_items_present_in_container = line[11]
        volume_of_items = line[12]
        Height_of_item = line[13]
        Length_of_item = line[14]
        Width_of_item = line[15]
        Cube_of_item = line[16]
        weight_of_each_item = line[17]
        total_time_spent_by_user_in_minutes = line[18]
        total_quantity_picked_by_user = line[19]
        Location = line[20]
        number_of_container_conveyor = line[21]
        station = line[22]
        day = line[23]
        datalist=datalist+[
            slno,
            Pick_Time,
            SKU,
            Actual_Quantity,
            User,
            manning,
            last_station_served_by_user,
            number_of_items_container,
            total_quantity_of_items_in_container,
            volume_of_items_present_in_container,
            Total_Weight_of_items_present_in_container,
            volume_of_items,
            Height_of_item,
            Length_of_item,
            Width_of_item,
            Cube_of_item,
            weight_of_each_item,
            total_time_spent_by_user_in_minutes,
            total_quantity_picked_by_user,
            Location,
            number_of_container_conveyor,
            station,
            day
            ]

    print(datalist)