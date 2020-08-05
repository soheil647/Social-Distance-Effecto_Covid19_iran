import os
import glob
import pandas as pd
import numpy as np


xlsx_extension = 'xlsx'
csv_extension = 'csv'
os.chdir("/home/sspc/Desktop/Datas/Citiyes")
all_excels = [i for i in glob.glob('*.{}'.format(xlsx_extension))]

combined_csv = pd.DataFrame()
frame = []
all_cities = [i for i in glob.glob('*') if i not in all_excels]

for city in all_cities:
    os.chdir("/home/sspc/Desktop/Datas/Citiyes/" + city)
    df = pd.read_csv(city + '.xlsx')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.set_index('date', inplace=True)
    frame.append(df)

new_df = frame[0]
for i in range(len(frame) - 1):
    new_df = new_df + frame[i + 1]

combined_csv = pd.concat(frame, keys=all_cities)

os.chdir("/home/sspc/Desktop/Datas/Citiyes/")
combined_csv.to_csv('cities_data.xlsx')
new_df.to_csv('country_data.xlsx')
