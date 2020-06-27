import pandas as pd
import os
from persiantools.jdatetime import JalaliDate
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

class PreProcess:
    def __init__(self, city_name='Tehran'):
        self.city_name = city_name

    @staticmethod
    def save_read_travel_data(city):
        df = pd.read_csv(city + '/Tehran.xlsx')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.drop(columns=df.columns[-1])
        for i in range(len(df)):
            df['date'][i] = JalaliDate(int(df['date'][i].split('/')[0]), int(df['date'][i].split('/')[1]),
                                       int(df['date'][i].split('/')[2])).to_gregorian()
        df.to_csv('Travel_data.xlsx')

    def process_input_data(self, drop=False):
        corona_data = pd.read_excel('Corona.xlsx')
        corona_data.set_index('date', inplace=True)
        input_city = "/home/sspc/Desktop/Datas/Citiyes/" + self.city_name
        os.chdir(input_city)
        data_file_city = pd.read_csv(self.city_name + '.xlsx')
        data_file_city = data_file_city.loc[:, ~data_file_city.columns.str.contains('^Unnamed')]
        data_file_city = data_file_city.drop(columns=data_file_city.columns[-1])
        for i in range(len(data_file_city)):
            data_file_city['date'][i] = JalaliDate(int(data_file_city['date'][i].split('/')[0]),
                                                   int(data_file_city['date'][i].split('/')[1]),
                                                   int(data_file_city['date'][i].split('/')[2])).to_gregorian()
        data_file_city.set_index('date', inplace=True)

        data = pd.concat([data_file_city, corona_data], axis=1).dropna()
        # return data
        features = data.drop('Daily New Cases', axis=1)
        target = data['Daily New Cases']

        if drop:

            # To Add Date to Columns in Day + month : '02-11 = 112
            # features['date'] = features.index
            # for i in range(len(features)):
            #     features['date'][i] = str(features['date'][i].day) + str(features['date'][i].month)

            # To drop Corona Columns
            features = features.drop('Daily Deaths', axis=1).drop('Daily Active Cases', axis=1).drop('Daily New Recoveries', axis=1)

            # features.iloc[1] = np.nan
            # features = features.interpolate(limit_direction='both', kind='cubic')

        return features, target
