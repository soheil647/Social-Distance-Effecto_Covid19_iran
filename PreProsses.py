import pandas as pd
import os
from persiantools.jdatetime import JalaliDate
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class PreProcess:
    def __init__(self, city_name='Tehran'):
        self.city_name = city_name

    def process_input_data(self, drop=False, city=False, country=False, plot=False, shift=0, normalize=False,
                           weak_days=False, plot_input=True):
        os.chdir("/home/sspc/Desktop/Datas/")
        corona_data = pd.read_excel('Corona.xlsx')
        corona_data.date = corona_data.apply(lambda x: x['date'] + pd.DateOffset(days=shift), axis=1)
        corona_data.set_index('date', inplace=True)

        air_data = pd.read_csv('Airplane.csv')
        air_data['Date'] = pd.date_range(start='21/1/2020', end='20/06/2020')
        air_data.set_index('Date', inplace=True)

        if city:
            os.chdir("/home/sspc/Desktop/Datas/Citiyes")
            data_file_city = pd.read_csv('cities_data.xlsx')

        if country:
            os.chdir("/home/sspc/Desktop/Datas/Citiyes")
            data_file_city = pd.read_csv('country_data.xlsx')

        else:
            input_city = "/home/sspc/Desktop/Datas/Citiyes/" + self.city_name
            os.chdir(input_city)
            data_file_city = pd.read_csv(self.city_name + '.xlsx')

        data_file_city = data_file_city.loc[:, ~data_file_city.columns.str.contains('^Unnamed')]
        data_file_city = data_file_city.drop(columns=data_file_city.columns[-1])
        for i in range(len(data_file_city)):
            data_file_city.loc[i, 'date'] = JalaliDate(int(data_file_city['date'][i].split('/')[0]),
                                                       int(data_file_city['date'][i].split('/')[1]),
                                                       int(data_file_city['date'][i].split('/')[2])).to_gregorian()
        data_file_city.set_index('date', inplace=True)

        data = pd.concat([data_file_city, air_data, corona_data], axis=1).dropna()

        if plot_input:
            data = data.reset_index(drop=True)

            def plot_parameter(input_feature, column_name):
                # use LaTeX fonts in the plot
                plt.title(column_name, fontsize=11)
                plt.plot(input_feature)
                plt.xlabel('Day', fontsize=11)
                plt.ylabel(column_name, fontsize=11)

                plt.savefig(column_name + '.pdf', bbox_inches='tight')
                plt.show()
            plot_parameter(data['Daily New Cases'], 'Daily New Cases')
            # plot_parameter(data['Daily Deaths'], 'Daily Deaths')
            # plot_parameter(data['Daily Active Cases'], 'Daily Active Cases')
            plot_parameter(data['total_cars'], 'Total Cars')
            plot_parameter(data['Total Passengers traveled'], 'Total Passengers traveled')




        features = data.drop('Daily New Cases', axis=1)
        target = data['Daily New Cases']

        if weak_days:
            print(features.index.weekday())
            features['day contribution'] = 0
            exit()

        if drop:
            features = features.drop('Daily Deaths', axis=1).drop('Daily Active Cases', axis=1).drop(
                'Daily New Recoveries', axis=1)

        if plot:
            norm_case = pd.DataFrame([float(i) / sum(target) for i in target])

            norm_total = [float(i) / sum(features.total) for i in features.total]
            title = 'Daily New Corona Virus Cases compare to Cars: ' + str(shift)
            plt.title(title)
            plt.plot(norm_case)
            plt.plot(norm_total)
            plt.xlabel('Day')
            plt.ylabel("Normalized Inputs")
            plt.legend(['New Case', 'Total Cars'])
            plt.show()

            norm_air = [float(i) / sum(features['Total Passengers traveled']) for i in features['Total Passengers traveled']]
            title = 'Daily New Corona Virus Cases compare to Airs: ' + str(shift)
            plt.title(title)
            plt.plot(norm_case)
            plt.plot(norm_air)
            plt.xlabel('Day')
            plt.ylabel("Normalized Inputs")
            plt.legend(['New Case', 'Total Passengers'])
            plt.show()


            title = 'Daily New Case, Total Cars, Total Passengers with shift: ' + str(shift)
            plt.title(title)
            plt.plot(norm_case)
            plt.plot(norm_total)
            plt.plot(norm_air)
            plt.xlabel('Day')
            plt.ylabel("Inputs")
            plt.legend(['New Case', 'Total cars', 'Total Passengers'])
            plt.show()

        if normalize:
            scaler = MinMaxScaler()
            features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

        return features, target
