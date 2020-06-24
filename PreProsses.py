import pandas as pd
from persiantools.jdatetime import JalaliDate


class PreProcess:
    def __init__(self, file_name='Travel_data.xlsx', cities='Tehran'):
        self.file_name = file_name

    @staticmethod
    def save_read_travel_data(city):
        df = pd.read_csv(city + '/Tehran.xlsx')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.drop(columns=df.columns[-1])
        for i in range(len(df)):
            df['date'][i] = JalaliDate(int(df['date'][i].split('/')[0]), int(df['date'][i].split('/')[1]),
                                       int(df['date'][i].split('/')[2])).to_gregorian()
        df.to_csv('Travel_data.xlsx')

    def process_input_data(self):
        data_file = pd.read_excel(self.file_name)

        features = data_file.drop('Daily New Cases', axis=1)
        features = features.loc[:, ~features.columns.str.contains('^Unnamed')]
        features = features.fillna(0)
        # features = features.drop('Daily Deaths', axis=1)
        # features = features.drop('Daily Active Cases', axis=1)
        # features = features.drop('Daily New Recoveries', axis=1)
        features = features.drop('date', axis=1)
        # features = features.drop('is_copy', axis=1)
        target = data_file['Daily New Cases']
        # target = pd.DataFrame(np.roll(target, -12))
        return features, target
