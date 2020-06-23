import os
import glob
import pandas as pd


def reject_outliers(data, m=1300):
    indexes = []
    for i in range(1, len(data)):
        if data[i] <= m:
            indexes.append(i)
    return indexes


def create_time(year, month, day):
    if day < 10:
        day = '0' + str(day)
    return str(year) + '/' + str(month) + '/' + str(day) + ' ' + '00:00:00'


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

my_columns = ['mehvar_code', 'mehvar_name', "start_time", "finish_time", 'duration', 'total', 'class1', 'class2',
              'class3', 'class4', 'class5', 'average_speed', 'speed_punish', 'distance_punish', 'sebghat_punish',
              'total_cars']

extra_files = ["Model.py", "FindTravels.py", "Travels", "Append_CSVs.py", "CorrectDatas.py"]
extension = 'xlsx'
os.chdir("/home/sspc/Desktop/Datas/")
all_excels = [i for i in glob.glob('*.{}'.format(extension))]

all_cities = [i for i in glob.glob('*') if i not in extra_files]
for city in all_cities:
    os.chdir("/home/sspc/Desktop/Datas/" + city)
    all_excels = [i for i in glob.glob('*.{}'.format(extension))]

    all_folders = [i for i in glob.glob('*') if i not in extra_files]
    for folder in all_folders:
        combined_csv = pd.DataFrame()
        frame = []
        os.chdir("/home/sspc/Desktop/Datas/" + city + "/" + folder)
        if os.path.isfile('./test.xlsx'):
            os.remove('test.xlsx')
        all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
        for file in all_filenames:
            df = pd.read_excel(file)
            df = df.iloc[1:]
            df.columns = my_columns
            df['is_copy'] = 0
            outlayers = reject_outliers(df['duration'])
            df = df.drop(outlayers).reset_index(drop=True)
            # print(df)
            k = 0
            i = 0
            # print(folder, file, len(df))
            while i < int(df['start_time'].str.split(pat='/')[len(df) - 1][2].split(' ')[0]):
                k = k + 1
                year = df['start_time'].str.split(pat='/')[i][0]
                month = df['start_time'].str.split(pat='/')[i][1]
                day = df['start_time'].str.split(pat='/')[i][2].split(' ')[0]
                # print(int(day), k)
                if int(day) == k:
                    i = i + 1
                    continue
                # df2 = concat([df.iloc[:2], line, df.iloc[2:]]).reset_index(drop=True)
                # print(df.loc[[i], :])
                line = df.loc[[i], :]
                line.start_time = create_time(year, month, k)
                line.finish_time = create_time(year, month, k+1)
                line.is_copy = 1
                df = pd.concat([df.iloc[:i], line, df.iloc[i:]]).reset_index(drop=True)
                i = i + 1
            frame.append(df)
    # os.chdir("/home/sspc/Desktop/Datas/Tehran")
        combined_csv = pd.concat(frame)
        combined_csv.to_csv("test.xlsx")