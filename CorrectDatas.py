import os
import glob
import pandas as pd


def reject_outliers(data, m=1300):
    indexes = []
    for i in range(1, len(data)+1):
        if data[i] <= m:
            indexes.append(i)
    return indexes


def create_time(year, month, day):
    if day < 10:
        day = '0' + str(day)
    return str(year) + '/' + str(month) + '/' + str(day) + ' ' + '00:00:00'


def find_max_day(month_name):
    if month_name == 'bahman':
        return 30
    if month_name == 'esfand':
        return 29
    if month_name == 'ordibehesht':
        return 31
    if month_name == 'farvardin':
        return 31


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

my_columns = ['mehvar_code', 'mehvar_name', "start_time", "finish_time", 'duration', 'total', 'class1', 'class2',
              'class3', 'class4', 'class5', 'average_speed', 'speed_punish', 'distance_punish', 'sebghat_punish',
              'total_cars']

extra_files = ["Model.py", "FindTravels.py", "Travels", "Append_CSVs.py", "CorrectDatas.py", 'Model.py']
extension = 'xlsx'
os.chdir("/home/sspc/Desktop/Datas/")
all_excels = [i for i in glob.glob('*.{}'.format(extension))]

all_cities = [i for i in glob.glob('*') if i not in all_excels and i not in extra_files]
for city in all_cities:
    os.chdir("/home/sspc/Desktop/Datas/" + city)
    all_excels = [i for i in glob.glob('*.{}'.format(extension))]

    all_folders = [i for i in glob.glob('*') if i not in all_excels]
    for folder in all_folders:
        combined_csv = pd.DataFrame()
        frame = []
        os.chdir("/home/sspc/Desktop/Datas/" + city + "/" + folder)
        if os.path.isfile(folder + '.xlsx'):
            os.remove(folder + '.xlsx')
        all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
        for file in all_filenames:
            # print(folder, file)
            df = pd.read_excel(file)
            df = df.iloc[1:]
            df.columns = my_columns
            df['is_copy'] = 0
            # print(len(df['duration']))
            year = df['start_time'].str.split(pat='/')[1][0]
            month = df['start_time'].str.split(pat='/')[1][1]
            outlayers = reject_outliers(df['duration'])
            df = df.drop(outlayers).reset_index(drop=True)
            if len(df) == 0:
                print(1)
                continue
            # print(df)
            k = 0
            i = 0
            # while i < int(df['start_time'].str.split(pat='/')[len(df) - 1][2].split(' ')[0]):
            max_day = find_max_day(folder.split(sep='_')[1])
            while i < max_day:
                k = k + 1
                if i >= len(df) and k != max_day:
                    line = df.loc[[i - 1], :]
                    line.start_time = create_time(year, month, k)
                    line.finish_time = create_time(year, month, k+1)
                    line.is_copy = 1
                    df = pd.concat([df.iloc[:i], line, df.iloc[i:]]).reset_index(drop=True)
                    i = i + 1
                    continue
                if k == max_day:
                    print(int(df['start_time'].str.split(pat='/')[len(df) - 1][2].split(' ')[0]))
                    # print(int(df['start_time'].str.split(pat='/')[i][2].split(' ')[0]))
                    if int(df['start_time'].str.split(pat='/')[len(df) - 1][2].split(' ')[0]) == max_day:
                        i = i + 1
                        continue
                    print("Yeap")
                    line = df.loc[[i - 1], :]
                    line.start_time = create_time(year, month, k)
                    if int(month) < 10:
                        month = '0' + str(int(month)+1)
                    elif month == 12:
                        month = str('01')
                    else:
                        month = str(int(month)+1)
                    line.finish_time = create_time(year, month, 1)
                    line.is_copy = 1
                    df = pd.concat([df.iloc[:i], line, df.iloc[i:]]).reset_index(drop=True)
                    i = i + 1
                    continue

                # print(folder, file)
                # print(i)
                # print(len(df))
                # print(df['start_time'].str.split(pat='/')[i][2].split(' ')[0])

                day = df['start_time'].str.split(pat='/')[i][2].split(' ')[0]
                if int(day) == k:
                    i = i + 1
                    continue

                line = df.loc[[i], :]
                line.start_time = create_time(year, month, k)
                line.finish_time = create_time(year, month, k+1)
                line.is_copy = 1
                df = pd.concat([df.iloc[:i], line, df.iloc[i:]]).reset_index(drop=True)
                i = i + 1

            frame.append(df)

        combined_csv = pd.concat(frame)
        combined_csv.to_csv(folder + '.xlsx')