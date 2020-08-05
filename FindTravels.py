import os
import glob
import pandas as pd


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

my_columns = ['mehvar_code', 'mehvar_name', "start_time", "finish_time", 'duration', 'total', 'class1', 'class2',
              'class3', 'class4', 'class5', 'average_speed', 'speed_punish', 'distance_punish', 'sebghat_punish',
              'total_cars']

extra_files = ["Model.py", "FindTravels.py", "Travels", "Append_CSVs.py", "CorrectDatas.py"]

xlsx_extension = 'xlsx'
csv_extension = 'csv'

os.chdir("/home/sspc/Desktop/Datas/Citiyes")
all_excels = [i for i in glob.glob('*.{}'.format(xlsx_extension))]
all_csv = [i for i in glob.glob('*.{}'.format(csv_extension))]

all_cities = [i for i in glob.glob('*') if i not in extra_files and i not in all_excels + all_csv]
for city in all_cities:
    os.chdir("/home/sspc/Desktop/Datas/Citiyes/" + city)
    all_excels = [i for i in glob.glob('*.{}'.format(xlsx_extension))]
    all_csv = [i for i in glob.glob('*.{}'.format(csv_extension))]

    all_folders = [i for i in glob.glob('*') if i not in all_excels + all_csv]
    for folder in all_folders:
        os.chdir("/home/sspc/Desktop/Datas/Citiyes/" + city + '/' + folder)
        combined_csv = pd.DataFrame()
        frame = []
        df = pd.read_csv(folder + '.xlsx')
        new_df = pd.DataFrame(columns=['date', 'class1', 'class2', 'class3', 'class4', 'class5', 'total', 'total_cars', 'is_copy'])
        for i in range(int(df['start_time'].str.split(pat='/')[len(df) - 1][2].split(' ')[0])):
            sum_columns = dict.fromkeys(
                ['class1', 'class2', 'class3', 'class4', 'class5', 'total', 'total_cars', 'is_copy'], 0)
            for j in range(len(df)):
                if int(df['start_time'].str.split(pat='/')[j][2].split(' ')[0]) == i + 1:
                    for key in sum_columns.keys():
                        sum_columns[key] += df[key][j]
            sum_columns['date'] = df['start_time'].str.split(pat=' ')[i][0]
            new_df = new_df.append(sum_columns, ignore_index=True)
            print("city: ", city, "day number: ", i, " month number: ", df['start_time'].str.split(pat='/')[i][1])
        os.chdir("/home/sspc/Desktop/Datas/Citiyes/" + city)
        if os.path.isfile(folder + '.xlsx'):
            print("Folder: ", folder, " Passed")
            continue
            # os.remove(folder + '.xlsx')
        new_df.to_csv(folder + '.xlsx')
