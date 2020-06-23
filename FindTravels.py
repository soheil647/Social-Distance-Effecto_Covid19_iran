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
extension = 'xlsx'
os.chdir("/home/sspc/Desktop/Datas/")
all_excels = [i for i in glob.glob('*.{}'.format(extension))]

all_cities = [i for i in glob.glob('*') if i not in extra_files]
for city in all_cities:
    os.chdir("/home/sspc/Desktop/Datas/" + city)
    all_excels = [i for i in glob.glob('*.{}'.format(extension))]

    all_folders = [i for i in glob.glob('*') if i not in extra_files]
    for folder in all_folders:
        os.chdir("/home/sspc/Desktop/Datas/" + city + '/' + folder)
        combined_csv = pd.DataFrame()
        frame = []
        df = pd.read_csv('test.xlsx')
        new_df = pd.DataFrame(columns=['date', 'class1', 'class2', 'class3', 'class4', 'class5', 'total', 'total_cars', 'is_copy'])
        for i in range(int(df['start_time'].str.split(pat='/')[len(df) - 1][2].split(' ')[0])):
            sum_columns = dict.fromkeys(
                ['class1', 'class2', 'class3', 'class4', 'class5', 'total', 'total_cars', 'is_copy'], 0)
            # this_date = df['start_time'].str.split(pat=' ')[i][0]
            for j in range(len(df)):
                if int(df['start_time'].str.split(pat='/')[j][2].split(' ')[0]) == i + 1:
                    # print(df['start_time'][j])
                    for key in sum_columns.keys():
                        sum_columns[key] += df[key][j]
            sum_columns['date'] = df['start_time'].str.split(pat=' ')[i][0]
            new_df = new_df.append(sum_columns, ignore_index=True)
            print(new_df)
            # days[this_date] = sum_columns
            # print(days)
        os.chdir("/home/sspc/Desktop/Datas/" + city)
        new_df.to_csv(folder + '_' + city + '.xlsx')
        # break
