import os
import glob
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def create_time(year, month, day):
    if day < 10:
        day = '0' + str(day)
    return str(year) + '/' + str(month) + '/' + str(day) + ' ' + '00:00:00'


def seperate_time(month_df):
    # month_df['زمان شروع'] = month_df['زمان شروع'].str.split(pat='/')
    year = month_df['زمان شروع'].str.split(pat='/')[1][0]
    month = month_df['زمان شروع'].str.split(pat='/')[1][1]
    month_df.loc[:, 'finish_time'] = month_df['زمان پایان'].str.split(pat='/').map(lambda x: x[2]).str.split(
        pat=' ').map(lambda x: x[1])
    month_df.loc[:, 'start_time'] = month_df['زمان شروع'].str.split(pat='/').map(lambda x: x[2]).str.split(pat=' ').map(
        lambda x: x[1])
    month_df.loc[:, 'finish_day'] = month_df['زمان پایان'].str.split(pat='/').map(lambda x: x[2]).str.split(
        pat=' ').map(lambda x: x[0])
    month_df.loc[:, 'start_day'] = month_df['زمان شروع'].str.split(pat='/').map(lambda x: x[2]).str.split(pat=' ').map(
        lambda x: x[0])
    code_mehvar = month_df['کد محور'].iloc[0]
    name = month_df['نام محور'].iloc[0]

    k = 0
    for i in range(len(month_df)):
        k = k + 1
        if int(month_df['start_day'].iloc[i]) == k:
            continue
        line = pd.DataFrame(
            {"کد محور": code_mehvar, 'نام محور': name, "زمان شروع": create_time(year, month, k),
             "زمان پایان": create_time(year, 10, k + 1), "finish_time": "00:00:00", "start_time": "00:00:00",
             "finish_day": k + 1,
             "start_day": k}, index=[k])
        month_df = pd.concat([month_df.iloc[:i], line, month_df.iloc[i:]]).reset_index(drop=True)
        return month_df

extra_files = ["MergeCSVs.py", "FindTravels.py", "Travels", "FillCSVs.py"]
extension = 'xlsx'
os.chdir("/home/sspc/Desktop/Datas/")
all_excels = [i for i in glob.glob('*.{}'.format(extension))]
# for excel in all_excels:
#     os.remove(excel)

all_cities = [i for i in glob.glob('*') if i not in extra_files]
for city in all_cities:
    os.chdir("/home/sspc/Desktop/Datas/" + city)
    all_excels = [i for i in glob.glob('*.{}'.format(extension))]
    # for excel in all_excels:
    #     os.remove(excel)

    all_folders = [i for i in glob.glob('*') if i not in extra_files]
    for folder in all_folders:
        os.chdir("/home/sspc/Desktop/Datas/" + city + "/" + folder)
        all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
        # combine all files in the list
        for i in range(len(all_filenames)):
            df = pd.read_excel(all_filenames[i])
            df.columns = df.iloc[0]
            df = df.iloc[1:]
            new_df = seperate_time(df)
            new_df.to_csv("test.xlsx", index=False)
            print(new_df)
            break
        break

        # combined_csv = pd.concat([pd.read_excel(all_filenames[0], encoding='utf-8')] + [pd.read_excel(f, encoding='utf-8').iloc[1:] for f in all_filenames[1:]])
        # combined_csv.columns = combined_csv.iloc[0]
        # combined_csv = combined_csv.iloc[1:]
        # combined_csv.sort_values(by=combined_csv.columns[1])
        # # export to csv
        # os.chdir("/home/sspc/Desktop/Datas/" + city)
        # combined_csv.to_csv(folder + ".xlsx", index=False)
