import os
import glob
import pandas as pd
import datetime


def Insert_row_(row_number, df, row_value):
    # Slice the upper half of the dataframe
    df1 = df[0:row_number]

    # Store the result of lower half of the dataframe
    df2 = df[row_number:]

    # Inser the row in the upper half dataframe
    df1.loc[row_number] = row_value

    # Concat the two dataframes
    df_result = pd.concat([df1, df2])

    # Reassign the index labels
    df_result.index = [*range(df_result.shape[0])]

    # Return the updated dataframe
    return df_result


def create_time(year, month, day):
    if day < 10:
        day = '0' + str(day)
    return str(year) + '/' + str(month) + '/' + str(day) + ' ' + '00:00:00'


def seperate_time(month_df):
    # month_df['زمان شروع'] = month_df['زمان شروع'].str.split(pat='/')
    print(month_df['زمان پایان'].str.split(pat='/').map(lambda x: x[2]))
    month_df.loc[:, 'finish_time'] = month_df['زمان پایان'].str.split(pat='/').map(lambda x: x[2]).str.split(
        pat=' ').map(lambda x: x[1])
    month_df.loc[:, 'start_time'] = month_df['زمان شروع'].str.split(pat='/').map(lambda x: x[2]).str.split(pat=' ').map(
        lambda x: x[1])
    month_df.loc[:, 'finish_day'] = month_df['زمان پایان'].str.split(pat='/').map(lambda x: x[2]).str.split(
        pat=' ').map(lambda x: x[0])
    month_df.loc[:, 'start_day'] = month_df['زمان شروع'].str.split(pat='/').map(lambda x: x[2]).str.split(pat=' ').map(
        lambda x: x[0])

    k = 0
    for i in range(len(month_df)):
        k = k + 1
        # print(month_df['کد محور'][i])
        if month_df['کد محور'][i] == 113253:
            if int(month_df['start_day'][i]) == k:
                continue
            line = pd.DataFrame(
                {"کد محور": 113253, "زمان شروع": create_time(1398, 10, k), "زمان پایان": create_time(1398, 10, k+1), "finish_time": "00:00:00", "start_time": "00:00:00", "finish_day": k + 1,
                 "start_day": k}, index=[k])
            # line = pd.DataFrame()
            print(create_time(1398, 9, k))
            print(month_df.loc[3].values)
            month_df = pd.concat([month_df.iloc[:i], line, month_df.iloc[i:]]).reset_index(drop=True)

    month_df.fillna(0, inplace=True)
    month_df.to_csv("test.xlsx", index=False)


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

extension = 'xlsx'
os.chdir("/home/sspc/Desktop/Datas/")
all_excels = [i for i in glob.glob('*.{}'.format(extension))]
for excel in all_excels:
    os.remove(excel)

extra_files = ["MergeCSVs.py", "FindTravels.py", "Travels"]
all_cities = [i for i in glob.glob('*') if i not in extra_files]
for city in all_cities:
    os.chdir("/home/sspc/Desktop/Datas/" + city)
    all_excels = [i for i in glob.glob('*.{}'.format(extension))]

    # for month_excel in all_excels:
    month_df = pd.read_csv(all_excels[0])
    # month_df = month_df.iloc[:, ::-1]
    # print(month_df.head())
    print(seperate_time(month_df))
