import os
import glob
import pandas as pd


def create_sorted_month(my_list):
    my_list = swapPositions(my_list, 2, 0)
    my_list = swapPositions(my_list, 2, 1)
    my_list = swapPositions(my_list, 2, 3)
    return my_list


def swapPositions(my_list, pos1, pos2):
    my_list[pos1], my_list[pos2] = my_list[pos2], my_list[pos1]
    return my_list

extra_files = ["Model.py", "FindTravels.py", "Travels", "Append_CSVs.py", "CorrectDatas.py"]
extension = 'xlsx'
os.chdir("/home/sspc/Desktop/Datas/")
all_excels = [i for i in glob.glob('*.{}'.format(extension))]

all_cities = [i for i in glob.glob('*') if i not in extra_files and i not in all_excels]
for city in all_cities:
    city_df = pd.DataFrame(columns=['date', 'class1', 'class2', 'class3', 'class4', 'class5', 'total', 'total_cars', 'is_copy'])
    os.chdir("/home/sspc/Desktop/Datas/" + city)
    all_excels = [i for i in glob.glob('*.{}'.format(extension)) if i != city + '.xlsx']
    all_excels = create_sorted_month(all_excels)
    print(all_excels)
    for excel in all_excels:
        city_df = city_df.append(pd.read_csv(excel), ignore_index=True)
    if os.path.isfile(city + '.xlsx'):
        os.remove(city + '.xlsx')
    city_df.to_csv(city + '.csv')
    city_df.to_csv(city + '.xlsx')
