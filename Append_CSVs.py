import os
import glob
import pandas as pd


extra_files = ["Model.py", "FindTravels.py", "Travels", "Append_CSVs.py", "CorrectDatas.py"]
extension = 'xlsx'
os.chdir("/home/sspc/Desktop/Datas/")

all_cities = [i for i in glob.glob('*') if i not in extra_files]
for city in all_cities:
    city_df = pd.DataFrame(columns=['date', 'class1', 'class2', 'class3', 'class4', 'class5', 'total', 'total_cars', 'is_copy'])
    os.chdir("/home/sspc/Desktop/Datas/" + city)
    all_excels = [i for i in glob.glob('*.{}'.format(extension))]
    for excel in all_excels[::-1]:
        city_df = city_df.append(pd.read_csv(excel), ignore_index=True)
    city_df.to_csv(city + '.xlsx')
