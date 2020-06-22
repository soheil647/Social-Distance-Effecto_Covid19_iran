import os
import glob
import pandas as pd

extra_files = ["MergeCSVs.py", "FindTravels.py", "Travels", "FillCSVs.py"]
extension = 'xlsx'
os.chdir("/home/sspc/Desktop/Datas/")
all_excels = [i for i in glob.glob('*.{}'.format(extension))]
for excel in all_excels:
    os.remove(excel)

all_cities = [i for i in glob.glob('*') if i not in extra_files]
for city in all_cities:
    os.chdir("/home/sspc/Desktop/Datas/" + city)
    all_excels = [i for i in glob.glob('*.{}'.format(extension))]
    for excel in all_excels:
        os.remove(excel)

    all_folders = [i for i in glob.glob('*') if i not in extra_files]
    for folder in all_folders:
        os.chdir("/home/sspc/Desktop/Datas/" + city + "/" + folder)
        all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
        # combine all files in the list
        combined_csv = pd.concat([pd.read_excel(all_filenames[0], encoding='utf-8')] + [pd.read_excel(f, encoding='utf-8').iloc[1:] for f in all_filenames[1:]])
        combined_csv.columns = combined_csv.iloc[0]
        combined_csv = combined_csv.iloc[1:]
        combined_csv.sort_values(by=combined_csv.columns[1])
        # export to csv
        os.chdir("/home/sspc/Desktop/Datas/" + city)
        combined_csv.to_csv(folder + ".xlsx", index=False)

