import pickle
import warnings

import pandas as pd

warnings.filterwarnings('ignore')

df = pd.read_csv('../data/data.csv')
df = df[df['Crop_Damage'] == 'Minimal Damage']
df.drop(['ID', 'Number_Doses_Week', 'Number_Weeks_Used', 'Number_Weeks_Quit', 'Crop_Damage'], axis=1, inplace=True)

'''
Estimated Insects Count:
Min 150 max 4097(or above)
'''

insect = int(input('Enter the estimated number of insect count on average on 1 plant '))

'''
Crop Type:
Rabi Crop
Kharif Crop
'''

crop_type = input('Enter crop type ')

'''
Soil Type:
Black-Cotton
Alluvial soil
'''

soil_type = input('Enter type of soil ')

'''
Season type:
Summer
Monsoon
Winter
'''

''' 
Pesticide Use Category
Insecticides 2
Bactericides 0
Herbicides 1
'''

season = input('Enter the weather ')

pest_dict = {2: 'Insecticides', 0: 'Bactericides', 1: 'Herbicides'}

filename = 'lightgbm_pest_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict([[insect, crop_type, soil_type, season]])
print(pest_dict[result[0]])

print('Lets find approximately how much of pesticide will you need')
row = input('How many rows does your plantation has ')
col = input('How many columns does your plantation has ')
plant_row = input('How many plants are there in a row on average')

total_plant = row * col * plant_row

