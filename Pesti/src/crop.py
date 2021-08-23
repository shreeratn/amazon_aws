import pickle
import warnings

warnings.filterwarnings('ignore')

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
area = int(input('Enter the area of your land in (sq meter) '))
row = int(input('Enter intended row spacing in your your plantation (cm) ')) / 100
col = int(input('Enter intended column spacing in your your plantation (cm) ')) / 100

plant_population = area / (row * col)

print('\n\nYou can plant about ', int(plant_population), ' plants in your land')
print('You will need ', 360 * area/10000, 'mL of', pest_dict[result[0]])