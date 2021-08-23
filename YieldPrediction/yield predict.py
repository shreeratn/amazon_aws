import pickle
import warnings

warnings.filterwarnings('ignore')

state = int(input('Enter State '))
dist = int(input('Enter District  '))
season = int(input('Enter Season  '))
crop = int(input('Enter Crop  '))
area = int(input('Enter land area  '))

filename = '../YieldPrediction/lightgbm_yield_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict([[state, dist, season, crop, area]])
if result[0] < 0:
    print('Check your inputs, if everything is right then the production predicted is negligible')
else:
    print('\n\n' + str(int(result[0])))
