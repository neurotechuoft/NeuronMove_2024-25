import matplotlib.pyplot as plt
import sklearn
import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')

x,y = 1,2 # Replace with actual data loading or generation

regressor = RandomForestRegressor(n_estimators=9, random_state=0, oob_score=True)

regressor.fit(x, y)