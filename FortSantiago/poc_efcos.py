# To ignore the imports warning
import warnings
warnings.filterwarnings("ignore")
# For the dataset loading 
import pandas as pd
# for arrays manupulation
import numpy as np
#for filling the missing values
from sklearn.impute import SimpleImputer
# For making train and test data 
from sklearn.model_selection import train_test_split
# For scaling the dataset 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
# Importing the machine learning models
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


# read csv files
df = pd.read_csv('FortSantiago_PortArea_Consolidated_Data.csv')
df.dropna(inplace=True)

# separate data into features and labels
X = df.drop(['FLOOD_HEIGHT_HIGHEST'], axis=1)
y= df['FLOOD_HEIGHT_HIGHEST']


# train test split
x_train,x_test,y_train,y_test=train_test_split(X.values,y.values,test_size=0.2,random_state=0)

# Standard Scaler
std = StandardScaler()
x_train=std.fit_transform(x_train)
x_test=std.transform(x_test)

# For Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=1400, min_samples_split= 5, min_samples_leaf = 4, max_features = 'sqrt', max_depth = 80, bootstrap = True)
rf_model.fit(x_train, y_train)
pred_rf = rf_model.predict(x_test)

# For ANN Regressor
ANN_model = MLPRegressor(hidden_layer_sizes=(20, 20), activation='relu', solver='adam', random_state=50)
ANN_model.fit(x_train, y_train)
pred_ANN = ANN_model.predict(x_test)

# For SVM Regression
SVR_model = SVR(kernel='rbf', C=1000) 
SVR_model.fit(x_train, y_train)
pred_SVR = SVR_model.predict(x_test)

# Saving the model
import pickle
pickle.dump(rf_model, open('rf_prediction.pkl', 'wb'))
pickle.dump(ANN_model, open('ANN_prediction.pkl', 'wb'))
pickle.dump(SVR_model, open('SVR_prediction.pkl', 'wb'))
pickle.dump(std, open('standscaler.pkl', 'wb'))