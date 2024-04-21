import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost.sklearn import XGBRegressor
# Generate sample data
df = pd.read_csv('tuning.csv')
X = df.iloc[:,0:3]
y0 = df.iloc[:,3]
y1=df.iloc[:,4]
y2=df.iloc[:,5]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
initial = 18
v = 5
a = 1
# Initialize and fit the XGBRegressor model
reg0 = XGBRegressor(n_estimators=50,
            max_depth=5,
            gamma = 0,
            importance_type='gain', ## gain, weight, cover, total_gain, total_cover
            reg_lambda = 1, ## tuning parameter of l2 penalty
            random_state=100).fit(X,y0)
reg1 = XGBRegressor(n_estimators=50,
            max_depth=5,
            gamma = 0,
            importance_type='gain', ## gain, weight, cover, total_gain, total_cover
            reg_lambda = 1, ## tuning parameter of l2 penalty
            random_state=100).fit(X,y1)
reg2 = XGBRegressor(n_estimators=50,
            max_depth=5,
            gamma = 0,
            importance_type='gain', ## gain, weight, cover, total_gain, total_cover
            reg_lambda = 1, ## tuning parameter of l2 penalty
            random_state=100).fit(X,y2)


# Predict on the test set
p_pred = reg0.predict([[initial,v,a]])
i_pred = reg1.predict([[initial,v,a]])
d_pred = reg2.predict([[initial,v,a]])
print(p_pred[0],i_pred[0],d_pred[0])