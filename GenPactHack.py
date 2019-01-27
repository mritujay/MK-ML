import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import statsmodels.api as sm
import matplotlib
import itertools
import pandas_profiling as pp

# import files 

train = pd.read_csv('C:/Users/mritunjay/dataScience/Python/AVHACK/train_GzS76OK/train.csv')
mealInfo = pd.read_csv('C:/Users/mritunjay/dataScience/Python/AVHACK/train_GzS76OK/meal_info.csv')
fulfilmentCenterInfo = pd.read_csv('C:/Users/mritunjay/dataScience/Python/AVHACK/train_GzS76OK/fulfilment_center_info.csv')
test = pd.read_csv('C:/Users/mritunjay/dataScience/Python/AVHACK/train_GzS76OK/test.csv')
train.info()
mealInfo.info()
fulfilmentCenterInfo.info()
train.head()
test['num_orders']=-999

df_rm = train.groupby('week')['num_orders'].sum()
df_rm.plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20);

## Concatnet train and test data
all_data = pd.concat([train,test],axis=0)
all_data.reset_index(inplace=True,drop=True)

# checking missing values
all_data.isnull().sum()
# no missing values
all_data.head()

# taking week 1 as 10-JAN-2016

#end_week = pd.to_datetime('10/01/16')
end_week = '10/01/16'
def nextdate(weekno):
    if(weekno >1):
        #end_week = end_week+(7*(weekno-1))
        end_week_new = datetime.datetime.strptime(end_week, "%d/%m/%y")+datetime.timedelta(days=7*(weekno-1))
    else:
        end_week_new = datetime.datetime.strptime(end_week, "%d/%m/%y")
    return end_week_new

all_data['week_day'] = all_data.apply(lambda x: nextdate(x['week']), axis=1)
all_data.drop(["week"],axis=1,inplace=True)
week = pd.DataFrame(index = range(0,all_data.shape[0]),columns=['week'])
week['week'] = all_data['week_day'].dt.week
all_data = all_data.join(week)
all_data.head()
all_data.info()
all_data.drop(["week_day"],axis=1,inplace=True)

# Pandas profiling 
pProfile = pp.ProfileReport(X_fdTrain)
pProfile.to_file("C:/Users/mritunjay/dataScience/Python/AVHACK/profileReport_1.html")
#base_price is highly correlated with checkout_price (ρ = 0.95339) Rejected

# Pandas profiling 
#pProfile = pp.ProfileReport(fdTrain)
#pProfile.to_file("C:/Users/mritunjay/dataScience/Python/AVHACK/profileReport.html")
#base_price is highly correlated with checkout_price (ρ = 0.95339) Rejected

all_data_new1 = pd.merge(all_data, fulfilmentCenterInfo, on='center_id')
all_data_new  = pd.merge(all_data_new1, mealInfo, on='meal_id')
all_data_new.info()
all_data_new.head()
all_data_new.drop(["center_id","meal_id","base_price"],axis=1,inplace=True)
all_data_new.head()



for col in ['emailer_for_promotion', 'homepage_featured', 'city_code', 'region_code','week']:
    all_data_new[col] = all_data_new[col].astype('object')
    
    
from sklearn.preprocessing import LabelEncoder

var_mod_le = ['city_code','region_code','center_type','category','cuisine','week']
var_mod_dum = ['emailer_for_promotion','homepage_featured','city_code','region_code','center_type','category','cuisine','week']

le = LabelEncoder()
for i in var_mod_dum:
    all_data_new[i] = le.fit_transform(all_data_new[i])
    
all_data_new.info()
all_data_new.head()  
all_data_new = pd.get_dummies(all_data_new,columns=['emailer_for_promotion','homepage_featured','city_code','region_code','center_type','category','cuisine','week'])
all_data_new.info()
all_data_new.head()


fdTrain = all_data_new[all_data_new['num_orders']!=-999]
fdTest = all_data_new[all_data_new['num_orders'] ==-999]
fdTest.drop(["num_orders"],axis=1,inplace=True)

#y_fdTrain=fdTrain['num_orders']
X_fdTrain = fdTrain
#X_fdTrain.drop(["num_orders"],axis=1,inplace=True)
X_fdTest = fdTest

target = 'num_orders'
IDcol = ['id']
from sklearn import cross_validation, metrics
import statsmodels.api as sm
def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
    X= sm.add_constant(dtrain[predictors])
    Y= dtrain[target]
    a1=sm.OLS(Y,X)
    a=a1.fit()
    print (a.summary())
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    #Perform cross-validation:
    cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20, scoring='mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    #Print model report:
    print ("\nModel Report")
    print ("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
    print ("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
    #Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])
    
    #Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index=False)



from sklearn.linear_model import LinearRegression,Ridge,Lasso

predictors = [x for x in X_fdTrain.columns if x not in [target]+IDcol]
alg1 = LinearRegression(normalize=True)
modelfit(alg1,X_fdTrain,X_fdTest,predictors,target,IDcol,'C:/Users/mritunjay/dataScience/Python/AVHACK/train_GzS76OK/alg1.csv')

coef1 = pd.Series(alg1.coef_,predictors).sort_values()
coef1.plot(kind='bar',title='Model Coefficients')
#Model Report
#RMSE : 297.6
#CV Score : Mean - 7.905e+12 | Std - 3.446e+13 | Min - 134.9 | Max - 1.581e+14

## Ridge Regression Model:

predictors = [x for x in X_fdTrain.columns if x not in [target]+IDcol]
alg2 = Ridge(alpha=0.05,normalize=True)
modelfit(alg2, X_fdTrain, X_fdTest, predictors, target, IDcol, 'C:/Users/mritunjay/dataScience/Python/AVHACK/train_GzS76OK/alg2.csv')
coef2 = pd.Series(alg2.coef_, predictors).sort_values()
coef2.plot(kind='bar', title='Model Coefficients')
#Model Report
#RMSE : 298.6
#CV Score : Mean - 289.4 | Std - 157.7 | Min - 128.4 | Max - 602

## Decision Tree Model


from sklearn.tree import DecisionTreeRegressor
predictors = [x for x in X_fdTrain.columns if x not in [target]+IDcol]
alg3 = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
modelfit(alg3, X_fdTrain, X_fdTest, predictors, target, IDcol, 'C:/Users/mritunjay/dataScience/Python/AVHACK/train_GzS76OK/alg3_week.csv')
coef3 = pd.Series(alg3.feature_importances_, predictors).sort_values(ascending=False)
coef3.plot(kind='bar', title='Feature Importances',figsize=(20,10), linewidth=5, fontsize=20)
#Model Report
#RMSE : 216.6
#CV Score : Mean - 254.7 | Std - 148.1 | Min - 47.99 | Max - 554.1

from sklearn.tree import DecisionTreeRegressor
predictors = [x for x in X_fdTrain.columns if x not in [target]+IDcol]
alg3_1 = DecisionTreeRegressor(max_depth=20, min_samples_leaf=15)
modelfit(alg3_1, X_fdTrain, X_fdTest, predictors, target, IDcol, 'C:/Users/mritunjay/dataScience/Python/AVHACK/train_GzS76OK/alg3_1_week.csv')
coef3_1 = pd.Series(alg3_1.feature_importances_, predictors).sort_values(ascending=False)
coef3_1.plot(kind='bar', title='Feature Importances',figsize=(20,10), linewidth=5, fontsize=20)
#Model Report
#RMSE : 179.1
#CV Score : Mean - 248.7 | Std - 143.3 | Min - 47.41 | Max - 547.1

from sklearn.tree import DecisionTreeRegressor
predictors = [x for x in X_fdTrain.columns if x not in [target]+IDcol]
alg3_2 = DecisionTreeRegressor(max_depth=25, min_samples_leaf=15)
modelfit(alg3_2, X_fdTrain, X_fdTest, predictors, target, IDcol, 'C:/Users/mritunjay/dataScience/Python/AVHACK/train_GzS76OK/alg3_1_week_25.csv')
coef3_2 = pd.Series(alg3_2.feature_importances_, predictors).sort_values(ascending=False)
coef3_2.plot(kind='bar', title='Feature Importances',figsize=(20,10), linewidth=5, fontsize=20)
#Model Report
#RMSE : 177.1
#CV Score : Mean - 248.8 | Std - 143.5 | Min - 47.37 | Max - 546.7

from sklearn.tree import DecisionTreeRegressor
predictors = [x for x in X_fdTrain.columns if x not in [target]+IDcol]
alg3_3 = DecisionTreeRegressor(max_depth=50, min_samples_leaf=15)
modelfit(alg3_3, X_fdTrain, X_fdTest, predictors, target, IDcol, 'C:/Users/mritunjay/dataScience/Python/AVHACK/train_GzS76OK/alg3_1_week_50.csv')
coef3_3 = pd.Series(alg3_3.feature_importances_, predictors).sort_values(ascending=False)
coef3_3.plot(kind='bar', title='Feature Importances',figsize=(20,10), linewidth=5, fontsize=20)

#Model Report
#RMSE : 176.6
#CV Score : Mean - 247.1 | Std - 144.1 | Min - 47.73 | Max - 546.3

## multicollenesar remove
HighlyCOrrItem = ['region_code_0','region_code_2','region_code_4','region_code_7']
from sklearn.tree import DecisionTreeRegressor
predictors = [x for x in X_fdTrain.columns if x not in [target]+IDcol+HighlyCOrrItem]
alg3_4 = DecisionTreeRegressor(max_depth=50, min_samples_leaf=15)
modelfit(alg3_4, X_fdTrain, X_fdTest, predictors, target, IDcol, 'C:/Users/mritunjay/dataScience/Python/AVHACK/train_GzS76OK/alg3_1_week_50_1.csv')
coef3_4 = pd.Series(alg3_4.feature_importances_, predictors).sort_values(ascending=False)
coef3_4.plot(kind='bar', title='Feature Importances',figsize=(20,10), linewidth=5, fontsize=20)

#Model Report
#RMSE : 176.6
#CV Score : Mean - 247 | Std - 143.6 | Min - 47.74 | Max - 546.3


## multicollenesar remove
HighlyCOrrItem = ['region_code_0','region_code_2','region_code_4','region_code_7','city_code_0','city_code_1','city_code_3','city_code_7','city_code_22','city_code_28','city_code_33','city_code_44','week_14','week_37','week_42','week_43','week_44','week_45','week_31','week_24','week_25','week_26','week_8','week_22','week_23','week_38','week_51']
from sklearn.tree import DecisionTreeRegressor
predictors = [x for x in X_fdTrain.columns if x not in [target]+IDcol+HighlyCOrrItem]
alg3_5 = DecisionTreeRegressor(max_depth=50, min_samples_leaf=15)
modelfit(alg3_5, X_fdTrain, X_fdTest, predictors, target, IDcol, 'C:/Users/mritunjay/dataScience/Python/AVHACK/train_GzS76OK/alg3_1_week_50_2.csv')
coef3_5 = pd.Series(alg3_5.feature_importances_, predictors).sort_values(ascending=False)
coef3_5.plot(kind='bar', title='Feature Importances',figsize=(20,10), linewidth=5, fontsize=20)

#Model Report
#RMSE : 178
#CV Score : Mean - 248.2 | Std - 143.4 | Min - 47.79 | Max - 546.7


from sklearn.ensemble import RandomForestRegressor
HighlyCOrrItem = ['region_code_0','region_code_2','region_code_4','region_code_7','city_code_0','city_code_1','city_code_3','city_code_7','city_code_18','city_code_22','city_code_28','city_code_33','city_code_44','week_13','week_14','week_37','week_42','week_43','week_44','week_45','week_31','week_24','week_25','week_26','week_8','week_22','week_23','week_38','week_51','week_35','week_39']
predictors = [x for x in X_fdTrain.columns if x not in [target]+IDcol+HighlyCOrrItem]
alg4_1 = RandomForestRegressor(n_estimators=200,max_depth=70, min_samples_leaf=15,n_jobs=4)
modelfit(alg4_1, X_fdTrain, X_fdTest, predictors, target, IDcol, 'C:/Users/mritunjay/dataScience/Python/AVHACK/train_GzS76OK/alg4_RF_50.csv')
coef4_1 = pd.Series(alg4_1.feature_importances_, predictors).sort_values(ascending=False)
coef4_1.plot(kind='bar', title='Feature Importances',figsize=(20,10), linewidth=5, fontsize=20)

#Model Report
#RMSE : 177.5
#CV Score : Mean - 237.1 | Std - 139.5 | Min - 45.83 | Max - 545.1
