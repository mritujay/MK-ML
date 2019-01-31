import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras as ke
from sklearn.metrics import accuracy_score
import os


root_dir = os.path.abspath("H:/DS/Python/ANN/LoanPrediction/")

trainFile = os.path.join(root_dir,'train.csv')
testFile = os.path.join(root_dir,'test.csv')


train = pd.read_csv(trainFile)
test = pd.read_csv(testFile)

train.info()

## many missing values avialable in dataset

# univariate analysys 

ls = train.Loan_Status.value_counts(normalize=True)
ls.plot.bar()


plt.figure(1)
plt.subplot(221)
train.Gender.value_counts(normalize=True).plot.bar(figsize=(15,10),title='Gender')
plt.subplot(222)
train.Married.value_counts(normalize=True).plot.bar(title='Married')
plt.subplot(223)
train.Education.value_counts(normalize=True).plot.bar(title='Education')
plt.subplot(224)
train.Self_Employed.value_counts(normalize=True).plot.bar(title='Self_Employed')
plt.show()

plt.figure(2)
plt.subplot(131)
train.Dependents.value_counts(normalize=True).plot.bar(figsize=(15,10),title='Dependents')
plt.subplot(132)
train.Property_Area.value_counts(normalize=True).plot.bar(title='Property_Area')
plt.show()


# Numerical variables

plt.figure(3)
plt.subplot(121)
train.ApplicantIncome.plot(kind='kde')
plt.subplot(122)
train.ApplicantIncome.plot.box(figsize=(16,5))
plt.show()

## Lots of outliers available and also its not normally distributed 
# maybe lot of ouliers are avaialble because of diffrent educational level of applicant

train.boxplot(column='ApplicantIncome', by = 'Education')
plt.suptitle("")

## large number of gratduate with higher income available in data

#3 Checking coapplicant income distribution

plt.figure(3)
plt.subplot(121)
train.CoapplicantIncome.plot(kind='kde')
plt.subplot(122)
train.CoapplicantIncome.plot.box(figsize=(16,5))
plt.show()

# here also we find same pattern as applicantIncome

# CHecking for loan AMount


plt.figure(3)
plt.subplot(121)
train.LoanAmount.plot(kind='kde')
plt.subplot(122)
train.LoanAmount.plot.box(figsize=(16,5))
plt.show()

# its looking fairly normal in nature however many outliers available


#################  BIVARIATE ANALYISIS ########################

Gender = pd.crosstab(train.Gender,train.Loan_Status)
GenderProportion = Gender.div(Gender.sum(1).astype('float'),axis=0)
GenderProportion.plot(kind='bar',stacked=True,figsize=(4,4))

## thr is no kuch difference between Male or female for whom loan was approved

#Checking for others

Married=pd.crosstab(train['Married'],train['Loan_Status'])
mp = Married.div(Married.sum(1).astype('float'),axis=0)
Dependents=pd.crosstab(train['Dependents'],train['Loan_Status'])
dp = Dependents.div(Dependents.sum(1).astype('float'),axis=0)
Education=pd.crosstab(train['Education'],train['Loan_Status'])
ep = Education.div(Education.sum(1).astype('float'),axis=0)
Self_Employed=pd.crosstab(train['Self_Employed'],train['Loan_Status'])
sp = Self_Employed.div(Self_Employed.sum(1).astype('float'),axis=0)

mp.plot(kind='bar',stacked=True,figsize=(4,4))
plt.show()
dp.plot(kind='bar',stacked=True,figsize=(4,4))
plt.show()
ep.plot(kind='bar',stacked=True,figsize=(4,4))
plt.show()
sp.plot(kind='bar',stacked=True,figsize=(4,4))
plt.show()


#Proportion of married applicants is higher for the approved loans.
#Distribution of applicants with 1 or 3+ dependents is similar across both the categories of Loan_Status.
#There is nothing significant we can infer from Self_Employed vs Loan_Status plot.


DcreditHist=pd.crosstab(train['Credit_History'],train['Loan_Status'])
chp = DcreditHist.div(DcreditHist.sum(1).astype('float'),axis=0)
chp.plot(kind='bar',stacked=True,figsize=(4,4))
plt.show()

#It seems people with credit history as 1 are more likely to get their loans approved.

propertyArea=pd.crosstab(train['Property_Area'],train['Loan_Status'])
pap = propertyArea.div(propertyArea.sum(1).astype('float'),axis=0)
pap.plot(kind='bar',stacked=True,figsize=(4,4))
plt.show()

#Proportion of loans getting approved in semiurban area is higher as compared to that in rural or urban areas.


## Numerical Independent variable vs target 


meanIncomeVsLoanApprovals = train.groupby('Loan_Status')['ApplicantIncome'].mean()
meanIncomeVsLoanApprovals.plot.bar()
meanIncomeVsLoanApprovals.plot(kind='bar',stacked=True,figsize=(4,4))


################  imputing missing values #########

train.isnull().sum()
train.info()

################  For categorical variables
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)

###############  For numerical variables

train['Loan_Amount_Term'].value_counts()
train.groupby('Loan_Amount_Term')['Loan_Amount_Term'].count()

# most of the applicant have taken loan for 360 months that is the mode so will impute with mode

train.Loan_Amount_Term.fillna(train.Loan_Amount_Term.mode()[0],inplace=True)

#Now we will see the LoanAmount variable. As it is a numerical variable, we can use mean or median 
#to impute the missing values. We will use median to fill the null values as earlier we saw 
#that loan amount have outliers so the mean will not be the proper approach as it is highly affected 
#by the presence of outliers.

#==============================================================================
# plt.figure(3)
# plt.subplot(121)
# train.LoanAmount.plot(kind='kde')
# plt.subplot(122)
# train.LoanAmount.plot.box(figsize=(16,5))
# plt.show()
#==============================================================================

train.LoanAmount.fillna(train.LoanAmount.median(),inplace=True)

# Checking all missing values are imputed or not
train.isnull().sum()

## imputing missing values in test set

test['Gender'].fillna(test['Gender'].mode()[0], inplace=True)
test['Married'].fillna(test['Married'].mode()[0], inplace=True)
test['Dependents'].fillna(test['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(test['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(test['Credit_History'].mode()[0], inplace=True)
test.Loan_Amount_Term.fillna(test.Loan_Amount_Term.mode()[0],inplace=True)
test.LoanAmount.fillna(test.LoanAmount.median(),inplace=True)

test.isnull().sum()

## Since LoanAmount has lots of otliers due to which distribution was right skewed . ouliers will change mean and statndard deviation 
#and will highlt affect distribution of data. so we would have to get rid of Outliers present in dataset

plt.figure(3)
plt.subplot(121)
np.log(train.LoanAmount).plot(kind='kde')
plt.subplot(122)
np.log(train.LoanAmount).plot.box(figsize=(16,5))
plt.show()


#  we can see that log od data has changed the distribution back to normal

train['LoanAmount_Log'] = np.log(train['LoanAmount']) #  train.LoanAmount.apply(lambda x: np.log(x))
test['LoanAmount_Log']  = np.log(test['LoanAmount'])


#==============================================================================
# Precision: It is a measure of correctness achieved in true prediction i.e. of observations labeled as true, how many are actually labeled true.
# Precision = TP / (TP + FP)
# 
# Recall(Sensitivity) - It is a measure of actual observations which are predicted correctly i.e. how many observations of true class are labeled correctly. It is also known as ‘Sensitivity’.
# Recall = TP / (TP + FN)
# 
# Specificity - It is a measure of how many observations of false class are labeled correctly.
# Specificity = TN / (TN + FP)
#==============================================================================


###############  model building ###############

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.utils import to_categorical
import keras
from keras.models import Sequential
from keras.layers import Dense

train.info()
train.head(2)

train_XX = np.array(train)

def Y_N_val(df):
    if(df['Loan_Status']=='Y'):
        return 1
    else:
        return 0


train['Loan_Status'] = train.apply(lambda x:Y_N_val(x),axis=1)


from sklearn.preprocessing import LabelEncoder

var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Credit_History','Loan_Amount_Term']

le = LabelEncoder()
for i in var_mod:
    train[i] = le.fit_transform(train[i])
    
le = LabelEncoder()
for i in var_mod:
    test[i] = le.fit_transform(test[i])
    
train = pd.get_dummies(train,columns=var_mod)
test = pd.get_dummies(test,columns=var_mod)

## LabelEcnoder


train_y = train.Loan_Status.values
train_x = train.drop(["Loan_ID","Loan_Status"],axis=1).as_matrix()
train_x = StandardScaler().fit_transform(train_x)



test_x = train.drop(["Loan_ID"],axis=1).as_matrix()
test_x = StandardScaler().fit_transform(train_x)

# Splitting the Training dataset into the Training set and Test set for validation purpose

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size = 0.2)


input_num_units = 31
hidden_num_units = 50
output_num_units = 1


#model = Sequential([
#        Dense(output_dim = 31, init = 'uniform', activation = 'relu', input_dim = 31),
#        #Dense(output_dim = 5, init = 'uniform', activation = 'relu'),
#        Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid', input_dim = 31)
#        ])
    
model = Sequential([
        Dense(output_dim = 31,activation = 'relu', input_dim = 31),
        Dense(output_dim = 20,activation = 'relu',input_dim = 31),
        Dense(output_dim = 1,activation = 'sigmoid', input_dim = 20)
        ])    

    #compile the model with necessary attributes

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# run the Model

trained_model = model.fit(X_train, y_train, nb_epoch=500, batch_size=1, validation_data=(X_test, y_test))
#trained_model = model.fit(X_train, y_train, nb_epoch=500, batch_size=10)

Prediction = model.predict_classes(test_x)

y_pred = model.predict(X_test)
y_pred = (y_pred >0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

confusion_matrix(y_test,np.array(model.predict_classes(X_test)))





