import pandas as pd
import numpy as np                     # For mathematical calculations
import seaborn as sns                  # For data visualization
import matplotlib.pyplot as plt        # For plotting graphs
%matplotlib inline
import warnings                        # To ignore any warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
test = pd.read_csv("test_Y3wMUE5_7gLdaTN.csv")

train_original = train.copy()
test_original = test.copy()

##  Understanding the Data

train.columns
test.columns

train.info()
train.dtypes
# =============================================================================
# 
# ##############################  UNIVARIATE ANALYSIS  #######################
# =============================================================================

#Target Variable
#We will first look at the target variable, i.e., Loan_Status. As it is a categorical variable, let us look at its frequency table, percentage distribution and bar plot.
#
#Frequency table of a variable will give us the count of each category in that variable.

train['Loan_Status'].value_counts()
# Normalize can be set to True to print proportions instead of number 
train['Loan_Status'].value_counts(normalize=True)

train['Loan_Status'].value_counts().plot.bar()

#Independent Variable (Categorical)

plt.figure(1)
plt.subplot(221)
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender')
plt.subplot(222)
train['Married'].value_counts(normalize=True).plot.bar(title= 'Married')
plt.subplot(223)
train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed')
plt.subplot(224)
train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History')
plt.show()

# =============================================================================
# #It can be inferred from the above bar plots that:
# #
# #80% applicants in the dataset are male.
# #Around 65% of the applicants in the dataset are married.
# #Around 15% applicants in the dataset are self employed.
# #Around 85% applicants have repaid their debts.
# =============================================================================

#Independent Variable (Ordinal)


plt.figure(1)
plt.subplot(131)
train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6), title= 'Dependents')
plt.subplot(132)
train['Education'].value_counts(normalize=True).plot.bar(title= 'Education')
plt.subplot(133)
train['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area')
plt.show()

#Independent Variable (Numerical)

#### Applicant income
plt.figure(1)
plt.subplot(121)
sns.distplot(train['ApplicantIncome']);
plt.subplot(122)
train['ApplicantIncome'].plot.box(figsize=(16,5))
plt.show()

#It can be inferred that most of the data in the distribution of applicant income is towards 
#left which means it is not normally distributed. We will try to make it normal in later sections 
#as algorithms works better if the data is normally distributed.
#
#The boxplot confirms the presence of a lot of outliers/extreme values. 
#This can be attributed to the income disparity in the society. Part of this can be driven by the fact that 
#we are looking at people with different education levels. Let us segregate them by Education:

train.boxplot(column='ApplicantIncome', by = 'Education')
plt.suptitle("")

# We can see that there are a higher number of graduates with very high incomes, 
#which are appearing to be the outliers.

# Let’s look at the Coapplicant income distribution.

plt.figure(1)
plt.subplot(121)
sns.distplot(train['CoapplicantIncome']);
plt.subplot(122)
train['CoapplicantIncome'].plot.box(figsize=(16,5))
plt.show()
#
#We see a similar distribution as that of the applicant income. Majority of coapplicant’s income ranges from 0 to 5000. 
#We also see a lot of outliers in the coapplicant income and it is not normally distributed.

# Let’s look at the distribution of LoanAmount variable.

plt.figure(1)
plt.subplot(121)
df=train.dropna()
sns.distplot(df['LoanAmount']);
plt.subplot(122)
train['LoanAmount'].plot.box(figsize=(16,5))
plt.show()

# =============================================================================
# We see a lot of outliers in this variable and the distribution is fairly normal. 
# We will treat the outliers in later sections.
# 
# =============================================================================
#Now we would like to know how well each feature correlate with Loan Status. 
#So, in the next section we will look at bivariate analysis.

# =============================================================================
# ##################################   BIVARIATE ANALYSIS    ########################
# =============================================================================



# =============================================================================
#    some of the hypotheses that we generated earlier:
# 
# Applicants with high income should have more chances of loan approval.
# Applicants who have repaid their previous debts should have higher chances of loan approval.
# Loan approval should also depend on the loan amount. If the loan amount is less, chances of loan approval should be high.
# Lesser the amount to be paid monthly to repay the loan, higher the chances of loan approval.
# =============================================================================

 # Lets try to test the above mentioned hypotheses using bivariate analysis
## Gender vs loan_status
Gender=pd.crosstab(train['Gender'],train['Loan_Status'])
Gender.plot(kind="bar", stacked=True, figsize=(4,4))
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

#It can be inferred that the proportion of male and female applicants is more or less 
#same for both approved and unapproved loans.

# plot remaining categorical var vs loan status

Married=pd.crosstab(train['Married'],train['Loan_Status'])
Dependents=pd.crosstab(train['Dependents'],train['Loan_Status'])
Education=pd.crosstab(train['Education'],train['Loan_Status'])
Self_Employed=pd.crosstab(train['Self_Employed'],train['Loan_Status'])

Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show()

Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.show()

Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show()

Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show()

#Proportion of married applicants is higher for the approved loans.
#Distribution of applicants with 1 or 3+ dependents is similar across both the categories of Loan_Status.
#There is nothing significant we can infer from Self_Employed vs Loan_Status plot.

Credit_History=pd.crosstab(train['Credit_History'],train['Loan_Status'])
Property_Area=pd.crosstab(train['Property_Area'],train['Loan_Status'])

Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show()

Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.show()

#It seems people with credit history as 1 are more likely to get their loans approved.
#Proportion of loans getting approved in semiurban area is higher as compared to that in rural or urban areas.

# =============================================================================
# #################  Numerical Independent Variable vs Target Variable  ###########
# =============================================================================


#We will try to find the mean income of people for which the loan has been approved vs the mean income 
#of people for which the loan has not been approved.

train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()

#mean income is not having any impact
#lets check in intervals

bins=[0,2500,4000,6000,81000]
group=['Low','Average','High', 'Very high']
train['Income_bin']=pd.cut(df['ApplicantIncome'],bins,labels=group)

Income_bin=pd.crosstab(train['Income_bin'],train['Loan_Status'])
Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('ApplicantIncome')
P = plt.ylabel('Percentage')

#It can be inferred that Applicant income does not affect the chances of loan approval which contradicts our hypothesis 
#in which we assumed that if the applicant income is high the chances of loan approval will also be high.


#check coapplicant income

bins=[0,1000,3000,42000]
group=['Low','Average','High']
train['Coapplicant_Income_bin']=pd.cut(df['CoapplicantIncome'],bins,labels=group)

Coapplicant_Income_bin=pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status'])
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('CoapplicantIncome')
plt.ylabel('Percentage')

#It shows that if coapplicant’s income is less the chances of loan approval are high. 
#But this does not look right. The possible reason behind this may be that most of 
#the applicants don’t have any coapplicant so the coapplicant income for such 
#applicants is 0 and hence the loan approval is not dependent on it. 
#So we can make a new variable in which we will combine the applicant’s and coapplicant’s income 
#to visualize the combined effect of income on loan approval.

#Let us combine the Applicant Income and Coapplicant Income and see the combined effect of 
#Total Income on the Loan_Status.

train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']
bins=[0,2500,4000,6000,81000]
group=['Low','Average','High', 'Very high']
train['Total_Income_bin']=pd.cut(train['Total_Income'],bins,labels=group)
Total_Income_bin=pd.crosstab(train['Total_Income_bin'],train['Loan_Status'])
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Total_Income')
P = plt.ylabel('Percentage')

#We can see that Proportion of loans getting approved for applicants having low Total_Income is very less 
#as compared to that of applicants with Average, High and Very High Income.

#Let’s visualize the Loan amount variable.

bins=[0,100,200,700]
group=['Low','Average','High']
train['LoanAmount_bin']=pd.cut(df['LoanAmount'],bins,labels=group)
LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status'])
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('LoanAmount')
P = plt.ylabel('Percentage')

#It can be seen that the proportion of approved loans is higher for Low and Average Loan Amount 
#as compared to that of High Loan Amount which supports our hypothesis in which we considered 
#that the chances of loan approval will be high when the loan amount is less.

train=train.drop(['Income_bin', 'Coapplicant_Income_bin', 'LoanAmount_bin', 'Total_Income_bin', 'Total_Income'], axis=1)

train['Dependents'].replace('3+', 3,inplace=True)
test['Dependents'].replace('3+', 3,inplace=True)
train['Loan_Status'].replace('N', 0,inplace=True)
train['Loan_Status'].replace('Y', 1,inplace=True)

matrix = train.corr()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");

#We see that the most correlated variables are (ApplicantIncome - LoanAmount) 
#and (Credit_History - Loan_Status). 
#LoanAmount is also correlated with CoapplicantIncome.
