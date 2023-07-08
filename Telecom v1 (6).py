#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import warnings
warnings.filterwarnings('ignore')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option("display.max_columns", 300)
pd.set_option("display.max_rows", 300)


# In[3]:


# read data
churn = pd.read_csv("C:/Assignment/Telecomchurn/Download/train/train.csv")


# In[4]:


churn.shape


# In[5]:


# look at initial rows of the data
churn.head(10)


# In[6]:


# feature type summary
churn.info(verbose=1)


# There are 99999 rows and 226 columns in the data. Lot of the columns are numeric type, but we need to inspect which are the categorical columns.

# In[7]:


# look at data statistics
churn.describe(include='all')


# In[8]:


import sweetviz as sv
sweet_report = sv.analyze(churn,pairwise_analysis="off")
sweet_report.show_html('sweet_report.html')


# In[9]:


# create backup of data
original = churn.copy()


# In[10]:


# create column name list by types of columns, 'id' is considered as mobile number which is unique
id_cols = ['id','circle_id']

date_cols = ['last_date_of_month_6',
             'last_date_of_month_7',
             'last_date_of_month_8',
             'date_of_last_rech_6',
             'date_of_last_rech_7',
             'date_of_last_rech_8',
             'date_of_last_rech_data_6',
             'date_of_last_rech_data_7',
             'date_of_last_rech_data_8',
            ]

cat_cols =  ['night_pck_user_6',
             'night_pck_user_7',
             'night_pck_user_8',
             'fb_user_6',
             'fb_user_7',
             'fb_user_8',
            ]

num_cols = [column for column in churn.columns if column not in id_cols + date_cols + cat_cols]

# print the number of columns in each list
print("#ID cols: %d\n#Date cols:%d\n#Numeric cols:%d\n#Category cols:%d" % (len(id_cols), len(date_cols), len(num_cols), len(cat_cols)))

# check if we have missed any column or not
print(len(id_cols) + len(date_cols) + len(num_cols) + len(cat_cols) == churn.shape[1])


# # Handling missing values

# In[11]:


# look at missing value ratio in each column
churn.isnull().sum()*100/churn.shape[0]


# # impute missing values

# ## i) Imputing with zeroes

# In[12]:


# some recharge columns have minimum value of 1 while some don't
recharge_cols = ['total_rech_data_6', 'total_rech_data_7', 'total_rech_data_8',
                 'count_rech_2g_6', 'count_rech_2g_7', 'count_rech_2g_8',
                 'count_rech_3g_6', 'count_rech_3g_7', 'count_rech_3g_8',
                 'max_rech_data_6', 'max_rech_data_7', 'max_rech_data_8',
                 'av_rech_amt_data_6', 'av_rech_amt_data_7', 'av_rech_amt_data_8',
                 ]

churn[recharge_cols].describe(include='all')


# In[13]:


# It is also observed that the recharge date and the recharge value are missing together which means the customer didn't recharge
churn.loc[churn.total_rech_data_6.isnull() & churn.date_of_last_rech_data_6.isnull(), ["total_rech_data_6", "date_of_last_rech_data_6"]].head(20)


# In the recharge variables where minumum value is 1, we can impute missing values with zeroes since it means customer did not recharge that month

# In[14]:


# create a list of recharge columns where we will impute missing values with zeroes
zero_impute = ['total_rech_data_6', 'total_rech_data_7', 'total_rech_data_8',
        'av_rech_amt_data_6', 'av_rech_amt_data_7', 'av_rech_amt_data_8',
        'max_rech_data_6', 'max_rech_data_7', 'max_rech_data_8',
       ]


# In[15]:


# impute missing values with 0
churn[zero_impute] = churn[zero_impute].apply(lambda x: x.fillna(0))


# In[16]:


# Validate missing values
print("Missing value ratio:\n")
print(churn[zero_impute].isnull().sum()*100/churn.shape[1])

# Print summary
print("\n\nSummary statistics\n")
print(churn[zero_impute].describe(include='all'))


# In[17]:


# drop id and date columns
print("Shape before dropping: ", churn.shape)
churn = churn.drop(id_cols + date_cols, axis=1)
print("Shape after dropping: ", churn.shape)


# ## ii) Replace NaN values in categorical variables

# We will replace missing values in the categorical values with '-1' where '-1' will be a new category.

# In[18]:


# replace missing values with '-1' in categorical columns
churn[cat_cols] = churn[cat_cols].apply(lambda x: x.fillna(-1))


# In[19]:


# missing value ratio
print("Missing value ratio:\n")
print(churn[cat_cols].isnull().sum()*100/churn.shape[0])


# ## iii) Drop variables with more than a given threshold of missing values

# In[20]:


initial_cols = churn.shape[1]

MISSING_THRESHOLD = 0.7

include_cols = list(churn.apply(lambda column: True if column.isnull().sum()/churn.shape[0] < MISSING_THRESHOLD else False))

drop_missing = pd.DataFrame({'features':churn.columns , 'include': include_cols})
drop_missing.loc[drop_missing.include == True,:]


# In[21]:


# drop columns
churn = churn.loc[:, include_cols]

dropped_cols = churn.shape[1] - initial_cols
print("{0} columns dropped.".format(dropped_cols))


# ## iv) imputing using MICE

# install fancyimpute package using [this](https://github.com/iskandr/fancyimpute) link and following the install instructions

# In[22]:


import cvxpy
from cvxopt import glpk

# Create optimization variables and problem
x = cvxpy.Variable(2)
objective = cvxpy.Minimize(cvxpy.sum(x))
constraints = [x >= 0, x <= 1]
problem = cvxpy.Problem(objective, constraints)

# Set GLPK solver explicitly
problem.solve(solver=cvxpy.GLPK)

# Print the optimal value and solution
print("Optimal value:", problem.value)
print("Optimal solution:", x.value)


# In[23]:


churn_cols = churn.columns
# using MICE technique to impute missing values in the rest of the columns
from fancyimpute import IterativeImputer as MICE
churn_imputed = MICE().fit_transform(churn)


# In[24]:


# convert imputed numpy array to pandas dataframe
churn = pd.DataFrame(churn_imputed, columns=churn_cols)
print(churn.isnull().sum()*100/churn.shape[0])


# # filter high-value customers

# ### calculate total data recharge amount

# In[25]:


# calculate the total data recharge amount for June and July --> number of recharges * average recharge amount
churn['total_data_rech_6'] = churn.total_rech_data_6 * churn.av_rech_amt_data_6
churn['total_data_rech_7'] = churn.total_rech_data_7 * churn.av_rech_amt_data_7


# ### add total data recharge and total recharge to get total combined recharge amount for a month

# In[26]:


# calculate total recharge amount for June and July --> call recharge amount + data recharge amount
churn['amt_data_6'] = churn.total_rech_amt_6 + churn.total_data_rech_6
churn['amt_data_7'] = churn.total_rech_amt_7 + churn.total_data_rech_7


# In[27]:


# calculate average recharge done by customer in June and July
churn['av_amt_data_6_7'] = (churn.amt_data_6 + churn.amt_data_7)/2


# In[28]:


# look at the 70th percentile recharge amount
print("Recharge amount at 70th percentile: {0}".format(churn.av_amt_data_6_7.quantile(0.7)))


# In[29]:


# retain only those customers who have recharged their mobiles with more than or equal to 70th percentile amount
churn_filtered = churn.loc[churn.av_amt_data_6_7 >= churn.av_amt_data_6_7.quantile(0.7), :]
churn_filtered = churn_filtered.reset_index(drop=True)
churn_filtered.shape


# In[30]:


# delete variables created to filter high-value customers
churn_filtered = churn_filtered.drop(['total_data_rech_6', 'total_data_rech_7',
                                      'amt_data_6', 'amt_data_7', 'av_amt_data_6_7'], axis=1)
churn_filtered.shape


# We're left with 30,001 rows after selecting the customers who have provided recharge value of more than or equal to the recharge value of the 70th percentile customer.

# # derive churn

# In[31]:


churn_filtered.head()


# In[32]:


churn_filtered['total_calls_mou_6'] = churn_filtered.total_ic_mou_6 + churn_filtered.total_og_mou_6
churn_filtered['total_calls_mou_7'] = churn_filtered.total_ic_mou_7 + churn_filtered.total_og_mou_7
churn_filtered['total_calls_mou_8'] = churn_filtered.total_ic_mou_8 + churn_filtered.total_og_mou_8


# In[33]:


# calculate total incoming and outgoing minutes of usage
churn_filtered['total_calls_mou_8'] = churn_filtered.total_ic_mou_8 + churn_filtered.total_og_mou_8


# In[34]:


# calculate 2g and 3g data consumption
churn_filtered['total_internet_mb_8'] =  churn_filtered.vol_2g_mb_8 + churn_filtered.vol_3g_mb_8


# In[35]:


# create churn variable: those who have not used either calls or internet in the month of September are customers who have churned

# 0 - not churn, 1 - churn
churn_filtered['churn'] = churn_filtered.apply(lambda row: 1 if (row.total_calls_mou_8 == 0 and row.total_internet_mb_8 == 0) else 0, axis=1)


# In[36]:


# delete derived variables
churn_filtered = churn_filtered.drop(['total_calls_mou_8', 'total_internet_mb_8'], axis=1)


# In[37]:


# change data type to category
churn_filtered.churn = churn_filtered.churn.astype("category")

# print churn ratio
print("Churn Ratio:")
print(churn_filtered.churn.value_counts()*100/churn_filtered.shape[0])


# # Calculate difference between 8th and previous months

# Let's derive some variables. The most important feature, in this situation, can be the difference between the 8th month and the previous months. The difference can be in patterns such as usage difference or recharge value difference. Let's calculate difference variable as the difference between 8th month and the average of 6th and 7th month.

# In[38]:


churn_filtered['arpu_diff'] = churn_filtered.arpu_8 - ((churn_filtered.arpu_6 + churn_filtered.arpu_7)/2)

churn_filtered['onnet_mou_diff'] = churn_filtered.onnet_mou_8 - ((churn_filtered.onnet_mou_6 + churn_filtered.onnet_mou_7)/2)

churn_filtered['offnet_mou_diff'] = churn_filtered.offnet_mou_8 - ((churn_filtered.offnet_mou_6 + churn_filtered.offnet_mou_7)/2)

churn_filtered['roam_ic_mou_diff'] = churn_filtered.roam_ic_mou_8 - ((churn_filtered.roam_ic_mou_6 + churn_filtered.roam_ic_mou_7)/2)

churn_filtered['roam_og_mou_diff'] = churn_filtered.roam_og_mou_8 - ((churn_filtered.roam_og_mou_6 + churn_filtered.roam_og_mou_7)/2)

churn_filtered['loc_og_mou_diff'] = churn_filtered.loc_og_mou_8 - ((churn_filtered.loc_og_mou_6 + churn_filtered.loc_og_mou_7)/2)

churn_filtered['std_og_mou_diff'] = churn_filtered.std_og_mou_8 - ((churn_filtered.std_og_mou_6 + churn_filtered.std_og_mou_7)/2)

churn_filtered['isd_og_mou_diff'] = churn_filtered.isd_og_mou_8 - ((churn_filtered.isd_og_mou_6 + churn_filtered.isd_og_mou_7)/2)

churn_filtered['spl_og_mou_diff'] = churn_filtered.spl_og_mou_8 - ((churn_filtered.spl_og_mou_6 + churn_filtered.spl_og_mou_7)/2)

churn_filtered['total_og_mou_diff'] = churn_filtered.total_og_mou_8 - ((churn_filtered.total_og_mou_6 + churn_filtered.total_og_mou_7)/2)

churn_filtered['loc_ic_mou_diff'] = churn_filtered.loc_ic_mou_8 - ((churn_filtered.loc_ic_mou_6 + churn_filtered.loc_ic_mou_7)/2)

churn_filtered['std_ic_mou_diff'] = churn_filtered.std_ic_mou_8 - ((churn_filtered.std_ic_mou_6 + churn_filtered.std_ic_mou_7)/2)

churn_filtered['isd_ic_mou_diff'] = churn_filtered.isd_ic_mou_8 - ((churn_filtered.isd_ic_mou_6 + churn_filtered.isd_ic_mou_7)/2)

churn_filtered['spl_ic_mou_diff'] = churn_filtered.spl_ic_mou_8 - ((churn_filtered.spl_ic_mou_6 + churn_filtered.spl_ic_mou_7)/2)

churn_filtered['total_ic_mou_diff'] = churn_filtered.total_ic_mou_8 - ((churn_filtered.total_ic_mou_6 + churn_filtered.total_ic_mou_7)/2)

churn_filtered['total_rech_num_diff'] = churn_filtered.total_rech_num_8 - ((churn_filtered.total_rech_num_6 + churn_filtered.total_rech_num_7)/2)

churn_filtered['total_rech_amt_diff'] = churn_filtered.total_rech_amt_8 - ((churn_filtered.total_rech_amt_6 + churn_filtered.total_rech_amt_7)/2)

churn_filtered['max_rech_amt_diff'] = churn_filtered.max_rech_amt_8 - ((churn_filtered.max_rech_amt_6 + churn_filtered.max_rech_amt_7)/2)

churn_filtered['total_rech_data_diff'] = churn_filtered.total_rech_data_8 - ((churn_filtered.total_rech_data_6 + churn_filtered.total_rech_data_7)/2)

churn_filtered['max_rech_data_diff'] = churn_filtered.max_rech_data_8 - ((churn_filtered.max_rech_data_6 + churn_filtered.max_rech_data_7)/2)

churn_filtered['av_rech_amt_data_diff'] = churn_filtered.av_rech_amt_data_8 - ((churn_filtered.av_rech_amt_data_6 + churn_filtered.av_rech_amt_data_7)/2)

churn_filtered['vol_2g_mb_diff'] = churn_filtered.vol_2g_mb_8 - ((churn_filtered.vol_2g_mb_6 + churn_filtered.vol_2g_mb_7)/2)

churn_filtered['vol_3g_mb_diff'] = churn_filtered.vol_3g_mb_8 - ((churn_filtered.vol_3g_mb_6 + churn_filtered.vol_3g_mb_7)/2)


# In[39]:


# let's look at summary of one of the difference variables
churn_filtered['total_og_mou_diff'].describe()


# ## delete columns that belong to the churn month (8th month)

# In[40]:


# delete all variables relating to 8th month
churn_filtered = churn_filtered.filter(regex='[^8]$', axis=1)
churn_filtered.shape


# In[41]:


# extract all names that end with 8
col_8_names = churn.filter(regex='8$', axis=1).columns

# update num_cols and cat_cols column name list
cat_cols = [col for col in cat_cols if col not in col_8_names]
cat_cols.append('churn')
num_cols = [col for col in churn_filtered.columns if col not in cat_cols]


# ## visualise data

# In[42]:


# change columns types
churn_filtered[num_cols] = churn_filtered[num_cols].apply(pd.to_numeric)
churn_filtered[cat_cols] = churn_filtered[cat_cols].apply(lambda column: column.astype("category"), axis=0)


# In[43]:


# create plotting functions
def data_type(variable):
    if variable.dtype == np.int64 or variable.dtype == np.float64:
        return 'numerical'
    elif variable.dtype == 'category':
        return 'categorical'

def univariate(variable, stats=True):

    if data_type(variable) == 'numerical':
        sns.distplot(variable)
        if stats == True:
            print(variable.describe())

    elif data_type(variable) == 'categorical':
        sns.countplot(variable)
        if stats == True:
            print(variable.value_counts())

    else:
        print("Invalid variable passed: either pass a numeric variable or a categorical vairable.")

def bivariate(var1, var2):
    if data_type(var1) == 'numerical' and data_type(var2) == 'numerical':
        sns.regplot(var1, var2)
    elif (data_type(var1) == 'categorical' and data_type(var2) == 'numerical') or (data_type(var1) == 'numerical' and data_type(var2) == 'categorical'):
        sns.boxplot(var1, var2)


# ## Univariate EDA

# In[44]:


univariate(churn.arpu_6)


# In[45]:


univariate(churn.loc_og_t2o_mou)


# In[46]:


univariate(churn.std_og_t2o_mou)


# In[47]:


univariate(churn.onnet_mou_8)


# In[48]:


univariate(churn.offnet_mou_8)


# Variables are very **skewed** towards the left.

# ## Bivariate EDA

# In[49]:


bivariate(churn_filtered.churn, churn_filtered.aon)


# In[50]:


bivariate(churn_filtered.aug_vbc_3g, churn_filtered.churn)


# In[51]:


bivariate(churn_filtered.spl_og_mou_7, churn_filtered.churn)


# In[52]:


pd.crosstab(churn_filtered.churn, churn_filtered.night_pck_user_7, normalize='columns')*100


# In[53]:


pd.crosstab(churn_filtered.churn, churn_filtered.sachet_3g_7)


# ### Cap outliers in all numeric variables with k-sigma technique

# In[54]:


def cap_outliers(array, k=3):
    upper_limit = array.mean() + k*array.std()
    lower_limit = array.mean() - k*array.std()
    array[array<lower_limit] = lower_limit
    array[array>upper_limit] = upper_limit
    return array


# In[55]:


# example of capping
sample_array = list(range(100))

# add outliers to the data
sample_array[0] = -9999
sample_array[99] = 9999

# cap outliers
sample_array = np.array(sample_array)
print("Array after capping outliers: \n", cap_outliers(sample_array, k=2))


# In[56]:


# cap outliers in the numeric columns
churn_filtered[num_cols] = churn_filtered[num_cols].apply(cap_outliers, axis=0)


# # Modelling

# ## i) Making predictions

# In[57]:


# import required libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC


# In[58]:


# from imblearn.metrics import sensitivity_specificity_support use skleran


# In[59]:


from sklearn.metrics import confusion_matrix

def sensitivity_specificity_support(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    support = tp + fn
    return sensitivity, specificity, support

# Example usage
y_true = [0, 1, 1, 0]
y_pred = [0, 0, 1, 1]
sensitivity, specificity, support = sensitivity_specificity_support(y_true, y_pred)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("Support:", support)


# In[60]:


# Separate the churned and non-churned customers
churned = churn_filtered[churn_filtered['churn'] == 1]
not_churned = churn_filtered[churn_filtered['churn'] == 0]

# Undersample the majority class (not churned) to balance the dataset
undersampled_not_churned = not_churned.sample(n=len(churned), random_state=42)

# Combine the undersampled not_churned and churned datasets
balanced_data = pd.concat([undersampled_not_churned, churned])

# Shuffle the rows of the balanced dataset
balanced_data = balanced_data.sample(frac=1, random_state=42)

# Split the balanced dataset into features and target
X = balanced_data.drop('churn', axis=1)  # Features
y = balanced_data['churn']  # Target


# ## Preprocessing data

# In[61]:


# change churn to numeric
churn_filtered['churn'] = pd.to_numeric(churn_filtered['churn'])


# ### Train Test split

# In[62]:


# divide data into train and test
X = churn_filtered.drop("churn", axis = 1)
y = churn_filtered.churn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 4, stratify = y)


# In[63]:


# print shapes of train and test sets
X_train.shape
y_train.shape
X_test.shape
y_test.shape


# ## Aggregating the categorical columns

# In[64]:


train = pd.concat([X_train, y_train], axis=1)

# aggregate the categorical variables
train.groupby('night_pck_user_6').churn.mean()
train.groupby('night_pck_user_7').churn.mean()

train.groupby('fb_user_6').churn.mean()
train.groupby('fb_user_7').churn.mean()


# In[65]:


# replace categories with aggregated values in each categorical column
mapping = {'night_pck_user_6' : {-1: 0.099165, 0: 0.066797, 1: 0.087838},
           'night_pck_user_7' : {-1: 0.115746, 0: 0.055494, 1: 0.051282},
           'fb_user_6'        : {-1: 0.099165, 0: 0.069460, 1: 0.067124},
           'fb_user_7'        : {-1: 0.115746, 0: 0.059305, 1: 0.055082},
          }     
X_train.replace(mapping, inplace = True)
X_test.replace(mapping, inplace = True)


# In[66]:


# check data type of categorical columns - make sure they are numeric
X_train[[col for col in cat_cols if col not in ['churn']]].info()


# ## PCA

# In[67]:


# apply pca to train data
pca = Pipeline([('scaler', StandardScaler()), ('pca', PCA())])


# In[68]:


pca.fit(X_train)
churn_pca = pca.fit_transform(X_train)


# In[69]:


# extract pca model from pipeline
pca = pca.named_steps['pca']

# look at explainded variance of PCA components
print(pd.Series(np.round(pca.explained_variance_ratio_.cumsum(), 4)*100))


# ~ 60 components explain 90% variance
# 
# ~ 80 components explain 95% variance

# In[70]:


# plot feature variance
features = range(pca.n_components_)
cumulative_variance = np.round(np.cumsum(pca.explained_variance_ratio_)*100, decimals=4)
plt.figure(figsize=(175/20,100/20)) # 100 elements on y-axis; 175 elements on x-axis; 20 is normalising factor
plt.plot(cumulative_variance)


# ## PCA and Logistic Regression

# In[71]:


# create pipeline
PCA_VARS = 60
steps = [('scaler', StandardScaler()),
         ("pca", PCA(n_components=PCA_VARS)),
         ("logistic", LogisticRegression(class_weight='balanced'))
        ]
pipeline = Pipeline(steps)


# In[72]:


# fit model
pipeline.fit(X_train, y_train)

# check score on train data
pipeline.score(X_train, y_train)


# ### Evaluate on test data

# In[73]:


print(X_train.columns.to_list())


# In[74]:


from sklearn.ensemble import RandomForestClassifier

# Create a Random Forest classifier
rf_model = RandomForestClassifier()

# Fit the model on the training data
rf_model.fit(X_train, y_train)

# Get feature importances
importances = rf_model.feature_importances_

# Create a dataframe to store feature importance values
feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})

# Sort the dataframe by feature importance in descending order
feature_importance = feature_importance.sort_values('Importance', ascending=False)
N = 10  # Define the number of top features to select
# Print the top N features
top_features = feature_importance['Feature'][:N].tolist()
print(top_features)


# In[75]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Extract the top selected features
top_features = ['isd_og_mou_diff', 'spl_ic_mou_diff', 'churn_probability', 'max_rech_amt_diff', 
                'total_rech_amt_diff', 'arpu_diff', 'roam_og_mou_7', 'roam_og_mou_diff', 'roam_ic_mou_diff', 'onnet_mou_diff']

# Select top features from X_train
X_train_selected = X_train[top_features]

# Scale the features
scaler = StandardScaler()
X_train_selected = scaler.fit_transform(X_train_selected)

# Fit logistic regression model with selected features
logistic_model = LogisticRegression(class_weight={0: 0.1, 1: 0.9})
logistic_model.fit(X_train_selected, y_train)

# Select top features from X_test
X_test_selected = X_test[top_features]

# Scale the features
X_test_selected = scaler.transform(X_test_selected)

# Predict churn on test data
y_pred = logistic_model.predict(X_test_selected)

# Get classification report
report = classification_report(y_test, y_pred, output_dict=True)

# Extract sensitivity and specificity
sensitivity = report['1']['recall']
specificity = report['0']['recall']

print("Sensitivity:", sensitivity)
print("Specificity:", specificity)


# In[76]:


from sklearn.metrics import classification_report

# predict churn on test data
y_pred = pipeline.predict(X_test)

# create classification report
report = classification_report(y_test, y_pred)
print(report)


# In[77]:


from sklearn.metrics import confusion_matrix

# Predict churn on test data
y_pred = pipeline.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Calculate sensitivity and specificity
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

# Calculate area under the ROC curve
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_prob)
print("AUC:", auc)


# In[78]:


from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# predict churn on test data
y_pred = pipeline.predict(X_test)

# create confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# calculate other performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:   ", round(accuracy, 2))
print("Precision:  ", round(precision, 2))
print("Recall:     ", round(recall, 2))
print("F1-Score:   ", round(f1, 2))

# check area under curve
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
print("AUC:    \t", round(roc_auc_score(y_test, y_pred_prob), 2))


# ### Hyperparameter tuning - PCA and Logistic Regression

# In[79]:


# class imbalance
y_train.value_counts()/y_train.shape


# In[80]:


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
# PCA
pca = PCA()

# logistic regression - the class weight is used to handle class imbalance - it adjusts the cost function
logistic = LogisticRegression(class_weight={0:0.1, 1: 0.9})

# create pipeline
steps = [("scaler", StandardScaler()),
         ("pca", pca),
         ("logistic", logistic)
        ]

# compile pipeline
pca_logistic = Pipeline(steps)

# hyperparameter space
params = {'pca__n_components': [60, 80], 'logistic__C': [0.1, 0.5, 1, 2, 3, 4, 5, 10], 'logistic__penalty': ['l1', 'l2']}

# create 5 folds
folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 4)

# create gridsearch object
model = GridSearchCV(estimator=pca_logistic, cv=folds, param_grid=params, scoring='roc_auc', n_jobs=1, verbose=1)


# In[81]:


# fit model
model.fit(X_train, y_train)


# In[82]:


# cross validation results
pd.DataFrame(model.cv_results_)


# In[83]:


# print best hyperparameters
print("Best AUC: ", model.best_score_)
print("Best hyperparameters: ", model.best_params_)


# In[85]:


from sklearn.metrics import confusion_matrix, roc_auc_score

# Predict churn on test data
y_pred = model.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Calculate sensitivity and specificity
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

# Calculate area under the ROC curve
y_pred_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_prob)
print("AUC:", auc)


# In[84]:


from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import sensitivity_specificity_support
# predict churn on test data
y_pred = model.predict(X_test)

# create onfusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# check sensitivity and specificity
sensitivity, specificity, _ = sensitivity_specificity_support(y_test, y_pred, average='binary')
print("Sensitivity: \t", round(sensitivity, 2), "\n", "Specificity: \t", round(specificity, 2), sep='')

# check area under curve
y_pred_prob = model.predict_proba(X_test)[:, 1]
print("AUC:    \t", round(roc_auc_score(y_test, y_pred_prob),2))


# ### Random Forest

# In[86]:


# random forest - the class weight is used to handle class imbalance - it adjusts the cost function
forest = RandomForestClassifier(class_weight={0:0.1, 1: 0.9}, n_jobs = -1)

# hyperparameter space
params = {"criterion": ['gini', 'entropy'], "max_features": ['auto', 0.4]}

# create 5 folds
folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 4)

# create gridsearch object
model = GridSearchCV(estimator=forest, cv=folds, param_grid=params, scoring='roc_auc', n_jobs=-1, verbose=1)


# In[87]:


# fit model
model.fit(X_train, y_train)


# In[88]:


# print best hyperparameters
print("Best AUC: ", model.best_score_)
print("Best hyperparameters: ", model.best_params_)


# In[90]:


from sklearn.metrics import confusion_matrix, roc_auc_score

# Predict churn on test data
y_pred = model.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Calculate sensitivity and specificity
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

# Calculate area under the ROC curve
y_pred_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_prob)
print("AUC:", auc)


# In[89]:


# predict churn on test data
y_pred = model.predict(X_test)

# create onfusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# check sensitivity and specificity
sensitivity, specificity, _ = sensitivity_specificity_support(y_test, y_pred, average='binary')
print("Sensitivity: \t", round(sensitivity, 2), "\n", "Specificity: \t", round(specificity, 2), sep='')

# check area under curve
y_pred_prob = model.predict_proba(X_test)[:, 1]
print("AUC:    \t", round(roc_auc_score(y_test, y_pred_prob),2))


# Poor sensitivity. The best model is PCA along with Logistic regression.

# ## ii) Choosing best features

# In[91]:


# run a random forest model on train data
max_features = int(round(np.sqrt(X_train.shape[1])))    # number of variables to consider to split each node
print(max_features)

rf_model = RandomForestClassifier(n_estimators=100, max_features=max_features, class_weight={0:0.1, 1: 0.9}, oob_score=True, random_state=4, verbose=1)


# In[92]:


# fit model
rf_model.fit(X_train, y_train)


# In[93]:


# OOB score
rf_model.oob_score_


# In[95]:


from sklearn.metrics import confusion_matrix, roc_auc_score

# Predict churn on test data
y_pred = rf_model.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Calculate sensitivity and specificity
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

# Calculate area under the ROC curve
y_pred_prob = rf_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_prob)
print("AUC:", auc)


# In[94]:


# predict churn on test data
y_pred = rf_model.predict(X_test)

# create onfusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# check sensitivity and specificity
sensitivity, specificity, _ = sensitivity_specificity_support(y_test, y_pred, average='binary')
print("Sensitivity: \t", round(sensitivity, 2), "\n", "Specificity: \t", round(specificity, 2), sep='')

# check area under curve
y_pred_prob = rf_model.predict_proba(X_test)[:, 1]
print("ROC:    \t", round(roc_auc_score(y_test, y_pred_prob),2))


# ### Feature Importance

# In[96]:


# predictors
features = churn_filtered.drop('churn', axis=1).columns

# feature_importance
importance = rf_model.feature_importances_

# create dataframe
feature_importance = pd.DataFrame({'variables': features, 'importance_percentage': importance*100})
feature_importance = feature_importance[['variables', 'importance_percentage']]

# sort features
feature_importance = feature_importance.sort_values('importance_percentage', ascending=False).reset_index(drop=True)
print("Sum of importance=", feature_importance.importance_percentage.sum())
feature_importance


# ### Extracting top 30 features

# In[97]:


# extract top 'n' features
top_n = 30
top_features = feature_importance.variables[0:top_n]


# In[98]:


# plot feature correlation
import seaborn as sns
plt.rcParams["figure.figsize"] =(10,10)
mycmap = sns.diverging_palette(199, 359, s=99, center="light", as_cmap=True)
sns.heatmap(data=X_train[top_features].corr(), center=0.0, cmap=mycmap)


# In[99]:


top_features = ['total_ic_mou_8', 'total_rech_amt_diff', 'total_og_mou_8', 'arpu_8', 'roam_ic_mou_8', 'roam_og_mou_8',
                'std_ic_mou_8', 'av_rech_amt_data_8', 'std_og_mou_8']
X_train = X_train[top_features]
X_test = X_test[top_features]


# # logistic regression
# steps = [('scaler', StandardScaler()),
#          ("logistic", LogisticRegression(class_weight={0:0.1, 1:0.9}))
#         ]
# 
# # compile pipeline
# logistic = Pipeline(steps)
# 
# # hyperparameter space
# params = {'logistic__C': [0.1, 0.5, 1, 2, 3, 4, 5, 10], 'logistic__penalty': ['l1', 'l2']}
# 
# # create 5 folds
# folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 4)
# 
# # create gridsearch object
# model = GridSearchCV(estimator=logistic, cv=folds, param_grid=params, scoring='roc_auc', n_jobs=-1, verbose=1)

# In[100]:


# fit model
model.fit(X_train, y_train)


# In[101]:


# print best hyperparameters
print("Best AUC: ", model.best_score_)
print("Best hyperparameters: ", model.best_params_)


# In[103]:


from sklearn.metrics import confusion_matrix, roc_auc_score

# Predict churn on test data
y_pred = model.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Calculate sensitivity and specificity
true_negative = cm[0, 0]
false_positive = cm[0, 1]
false_negative = cm[1, 0]
true_positive = cm[1, 1]

sensitivity = true_positive / (true_positive + false_negative)
specificity = true_negative / (true_negative + false_positive)

print("Sensitivity:", round(sensitivity, 2))
print("Specificity:", round(specificity, 2))

# Calculate area under the ROC curve
y_pred_prob = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_prob)
print("ROC AUC:", round(roc_auc, 2))


# In[102]:


# predict churn on test data
y_pred = model.predict(X_test)

# create onfusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# check sensitivity and specificity
sensitivity, specificity, _ = sensitivity_specificity_support(y_test, y_pred, average='binary')
print("Sensitivity: \t", round(sensitivity, 2), "\n", "Specificity: \t", round(specificity, 2), sep='')

# check area under curve
y_pred_prob = model.predict_proba(X_test)[:, 1]
print("ROC:    \t", round(roc_auc_score(y_test, y_pred_prob),2))


# ### Extract the intercept and the coefficients from the logistic model

# In[105]:


logistic_model = model


# In[104]:


logistic_model = model.best_estimator_.named_steps['logistic']


# In[108]:


logistic_model = model.best_estimator_
intercept = logistic_model.intercept_
intercept_df = pd.DataFrame(intercept.reshape((1, 1)), columns=['intercept'])


# In[106]:


# intercept
intercept_df = pd.DataFrame(logistic_model.intercept_.reshape((1,1)), columns = ['intercept'])


# In[109]:


# coefficients
coefficients = logistic_model.coef_.reshape((9, 1)).tolist()
coefficients = [val for sublist in coefficients for val in sublist]
coefficients = [round(coefficient, 3) for coefficient in coefficients]

logistic_features = list(X_train.columns)
coefficients_df = pd.DataFrame(logistic_model.coef_, columns=logistic_features)


# In[110]:


# concatenate dataframes
coefficients = pd.concat([intercept_df, coefficients_df], axis=1)
coefficients


# ## Business Insights
# 
# * Telecom company needs to pay attention to the roaming rates. They need to provide good offers to the customers who are using services from a roaming zone.
# * The company needs to focus on the STD and ISD rates. Perhaps, the rates are too high. Provide them with some kind of STD and ISD packages.
# * To look into both of the issues stated above, it is desired that the telecom company collects customer query and complaint data and work on their services according to the needs of customers.

# In[ ]:




