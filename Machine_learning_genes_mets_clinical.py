#!/usr/bin/env python
# coding: utf-8

# In[1]:


# load library
import numpy as np
import pandas as pd


# In[2]:


input_dir = '/.../Integration_omics/Data'


# In[3]:


# load data with pandas
data = pd.read_csv(input_dir + '/validated_sig_genes_mets.csv')


# In[4]:


data['Cortisol_Group'] = data['Cortisol'] > 18


# In[5]:


data['Cortisol_Group'].value_counts()


# In[6]:


data.head()


# In[7]:


gene_met_list = list(data.loc[:,'MAN1C1':'M100022127'].columns)
len(gene_met_list) # 76: 19 mets and 57 genes


# In[8]:


num_cols = ['Age_at_plasma_collection_date',
            'Corticosteroids_total_number_of_prescriptions_within_5y_log'] + gene_met_list  # this list is for symmetric numeric columns

cat_cols = ['Gender_impute_all',  
            'Closest_collect_date_smoking_status', 
            'BMI_median_closest_measure_date_to_collect_date_category',
            'ICS_Dose_Classification_5Y_Median', 
            'Cortisol_closest_date_collect_date_gap_abs_quartile', 
            'Any_Bronchiectasis_Existence_Yes_No',
            'Any_Chronic_Bronchitis_Existence_Yes_No']  # this list for the class columns

target = 'Cortisol_Group'    # this is the name of the target


# In[9]:


# processing pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import set_config

set_config(transform_output="pandas")

import numpy as np

num_pipeline = Pipeline([                           # now we need a small pipeline for numeric columns since it has two steps
    ('impute', SimpleImputer(strategy='median')),   # this step will impute missing values using column medians
    ('standardize', StandardScaler())               # this step will scale all numeric columns
])



processing_pipeline = ColumnTransformer([      # this transformer merges the processed numeric columns and class columns
    ('numeric', num_pipeline, num_cols),                                                       # numeric columns
    ('class', OneHotEncoder(max_categories=5, handle_unknown='infrequent_if_exist', sparse_output=False), cat_cols) #encoder to transform class columns to numeric, this will automatically handle missing data
  ])


# In[10]:


# put target name in target variable
target = 'Cortisol_Group'


# In[11]:


#import
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


# In[12]:


#result list
results = []


# In[13]:


#loop 100 runs with set seeds
for seed in range(111,11101,111):
#for seed in range(1,2):
    #set seed
    np.random.seed(seed)

    #train test split
    train_data, test_data = train_test_split(data, test_size = 0.3)
    train_processed = processing_pipeline.fit_transform(train_data)
    n_features = train_processed.shape[1]
    
    #logistic
    logistic = LogisticRegression(max_iter=10000)
    param_grid = [{'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1 , 5, 10, 50, 100]}]
    grid_search = GridSearchCV(logistic, param_grid, cv=5, scoring='roc_auc', return_train_score=True)

    logistic_pipeline = Pipeline([
        ('processing', processing_pipeline),
        ('modeling', grid_search)
    ])

    logistic_pipeline.fit(train_data, train_data[target])
    logistic_train_auc = logistic_pipeline['modeling'].best_score_
    logistic_test_auc = logistic_pipeline.score(test_data, test_data[target])

    #SVM
    svc = SVC()
    param_grid = [{
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'kernel' : ['rbf'],
        'gamma' : [0.001, 0.01, 0.1, 1, 10, 100]
    }]

    grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='roc_auc', return_train_score=True)

    svc_pipeline = Pipeline([
        ('processing', processing_pipeline),
        ('modeling', grid_search)
    ])

    svc_pipeline.fit(train_data, train_data[target])
    svc_train_auc = svc_pipeline['modeling'].best_score_
    svc_test_auc = svc_pipeline.score(test_data, test_data[target])

    #RF
    param_grid = [{
        'max_depth': [3, 4, 5],
        'min_samples_split' : [0.05, 0.1, 0.2],
        'min_samples_leaf' : [0.05, 0.1, 0.2],
        'n_estimators': [50, 100]
    }]

    forest = RandomForestClassifier()

    grid_search = GridSearchCV(forest, param_grid, cv=5, scoring='roc_auc', return_train_score=True)

    forest_pipeline = Pipeline([
        ('processing', processing_pipeline),
        ('modeling', grid_search)
    ])

    forest_pipeline.fit(train_data, train_data[target])
    forest_train_auc = forest_pipeline['modeling'].best_score_
    forest_test_auc = forest_pipeline.score(test_data, test_data[target])

    #GB (use same parameter grid with RF, so no needs to redefine)
    gbc = GradientBoostingClassifier()

    grid_search = GridSearchCV(gbc, param_grid, cv=5, scoring='roc_auc', return_train_score=True)

    gbc_pipeline = Pipeline([
        ('processing', processing_pipeline),
        ('modeling', grid_search)
    ])

    gbc_pipeline.fit(train_data, train_data[target])
    gbc_train_auc = gbc_pipeline['modeling'].best_score_
    gbc_test_auc = gbc_pipeline.score(test_data, test_data[target])

    #MLP
    param_grid = [{
        'hidden_layer_sizes' : [[n_features // 2, n_features // 2],
                                [n_features // 2, n_features // 2, n_features // 2],
                                [n_features, n_features],
                                [n_features, n_features, n_features],
                                [n_features*2, n_features*2],
                                [n_features*2, n_features*2, n_features*2]],
        'alpha' : [0.001, 0.01, 0.1, 1, 10]                                    #regularization terms
    }]

    mlp = MLPClassifier(max_iter=10000)
    grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='roc_auc', return_train_score=True)

    mlp_pipeline = Pipeline([
        ('processing', processing_pipeline),
        ('modeling', grid_search)
    ])

    mlp_pipeline.fit(train_data, train_data[target])
    mlp_train_auc = mlp_pipeline['modeling'].best_score_
    mlp_test_auc = mlp_pipeline.score(test_data, test_data[target])
    results.append([seed, 
                    logistic_train_auc,
                    logistic_test_auc,
                    svc_train_auc,
                    svc_test_auc,
                    forest_train_auc,
                    forest_test_auc,
                    gbc_train_auc,
                    gbc_test_auc,
                    mlp_train_auc,
                    mlp_test_auc])


# In[ ]:


results_df = pd.DataFrame(results)
results_df.columns = ['seed', 
                      'logistic_train_auc',
                      'logistic_test_auc',
                      'svc_train_auc',
                      'svc_test_auc',
                      'forest_train_auc',
                      'forest_test_auc',
                      'gbc_train_auc',
                      'gbc_test_auc',
                      'mlp_train_auc',
                      'mlp_test_auc']


# In[ ]:


output_dir = '...'
results_df.to_csv(output_dir + 'validated_genes_mets_clinical_machine_learning_result.csv', index = None)


# In[ ]:




