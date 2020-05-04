import pandas as pd
from sklearn.model_selection import cross_val_score, RandomizedSearchCV


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

#Feature engineering of trained data
train_data.dropna(subset=['BsmtExposure'], axis= 0)
#train_data.drop(index='NA', columns='BsmtExposure', axis=0)
train_data['LotFrontage'] = train_data['LotFrontage'].fillna(train_data['LotFrontage'].mean())
#train_data['MasVnrArea'] = train_data['MasVnrArea'].fillna(train_data['MasVnrArea'].mean())
train_data['GarageYrBlt'] = train_data['GarageYrBlt'].fillna(train_data['GarageYrBlt'].mean())
#train_data['MasVnrType'] = train_data['MasVnrType'].fillna(train_data['MasVnrType'].mode()[0])
train_data['BsmtQual'] = train_data['BsmtQual'].fillna(train_data['BsmtQual'].mode()[0])
train_data['BsmtCond'] = train_data['BsmtCond'].fillna(train_data['BsmtCond'].mode()[0])
#train_data['BsmtExposure'] = train_data['BsmtExposure'].fillna(train_data['BsmtExposure'].mode()[0])

train_data['BsmtFinType1'] = train_data['BsmtFinType1'].fillna(train_data['BsmtFinType1'].mode()[0])
train_data['BsmtFinType2'] = train_data['BsmtFinType2'].fillna(train_data['BsmtFinType2'].mode()[0])
train_data['Electrical'] = train_data['Electrical'].fillna(train_data['Electrical'].mode()[0])
train_data['FireplaceQu'] = train_data['FireplaceQu'].fillna(train_data['FireplaceQu'].mode()[0])
train_data['GarageType'] = train_data['GarageType'].fillna(train_data['GarageType'].mode()[0])
train_data['GarageFinish'] = train_data['GarageFinish'].fillna(train_data['GarageFinish'].mode()[0])
train_data['GarageQual'] = train_data['GarageQual'].fillna(train_data['GarageQual'].mode()[0])
train_data['GarageCond'] = train_data['GarageCond'].fillna(train_data['GarageCond'].mode()[0])
train_data.drop(['Alley'], axis=1, inplace=True)
train_data.drop(['PoolQC'], axis=1, inplace=True)
train_data.drop(['Fence'], axis=1, inplace=True)
train_data.drop(['MiscFeature', 'Id', 'MasVnrArea', 'MasVnrType'], axis=1, inplace=True)

#sns.heatmap(train_data.isnull(), yticklabels=False,cbar= False, cmap='coolwarm')
#print(train_data.info())
#plt.show()

#print(train_data.shape)
#print(train_data.isnull().sum())
#sns.heatmap(train_data.isnull(), yticklabels=False,cbar= False, cmap='coolwarm')
#sns.heatmap(test_data.isnull(), yticklabels=False,cbar= False, cmap='coolwarm')
#print(test_data.info())
#plt.show()

#feature engineering of test data
test_data.drop(['Alley', 'PoolQC','MiscVal','Fence','MiscFeature'], axis=1, inplace=True)
test_data['Utilities']=test_data['Utilities'].fillna(test_data['Utilities'].mode()[0])
test_data['Exterior1st']=test_data['Exterior1st'].fillna(test_data['Exterior1st'].mode()[0])
test_data['Exterior2nd']=test_data['Exterior2nd'].fillna(test_data['Exterior2nd'].mode()[0])
test_data['BsmtFinType1']=test_data['BsmtFinType1'].fillna(test_data['BsmtFinType1'].mode()[0])

test_data['MSZoning']=test_data['MSZoning'].fillna(test_data['MSZoning'].mode()[0])
test_data['BsmtFinSF1']=test_data['BsmtFinSF1'].fillna(test_data['BsmtFinSF1'].mean())
test_data['BsmtFinSF2']=test_data['BsmtFinSF2'].fillna(test_data['BsmtFinSF2'].mean())
test_data['LotFrontage']=test_data['LotFrontage'].fillna(test_data['LotFrontage'].mean())
test_data['BsmtUnfSF']=test_data['BsmtUnfSF'].fillna(test_data['BsmtUnfSF'].mean())
test_data['TotalBsmtSF']=test_data['TotalBsmtSF'].fillna(test_data['TotalBsmtSF'].mean())
test_data['BsmtFullBath']=test_data['BsmtFullBath'].fillna(test_data['BsmtFullBath'].mode()[0])
test_data['BsmtHalfBath']=test_data['BsmtHalfBath'].fillna(test_data['BsmtHalfBath'].mode()[0])
test_data['KitchenQual']=test_data['KitchenQual'].fillna(test_data['KitchenQual'].mode()[0])
test_data['Functional']=test_data['Functional'].fillna(test_data['Functional'].mode()[0])
test_data['GarageCars']=test_data['GarageCars'].fillna(test_data['GarageCars'].mean())
test_data['GarageArea']=test_data['GarageArea'].fillna(test_data['GarageArea'].mean())
test_data['GarageYrBlt']=test_data['GarageYrBlt'].fillna(test_data['GarageYrBlt'].mean())
test_data['SaleType']=test_data['SaleType'].fillna(test_data['SaleType'].mode()[0])
test_data['BsmtCond']=test_data['BsmtCond'].fillna(test_data['BsmtCond'].mode()[0])
test_data['BsmtQual']=test_data['BsmtQual'].fillna(test_data['BsmtQual'].mode()[0])
test_data['BsmtQual']=test_data['BsmtQual'].fillna(test_data['BsmtQual'].mode()[0])
test_data['FireplaceQu']=test_data['FireplaceQu'].fillna(test_data['FireplaceQu'].mode()[0])
test_data['GarageType']=test_data['GarageType'].fillna(test_data['GarageType'].mode()[0])
test_data['GarageFinish']=test_data['GarageFinish'].fillna(test_data['GarageFinish'].mode()[0])
test_data['GarageQual']=test_data['GarageQual'].fillna(test_data['GarageQual'].mode()[0])

#sns.heatmap(test_data.isnull(), yticklabels=False,cbar= False, cmap='coolwarm')
#print(test_data.info())
#print(test_data.loc[:, test_data.isnull().any()].head())
#plt.show()
#print(test_data.shape)

#load test data
test_data.to_csv('formulatedtest.csv',index=False)
test_df = pd.read_csv("formulatedtest.csv")
final_df=pd.concat([train_data,test_df],axis=0,sort = False)

#Feature extraction
columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']



df_final = final_df
i = 0
for fields in columns:

    #print(fields)
    df1 = pd.get_dummies(final_df[fields], drop_first=True)

    final_df.drop([fields], axis=1, inplace=True)
    if i == 0:
        df_final = df1.copy()
    else:

        df_final = pd.concat([df_final, df1], axis=1)
    i = i + 1


final_df = pd.concat([final_df, df_final], axis=1)
#print(df_final.head())
main_df = train_data.copy()
final_df = final_df.loc[:,~final_df.columns.duplicated()]
df_train = final_df.iloc[:1460,:]
df_test = final_df.iloc[1460:,:]
print(df_test.shape)
print(test_data.shape)
X_Train = df_train.drop(['SalePrice'],axis= 1)
print(X_Train.shape)
Y_Train = df_train['SalePrice']

#train model
import xgboost
classifier = xgboost.XGBRegressor()
classifier.fit(X_Train,Y_Train)
import pickle
filename = 'final_model.pkl'
pickle.dump(classifier, open(filename, 'wb'))
#y_pred = classifier.predict(test_data)
#df_test.drop(['SalePrice'],axis=1,inplace=True)

y_pred = classifier.predict(df_test.drop(['SalePrice'],axis=1))
#y_pred_train = classifier.predict(X_Train)

#print(y_pred_train)
pred= pd.DataFrame(y_pred)
sub_df= pd.read_csv('sample_submission.csv')
datasets= pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns=['Id','SalePrice']
datasets.to_csv('sample_submission.csv',index=False)

#scores = cross_val_score(X_Train, Y_Train, cv=5, scoring='F1_macro')
#print(scores.mean())
#print('Training score is ', f1_score(y_pred_train, Y_Train))
#print('Testing score is ', f1_score(y_pred, Y_Train))
#Hyperparameter Optimization
n_estimator = [100,200,300,400,500,600,900,1100,1500]
max_depth = [2,3,4,5,10,12,15]
booster = ['gbtree', 'gblinear']
learning_rate = [0.005,0.001, 0.01,0.1,0.15,0.20]
min_child_weights = [1,2,3,4]
base_score=[0.25,0.5,0.75,1]
#define grid of hyperparameter to search
hyperparamter_grid = {'n_estimators' : n_estimator,
                      'max_depth' : max_depth,
                      'learning_rate' : learning_rate,
                      'min_child_weight' : min_child_weights,
                      'booster' : booster,
                      'base_score': base_score}

#Set up random search
random_cv = RandomizedSearchCV(estimator=classifier, param_distributions= hyperparamter_grid,
                               cv=5, n_iter=50,
                               scoring='neg_mean_absolute_error', n_jobs=4,
                               verbose=5, return_train_score=True, random_state=42)
random_cv.fit(X_Train,Y_Train)
print(random_cv.best_estimator_)

classifier = xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints=None,
             learning_rate=0.1, max_delta_step=0, max_depth=2,
             min_child_weight=2, monotone_constraints=None,
             n_estimators=500, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,
             validate_parameters=False, verbosity=None)
classifier.fit(X_Train, Y_Train)

#load model in pickle file
import pickle
filename = 'final_model.pkl'
pickle.dump(classifier,open(filename, 'wb'))
#df_test.drop(['SalePrice'],axis=1,inplace=True)

y_pred = classifier.predict(df_test.drop(['SalePrice'],axis=1))
print(y_pred)
pred = pd.DataFrame(y_pred)
sub_df = pd.read_csv('sample_submission.csv')
datasets = pd.concat([sub_df['Id'],pred],axis = 1)
datasets.columns = ['Id', 'SalePrice']
datasets.to_csv('sample_submission.csv', index = False)







