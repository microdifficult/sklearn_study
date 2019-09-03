
# coding: utf-8

# In[232]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[233]:


df=pd.read_csv('C:/Users/Administrator/Scikit_learn/HOUSEPRICE/train.csv',delimiter=',',header=0)
#用/
df_backup=df


# In[234]:


df.describe()


# In[235]:


df.info()
print(df.Alley.fillna('No_access',inplace=True))
#fillna不改变原数据

print(df.PoolQC.fillna('No_pool',inplace=True))
print(df.MiscFeature.fillna('None',inplace=True))
print(df.Fence.fillna('No_fence',inplace=True))
print(df.FireplaceQu.fillna('No_fireplace',inplace=True))


# In[236]:


df.corr()


# In[237]:


df_corr=df.corr()


# In[238]:


print(df_corr.sort_values(by='SalePrice'))


# In[239]:


plt.figure(figsize=(10,8)) 
plt.hist(df['SalePrice'],bins=200)
plt.xlabel('HousePrice')
plt.ylabel('Counts')
plt.show()

plt.figure(figsize=(10,8)) 
df['SalePrice'].plot(kind='kde')
#plot是pylab的函数，kind(line,bar,kde,barh)
plt.xlabel('HousePrice')
plt.ylabel('Counts')
plt.xlim([0,1000000])
plt.show()


# In[240]:


df.SalePrice.value_counts()
df.MSSubClass.value_counts().plot(kind='bar',figsize=(10,8))
sta_MSSubClass=df.MSSubClass.value_counts().index
sta_MSSubClass=list(sta_MSSubClass.values)
print(sta_MSSubClass)
plt.show()



type(sta_MSSubClass)
#获取变量类型
plt.figure(figsize=(10,8)) 
for i in sta_MSSubClass:
    plt.hist(df.SalePrice[df.MSSubClass==i],bins=50)  
   
plt.xlim([0,1000000])
plt.legend(['20','60','50','120','30','160','70','80','90','190','85','75','45','180','40'])
plt.tight_layout()
plt.show()
for i in sta_MSSubClass:
    df.SalePrice[df.MSSubClass==i].plot(kind='kde',figsize=(10,10))
plt.xlim([0,1000000])
plt.legend(['20','60','50','120','30','160','70','80','90','190','85','75','45','180','40'])
plt.show()

plt.figure(figsize=(20,15)) 
for i in sta_MSSubClass:
    plt.subplot(3,5,sta_MSSubClass.index(i)+1)
    plt.hist(df.SalePrice[df.MSSubClass==i],bins=50)  
    plt.title(i,fontsize=5)
    plt.xlim([0,1000000])
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

#plt.legend(['20','60','50','120','30','160','70','80','90','190','85','75','45','180','40'])
plt.tight_layout()
plt.savefig('YES.png',dpi=1000)
plt.style.use('ggplot')
plt.show()

#subplot 可以用（3,5,i）的方式
#在 plt.show() 之前调用 plt.savefig()
#plt.tight_layout()设置子图间距

plt.figure(figsize=(20,15)) 
for i in sta_MSSubClass:
    plt.subplot(3,5,sta_MSSubClass.index(i)+1)
    plt.boxplot(x=df.SalePrice[df.MSSubClass==i])

    plt.title(i,fontsize=5)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

plt.tight_layout()
plt.style.use('ggplot')
plt.show()
#先设置大小


# In[241]:


df.MSZoning.value_counts().plot(kind='bar',figsize=(10,7))
plt.show()
#plt.figure(figsize=(20,15)) 
sta_MSZoning=df.MSZoning.value_counts().index
type(sta_MSZoning)
sta_MSZoning=list(sta_MSZoning.values)
for i in sta_MSZoning:
    df.SalePrice[df.MSZoning==i].plot(kind='kde',figsize=(10,7))
plt.xlim([0,1000000])
plt.legend(['RL','RM','FV','RH','C(ALL)'])
plt.tight_layout()
plt.show()
plt.figure(figsize=(10,7)) 
for i in sta_MSZoning:
    plt.subplot(1,5,sta_MSZoning.index(i)+1)
    plt.boxplot(x=df.SalePrice[df.MSZoning==i])
    plt.title(i,fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

plt.tight_layout()
plt.style.use('ggplot')
plt.show()


# In[242]:


dff_continuous=df.copy()
types=dff_continuous.dtypes
types=types[types!=object]
print(types)
print(types.index)
for i in types.index:
    plt.figure(figsize=(10,7))
    plt.scatter(df[i],df['SalePrice'])
    plt.title(i)
    plt.tight_layout()
    plt.style.use('ggplot')
    plt.show()


# In[243]:


dff_uncontinuous=df.copy()
#dff_uncontinuous = dff_uncontinuous.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'], axis=1)
types=dff_uncontinuous.dtypes
types2=types[types==object]
print(types2)
for i in types2.index:   
    df[i].value_counts().plot(kind='bar',figsize=(10,8))
    plt.title(i)
    plt.tight_layout()
    plt.style.use('ggplot')
    plt.show()
    plt.figure(figsize=(10,7))
    plt.scatter(df[i],df['SalePrice'])
    plt.title(i)
    plt.tight_layout()
    plt.style.use('ggplot')
    plt.show()


# In[244]:


df_backup.info()
df=df_backup


# In[245]:


df.corr()


# In[246]:


from sklearn.neighbors import KNeighborsRegressor
LotFrontAge=df[['MSSubClass','LotArea','1stFlrSF','GrLivArea','LotFrontage']]
known_LotFrontage=LotFrontAge[LotFrontAge.LotFrontage.notnull()].as_matrix()
unknown_LotFrontage=LotFrontAge[LotFrontAge.LotFrontage.isnull()].as_matrix()
X_train=known_LotFrontage[:,0:-1]
y_train=known_LotFrontage[:,-1]


# In[247]:


clf=KNeighborsRegressor()
clf.fit(X_train,y_train)
predict_unknown_LotFrontage=clf.predict(unknown_LotFrontage[:,0:-1])
unknown_LotFrontage=[unknown_LotFrontage,predict_unknown_LotFrontage]
df.loc[(df.LotFrontage.isnull()),'LotFrontage']=predict_unknown_LotFrontage
#df.loc[行，列]


# In[248]:


df.iloc[:,0:11]
#loc：通过选取行（列）标签索引数据 
#iloc：通过选取行（列）位置编号索引数据 
#ix：既可以通过行（列）标签索引数据，也可以通过行（列）位置编号索引数据


# In[249]:


df.info()


# In[250]:


plt.scatter(df.MasVnrType,df.MasVnrArea)
plt.show()
df.MasVnrType.value_counts()

from sklearn.preprocessing import LabelBinarizer,OneHotEncoder,LabelEncoder
lr=LabelBinarizer()
le=LabelEncoder()
#df['Street']=lr.fit_transform(df['Street'])
#df['Utilities']=lr.fit_transform(df['Utilities'])
#df['CentralAir']=lr.fit_transform(df['CentralAir'])
le=LabelEncoder()
#for i in types2.index:
    #df[i]=le.fit_transform(df[i])
types=df.dtypes
types2=types[types==object]
kk=df.ExterQual.value_counts()
lr.fit(kk.index)

# In[251]:


from sklearn.preprocessing import LabelBinarizer,OneHotEncoder,LabelEncoder,StandardScaler
le=LabelEncoder()


# In[252]:


types=df.dtypes
types2=types[types==object]


# In[253]:


#for i in types2.index:
    #df[i]=le.fit_transform(df[i])


# In[254]:


df.info()
types2.index


# In[255]:


df.MasVnrType.fillna('No_data',inplace=True)
df.BsmtQual.fillna('No_data',inplace=True)
df.BsmtCond.fillna('No_data',inplace=True)
df.BsmtExposure.fillna('No_data',inplace=True)
df.BsmtFinType1.fillna('No_data',inplace=True)
df.BsmtFinType2.fillna('No_data',inplace=True)
df.GarageType.fillna('No_data',inplace=True)
df.GarageFinish.fillna('No_data',inplace=True)
df.GarageQual.fillna('No_data',inplace=True)
df.GarageCond.fillna('No_data',inplace=True)
df.Electrical.fillna('No_data',inplace=True)


# In[256]:


df.MasVnrType.value_counts()


# In[257]:


for i in types2.index:
    df[i]=le.fit_transform(df[i])
#必须先搞缺失值，包括类型和值，否则不匹配


# In[258]:


df.info()


# In[259]:


print(df.MSSubClass)


# In[260]:


Ohe=OneHotEncoder()


# In[261]:


Ohe.fit_transform(df_backup.MSSubClass.reshape(-1,1))
bb=Ohe.transform(df_backup.MSSubClass.reshape(-1,1)).toarray()


# In[262]:


dummies_MSSubClass=pd.get_dummies(df_backup['MSSubClass'],prefix='MSSubClass')
dummies_MSZoning=pd.get_dummies(df_backup['MSZoning'],prefix='MSZoning')
dummies_Street=pd.get_dummies(df_backup['Street'],prefix='Street')
dummies_Alley=pd.get_dummies(df_backup['Alley'],prefix='Alley')
dummies_LotShape=pd.get_dummies(df_backup['LotShape'],prefix='MSSubClass')
dummies_LandContour=pd.get_dummies(df_backup['LandContour'],prefix='LandContour')
dummies_Utilities=pd.get_dummies(df_backup['Utilities'],prefix='Utilities')
dummies_LotConfig=pd.get_dummies(df_backup['LotConfig'],prefix='LotConfig')
dummies_LandSlope=pd.get_dummies(df_backup['LandSlope'],prefix='MSSubClass')
dummies_Neighborhood=pd.get_dummies(df_backup['Neighborhood'],prefix='Neighborhood')
dummies_Condition1=pd.get_dummies(df_backup['Condition1'],prefix='Condition1')
dummies_Condition2=pd.get_dummies(df_backup['Condition2'],prefix='Condition2')
dummies_BldgType=pd.get_dummies(df_backup['BldgType'],prefix='BldgType')

dummies_HouseStyle=pd.get_dummies(df_backup['HouseStyle'],prefix='HouseStyle')
dummies_OverallQual=pd.get_dummies(df_backup['OverallQual'],prefix='OverallQual')
dummies_OverallCond=pd.get_dummies(df_backup['OverallCond'],prefix='OverallCond')
dummies_RoofStyle=pd.get_dummies(df_backup['RoofStyle'],prefix='RoofStyle')
dummies_RoofMatl=pd.get_dummies(df_backup['RoofMatl'],prefix='RoofMatl')
dummies_Exterior1st=pd.get_dummies(df_backup['Exterior1st'],prefix='Exterior1st')
dummies_Exterior2nd=pd.get_dummies(df_backup['Exterior2nd'],prefix='Exterior2nd')
dummies_MasVnrType=pd.get_dummies(df_backup['MasVnrType'],prefix='MasVnrType')
dummies_ExterQual=pd.get_dummies(df_backup['ExterQual'],prefix='ExterQual')
dummies_ExterCond=pd.get_dummies(df_backup['ExterCond'],prefix='ExterCond')
dummies_Foundation=pd.get_dummies(df_backup['Foundation'],prefix='Foundation')
dummies_BsmtQual=pd.get_dummies(df_backup['BsmtQual'],prefix='BsmtQual')
dummies_BsmtCond=pd.get_dummies(df_backup['BsmtCond'],prefix='BsmtCond')
dummies_BsmtExposure=pd.get_dummies(df_backup['BsmtExposure'],prefix='BsmtExposure')
dummies_BsmtFinType1=pd.get_dummies(df_backup['BsmtFinType1'],prefix='BsmtFinType1')
dummies_BsmtFinType2=pd.get_dummies(df_backup['BsmtFinType2'],prefix='BsmtFinType2')
dummies_Heating=pd.get_dummies(df_backup['Heating'],prefix='Heating')
dummies_HeatingQC=pd.get_dummies(df_backup['HeatingQC'],prefix='HeatingQC')
dummies_Electrical=pd.get_dummies(df_backup['Electrical'],prefix='Electrical')
dummies_BsmtFullBath=pd.get_dummies(df_backup['BsmtFullBath'],prefix='BsmtFullBath')
dummies_BsmtHalfBath=pd.get_dummies(df_backup['BsmtHalfBath'],prefix='BsmtHalfBath')
dummies_FullBath=pd.get_dummies(df_backup['FullBath'],prefix='FullBath')
dummies_HalfBath=pd.get_dummies(df_backup['HalfBath'],prefix='HalfBath')
dummies_BedroomAbvGr=pd.get_dummies(df_backup['BedroomAbvGr'],prefix='BedroomAbvGr')
dummies_KitchenAbvGr=pd.get_dummies(df_backup['KitchenAbvGr'],prefix='KitchenAbvGr')
dummies_KitchenQual=pd.get_dummies(df_backup['KitchenQual'],prefix='KitchenQual')
dummies_TotRmsAbvGrd=pd.get_dummies(df_backup['TotRmsAbvGrd'],prefix='TotRmsAbvGrd')
dummies_Functional=pd.get_dummies(df_backup['Functional'],prefix='Functional')
dummies_Fireplaces=pd.get_dummies(df_backup['Fireplaces'],prefix='Fireplaces')
dummies_FireplaceQu=pd.get_dummies(df_backup['FireplaceQu'],prefix='FireplaceQu')

dummies_GarageType=pd.get_dummies(df_backup['GarageType'],prefix='GarageType')
dummies_GarageFinish=pd.get_dummies(df_backup['GarageFinish'],prefix='GarageFinish')
dummies_GarageCars=pd.get_dummies(df_backup['GarageCars'],prefix='GarageCars')
dummies_GarageQual=pd.get_dummies(df_backup['GarageQual'],prefix='GarageQual')
dummies_GarageCond=pd.get_dummies(df_backup['GarageCond'],prefix='GarageCond')
dummies_PavedDrive=pd.get_dummies(df_backup['PavedDrive'],prefix='PavedDrive')
dummies_PoolQC=pd.get_dummies(df_backup['PoolQC'],prefix='PoolQC')
dummies_Fence=pd.get_dummies(df_backup['Fence'],prefix='Fence')
dummies_MiscFeature=pd.get_dummies(df_backup['MiscFeature'],prefix='MiscFeature')
dummies_SaleType=pd.get_dummies(df_backup['SaleType'],prefix='SaleType')
dummies_SaleCondition=pd.get_dummies(df_backup['SaleCondition'],prefix='SaleCondition')



# In[263]:


df_onehotencoder=pd.concat([df_backup,dummies_MSSubClass,
                            dummies_MSZoning,dummies_Street,dummies_Alley,dummies_LotShape,dummies_LandContour,dummies_Utilities,
                            dummies_LotConfig,dummies_LandSlope,dummies_Neighborhood,dummies_Condition1,dummies_Condition2,dummies_BldgType,
                            dummies_HouseStyle,dummies_OverallQual,dummies_OverallCond,dummies_RoofStyle,dummies_RoofMatl,dummies_Exterior1st,
                            dummies_Exterior2nd,dummies_MasVnrType,dummies_ExterQual,dummies_ExterCond,dummies_Foundation,dummies_BsmtQual,
                            dummies_BsmtCond,dummies_BsmtExposure,dummies_BsmtFinType1,dummies_BsmtFinType2,dummies_Heating,dummies_HeatingQC,
                            dummies_Electrical,dummies_BsmtFullBath,dummies_BsmtHalfBath,dummies_FullBath,dummies_HalfBath,dummies_BedroomAbvGr,
                            dummies_KitchenAbvGr,dummies_KitchenQual,dummies_TotRmsAbvGrd,dummies_Functional,dummies_Fireplaces,dummies_FireplaceQu
 ],axis=1)
df_onehotencoder.drop(['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1',
         'Condition2','BldgType','HouseStyle','OverallQual','OverallCond','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
         'ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC',
         'Electrical','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','KitchenQual','TotRmsAbvGrd',
         'Functional','Fireplaces','FireplaceQu'
],inplace=True,axis=1)
df_onehotencoder


# In[264]:


df_onehotencoder_copy=df_onehotencoder.copy()


# In[265]:


from sklearn.preprocessing import StandardScaler,scale
df_onehotencoder['LotFrontage']=scale(df_onehotencoder['LotFrontage'])
df_onehotencoder['LotArea']=scale(df_onehotencoder['LotArea'])


# In[266]:


print(df.GarageCars)


# In[267]:






df_onehotencoder['YearBuilt']=scale(df_onehotencoder['YearBuilt'])
df_onehotencoder['YearRemodAdd']=scale(df_onehotencoder['YearRemodAdd'])
#df_onehotencoder['MasVnrArea']=scale(df_onehotencoder['MasVnrArea'])
df_onehotencoder['BsmtFinSF1']=scale(df_onehotencoder['BsmtFinSF1'])
df_onehotencoder['BsmtFinSF2']=scale(df_onehotencoder['BsmtFinSF2'])
df_onehotencoder['BsmtUnfSF']=scale(df_onehotencoder['BsmtUnfSF'])
df_onehotencoder['TotalBsmtSF']=scale(df_onehotencoder['TotalBsmtSF'])
df_onehotencoder['1stFlrSF']=scale(df_onehotencoder['1stFlrSF'])
df_onehotencoder['2ndFlrSF']=scale(df_onehotencoder['2ndFlrSF'])
df_onehotencoder['LowQualFinSF']=scale(df_onehotencoder['LowQualFinSF'])
df_onehotencoder['GrLivArea']=scale(df_onehotencoder['GrLivArea'])
#df_onehotencoder['GarageYrBlt']=scale(df_onehotencoder['GarageYrBlt'])
df_onehotencoder['GarageArea']=scale(df_onehotencoder['GarageArea'])
df_onehotencoder['WoodDeckSF']=scale(df_onehotencoder['WoodDeckSF'])
df_onehotencoder['OpenPorchSF']=scale(df_onehotencoder['OpenPorchSF'])
df_onehotencoder['EnclosedPorch']=scale(df_onehotencoder['EnclosedPorch'])
df_onehotencoder['3SsnPorch']=scale(df_onehotencoder['3SsnPorch'])
df_onehotencoder['PoolArea']=scale(df_onehotencoder['PoolArea'])
df_onehotencoder['MoSold']=scale(df_onehotencoder['MoSold'])
df_onehotencoder['YrSold']=scale(df_onehotencoder['YrSold'])






# In[268]:


#填补缺失值
df_onehotencoder['MasVnrArea']= df_onehotencoder_copy.MasVnrArea.fillna(df_onehotencoder.MasVnrArea.median())
df_onehotencoder['GarageYrBlt']= df_onehotencoder_copy.GarageYrBlt.fillna(df_onehotencoder.GarageYrBlt.median())


df_onehotencoder['MasVnrArea']=scale(df_onehotencoder['MasVnrArea'])
df_onehotencoder['GarageYrBlt']=scale(df_onehotencoder['GarageYrBlt'])


# In[269]:


df_onehotencoder.describe()
df_onehotencoder['SalePrice']=scale(df_onehotencoder['SalePrice'])


# In[270]:


df_onehotencoder


# In[283]:


from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.neighbors import KNeighborsRegressor
y=df_onehotencoder['SalePrice']
X=df_onehotencoder.drop(['SalePrice'],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7,random_state=13)


# In[286]:


y_train


# In[273]:


clf_KNN=KNeighborsRegressor()
clf_KNN.fit(X_train,y_train)
predicts_knn=clf_KNN.predict(X_test)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
cross_val_score(clf_KNN,X,y,cv=5,scoring='r2')

print('the r2_score is %s \tmean_absolute_error is %s \t the mean_squared_error is %s'%(r2_score(y_test,predicts_knn),mean_absolute_error(y_test,predicts_knn),mean_squared_error(y_test,predicts_knn)))


# In[274]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
cross_val_score(lr,X,y,cv=5)
lr.fit(X_train,y_train)
predicts_lr=lr.predict(X_test)
print('the r2_score is %s \tmean_absolute_error is %s \t the mean_squared_error is %s'%(r2_score(y_test,predicts_lr),mean_absolute_error(y_test,predicts_lr),mean_squared_error(y_test,predicts_lr)))


# In[275]:


from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
print(cross_val_score(dt,X,y,cv=5))
dt.fit(X_train,y_train)
predicts_dt=dt.predict(X_test)
print('the r2_score is %s \tmean_absolute_error is %s \t the mean_squared_error is %s'%(r2_score(y_test,predicts_dt),mean_absolute_error(y_test,predicts_dt),mean_squared_error(y_test,predicts_dt)))


# In[276]:


from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(n_estimators=150)
print(cross_val_score(rfr,X,y,cv=5))
rfr.fit(X_train,y_train)
predicts_rfr=rfr.predict(X_test)
print('the r2_score is %s \tmean_absolute_error is %s \t the mean_squared_error is %s'%(r2_score(y_test,predicts_rfr),mean_absolute_error(y_test,predicts_rfr),mean_squared_error(y_test,predicts_rfr)))
plt.plot(range(1,151),[accuracy for accuracy in abr.staged_score(X_test,y_test)])
plt.show()


# In[277]:


from sklearn.ensemble import AdaBoostRegressor
abr=AdaBoostRegressor(n_estimators=150)
print(cross_val_score(abr,X,y,cv=5))
abr.fit(X_train,y_train)
predicts_abr=abr.predict(X_test)
print('the r2_score is %s \tmean_absolute_error is %s \t the mean_squared_error is %s'%(r2_score(y_test,predicts_abr),mean_absolute_error(y_test,predicts_abr),mean_squared_error(y_test,predicts_abr)))
plt.plot(range(1,151),[accuracy for accuracy in abr.staged_score(X_test,y_test)])
plt.show()


# In[278]:


dt.feature_importances_


# In[279]:


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


# In[280]:


tuple(np.array(range(1,150)))


# In[295]:


pipeline=Pipeline([
    ('plf',RandomForestRegressor())
])


# In[306]:


parameters={
    #'plf__n_estimators':tuple(np.array(range(1,50))),
    'plf__criterion':('mse','mae'),
    'plf__max_depth':(150,155,160),
    'plf__min_samples_split':(2,3),
    'plf__min_samples_leaf':(1,2,3), 
    #'plf__bootstrap':(True,False),
}


# In[307]:


grid_search=GridSearchCV(pipeline,parameters,n_jobs=-1,verbose=1,cv=3)
grid_search.fit(X_train,y_train)


# In[308]:


print('Best score: %0.3f' %grid_search.best_score_)
print('Best parameter set:')
best_parameters=grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print('\t%s:%r'%(param_name ,best_parameters[param_name]))


# In[344]:


predict_grid=grid_search.predict(X_test)


# In[339]:


coeff=pd.DataFrame({"columns":list(X_train.columns),"coef":list(rfr.feature_importances_.T)})
coeff.sort_values('coef', inplace=True)
coeff


# In[319]:


np.size(list(rfr.feature_importances_.T))


# In[346]:


plt.scatter(y_test,predict_grid)
plt.axis([-2,5,-2,5])
plt.show()


# In[350]:


from sklearn.learning_curve import learning_curve
train_sizes,train_score,test_score = learning_curve(grid_search,X_train,y_train,train_sizes=[0.1,0.2,0.4,0.6,0.8,1],cv=3)
train_error =  1- np.mean(train_score,axis=1)
test_error = 1- np.mean(test_score,axis=1)



# In[352]:


plt.plot(train_sizes,1-train_error,'o-',color = 'r',label = 'training')
plt.plot(train_sizes,1-test_error,'o-',color = 'g',label = 'testing')
plt.legend(loc='best')
plt.xlabel('traing examples')
plt.ylabel('Accuracy')
plt.show()


# In[368]:


from mlxtend.regressor import StackingRegressor
lr = LinearRegression()
sclf = StackingRegressor(regressors=[grid_search, abr, rfr], meta_regressor=lr)
print('3-fold cross validation:\n')
for clf, label in zip([grid_search, abr, rfr, sclf], 
                      ['grid_search', 
                       'abr', 
                       'rfr',
                       'StackingClassifier']):
    scores = cross_val_score(clf, X, y)

 
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
sclf.fit(X_train,y_train)
predictions=sclf.predict(X_test)


# In[370]:


train_sizes,train_score,test_score = learning_curve(sclf,X,y,train_sizes=[0.1,0.2,0.4,0.6,0.8,1],cv=3)
train_error =  1- np.mean(train_score,axis=1)
test_error = 1- np.mean(test_score,axis=1)
plt.plot(train_sizes,1-train_error,'o-',color = 'r',label = 'training')
plt.plot(train_sizes,1-test_error,'o-',color = 'g',label = 'testing')
plt.legend(loc='best')
plt.xlabel('traing examples')
plt.ylabel('Accuracy')
plt.show()

