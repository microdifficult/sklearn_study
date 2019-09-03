from feature_selector import FeatureSelector
import pandas as pd
import numpy as np

# numpy and pandas for data manipulation
import numpy as np

# model used for feature importances
import lightgbm as lgb

# utility for early stopping with a validation set
from sklearn.model_selection import train_test_split

# visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# memory management
import gc

# utilities
from itertools import chain



#Example Dataset

train = pd.read_csv('C:/Users/Administrator/Scikit_learn/feature-selector-master/credit_example.csv')
train_labels = train['TARGET']

#pd.head，默认查看前五行数据。因为train已经是pd因此直接.head即可
print(train.head())    

#pandas.drop的用法，删除Target列
train = train.drop(['TARGET'],axis=1)     

#对于pandas，行标为index，列表为columns
#如常用df = pd.DataFrame(np.random.randn(5,3),index = list('abcde'),columns = ['one','two','three'])

#Create the Instance
fs = FeatureSelector(data = train, labels = train_labels)




#   1   Missing Values

fs.identify_missing(missing_threshold=0.6)


#The features identified for removal can be accessed through the ops dictionary of the FeatureSelector object.
missing_features = fs.ops['missing']
print(missing_features[:20])

fs.plot_missing()         #在每一个画图的后面加上plt.show即可
plt.show()
print(fs.missing_stats.head(20))




#   2   Single Unique Value



fs.identify_single_unique()

single_unique = fs.ops['single_unique']
print(single_unique)


fs.plot_unique()         #画图都不好用
plt.show()
print(fs.unique_stats.sample(5))



#   3   Collinear (highly correlated) Feature

fs.identify_collinear(correlation_threshold=0.975)
correlated_features = fs.ops['collinear']
correlated_features[:5]

fs.plot_collinear()
plt.show()

fs.plot_collinear(plot_all=True)
plt.show()

fs.identify_collinear(correlation_threshold=0.98)
fs.plot_collinear()
plt.show()


print(fs.record_collinear.head())




#   4. Zero Importance Features：one-hot coding 主要用于0相关性特征的识别

fs.identify_zero_importance(task = 'classification', eval_metric = 'auc', 
                            n_iterations = 10, early_stopping = True)
one_hot_features = fs.one_hot_features    #one-hot哑变量
base_features = fs.base_features           #原有变量
print('There are %d original features' % len(base_features))
print('There are %d one-hot features' % len(one_hot_features))
#There are 121 original features
#There are 134 one-hot features

print(fs.data_all.head(10))
#加一起就是255个变量


zero_importance_features = fs.ops['zero_importance']
print(zero_importance_features[1:15])

fs.plot_feature_importances(threshold = 0.99, plot_n = 12)
plt.show()

print(fs.feature_importances.head(10))


one_hundred_features = list(fs.feature_importances.loc[:99, 'feature'])
len(one_hundred_features)




#    5. Low Importance Features


# When using this method, we must have already run identify_zero_importance and 
# need to pass in a cumulative_importance that accounts for that fraction of total feature importance.

fs.identify_low_importance(cumulative_importance = 0.99)

low_importance_features = fs.ops['low_importance']
print(low_importance_features[:5])

fs.plot_feature_importances(threshold = 0.99, plot_n = 12)
plt.show()




# 6   Removing Features

# Removing Features:    This method returns the resulting data which we can then use for machine learning. 
#                       The original data will still be accessible in the data attribute of the Feature Selector.

train_no_missing = fs.remove(methods = ['missing'])    #以鉴别17种
train_no_missing_zero = fs.remove(methods = ['missing', 'zero_importance'])   #已经鉴别66+17=83种



all_to_remove = fs.check_removal()    #检查所有要删除的features
print(all_to_remove[0:])


train_removed = fs.remove(methods = 'all')   #删除所有的不好的features


# 7   Handling One-Hot Features

train_removed_all = fs.remove(methods = 'all', keep_one_hot=False)

print('Original Number of Features', train.shape[1])
print('Final Number of Features: ', train_removed_all.shape[1])




# 8 Alternative Option for Using all Methods ：一个命令全部做完
fs = FeatureSelector(data = train, labels = train_labels)

fs.identify_all(selection_params = {'missing_threshold': 0.6, 'correlation_threshold': 0.98, 
                                    'task': 'classification', 'eval_metric': 'auc', 
                                     'cumulative_importance': 0.99})
train_removed_all_once = fs.remove(methods = 'all', keep_one_hot = True)

fs.feature_importances.head()

#变量数可能不对应的原因，由于0相关和弱相关的数量在变化因此，可能会有偏差
#  There is a slight discrepancy between the number of features removed because the feature importances have changed.
#                         The number of features identified for removal by the missing, single_unique, and collinear will stay the same because they are deterministic, 
#                         but the number of features from zero_importance and low_importance may vary due to training a model multiple times.