import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('housing.csv')
print(data)
data.info()
data.dropna(inplace=True)
data.info()
from sklearn.model_selection import train_test_split
x = data.drop(['median_house_value'],axis=1)
y = data ['median_house_value']
from sklearn.model_selection import train_test_split
import pandas as pd
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
train_data = pd.concat([x_train, y_train], axis=1)

train_data.hist(figsize=(15,8))
plt.show()
numeric_train_data = train_data.select_dtypes(include=[np.number])
correlation_matrix = numeric_train_data.corr()
plt.figure(figsize=(15,8))
sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu")
plt.show()
train_data['total_rooms'] = np.log(train_data['total_rooms'] + 1)
train_data['total_bedrooms'] = np.log(train_data['total_bedrooms'] + 1)
train_data['population'] = np.log(train_data['population'] + 1)
train_data['households'] = np.log(train_data['households'] + 1)
train_data.hist(figsize=(15,8))
plt.show()
train_data = train_data.join(
    pd.get_dummies(train_data.ocean_proximity).astype(int)
).drop(['ocean_proximity'], axis=1)
plt.figure(figsize=(18,10))
sns.heatmap(train_data.select_dtypes(include=[np.number]).corr(), annot=True, cmap="YlGnBu")
plt.show()
plt.figure(figsize=(15,8))
sns.scatterplot(x="latitude", y="longitude", data=train_data, hue="median_house_value", palette="coolwarm")
plt.show()
train_data['bedroom_ratio'] = train_data['total_bedrooms'] / train_data['total_rooms']
train_data['household_rooms'] = train_data['total_rooms'] / train_data['households']
plt.figure(figsize=(15,8))
sns.heatmap(train_data.select_dtypes(include=[np.number]).corr(), annot=True, cmap="YlGnBu")
plt.show()
#  LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train, y_train = train_data.drop(['median_house_value'], axis=1),train_data['median_house_value']
x_train_s = scaler.fit_transform(x_train)
reg = LinearRegression()
reg.fit(x_train_s, y_train)
test_data = x_test.join(y_test)

test_data['total_rooms'] = np.log(test_data['total_rooms'] + 1)
test_data['total_bedrooms'] = np.log(test_data['total_bedrooms'] + 1)
test_data['population'] = np.log(test_data['population'] + 1)
test_data['households'] = np.log(test_data['households'] + 1)

test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity)).drop(['ocean_proximity'], axis=1)
y_test = test_data['median_house_value']
test_data = test_data.reindex(columns=train_data.drop(['median_house_value'], axis=1).columns, fill_value=0)
test_data['bedroom_ratio'] = test_data['total_bedrooms'] / test_data['total_rooms']
test_data['household_rooms'] = test_data['total_rooms'] / test_data['households']
x_test = test_data
x_train_s = scaler.transform(x_test)
reg_score = reg.score(x_train_s, y_test)
print("Linear Regression R^2 score:", reg_score)
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor()
forest.fit(x_train, y_train)
forest_score = forest.score(x_test, y_test)
print("Random Forest R^2 score:", forest_score)
