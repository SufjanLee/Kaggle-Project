# Kaggle-Project

## content
1.House Price Prediction

---

## 1. House price prediction
### Process Details:
### 1. Show the overall report
`pandas_profiling` is used for an overall report of the dataset
```
import pandas as pd
import pandas_profiling
df = pd.read_csv("")
df.profile_report()
```
we see **32 numeric attributes**, **48 categorical attributes** and only **1 boolean attribute**. And the report also show the **missing values** and **zero values**, which are super convenient for further data preprocessing process.
### 2. Do Preprocessing
I decided to drop the attribute that most of values are zeros or null. 
```
# delete the attribute with zeros and null values.
df_new = df.drop(['Alley', '3SsnPorch', 'BsmtFinSF2', 'EnclosedPorch', 'Fence', 'FireplaceQu', 'Id', 'LowQualFinSF', 'MasVnrArea', 'MiscFeature', 'MiscVal', 'PoolQC', 'YrSold'], axis=1)
```
To get the more reliable result from the data. I use ***correlation matrix (Pearson Correlation)*** to find the attributes that may influence the Price.
But the correlation matrix **excludes** the categorical attributes, so I need to use `preprocessing.LabelEcoder()` to transfer the label first.

In pandas, the **dtypes** of categorical attribute is **"object"**.
```
# select 'Categorical attributes'
cat_ls = []  # store categorical attribute name in list
for i in list(df_new.columns.values):  # get the attribute name in DataFrame
    if df["{}".format(i)].dtypes == 'object':
        cat_ls.append(i)
```
Using more frequently value to fill the null values if the attributes are categorical. 
`isnull` returns boolean value, use `.sum()` to count the total numbers. `fillna()` is used for filling null values. `index(0)` is the most frequently value.
```
# 'test code': get the most frequently value in an attribute.
df_new['MSZoning'].value_counts().index[0]
# output: "RL"
```
```
# use the most frequently value instead of null.
for i in cat_ls:
    if df_new[i].isnull().sum() > 0:
        df_new[i].fillna(df_new[i].value_counts().index[0])
    else:
        continue
```
If the attributes are **numeric type**, fill null value with **mean values**. 
```
# fill null with mean values.
df_new['GarageArea'].fillna(df_new['GarageArea'].mean())
print(df_new['GarageArea'].mean())
df_new['GarageArea'].isnull().sum()
```
According to the **correlation matrix**, I select the value **greater than** +0.6 and *less than* -0.6. 
```
# plot corr matrix
corrmat = df_new.corr()
f, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(corrmat, vmax=0.8, square=True)

# show the corr list.
# 对比Pearson R and filter range [-0.6 - 0.6]
corr = df_new.corr()
corr_price = corr["SalePrice"]
corr_price[(corr_price>=0.6) | (corr_price<=-0.6)]
```
return a selected correlation list as follows:
```
OverallQual    0.790982
ExterQual     -0.636884
BsmtQual      -0.620886
TotalBsmtSF    0.613581
1stFlrSF       0.605852
GrLivArea      0.708624
GarageCars     0.640409
GarageArea     0.623431
SalePrice      1.000000
Name: SalePrice, dtype: float64
```
I find that ***"GarageArea"*** and ***"GarageCars"*** are talking the same feature, and also ***"1stFlrSF"*** and ***"GrLivArea"*** are the same. So just need contain one respectively. According to the Pearson Corrlation value, I finally keep "GarageCars" and "GrLivArea".
```
# 删除"GarageArea", "1stFlrSF"
df1 = df_new[["OverallQual","GrLivArea", "GarageCars", "TotalBsmtSF" , "ExterQual", "BsmtQual", "SalePrice"]]
```
To make sure the attributes are reliable, I plot the data with boxplot.
```
figs, axes = plt.subplots(6, 1, figsize=(18, 35))
sns.boxplot(x=x1, y=df1['SalePrice'], data=df1, ax=axes[0])
sns.boxplot(x=x2, y=df1['SalePrice'], data=df1, ax=axes[1])
sns.boxplot(x=x3, y=df1['SalePrice'], data=df1, ax=axes[2])
sns.boxplot(x=x4, y=df1['SalePrice'], data=df1, ax=axes[3])
sns.boxplot(x=x5, y=df1['SalePrice'], data=df1, ax=axes[4])
sns.boxplot(x=x6, y=df1['SalePrice'], data=df1, ax=axes[5])
```
### 3. Build a model
first, split data into the attributes(x_train) and target(y_train).
And seperated into two parts, train set(70%) and test set(30%).
Before we build the model, we need to standardize the data first.
```
# standard data
scaler = StandardScaler()
```
train datset need `fit_transform()`.
```
x_train_scaled = scaler.fit_transform(x_train)
```
test dataset only need `transform()`.
```
x_test_scaled = scaler.transform(x_test)
```
Because this is a regression problem, So I use **SVR**, **LinearRegression**, **RandomForestRegression**, **BayesianRidge** to model the data. 
```
estimators = {
    "SVR": SVR(),
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=500),
    "BayesianRidge": BayesianRidge()
}
```
And use **'MSE'** to evaluate the performance on test set. Due to the result, I choose `RandomForestRegressor` as my model.
```
OUTPUT:
SVR_Precision: -0.05701409359496833
MSE: 374429209.83862865
LinearRegression_Precision: 0.7181594934328378
MSE: 10664847.893057393
RandomForestRegressor_Precision: 0.8174083989281499
MSE: 8364557.191390596
BayesianRidge_Precision: 0.7185297306522369
MSE: 10646310.284484206
```
