import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv('IRIS.csv')
# print(df.head()) //sepal_length,sepal_width,petal_length,petal_width,species
# print(df.isnull().sum()) // No Null values
# print(df.info()) // all numeric expect species
# print(df['species'].nunique()) // 3 unique 


# numeric_df = df.drop(columns=['species'], errors='ignore')
# plt.figure(figsize=(10, 6)) 
# plt.boxplot(numeric_df.values, labels=numeric_df.columns)
# plt.title('Boxplot of Iris Features')
# plt.ylabel('Values (cm)')
# plt.show()


# plt.figure(figsize=(10, 6)) 
# plt.hist(numeric_df.values, label=numeric_df.columns, alpha=0.7, bins=15)
# plt.title('Histogram Distribution of Iris Features')
# plt.xlabel('Values (cm)')
# plt.ylabel('Frequency')
# plt.legend()
# plt.show()

# Encoding  categorical to numerical encoded data
df['species'] = df['species'].map({
    'Iris-setosa': 0, 
    'Iris-versicolor': 1, 
    'Iris-virginica': 2
})

# print(df.sample())

# IQR capping to eliminate Outliers
features = df.columns[:-1] 

for col in features:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    df[col] = np.clip(df[col], lower_bound, upper_bound)

def train_test_split_scratch(df, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = df.index.tolist()
    np.random.shuffle(indices)
    
    split_index = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[indices[:split_index]]
    test_df = df.iloc[indices[split_index:]]
    
    features = df.columns[:-1]
    
    X_train = train_df[features].values
    X_test = test_df[features].values
    y_train = train_df['species'].values
    y_test = test_df['species'].values
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split_scratch(df, test_size=0.2, random_state=42)

def min_max_scale_scratch(X_train, X_test):
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    
    denominator = np.where(x_max - x_min == 0, 1, x_max - x_min)

    X_train_scaled = (X_train - x_min) / denominator
    X_test_scaled = (X_test - x_min) / denominator
    
    return X_train_scaled, X_test_scaled

X_train_scaled, X_test_scaled = min_max_scale_scratch(X_train, X_test)


train_data = np.hstack((X_train_scaled, y_train.reshape(-1, 1)))
test_data = np.hstack((X_test_scaled, y_test.reshape(-1, 1)))

columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

train_df = pd.DataFrame(train_data, columns=columns)
test_df = pd.DataFrame(test_data, columns=columns)

train_df.to_csv("iris_train.csv", index=False)
test_df.to_csv("iris_test.csv", index=False)