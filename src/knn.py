import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pprint import pprint

PASSENGER_ID = 'PassengerId'
SURVIVED = 'Survived'
PCLASS = 'Pclass'
NAME = 'Name'
SEX = 'Sex'
AGE = 'Age'
SIBLINGS_SPOUSE = 'SibSp'
PARCH = 'Parch'
TICKET = 'Ticket'
FARE = 'Fare'
CABIN = 'Cabin'
EMBARKED = 'Embarked'


dataset = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
pprint(dataset.head())  # return first 5 rows (default 5)
print(dataset.dtypes)
"""
[5 rows x 12 columns]
PassengerId      int64
Survived         int64
Pclass           int64 --> might be useful
Name            object
Sex             object --> might be useful
Age            float64 --> might be useful
SibSp            int64 --> might be useful
Parch            int64 --> might be useful
Ticket          object --> might be useful
Fare           float64 --> might be useful
Cabin           object --> might be useful
Embarked        object --> might be useful
"""

# print statistics data
print(dataset.describe())

"""
The MEAN of Survived is 0.383838 => there were 38.38% people survived

       PassengerId    Survived      Pclass  ...       SibSp       Parch        Fare
count   891.000000  891.000000  891.000000  ...  891.000000  891.000000  891.000000
mean    446.000000    0.383838    2.308642  ...    0.523008    0.381594   32.204208
std     257.353842    0.486592    0.836071  ...    1.102743    0.806057   49.693429
min       1.000000    0.000000    1.000000  ...    0.000000    0.000000    0.000000
25%     223.500000    0.000000    2.000000  ...    0.000000    0.000000    7.910400
50%     446.000000    0.000000    3.000000  ...    0.000000    0.000000   14.454200
75%     668.500000    1.000000    3.000000  ...    1.000000    0.000000   31.000000
max     891.000000    1.000000    3.000000  ...    8.000000    6.000000  512.329200
"""

# Feature engineering

print(dataset.shape)  # (891, 12) -> 891 passengers, 12 attributes

# leverage chart to decide which column is useful

survived_m = dataset.Survived[dataset.Sex == 'male'].value_counts()
survived_f = dataset.Survived[dataset.Sex == 'female'].value_counts()

df = pd.DataFrame({'male': survived_m, 'female': survived_f})
df.plot(kind='bar', stacked=True)  # stack on
plt.title('Survivor by sex')
plt.xlabel(SURVIVED)
plt.ylabel('count')
plt.show()

dataset.Age.hist()  # histogram
plt.ylabel("Number")
plt.xlabel(AGE)
plt.title('Age distribution')
plt.show()

dataset.Age[dataset.Survived == 1].hist()  # histogram
plt.ylabel("Number")
plt.xlabel(AGE)
plt.title('Survivor age distribution')
plt.show()

dataset.Age[dataset.Survived == 0].hist()  # histogram
plt.ylabel("Number")
plt.xlabel(AGE)
plt.title('Victim age distribution')
plt.show()

# Fare
dataset.Fare.hist()  # histogram
plt.ylabel("Number")
plt.xlabel(FARE)
plt.title('Fare distribution')
plt.show()

dataset.Fare[dataset.Survived == 1].hist()  # histogram
plt.ylabel("Number")
plt.xlabel(FARE)
plt.title('Survivor fare distribution')
plt.show()

dataset.Fare[dataset.Survived == 0].hist()  # histogram
plt.ylabel("Number")
plt.xlabel(FARE)
plt.title('Victim age distribution')
plt.show()

Survived_S = dataset.Survived[dataset[EMBARKED] == 'S'].value_counts()
Survived_C = dataset.Survived[dataset[EMBARKED] == 'C'].value_counts()
Survived_Q = dataset.Survived[dataset[EMBARKED] == 'Q'].value_counts()

df = pd.DataFrame({'S': Survived_S, 'C': Survived_C, 'Q': Survived_Q})
df.plot(kind='bar', stacked=True)
plt.title("Survivor by Embarked")
plt.xlabel("Survival")
plt.ylabel("count")
plt.show()

# pclass, exe, age, fare, embarked is useful, reserved
label = dataset['Survived']
data = dataset[[PCLASS, SEX, AGE, FARE, EMBARKED]]
test_data = test_data[[PCLASS, SEX, AGE, FARE, EMBARKED]]

print(data.shape)
print(data)


# process NaN data
def fill_nan(data):
    data_copy = data.copy(deep=True)
    data_copy[AGE] = data_copy[AGE].fillna(data_copy[AGE].median())
    data_copy[FARE] = data_copy[FARE].fillna(data_copy[FARE].median())
    data_copy[PCLASS] = data_copy[PCLASS].fillna(data_copy[PCLASS].median())
    data_copy[SEX] = data_copy[SEX].fillna('male')
    data_copy[EMBARKED] = data_copy[EMBARKED].fillna('S')
    return data_copy


data_no_nan = fill_nan(data)
test_data_no_nan = fill_nan(test_data)
print("data.isnull().values.any(): " + str(data.isnull().values.any()))  # True if there is NaN value
print("data_no_nan.isnull().values.any(): " + str(data_no_nan.isnull().values.any()))


# Process string data
def convert_sex_to_int(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[data_copy['Sex'] == 'female', 'Sex'] = 0
    data_copy.loc[data_copy['Sex'] == 'male', 'Sex'] = 1
    return data_copy


data_after_conversion = convert_sex_to_int(data_no_nan)
test_data_after_conversion = convert_sex_to_int(test_data_no_nan)
print("Data after conversion sex to int:")
print(data_after_conversion)


def convert_embarked_to_int(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[data_copy['Embarked'] == 'S', 'Embarked'] = 0
    data_copy.loc[data_copy['Embarked'] == 'C', 'Embarked'] = 1
    data_copy.loc[data_copy['Embarked'] == 'Q', 'Embarked'] = 2
    return data_copy


data_after_conversion = convert_embarked_to_int(data_after_conversion)
test_data_after_conversion = convert_embarked_to_int(test_data_after_conversion)
print("Test data after conversion:")
print(test_data_after_conversion)
# End of Feature engineering

#  Training: Create Training data set and validation data set
X_train_origin = data_after_conversion
y_train_origin = label
X_test = test_data_after_conversion

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train_origin, y_train_origin, random_state=0, test_size=0.2)
print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

k_range = range(1, 51)  # just try 50 now
k_scores = []
max_score = 0
best_k = 0
for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)  # train model
    pred = model.predict(X_valid)
    score = accuracy_score(y_valid, pred)
    if score > max_score:
        max_score = score
        best_k = k
    print('k = {}, score = {}'.format(k, score))
    k_scores.append(score)

print("[max_score:best_k] = " + str(max_score) + ":" + str(best_k))
plt.plot(k_range, k_scores)
plt.xlabel('K')
plt.ylabel('Accuracy on validation set')
plt.show()
print(np.array(k_scores).argsort())


# Test
k = 33
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train_origin, y_train_origin)
result = model.predict(X_test)

print("result:")
print(str(result))  # Save and upload this result to Kaggle!
