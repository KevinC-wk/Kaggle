#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Getting more Information on the data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train.set_index(['PassengerId'],inplace=True)
test.set_index(['PassengerId'], inplace=True)
titanic_data = pd.concat([train, test])
train.head()
print('------------------------------------')
train.info()

#Feature Engineering
#From the info we can tell that the Age, Embarked, and Cabin values are missing
#Intuiton tells me that passengerclass and survival rate may be correlated
train.groupby('Pclass').Survived.mean().plot(kind='bar')
#As expected we see that the ones who have a higher passenger class have a 
#higher chance of survival. Let's investigate the name column 

titanic_data['Surname'] = titanic_data['Name'].str.extract('([A-Za-z]+)\.', 
            expand = False)

#Lets take a look at the surnamesof the passengers
titanic_data['Surname'].value_counts()
#We see here that there are multiple unique names. Let's narrow down the names
#into categories and replace them.
rare = ['Dr', 'Rev', 'Col', 'Major', 'Capt', 'Lady', 'Don', 'Master',
              'Jonkheer', 'Countess', 'Dona']

titanic_data['Surname'] = titanic_data['Surname'].replace(to_replace = rare,
             value = 'Rare')

titanic_data['Surname'][titanic_data['Surname'] == 'Ms'] = 'Miss' 
titanic_data['Surname'][titanic_data['Surname'] == 'Mlle'] = 'Miss'
titanic_data['Surname'][titanic_data['Surname'] == 'Mme'] = 'Mrs'
titanic_data['Surname'][titanic_data['Surname'] == 'Sir'] = 'Mr'

titanic_data['Surname'].value_counts()

#So now we broke down the names into categories, let's see if title and survival
#rate are related.

train = titanic_data.loc[1:891]
sns.factorplot('Surname','Survived', hue='Pclass', data=train)

#From what can notice here, is that most married women of class 1 and 2 suriveved
#People of class 3 did not survive much, but surname and class have a huge 
#factor of debating who survived. Since married woman have a higher chance
#of surviving, lets explore if families have a higher chance.

titanic_data['Fsize'] = titanic_data['Parch'] + titanic_data['SibSp'] + 1
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 
            'female': 1})
#Family Size, plus one for the current passenger

train = titanic_data.loc[1:891]
test = titanic_data.loc[892:]
sns.countplot(x='Fsize', hue='Survived', data=train)
#Most individuals aboard ship sadly, ended up falling, but average values for
#families seemed to have survived the ship. However as the size of the family
#increases their chances of survival decrease

#Now lets explore the Embarked Column as it is missing 2 values
train[train['Embarked'].isnull()]
#It seems that they share a common fare and class, and which they both survived
#Lets plot the fare and class together


plot = sns.boxplot(x='Embarked', y='Fare', hue='Pclass', data=train)
#Looking at the box plot the median fare for embarkment Value C is 80$ for the
#Pclass of value 1. I beleive we can put the N/A values as 'C' for embarkment
train['Embarked'] = train['Embarked'].fillna('C')

#Now Since we have two categorial variables we need to create dummy variables
#for them.

train = pd.get_dummies(train, columns=['Embarked', 'Surname'], drop_first=True)
train_y = train['Survived']
train_X = train.drop(['SibSp', 'Survived', 'Ticket'], axis=1)
test = pd.get_dummies(test, columns= ['Embarked', 'Surname'], drop_first = True)
test_X = test.drop(['SibSp', 'Survived', 'Ticket'], axis=1)

#Now to create the model and to predict
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy',
                                    random_state = 0)
classifier.fit(train_X.iloc[:, 5:],train_y)

#Prediction
y_pred = classifier.predict(test_X.iloc[:, 5:])

prediction = pd.DataFrame(data=y_pred, columns = ['Survived'])
prediction['PassengerId'] = test_X.index.values

submission = prediction.to_csv("submission.csv")













