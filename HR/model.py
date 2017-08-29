#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

hr_data = pd.read_csv('HR_comma_sep.csv')
hr_data.isnull().sum()

#So now we know that no columns are empty we are not missing any data

hr_data.head()
hr_data.info()

#Lets rename some variable for readiblity and easy access
names = {'satisfaction_level' : 'satisfaction',
         'last_evaluation': 'evaluation',
         'number_project': 'projectCount',
         'average_montly_hours' : 'averageMonthlyHours',
         'time_spend_company' : 'yearsAtCompany',
         'Work_accident' : 'accident',
         'promotion_last_5years': 'promotion',
         'left': 'turnover',
         'sales': 'department'}
hr_data.rename(columns=names, inplace=True)

#Moving the turnover column to the very end
cols = list(hr_data)
cols.insert(9, cols.pop(cols.index('turnover')))
hr_data = hr_data.loc[:, cols]

#First of all let's explore the amount of turnover, we have in our dataset

turnoverStats = hr_data['turnover'].value_counts() / 14999
turnoverStats

#76% of people in decided to stay, whereas 24% left. The question I am hoping
#to find more information about is what is influencing the 24% to leave

#First thing is to plot the heatmap 
corrmat = hr_data.corr()
sns.heatmap(corrmat)
sns.plt.title('Heatmap of Correlation Matrix')

#From what we can see here, the satisfaction and turnover rate are negatively 
#correlated. This makes sense as if an employee is satisfied with their 
#position they are less inclined to leave. One investigation we can pursue
#are the factors that lead to an employee satisfaction.
#Alos those who have accidents are more likey to leave their position
#Projectcount and number of hours spent are correlated to the evaluation

fig, axes = plt.subplot(2,3, figsize=(10,5))

