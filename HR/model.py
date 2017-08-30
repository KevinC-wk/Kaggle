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

#First thing is to plot the heatmap of the correlation matrix to see
#how each feature is related
corrmat = hr_data.corr()
sns.heatmap(corrmat)
sns.plt.title('Heatmap of Correlation Matrix')

#From what we can see here, the satisfaction and turnover rate are negatively 
#correlated. This makes sense as if an employee is satisfied with their 
#position they are less inclined to leave. One investigation we can pursue
#are the factors that lead to an employee satisfaction.
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15,6))
sns.axes_style('darkgrid')
sns.distplot(hr_data['satisfaction'], kde=False, color='r', ax=ax1).set_title('Satisfaction Distribution')
sns.distplot(hr_data['evaluation'], kde=False, color='m', ax=ax2).set_title('Evaluation Distribution')
sns.distplot(hr_data['averageMonthlyHours'], kde=False, color='b', ax=ax3).set_title('Monthly Hours Distribution')

#From what see see here, there is a decent population who are unsatisfied with thier job
#and there is a spike in satisfcations.
#We can also see that there is a large amount of employees working overtime,
#this could also lead to dissatisifiation in the job at hand

#Lets see which departments have the highest turnover
sns.countplot(y='department', hue='turnover', data=hr_data).set_title('Department Turnover Distribution')
#From here we see that sales, technical, and support, have the highest turnover
#rate, wheeras management has the lowest. The question at hand is can we pinpoint
#the causes of their turnover

#Compensation
#Are the departments being compensated accordingly for their time
sns.countplot(y='salary', hue='turnover', data=hr_data).set_title('Salary Turnover Distribution')
sns.countplot(y='department', hue='salary', data=hr_data)

sns.boxplot(x='salary', y='yearsAtCompany', hue='turnover', data=hr_data)


#Lets see how many projects each department works on
sns.boxplot(x='projectCount', y='averageMonthlyHours', hue='turnover', data=hr_data)

sns.boxplot(x='projectCount', y='evaluation', hue='turnover', data=hr_data)

#Years vs satisfcation 
sns.boxplot(x='yearsAtCompany' y='satisfaction', hue='turnover', data=hr_data)


#Turnover vs monthly hours
fig = plt.figure(figsize=(10,6))
ax = sns.kdeplot(hr_data.loc[(hr_data['turnover'] == 0), 'averageMonthlyHours'], shade=True, label= 'no turnover')
ax = sns.kdeplot(hr_data.loc[(hr_data['turnover'] == 1), 'averageMonthlyHours'], shade=True, label= 'turnover')
plt.title('Density Distribution of Average Monthly Hours, (Turnover vs No Turnover)')
sns.lmplot(x='evaluation', y='satisfaction', data=hr_data, hue='turnover', fit_reg=False)




