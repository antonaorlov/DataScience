import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

#open file name
file_name='NYPD_Complaint_Data_Historic.csv'
df=pd.read_csv(file_name)
#print(df.head())
#renaming all columns to be lowercase
df.rename(columns=str.lower,inplace=True)
#print(df.head())
#checking if there are duplicates found
duplicates=df.columns.duplicated()
if duplicates.any():
    print("there are duplicates")
else: 
    print("none")
    
#convert date time columns to datetime objects
df['cmplnt_fr_dt']=pd.to_datetime(df['cmplnt_fr_dt'], errors='coerce')
df['cmplnt_fr_tm'] = pd.to_datetime(df['cmplnt_fr_tm'], format='%H:%M:%S').dt.time
#check current data types

current_data=df[['addr_pct_cd', 'ky_cd', 'pd_cd']].dtypes
for col in ['addr_pct_cd', 'ky_cd', 'pd_cd']:
   if pd.api.types.is_float_dtype(df[col]):
       if df[col].notnull().all():
           df[col]=df[col].astype(int)
update_data=df[['addr_pct_cd', 'ky_cd', 'pd_cd']].dtypes
#print(current_data,update_data, df[['cmplnt_fr_dt','cmplnt_fr_tm']].head())


#removing null values
df=df[df['boro_nm']!='(null)']
df=df[df['susp_race']!='(null)']
df=df[df['ofns_desc']!='(null)']

#removing D, E, L from victom sex description since there is no info on what these values represent
df=df[~df['vic_sex'].isin(['D','E','L'])]
remanining_vic_sex=df['vic_sex'].unique()

#dropping unecesary columns 
df=df.drop('cmplnt_num',axis=1)
#print(df.head(3))

#create bar chart of crime frequencies by borough
counts_of_crime=df['boro_nm'].value_counts().reset_index()
counts_of_crime.columns=['Borough','Crime Count']
plt.figure(figsize=(8,8))
sns.barplot(x='Borough',y='Crime Count', hue='Borough', data=counts_of_crime, palette='viridis', legend=False)
plt.title('Crime Frequencies by Borough')
plt.xlabel('Borough', fontsize=10)
plt.ylabel('Number of Crimes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



#creating heatmap showing concentration of different type sof crime across different times of day
#Getting time of day splitting to Morning: 5am-11:59am, Afternoon: 12pm:4:59pm, Evening: 5pm:8:59pm, Night: 9pm:4:59am

def time_of_day(time):
    if time>=datetime.time(5,0) and time< datetime.time(12,0):
        return 'Morning'
    elif time>=datetime.time(12,0) and time < datetime.time(17,0):
        return 'Afternoon'
    elif time>=datetime.time(17,0) and time < datetime.time(21,0):
        return 'Evening'
    else:
        return 'Night'
#making new time_of_day column
df['time_of_day']=df['cmplnt_fr_tm'].apply(time_of_day)
#group ofns_desc and time_of_day column count occurence
crime_time_heatmap=df.groupby(['ofns_desc', 'time_of_day']).size().unstack(fill_value=0)
#get frequenzie better color shading
crime_time_heatmap=crime_time_heatmap.apply(lambda x:x/x.sum(),axis=1)
#reorder table columns
time_order=['Morning','Afternoon','Evening','Night']
crime_time_heatmap=crime_time_heatmap.reindex(columns=time_order)
#plot heatmap
plt.figure(figsize=(10,12))
sns.heatmap(crime_time_heatmap, annot=True, fmt=".2f", cmap='viridis')
plt.title('Heatmap of Crime Type by Time of Day', fontsize=12)
plt.ylabel('Type of Crime', fontsize=12)
plt.xlabel('Time of Day',fontsize=12)
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()



df['day_of_week'] = df['cmplnt_fr_dt'].dt.day_name()
crime_by_day=df.groupby('day_of_week').size().reset_index(name='crime_count')

#line plot
df=df[(df['cmplnt_fr_dt'].dt.year>=2020) & (df['cmplnt_fr_dt'].dt.year<=2023)]
montly_crimes=df.set_index('cmplnt_fr_dt').resample('M').size()
plt.figure(figsize=(14,7))
plt.plot(montly_crimes.index, montly_crimes, marker='o', linestyle='-', color='blue')
plt.title('Montly Count Crime from 2020 to 2023')
plt.xlabel('Month')
plt.ylabel('Number of Crimes')
plt.grid(True)
plt.show()

#Training model
#encoding categorical features
df=df.drop(columns=['cmplnt_fr_dt', 'cmplnt_fr_tm'])
nonfeature=['boro_nm','addr_pct_cd']
feature=[col for col in df.columns if col not in nonfeature]
le=LabelEncoder()
for col in feature:
    df[col]=le.fit_transform(df[col])
#encode=pd.get_dummies(df,columns=feature, drop_first=True, sparse=True)
X=df.drop(columns=nonfeature)
#df=pd.get_dummies(df,columns=['ky_cd','ofns_desc'])
#X=df.drop(columns=['boro_nm','addr_pct_cd'])
y=df['boro_nm'] #df['ADDR_PCT_CD] next
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
#Forest classsifier
model = RandomForestClassifier(
    n_estimators=10,
    max_depth=10,
    min_samples_split=50,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    )
model.fit(X_train, y_train)
predict=model.predict(X_test)
cm=confusion_matrix(y_test, predict)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
