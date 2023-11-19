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
import folium 
from folium.plugins import TimestampedGeoJson


#file_name='NYPD_Complaint_Data_Historic.csv'

def prepare_data(file_name):
    """
    Reads CSV file, renames all columns to lowercase, checks for duplicate column,
    removes null values of columns, dropping unecesary columns, 
    removing D, E, L from victom sex description 
    since there is no info on what these values represent

    :param file_name: file containing OpenData NYPD complaint Data
    
    :return: DataFrame
    """
    #read file name
    df=pd.read_csv(file_name)
    #rename columns names to lowercase for easy access
    df.rename(columns=str.lower,inplace=True)
    #check for duplicates if found remove them
    duplicates=df.columns.duplicated()
    if duplicates.any():
        print("there are duplicates")
        df=df.loc[:, ~duplicates]
    else: 
        print("no duplicates found")
    
    # #removing null values
    df=df[df['boro_nm']!='(null)']
    df=df[df['susp_race']!='(null)']
    df=df[df['ofns_desc']!='(null)']
    
    # #dropping unecesary columns 
    df=df.drop(['cmplnt_num','pd_desc','crm_atpt_cptd_cd', 'law_cat_cd', 'susp_age_group', 'susp_sex', 'lat_lon', 'patrol_boro', 'vic_age_group', 'vic_race','zip codes'], axis=1)
    
    #removing D, E, L from victom sex description since there is no info on what these values represent
    df=df[~df['vic_sex'].isin(['D','E','L'])]
    remanining_vic_sex=df['vic_sex'].unique()
    

    return df



def filter_by_date(df,start_year,end_year):
    """
    Converts 'cmplnt_fr_dt' to datetime, filters Dataframe by given date range,
    and converts 'cmplnt_fr_tm' to time.
    
    #param df: Dataframe containing NYPD Complaint Data
    #param start_year: starting year for date range
    #param end_year: Ending year for data range
    
    return Filtered DataFrame

    """
    #convert 'cmp;nt_fr_dt' to datetime
    df['cmplnt_fr_dt']=pd.to_datetime(df['cmplnt_fr_dt'], errors='coerce', format='%m/%d/%Y')
    #filter data frame by specific date range
    filtered = df[(df['cmplnt_fr_dt'].dt.year>=start_year) & (df['cmplnt_fr_dt'].dt.year<=end_year)]
    #convert 'cmplnt_fr_tm' to time
    filtered['cmplnt_fr_tm'] = pd.to_datetime(filtered['cmplnt_fr_tm'], errors='coerce', format='%H:%M:%S').dt.time

    return filtered

def convert_float_to_int(df,columns):
    """
    Converts specified float columns to integers containing no null values

    #param df: Dataframe to be processes
    #param columns: List of columns names to converted from float to int
    
    #return: DataFrame with updated column types
    """
    for col in columns:
        if pd.api.types.is_float_dtype(df[col]):
            if df[col].notnull().all():
                df[col]=df[col].astype(int)
    return df
    
    
def plot_crime_frequencies_by_borough(df,borough):
    """
    Creates bar chart of crime frequencies by borough

    #param df: Dataframe containing NYPD Complaint Data
    #param borough: name of column representing boroughs
    """
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
    
def plot_monthly_crime_trends(df,date_col):
    """
    Creates line plot of monthly crime counts within a sepecified date range

    #param df: DataFrame containing NYPD Complaint Data
    #param date_col: Name of column representing dates
    """
    #filter DataFrame for specified date range
    df=df[(df['cmplnt_fr_dt'].dt.year>=2020) & (df['cmplnt_fr_dt'].dt.year<=2023)]
    
    #calculate monthly crimes
    montly_crimes=df.set_index('cmplnt_fr_dt').resample('M').size()
    
    #plot
    plt.figure(figsize=(14,7))
    plt.plot(montly_crimes.index, montly_crimes, marker='o', linestyle='-', color='blue')
    plt.title('Montly Count Crime from 2020 to 2023')
    plt.xlabel('Month')
    plt.ylabel('Number of Crimes')
    plt.grid(True)
    plt.show()
    
    
def train_evauluate_model(df,non_features, test_size=0.2, random_state=42):
    """
    Trains RandomForestClassifier and evaluate using a confusion matrix

    #param df:DataFrame containing NYPD Complaint Data
    #param non_features: List of columns names to be excluded from features
    #param test_size: Fraction of data use on test set
    #param random_state: Random state for reproducibility
    """
    #encode categorical features
    feature=[col for col in df.columns if col not in non_features]
    le=LabelEncoder()
    for col in feature:
        df[col]=le.fit_transform(df[col])
    
    #prepare data
    X=df.drop(columns=non_features)
    Y=df[non_features[0]]
    X_train, X_test, Y_train, y_test = train_test_split(X,Y, test_size=test_size, random_state=random_state)
    
    #Forest classsifier
    model = RandomForestClassifier(
        n_estimators=10,
         max_depth=10,
         min_samples_split=50,
         max_features='sqrt',
         random_state=random_state,
         n_jobs=-1,
         )
    model.fit(X_train, Y_train)
    
    #predict and evaulate
    predict=model.predict(X_test)
    cm=confusion_matrix(y_test, predict, normalize='true')*100
    
    #plot confusion matrix
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='coolwarm', vmin=0, vmax=100)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def create_crimte_map(df,date_col,lat_col,long_col, start_year, end_year, sample_size=1000,file_path='nyc_crime_map.html'):
    """
    Create folium map with crime data from specified data range

    Args:
        df (_type_):DataFrame containing NYPD Complaint Data
        date_col: Name of column with date information
        lat_col: Name of columns of latitude
        long_col: Name of column of Longitude
        start_year: Start year of filtering data
        end_year: End year of filtering data
        sample_size): Number of points to sample for map
        file_path: file path to save map
    """
    #convert date column and filter data
    df[date_col]=pd.to_datetime(df[date_col], errors='coerce', format='%m/%d%Y')
    filtered_data=df[(df[date_col].dt.year>=start_year) & (df[date_col].dt.year<=end_year)]
    filtered=filtered_data.dropna(subset=[lat_col, long_col])
    
    #sample data
    sample =filtered.sample(n=min(1000, len(filtered)), random_state=1)
    
    #create map
    nyc_map=folium.Map(location=[sample[lat_col].mean(), sample[long_col].mean()], zoom_start=11)
    for idx, row in sample.iterrows():
        folium.CircleMarker(
            location=[row[lat_col], row[long_col]],
            radius=2,
            color='red',
            fill=True,
            fill_color='red',
        ).add_to(nyc_map)
        
    #Save map
    nyc_map.save(file_path)
    
    
   

file_name='NYPD_Complaint_Data_Historic.csv'
df=prepare_data(file_name)

filter_data=filter_by_date(df,2020,2023)

#convert_columns=['addr_pct_cd', 'ky_cd', 'pd_cd']
#filter_data=convert_columns(df,convert_columns)

plot_crime_frequencies_by_borough(filter_data, 'boro_nm')

plot_monthly_crime_trends(df, 'cmplnt_fr_dt')

non_feature_cols=['boro_nm','addr_pct_cd']
train_evauluate_model(filter_data, non_feature_cols)

create_crimte_map(df,'cmplnt_fr_dt','latitude', 'longitude', 2020, 2023, file_path='C:/Users/Raptorial/Desktop/nyc_crime_map.html')
    
   
 
    #df['cmplnt_fr_dt']=pd.to_datetime(df['cmplnt_fr_dt'], errors='coerce', format='%m/%d/$Y')
#filtered_data=df[(df['cmplnt_fr_dt'].dt.year>=2020) & (df['cmplnt_fr_dt'].dt.year<=2023)]
#filtered=filtered_data.dropna(subset=['latitude', 'longitude'])





# #df=pd.get_dummies(df,columns=['ky_cd','ofns_desc'])
# #X=df.drop(columns=['boro_nm','addr_pct_cd'])
# y=df['boro_nm'] #df['ADDR_PCT_CD] next


# model.fit(X_train, y_train)

# df['day_of_week'] = df['cmplnt_fr_dt'].dt.day_name()
# crime_by_day=df.groupby('day_of_week').size().reset_index(name='crime_count')



#Training model
#encoding categorical features
# df=df.drop(columns=['cmplnt_fr_dt', 'cmplnt_fr_tm'])
# nonfeature=['boro_nm','addr_pct_cd']
# feature=[col for col in df.columns if col not in nonfeature]
# le=LabelEncoder()
# for col in feature:
#     df[col]=le.fit_transform(df[col])
# #encode=pd.get_dummies(df,columns=feature, drop_first=True, sparse=True)
# X=df.drop(columns=nonfeature)
# #df=pd.get_dummies(df,columns=['ky_cd','ofns_desc'])
# #X=df.drop(columns=['boro_nm','addr_pct_cd'])
# y=df['boro_nm'] #df['ADDR_PCT_CD] next
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
# #Forest classsifier
# model = RandomForestClassifier(
#     n_estimators=10,
#     max_depth=10,
#     min_samples_split=50,
#     max_features='sqrt',
#     random_state=42,
#     n_jobs=-1,
#     )
# model.fit(X_train, y_train)
# predict=model.predict(X_test)
# cm=confusion_matrix(y_test, predict, normalize='true')*100
# plt.figure(figsize=(10,7))
# sns.heatmap(cm, annot=True, fmt='.2f', cmap='coolwarm', vmin=0, vmax=100)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()




#filtered['year']=filtered['cmplnt_fr_dt'].dt.year

# feautures =[]
# for year in filtered['year'].unique():
#     year_data=filtered[filtered['year']==year]
#     feautures.extend([
#         {
#         'type':'Feature',
#         'geometry':{
#             'type':'Point',
#             'coordinates':[row['longitude'], row['latitude']]
#         },
#         'properties':{
#             'time':f'{year}-01-01T00:00:00Z',
#             'style':{'color':''},
#             'icon':'circle',
#             'iconstyle':{
#                 'fillColor':'red',
#                 'fillOpacity':0.6,
#                 'stroke':False,
#                 'radius':5
#             },
#         },
#         } for idx, row in year_data.iterrows()
#     ])

# TimestampedGeoJson({
#     'type':'FeatureCollection',
#     'features':feautures,
# }, period='P1Y',
#    add_last_point=True,
#    auto_play=False,
#    loop=False,
#    max_speed=1,
#    loop_button=True,
#    date_options='YYYY',
#    time_slider_drag_update=True).add_to(nyc_map)
# nyc_map.save('C:/Users/Raptorial/Desktop/nyc_crime_map.html')
    
    

    






#creating heatmap showing concentration of different type sof crime across different times of day
#Getting time of day splitting to Morning: 5am-11:59am, Afternoon: 12pm:4:59pm, Evening: 5pm:8:59pm, Night: 9pm:4:59am

# def time_of_day(time):
#     if time>=datetime.time(5,0) and time< datetime.time(12,0):
#         return 'Morning'
#     elif time>=datetime.time(12,0) and time < datetime.time(17,0):
#         return 'Afternoon'
#     elif time>=datetime.time(17,0) and time < datetime.time(21,0):
#         return 'Evening'
#     else:
#         return 'Night'
# #making new time_of_day column
# df['time_of_day']=df['cmplnt_fr_tm'].apply(time_of_day)
# #group ofns_desc and time_of_day column count occurence
# crime_time_heatmap=df.groupby(['ofns_desc', 'time_of_day']).size().unstack(fill_value=0)
# #get frequenzie better color shading
# crime_time_heatmap=crime_time_heatmap.apply(lambda x:x/x.sum(),axis=1)
# #reorder table columns
# time_order=['Morning','Afternoon','Evening','Night']
# crime_time_heatmap=crime_time_heatmap.reindex(columns=time_order)
# #plot heatmap
# plt.figure(figsize=(10,12))
# sns.heatmap(crime_time_heatmap, annot=True, fmt=".2f", cmap='viridis')
# plt.title('Heatmap of Crime Type by Time of Day', fontsize=12)
# plt.ylabel('Type of Crime', fontsize=12)
# plt.xlabel('Time of Day',fontsize=12)
# plt.tight_layout()
# plt.subplots_adjust(top=0.95)
# plt.show()




#check current data types



# current_data=df[['addr_pct_cd', 'ky_cd', 'pd_cd']].dtypes

# update_data=df[['addr_pct_cd', 'ky_cd', 'pd_cd']].dtypes
#print(current_data,update_data, df[['cmplnt_fr_dt','cmplnt_fr_tm']].head())



