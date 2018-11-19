from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import  SimpleImputer
import datetime
import pandas as pd
import csv
import time
import numpy as np

# transform dataframe to convert date and time to ordinal(absolute numerical) values
def transformDataframe(df):
    for i in range(0, len(df)):
        datetime_original = df['pickup_datetime'][i]
        datetime_detailed = datetime_original.split()
        # ordinal date
        date_ordinal = datetime.datetime.strptime(datetime_detailed[0], '%Y-%m-%d').date().toordinal()
        time_ordinal = time.strptime(datetime_detailed[1], '%H:%M:%S')
        # ordinal time
        time_ordinal = datetime.timedelta(hours=time_ordinal.tm_hour, minutes=time_ordinal.tm_min,
                                          seconds=time_ordinal.tm_sec).total_seconds()
        df['pickup_datetime'][i] = date_ordinal
        df['pickup_time'][i] = time_ordinal
    return df

taxi_features = ['pickup_datetime', 'pickup_time', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
# Since the file is very large, lets read it in chunks of 10000
tp = pd.read_csv('train.csv', iterator=True, chunksize=10000)
#<pandas.io.parsers.TextFileReader object at 0x00000000150E0048>
taxi_data_train = pd.concat(tp, ignore_index=True)
# extra column that will be used to store ordinal time
taxi_data_train["pickup_time"] = np.nan
taxi_data_train = transformDataframe(taxi_data_train)
X_train = taxi_data_train[taxi_features]
impute = SimpleImputer()
X_train = impute.fit_transform(X_train)
Y_train = taxi_data_train.fare_amount
taxi_model_train = RandomForestRegressor(random_state = 1)
taxi_model_train.fit(X_train, Y_train)
taxi_data = pd.read_csv('test.csv')
taxi_data["pickup_time"] = np.nan
taxi_data = transformDataframe(taxi_data)
keys = taxi_data.values.T[0].tolist()
X = taxi_data[taxi_features]
X = impute.fit_transform(X)
predictions = taxi_model_train.predict(X)
with open('nyc_taxi.csv', 'a', newline='' ) as csvfile:
    filewriter = csv.writer(csvfile)
    filewriter.writerow(['key', 'fare_amount'])
    # write to output file
    for i in range(0, 9914): #9916
        filewriter.writerow([keys[i], predictions[i]])
    print('over and out')