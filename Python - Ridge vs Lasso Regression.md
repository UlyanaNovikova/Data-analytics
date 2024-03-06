# NYC Taxi Trip Duration Prediction: Ridge vs Lasso Regression

This project aims at predicting cab trip durations using machine learning on cab trip data in New York City. We explore temporal patterns, apply Ridge and Lasso regression models with different preprocessing strategies and perform grid search to optimize their hyperparameters.

We will be working with data from the [New York City Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration/overview) competition, which was about predicting the duration of a taxi trip.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from warnings import filterwarnings
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV

filterwarnings('ignore')
%matplotlib inline
```

### Descriptive statistics


```python
train = pd.read_csv('train.csv')
```


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>vendor_id</th>
      <th>pickup_datetime</th>
      <th>dropoff_datetime</th>
      <th>passenger_count</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>store_and_fwd_flag</th>
      <th>trip_duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id2875421</td>
      <td>2</td>
      <td>2016-03-14 17:24:55</td>
      <td>2016-03-14 17:32:30</td>
      <td>1</td>
      <td>-73.982155</td>
      <td>40.767937</td>
      <td>-73.964630</td>
      <td>40.765602</td>
      <td>N</td>
      <td>455</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id2377394</td>
      <td>1</td>
      <td>2016-06-12 00:43:35</td>
      <td>2016-06-12 00:54:38</td>
      <td>1</td>
      <td>-73.980415</td>
      <td>40.738564</td>
      <td>-73.999481</td>
      <td>40.731152</td>
      <td>N</td>
      <td>663</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id3858529</td>
      <td>2</td>
      <td>2016-01-19 11:35:24</td>
      <td>2016-01-19 12:10:48</td>
      <td>1</td>
      <td>-73.979027</td>
      <td>40.763939</td>
      <td>-74.005333</td>
      <td>40.710087</td>
      <td>N</td>
      <td>2124</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id3504673</td>
      <td>2</td>
      <td>2016-04-06 19:32:31</td>
      <td>2016-04-06 19:39:40</td>
      <td>1</td>
      <td>-74.010040</td>
      <td>40.719971</td>
      <td>-74.012268</td>
      <td>40.706718</td>
      <td>N</td>
      <td>429</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id2181028</td>
      <td>2</td>
      <td>2016-03-26 13:30:55</td>
      <td>2016-03-26 13:38:10</td>
      <td>1</td>
      <td>-73.973053</td>
      <td>40.793209</td>
      <td>-73.972923</td>
      <td>40.782520</td>
      <td>N</td>
      <td>435</td>
    </tr>
  </tbody>
</table>
</div>



`dropoff_datetime` was added only to the training sample, so, this column cannot be used, let's delete it. `pickup_datetime` contains the date and time the trip started. Let's convert dates to `datetime` objects


```python
train = train.drop('dropoff_datetime', axis=1)
train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)

target = 'trip_duration'
```

The `trip_duration` column is the target value we want to predict. Let's look at the distribution of the target in the training sample.

The original distribution is skewed. By using a logarithmic scale we compress the high values, making the visualization more informative.


```python
plt.rcParams["figure.figsize"] = (14,6)
sns.set_palette("Set1")
fig, ax = plt.subplots(ncols=2)

sns.histplot(data=train, x=target, bins=20, ax=ax[0])
sns.histplot(data=train, x=target, bins=50, ax=ax[1], log_scale=True)

ax[0].set_title('Target variable distribution histogram', dict(size=14))
ax[0].set_xlabel('Trip duration', dict(size=13))
ax[0].set_ylabel('Frequency', dict(size=13))

ax[1].set_title('Log target variable distribution histogram', dict(size=14))
ax[1].set_xlabel('Trip duration, log scale', dict(size=13))
_ = ax[1].set_ylabel('Frequency', dict(size=13))
```


    
![png](output_11_0.png)
    


Let's add the `log_trip_duration` column to our sample and draw a histogram of the modified target for the training sample.


```python
train['log_trip_duration'] = np.log1p(train[target])
train = train.drop(columns=[target])
target = 'log_trip_duration'
```


```python
sns.set_palette("Set1")
plt.title('Distribution histogram of the new target variable', dict(size=14))
plt.xlabel('Log trip duration', dict(size=13))
plt.ylabel('Frequency', dict(size=13))
_ = sns.histplot(data=train, x=target, bins=100)
```


    
![png](output_14_0.png)
    


Let's calculate the value of the metric with the best constant prediction


```python
def RMSE(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean() ** .5
```


```python
const_model = train[target].mean()
print('Best constant prediction value: {}'.format(RMSE(train[target], const_model)))
```

    Best constant prediction value: 0.7957649731869667


Let's see how many trips there were on each day


```python
plt.rcParams["figure.figsize"] = (16,6)
train['day_of_year'] = train.pickup_datetime.dt.day_of_year

sns.countplot(x=train['day_of_year'])

plt.title('Number of trips in a day', dict(size=14))
plt.xlabel('Date', dict(size=13))
plt.ylabel('Number of trips', dict(size=13))

_ = plt.xticks(np.arange(0, 181, 6), np.unique(train.pickup_datetime.dt.date)[::6], rotation=60)
```


    
![png](output_19_0.png)
    


Lets find the dates on which there were the smallest number of trips


```python
train['day_of_year'] = train.pickup_datetime.dt.to_period("D")
daily_trip_counts = train['day_of_year'].value_counts().sort_index()
min_two_dates = daily_trip_counts.nsmallest(3).index
print("Dates with the fewest trips:")
min_two_dates
```

    Dates with the fewest trips:





    PeriodIndex(['2016-01-23', '2016-01-24', '2016-05-30'], dtype='period[D]')



Therefore, the lowest number of trips occurred on these dates: 2016-01-23, 2016-01-24 and 2016-05-30. At this time in January, there were snowstorms in New York that brought traffic to a stop. At 2016-05-30 there was a memorial which could block traffic.

Plot the number of trips versus the day of the week and the hours of the day


```python
train['hour'] = train.pickup_datetime.dt.hour
train['day_of_week'] = train.pickup_datetime.dt.day_of_week
train['day_of_year'] = train.pickup_datetime.dt.day_of_year
train['month_name'] = train.pickup_datetime.dt.month_name()
```


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>vendor_id</th>
      <th>pickup_datetime</th>
      <th>passenger_count</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>store_and_fwd_flag</th>
      <th>log_trip_duration</th>
      <th>day_of_year</th>
      <th>hour</th>
      <th>day_of_week</th>
      <th>month_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id2875421</td>
      <td>2</td>
      <td>2016-03-14 17:24:55</td>
      <td>1</td>
      <td>-73.982155</td>
      <td>40.767937</td>
      <td>-73.964630</td>
      <td>40.765602</td>
      <td>N</td>
      <td>6.122493</td>
      <td>74</td>
      <td>17</td>
      <td>0</td>
      <td>March</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id2377394</td>
      <td>1</td>
      <td>2016-06-12 00:43:35</td>
      <td>1</td>
      <td>-73.980415</td>
      <td>40.738564</td>
      <td>-73.999481</td>
      <td>40.731152</td>
      <td>N</td>
      <td>6.498282</td>
      <td>164</td>
      <td>0</td>
      <td>6</td>
      <td>June</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id3858529</td>
      <td>2</td>
      <td>2016-01-19 11:35:24</td>
      <td>1</td>
      <td>-73.979027</td>
      <td>40.763939</td>
      <td>-74.005333</td>
      <td>40.710087</td>
      <td>N</td>
      <td>7.661527</td>
      <td>19</td>
      <td>11</td>
      <td>1</td>
      <td>January</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id3504673</td>
      <td>2</td>
      <td>2016-04-06 19:32:31</td>
      <td>1</td>
      <td>-74.010040</td>
      <td>40.719971</td>
      <td>-74.012268</td>
      <td>40.706718</td>
      <td>N</td>
      <td>6.063785</td>
      <td>97</td>
      <td>19</td>
      <td>2</td>
      <td>April</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id2181028</td>
      <td>2</td>
      <td>2016-03-26 13:30:55</td>
      <td>1</td>
      <td>-73.973053</td>
      <td>40.793209</td>
      <td>-73.972923</td>
      <td>40.782520</td>
      <td>N</td>
      <td>6.077642</td>
      <td>86</td>
      <td>13</td>
      <td>5</td>
      <td>March</td>
    </tr>
  </tbody>
</table>
</div>




```python
tmp_data = train.groupby('hour').id.count()\
.reset_index().rename({
    'hour': 'Hour',
    'id': 'Number of trips that day'},
    axis=1
)

sns.relplot(data=tmp_data,
            x='Hour',
            y='Number of trips that day',
            kind='line')
_ = plt.title('Number of trips distribution by time of day', dict(size=14))
```


    
![png](output_26_0.png)
    



```python
tmp_data_weekday = train.groupby('day_of_week').id.count() \
    .reset_index().rename({
        'day_of_week': 'Day of Week',
        'id': 'Number of Trips'},
        axis=1
    )

sns.relplot(data=tmp_data_weekday,
            x='Day of Week',
            y='Number of Trips',
            kind='line',
            height=6,
            aspect=1.3)

plt.xticks(tmp_data_weekday['Day of Week'], ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

_ = plt.title('Number of trips distribution by day of week', dict(size=14))

```


    
![png](output_27_0.png)
    


Let's draw on one graph the dependence of the number of trips on the hour of the day for different months Similarly,we draw the dependence of the number of trips on the hour of the day for different days of the week.


```python
tmp_data = train.groupby(['month_name', 'hour']).id.count()\
.reset_index().rename({
    'month_name': 'Month name',
    'hour': 'Hour',
    'id': 'Number of trips'},
    axis=1
)

sns.relplot(data=tmp_data,
            x='Hour',
            y='Number of trips',
            hue='Month name',
            kind='line')
_ = plt.title('Distribution of the number of trips by hour of the day and month', dict(size=14))
```


    
![png](output_29_0.png)
    



```python
train['day_name'] = train['pickup_datetime'].dt.day_name()

tmp_data_weekday = train.groupby(['day_name', 'hour']).id.count() \
    .reset_index().rename({
        'day_name': 'Day of week',
        'hour': 'Hour',
        'id': 'Number of trips'},
        axis=1
    )

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

sns.relplot(data=tmp_data_weekday,
            x='Hour',
            y='Number of trips',
            hue='Day of week',
            kind='line',
            hue_order=day_order)

_ = plt.title('Distribution of the number of trips by hour of the day and day of the week', size=14)
```


    
![png](output_30_0.png)
    


The most common travel time is between 15 and 20 hours. It may be caused by the typical evening rush hours when individuals leave work. The least common travel time is between 0 and 5 hours, when most people sleep. On Sundays travel stays low throughout the day, suggesting that many people prefer to rest on Sundays compared to other weekdays. The number of trips tends to be lower in January compared to other months, probably influenced by problematic weather conditions during this winter month.

Let's split the sample into train and test (7:3). For the train sample, we plot the mean logarithm of travel time versus day of the week. Then we do the same, but for the hour of the day and the day of the year.


```python
train = train.drop('month_name', axis=1)
train['month'] = train.pickup_datetime.dt.month

train['is_anomaly_jan'] = ((23 <= train.day_of_year) & (train.day_of_year <= 26)).astype('int')
train['is_anomaly_may'] = (train.day_of_year == 151).astype('int')
```


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>vendor_id</th>
      <th>pickup_datetime</th>
      <th>passenger_count</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>store_and_fwd_flag</th>
      <th>log_trip_duration</th>
      <th>day_of_year</th>
      <th>hour</th>
      <th>day_of_week</th>
      <th>day_name</th>
      <th>month</th>
      <th>is_anomaly_jan</th>
      <th>is_anomaly_may</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id2875421</td>
      <td>2</td>
      <td>2016-03-14 17:24:55</td>
      <td>1</td>
      <td>-73.982155</td>
      <td>40.767937</td>
      <td>-73.964630</td>
      <td>40.765602</td>
      <td>N</td>
      <td>6.122493</td>
      <td>74</td>
      <td>17</td>
      <td>0</td>
      <td>Monday</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id2377394</td>
      <td>1</td>
      <td>2016-06-12 00:43:35</td>
      <td>1</td>
      <td>-73.980415</td>
      <td>40.738564</td>
      <td>-73.999481</td>
      <td>40.731152</td>
      <td>N</td>
      <td>6.498282</td>
      <td>164</td>
      <td>0</td>
      <td>6</td>
      <td>Sunday</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id3858529</td>
      <td>2</td>
      <td>2016-01-19 11:35:24</td>
      <td>1</td>
      <td>-73.979027</td>
      <td>40.763939</td>
      <td>-74.005333</td>
      <td>40.710087</td>
      <td>N</td>
      <td>7.661527</td>
      <td>19</td>
      <td>11</td>
      <td>1</td>
      <td>Tuesday</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id3504673</td>
      <td>2</td>
      <td>2016-04-06 19:32:31</td>
      <td>1</td>
      <td>-74.010040</td>
      <td>40.719971</td>
      <td>-74.012268</td>
      <td>40.706718</td>
      <td>N</td>
      <td>6.063785</td>
      <td>97</td>
      <td>19</td>
      <td>2</td>
      <td>Wednesday</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id2181028</td>
      <td>2</td>
      <td>2016-03-26 13:30:55</td>
      <td>1</td>
      <td>-73.973053</td>
      <td>40.793209</td>
      <td>-73.972923</td>
      <td>40.782520</td>
      <td>N</td>
      <td>6.077642</td>
      <td>86</td>
      <td>13</td>
      <td>5</td>
      <td>Saturday</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_data, test_data = train_test_split(train, test_size=0.3, random_state=10)
```


```python
mean_log_travel_time_by_day = train_data.groupby('day_of_week')['log_trip_duration'].mean()

plt.figure(figsize=(10, 6))
sns.lineplot(x=mean_log_travel_time_by_day.index, y=mean_log_travel_time_by_day.values)
plt.title('Average trip duration by the number of day in a week (train sample)')
plt.xlabel('Day of the week')
plt.ylabel('Mean logarithm of travel time')
plt.xticks(range(7), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.show()
```


    
![png](output_36_0.png)
    



```python
mean_log_travel_time_by_hour = train_data.groupby('hour')['log_trip_duration'].mean()

plt.figure(figsize=(10, 6))
sns.lineplot(x=mean_log_travel_time_by_hour.index, y=mean_log_travel_time_by_hour.values)
plt.title('Average trip duration by the hour of day (train sample)')
plt.xlabel('Hour')
plt.ylabel('Mean logarithm of travel time')
plt.show()
```


    
![png](output_37_0.png)
    



```python
plt.figure(figsize=(16, 4))
sns.barplot(x='day_of_year', y='log_trip_duration', data=train_data, ci=None)
plt.title('Average trip duration by the day of the year (train sample)')
plt.xlabel('Day of the year')
plt.ylabel('Mean logarithm of travel time')

_ = plt.xticks(np.arange(0, 181, 6), np.unique(train_data.pickup_datetime.dt.date)[::6], rotation=60)
```


    
![png](output_38_0.png)
    


1. The graphs of the target depending on the day of the week and on the hour of the day are similar to those for the number of trips. Both show lower activity on Mondays and Sundays, reduced trips around 5 AM, increased activity on Wednesdays, Thursdays and Fridays, and heightened trips in the second half of the day. These patterns likely reflect underlying human behavior, commuting habits, and societal rhythms. The logarithmic transformation captures the relative changes in travel time and also preserves the observed patterns related to daily and hourly variations.

2. The logarithmic transformation in the plot of log_trip_duration mitigates the visibility of anomalies observed in the original trip count graphs, such as the lowest activity on specific dates ('2016-01-23', '2016-01-24', '2016-05-30'). This is because the transformation compresses extreme values, making them less apparent in the transformed scale. The absence of these anomalies in the log-transformed graph underscores the impact of the chosen transformation on highlighting broader temporal trends rather than specific anomalous dates.

### Predicting the target variable

Let's train `Ridge` regression with default parameters by encoding all categorical features with `OneHotEncoder`. We scale numerical features with `StandardScaler`. 


```python
def make_pipeline(categorical, numeric, model):
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
       transformers=[
        ('numeric', numeric_transformer, numeric),
        ('categorical', categorical_transformer, categorical),
    ])
    pipeline = Pipeline(steps = [
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    return pipeline

def fit_pipeline(pipeline, train_data, test_data):
    pipeline.fit(train_data.drop(target, axis=1), train_data[target])

    print('Train RMSE is {}'.format(RMSE(train_data[target], pipeline.predict(train_data.drop(target, axis=1)))))
    print('Test RMSE is {}'.format(RMSE(test_data[target], pipeline.predict(test_data.drop(target, axis=1)))))
```


```python
train_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>vendor_id</th>
      <th>pickup_datetime</th>
      <th>passenger_count</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>store_and_fwd_flag</th>
      <th>log_trip_duration</th>
      <th>day_of_year</th>
      <th>hour</th>
      <th>day_of_week</th>
      <th>day_name</th>
      <th>month</th>
      <th>is_anomaly_jan</th>
      <th>is_anomaly_may</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>824746</th>
      <td>id2365163</td>
      <td>2</td>
      <td>2016-03-27 00:25:29</td>
      <td>1.016645</td>
      <td>-0.295620</td>
      <td>-0.261509</td>
      <td>-0.180849</td>
      <td>-1.145803</td>
      <td>N</td>
      <td>7.134891</td>
      <td>-0.092859</td>
      <td>-2.126174</td>
      <td>6</td>
      <td>Sunday</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>645821</th>
      <td>id3126187</td>
      <td>2</td>
      <td>2016-03-18 20:44:12</td>
      <td>-0.505419</td>
      <td>-0.206681</td>
      <td>-0.275019</td>
      <td>-0.048720</td>
      <td>0.886502</td>
      <td>N</td>
      <td>6.878326</td>
      <td>-0.267370</td>
      <td>0.999014</td>
      <td>4</td>
      <td>Friday</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>691846</th>
      <td>id2588127</td>
      <td>2</td>
      <td>2016-04-15 18:26:12</td>
      <td>0.255613</td>
      <td>0.014655</td>
      <td>-0.005959</td>
      <td>1.409618</td>
      <td>0.462631</td>
      <td>N</td>
      <td>7.642524</td>
      <td>0.275554</td>
      <td>0.686495</td>
      <td>4</td>
      <td>Friday</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1147931</th>
      <td>id0915715</td>
      <td>2</td>
      <td>2016-03-24 14:28:34</td>
      <td>-0.505419</td>
      <td>-0.111301</td>
      <td>0.779694</td>
      <td>-0.021542</td>
      <td>0.254628</td>
      <td>N</td>
      <td>6.888572</td>
      <td>-0.151029</td>
      <td>0.061457</td>
      <td>3</td>
      <td>Thursday</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>183569</th>
      <td>id3758776</td>
      <td>1</td>
      <td>2016-04-28 13:53:52</td>
      <td>-0.505419</td>
      <td>-0.003133</td>
      <td>0.403698</td>
      <td>0.081675</td>
      <td>0.257498</td>
      <td>N</td>
      <td>6.177944</td>
      <td>0.527626</td>
      <td>-0.094802</td>
      <td>3</td>
      <td>Thursday</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
categorical_columns = ['vendor_id', 'month', 'is_anomaly_jan', 'is_anomaly_may', 'day_of_week']
numeric_columns = ['passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'day_of_year', 'dropoff_latitude', 'hour']
pipeline = make_pipeline(categorical_columns, numeric_columns, Ridge())
fit_pipeline(pipeline, *train_test_split(train, random_state=42, test_size=0.3))
```

    Train RMSE is 0.7757660446952792
    Test RMSE is 0.7766974606491961


Let's use other way of categorical features encoding and normalization for numerical features and train `Ridge` and `Lasso` regression


```python
def make_pipeline(categorical, numeric, model):
    numeric_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric),
            ('categorical', categorical_transformer, categorical),
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    return pipeline

ridge_pipeline_new = make_pipeline(categorical_columns, numeric_columns, Ridge())
lasso_pipeline_new = make_pipeline(categorical_columns, numeric_columns, Lasso())

fit_pipeline(ridge_pipeline_new, *train_test_split(train, random_state=42, test_size=0.3))

fit_pipeline(lasso_pipeline_new, *train_test_split(train, random_state=42, test_size=0.3))
```

    Train RMSE is 0.7786118432667183
    Test RMSE is 0.7780836488959478
    Train RMSE is 0.7960728728509628
    Test RMSE is 0.7950269555526899


The lower RMSE values for Ridge Regression suggest that it has better predictive performance compared to Lasso Regression. It can be because Ridge Regression's L2 regularization prevents overfitting by penalizing large coefficients, while Lasso Regression's L1 regularization induces sparsity, making it less flexible. One-Hot Encoding ans standart scaling turned out to be the best options with the lowest Test RMSE, suggesting that it effectively captured the categorical information. The other encoding options, might have shown worse quality due to assumptions about ordinal relationships or sensitivity to outliers in scaling.

### Grid search for `Ridge` and `Lasso`

Let's find the optimal values of the regularization parameter for `Ridge` and `Lasso`


```python
# Split the training sample into train and validation sets (80% train, 20% validation)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Separate features and target variable
X_train = train_data.drop(target, axis=1)
y_train = train_data[target]

X_val = val_data.drop(target, axis=1)
y_val = val_data[target]

X_test = test_data.drop(target, axis=1)
y_test = test_data[target]

# Grid search for Ridge Regression
ridge_alphas = np.logspace(-2, 3, 20)
ridge_params = {'regressor__alpha': ridge_alphas}
ridge_grid = GridSearchCV(make_pipeline(categorical_columns, numeric_columns, Ridge()), param_grid=ridge_params, scoring='neg_mean_squared_error', cv=5)
ridge_grid.fit(X_train, y_train)

# Grid search for Lasso Regression
lasso_alphas = np.logspace(-2, 3, 20)
lasso_params = {'regressor__alpha': lasso_alphas}
lasso_grid = GridSearchCV(make_pipeline(categorical_columns, numeric_columns, Lasso()), param_grid=lasso_params, scoring='neg_mean_squared_error', cv=5)
lasso_grid.fit(X_train, y_train)

# Get the best models
best_ridge_model = ridge_grid.best_estimator_
best_lasso_model = lasso_grid.best_estimator_
best_ridge_alpha = ridge_grid.best_params_['regressor__alpha']
print(f'Best Ridge Alpha: {best_ridge_alpha}')
best_lasso_alpha = lasso_grid.best_params_['regressor__alpha']
print(f'Best Lasso Alpha: {best_lasso_alpha}')

# Evaluate on the test set
ridge_test_rmse = RMSE(y_test, best_ridge_model.predict(X_test))
lasso_test_rmse = RMSE(y_test, best_lasso_model.predict(X_test))

print('Ridge Regression Test RMSE: {}'.format(ridge_test_rmse))
print('Lasso Regression Test RMSE: {}'.format(lasso_test_rmse))
```

    Best Ridge Alpha: 0.01
    Best Lasso Alpha: 0.01
    Ridge Regression Test RMSE: 0.7717992764722363
    Lasso Regression Test RMSE: 0.7931257597976528


Ridge Regression model with appha=0,01 has the better quality than Lasso Regression. 
