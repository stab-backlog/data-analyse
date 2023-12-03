import pandas as pd

df = pd.read_csv('ecom_yl.csv').rename(mapper=lambda x: x.replace(' ', '_').lower(), axis=1)

# print(df.isnull().sum(), end='\n\n')
# print(*[df[i].value_counts() for i in df.columns], sep=f'\n{"-" * 10}\n')
df.fillna(value={'revenue': 0}, inplace=True)

df['promo_code'] = df['promo_code'].apply(lambda x: 0 if x != 1 else x)

import difflib
import re


def merge_similar_strings(row, lst, flag=0):
    def replace_russian_letters(text):
        table = {ord('\N{CYRILLIC CAPITAL LETTER A}'): ord('\N{LATIN CAPITAL LETTER A}'),
                 ord('\N{CYRILLIC CAPITAL LETTER EM}'): ord('\N{LATIN CAPITAL LETTER M}'),
                 ord('\N{CYRILLIC CAPITAL LETTER EN}'): ord('\N{LATIN CAPITAL LETTER H}'),
                 ord('\N{CYRILLIC CAPITAL LETTER ER}'): ord('\N{LATIN CAPITAL LETTER P}'),
                 ord('\N{CYRILLIC CAPITAL LETTER ES}'): ord('\N{LATIN CAPITAL LETTER C}'),
                 ord('\N{CYRILLIC CAPITAL LETTER HA}'): ord('\N{LATIN CAPITAL LETTER X}'),
                 ord('\N{CYRILLIC CAPITAL LETTER IE}'): ord('\N{LATIN CAPITAL LETTER E}'),
                 ord('\N{CYRILLIC CAPITAL LETTER KA}'): ord('\N{LATIN CAPITAL LETTER K}'),
                 ord('\N{CYRILLIC CAPITAL LETTER O}'): ord('\N{LATIN CAPITAL LETTER O}'),
                 ord('\N{CYRILLIC CAPITAL LETTER TE}'): ord('\N{LATIN CAPITAL LETTER T}'),
                 ord('\N{CYRILLIC CAPITAL LETTER U}'): ord('\N{LATIN CAPITAL LETTER Y}'),
                 ord('\N{CYRILLIC CAPITAL LETTER VE}'): ord('\N{LATIN CAPITAL LETTER B}'),
                 ord('\N{CYRILLIC SMALL LETTER A}'): ord('\N{LATIN SMALL LETTER A}'),
                 ord('\N{CYRILLIC SMALL LETTER EM}'): ord('\N{LATIN SMALL LETTER M}'),
                 ord('\N{CYRILLIC SMALL LETTER EN}'): ord('\N{LATIN SMALL LETTER H}'),
                 ord('\N{CYRILLIC SMALL LETTER ER}'): ord('\N{LATIN SMALL LETTER P}'),
                 ord('\N{CYRILLIC SMALL LETTER ES}'): ord('\N{LATIN SMALL LETTER C}'),
                 ord('\N{CYRILLIC SMALL LETTER HA}'): ord('\N{LATIN SMALL LETTER X}'),
                 ord('\N{CYRILLIC SMALL LETTER IE}'): ord('\N{LATIN SMALL LETTER E}'),
                 ord('\N{CYRILLIC SMALL LETTER KA}'): ord('\N{LATIN SMALL LETTER K}'),
                 ord('\N{CYRILLIC SMALL LETTER O}'): ord('\N{LATIN SMALL LETTER O}'),
                 ord('\N{CYRILLIC SMALL LETTER TE}'): ord('\N{LATIN SMALL LETTER T}'),
                 ord('\N{CYRILLIC SMALL LETTER U}'): ord('\N{LATIN SMALL LETTER Y}'),
                 ord('\N{CYRILLIC SMALL LETTER VE}'): ord('\N{LATIN SMALL LETTER B}')}
        if flag == 0:
            table = {}
        return re.sub(r'((?P<ch>\w)(?P=ch)+)', r'\2', text.translate(table)).title()

    str_row = replace_russian_letters(row)
    closest_match = difflib.get_close_matches(str_row, lst, n=1)
    return closest_match[0] if closest_match else str_row


region_lst = ['United States', 'Uk', 'Germany', 'France']
df['region'].fillna(value='NaN', inplace=True)
df['region'] = df.apply(lambda x: merge_similar_strings(x['region'], region_lst, flag=1), axis=1)

channel_lst = ['социальные сети ', 'organic', 'контекстная реклама', 'реклама у блогеров', 'email-рассылки']
df['channel'].fillna(value='NaN', inplace=True)
df['channel'] = df.apply(lambda x: merge_similar_strings(x['channel'], channel_lst), axis=1)

df.drop_duplicates().reset_index(drop=True)

# print(*[df[i].value_counts() for i in df.columns], sep=f'\n{"-" * 10}\n')

df['session_start'] = pd.to_datetime(df['session_start'])
df['session_end'] = pd.to_datetime(df['session_end'])
df['session_date'] = pd.to_datetime(df['session_date'])
df['order_dt'] = pd.to_datetime(df['order_dt'])

df['total_revenue'] = df['revenue'] * (1 - df['promo_code'] * 0.1)


def get_time_of_day(hour):
    if 6 <= hour < 10:
        return 'утро'
    elif 10 <= hour < 17:
        return 'день'
    elif 17 <= hour < 22:
        return 'вечер'
    else:
        return 'ночь'


df['time_of_day'] = df['hour_of_day'].apply(get_time_of_day)

df['payer'] = df['revenue'].apply(lambda x: 1 if x > 0 else 0)

sales_by_region = df['region'].value_counts(normalize=True)
sales_by_channel = df['channel'].value_counts(normalize=True)
sales_by_device = df['device'].value_counts(normalize=True)

print(sales_by_region, sales_by_channel, sales_by_device, sep='\n\n', end='\n------------------------------\n')

users_by_region = df.groupby(['region', 'payer'])['user_id'].nunique()
users_by_channel = df.groupby(['channel', 'payer'])['user_id'].nunique()
users_by_device = df.groupby(['device', 'payer'])['user_id'].nunique()

print(users_by_region, users_by_channel, users_by_device, sep='\n\n', end='\n------------------------------\n')

purchase_by_payment_type = df['payment_type'].value_counts()

print(purchase_by_payment_type, sep='\n\n', end='\n------------------------------\n')

average_check = df['total_revenue'].mean()
average_purchases_per_user = df.groupby('user_id')['order_dt'].nunique().mean()
average_session_duration_by_channel = df.groupby('channel')['sessiondurationsec'].mean()
average_session_duration_by_device = df.groupby('device')['sessiondurationsec'].mean()

print(average_check, average_purchases_per_user,
      average_session_duration_by_channel, average_session_duration_by_device,
      sep='\n\n', end='\n------------------------------\n')

top_channels_by_average_check = df.groupby('channel')['total_revenue'].mean().nlargest(3)
top_regions_by_average_check = df.groupby('region')['total_revenue'].mean().nlargest(3)
top_months_by_average_check = df.groupby(['region', 'month'])['total_revenue'].mean().nlargest(3)

print(top_channels_by_average_check, top_regions_by_average_check, top_months_by_average_check,
      sep='\n\n', end='\n------------------------------\n')

mau_by_channel = df.groupby(['month', 'channel'])['user_id'].nunique().reset_index()
top_channels_by_mau = mau_by_channel.groupby('channel')['user_id'].sum().nlargest(3)

print(top_channels_by_mau, sep='\n\n', end='\n------------------------------\n')

table_by_channel = df.groupby('channel').agg(
    {'user_id': 'count', 'payer': 'sum', 'total_revenue': 'sum'})

print(table_by_channel, sep='\n\n', end='\n------------------------------\n')

###############

from scipy import stats

device_purchases = df.groupby(['region', 'device'])['revenue'].count().reset_index()
device_purchases = device_purchases.pivot(index='region', columns='device', values='revenue')
device_purchases.fillna(0, inplace=True)
device_purchases_test = stats.chi2_contingency(device_purchases)
print(device_purchases_test[1])

channel_purchases = df.groupby(['region', 'channel'])['revenue'].count().reset_index()
channel_purchases = channel_purchases.pivot(index='region', columns='channel', values='revenue')
channel_purchases.fillna(0, inplace=True)
channel_purchases_test = stats.chi2_contingency(channel_purchases)
print(channel_purchases_test[1])

region_revenue = df.groupby('region')['revenue'].mean()
region_revenue_test = stats.f_oneway(*[region_revenue.values for region in region_revenue.index])
print(region_revenue_test[1])

channel_revenue = df.groupby('channel')['revenue'].mean()
channel_revenue_test = stats.f_oneway(*[channel_revenue.values for channel in channel_revenue.index])
print(channel_revenue_test[1])

hour_revenue = df.groupby('hour_of_day')['revenue'].mean()
hour_revenue_test = stats.f_oneway(*[hour_revenue.values for hour in hour_revenue.index])
print(hour_revenue_test[1])

session_duration = df['sessiondurationsec']
session_revenue = df['revenue']
session_revenue_corr = session_duration.corr(session_revenue)
print(session_revenue_corr)

###############

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

X = df[['region', 'device', 'channel', 'sessiondurationsec', 'month', 'day', 'hour_of_day', 'promo_code']]
y = df['revenue']
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"r2: {round(r2, 2)}; mae: {round(mae, 2)}")  # -> хорошо соответствует, но плохо прогнозирует
