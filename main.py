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

print('\n' * 5, 'THAT\'S WHAT I NEED', '\n' * 5)

from scipy import stats

"""
 Влияет ли тип устройства на количество покупок в день по каждому региону?
 Влияет ли тип рекламного канала на количество покупок в день по каждому региону?
 Проверить гипотезу о том, что средний чек отличается в зависимости от региона?!
 Проверить гипотезу о том, что средний чек отличается в зависимости от рекламного канала?!
 Проверить гипотезу о том, что средний чек отличается в зависимости от времени суток?!
 Есть ли взаимосвязь между продолжительностью сессии с суммой покупок?
 Проверить гипотезу о том, что средний чек отличается в зависимости от длительности сессии?!
 Есть ли взаимосвязь между типом утройства с типом оплаты?!
 Влияет ли день недели визита на час визита по каждому региону?
 """

pt_device = pd.pivot_table(df, values='revenue', index='region', columns='device', aggfunc='count')
print(pt_device.to_string())

pt_channel = pd.pivot_table(df, values='revenue', index='region', columns='channel', aggfunc='count')
print(pt_channel.to_string())

grouped_region = df.groupby('region')['revenue']
regions = []
regions_mean_revenues = []
for region, revenue in grouped_region:
    regions.append(region)
    regions_mean_revenues.append(revenue.mean())
print(regions)
for i in range(len(regions)):
    for j in range(i + 1, len(regions)):
        region1 = regions[i]
        region2 = regions[j]

        revenue1 = grouped_region.get_group(region1)
        revenue2 = grouped_region.get_group(region2)

        t_statistic, p_value = stats.ttest_ind(revenue1, revenue2)
        rounded = round(p_value, 3)
        print(f"P-value {region1} и {region2}: {rounded} & {rounded <= 0.05}")

grouped_channel = df.groupby('channel')['revenue']
channels = []
channels_mean_revenues = []
for channel, revenue in grouped_channel:
    channels.append(channel)
    channels_mean_revenues.append(revenue.mean())
print(channels)
for i in range(len(channels)):
    for j in range(i + 1, len(channels)):
        channel1 = channels[i]
        channel2 = channels[j]

        revenue1 = grouped_channel.get_group(channel1)
        revenue2 = grouped_channel.get_group(channel2)

        t_statistic, p_value = stats.ttest_ind(revenue1, revenue2)
        rounded = round(p_value, 3)
        print(f"P-value {channel1} и {channel2}: {rounded} & {rounded <= 0.05}")

grouped_time = df.groupby('time_of_day')['revenue']
times = []
times_mean_revenues = []
for time, revenue in grouped_time:
    times.append(time)
    times_mean_revenues.append(revenue.mean())
print(times)
for i in range(len(times)):
    for j in range(i + 1, len(times)):
        time1 = times[i]
        time2 = times[j]

        revenue1 = grouped_time.get_group(time1)
        revenue2 = grouped_time.get_group(time2)

        t_statistic, p_value = stats.ttest_ind(revenue1, revenue2)
        rounded = round(p_value, 3)
        print(f"P-value {time1} и {time2}: {rounded} & {rounded <= 0.05}")

correlation = df['sessiondurationsec'].corr(df['revenue'])
print('corr', round(correlation, 3))

short_sessions = df[df['sessiondurationsec'] < 600]['revenue']
long_sessions = df[df['sessiondurationsec'] >= 600]['revenue']

t_statistic, p_value = stats.ttest_ind(short_sessions, long_sessions)
rounded = round(p_value, 3)
print(f"P-value short_sessions и long_sessions: {rounded} & {rounded <= 0.05}")

cross_table = pd.crosstab(df['device'], df['payment_type'])
chi2, p_value, _, _ = stats.chi2_contingency(cross_table)
rounded = round(p_value, 3)
print(f"P-value device и payment_type: {rounded} & {rounded <= 0.05}")


pt_day = pd.pivot_table(df, values='hour_of_day', index='region', columns='day', aggfunc='count')
print(pt_day.to_string())

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
