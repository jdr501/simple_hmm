import pandas as pd


# import data
df = pd.read_csv('slowdown.csv')
# changing year to a readable quarterly date
df['year'] = pd.date_range("1970-01-01",freq="Q", periods=len(df)).to_period('Q')
# select the variables we need
df = df.loc[:, ['year', 'OIL', 'YUS', 'CPUS', 'SUS']]
df['YUS'] = 100*df['YUS']
df['CPUS'] = 100*df['CPUS']
# Take subset of data to match the paper
df = df[df['year'] >= pd.Period('1980Q1')]
df.set_index('year', inplace=True)

