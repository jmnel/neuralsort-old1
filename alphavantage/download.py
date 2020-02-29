from alpha_vantage.timeseries import TimeSeries
from pprint import pprint
import matplotlib
matplotlib.use('Qt5Cairo')
import matplotlib.pyplot as plt

api_key = '5SSRQTB3PVHK28H9'

ts = TimeSeries(api_key, output_format='pandas')

syms = ['MMM', 'AXP', 'AAPL', 'BA', 'CAT',
        'CVX', 'CSCO', 'KO', 'DIS', 'DOW',
        'XOM', 'GS', 'HD', 'GS', 'HD',
        'IBM', 'INTC', 'JNJ', 'MCD', 'MRK',
        'MSFT', 'NKE', 'PFE', 'PG', 'TRV',
        'UTX', 'UHN', 'VZ', 'V', 'WBA', 'WMT']

data, meta = ts.get_daily(symbol='MSFT', outputsize='full')
#data, meta = ts.get_intraday(symbol='MSFT', interval='1min', outputsize='full')

print(meta)
pprint(data.tail(2))

data['4. close'].plot()

print(len(data))

plt.show()
