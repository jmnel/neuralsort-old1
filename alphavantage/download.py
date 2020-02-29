from alpha_vantage.timeseries import TimeSeries
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor
from time import sleep, time

pool = ThreadPoolExecutor(1)

t_last_batch = time()
batch_quota = 0

requests = list()

proc = None


def process():

    print('process called')

    global batch_quota
    global t_last_batch

    while len(requests) > 0:

        #        if True:
        if batch_quota < 2:
            batch_quota += 1
            print(batch_quota)
            req = requests.pop(0)
            print(f'processing {req}')
            sleep(2)
            print('item done')
        else:
            print('batch quota exceeded.. waiting')
            t_elapsed = time() - t_last_batch
            print(t_elapsed)
            if time() - t_last_batch > 20.0:
                print('doing next batch')
                batch_quota = 0
                t_last_batch = time()
            else:
                sleep(10.0)

    sleep(2)
    return 'done'


def get(symbol):
    print(f'adding request for {symbol}')
    requests.append(symbol)
    global proc
    if proc is None:
        proc = pool.submit(process)


def resulta():
    global proc
    return proc


a = get('FOO')
b = get('BAR')
c = get('FooBAR')

d = resulta()
while not d.done():
    print('waiting')
    sleep(0.5)

print('done with proc')

# class AlphaVantageDownloader:

#    _daily_quota = 0
#    _minute_quota = 0


#    def __init__(api_key: str,
#                 minute_api_limit=5,
#                 daily_api_limit=500,
#                 wait_daily_quote=False):

#        self._minute_api_limit = minute_api_limit
#        self._daily_api_limit = daily_api_limit

#    def request_daily_adjusted(symbol: str):


#api_key = '5SSRQTB3PVHK28H9'

#ts = TimeSeries(api_key, output_format='pandas')

# syms = ['MMM', 'AXP', 'AAPL', 'BA', 'CAT',
#        'CVX', 'CSCO', 'KO', 'DIS', 'DOW',
#        'XOM', 'GS', 'HD', 'GS', 'HD',
#        'IBM', 'INTC', 'JNJ', 'MCD', 'MRK',
#        'MSFT', 'NKE', 'PFE', 'PG', 'TRV',
#        'UTX', 'UHN', 'VZ', 'V', 'WBA', 'WMT']

#data, meta = ts.get_
#data, meta = ts.get_daily_adjusted(symbol='MSFT', outputsize='full')
#data, meta = ts.get_intraday(symbol='MSFT', interval='1min', outputsize='full')

# print(meta)

# print(data)

# for m in meta:
#    print(m)
# print(meta)
# pprint(data.head(4))

# pprint(data.tail(4))

#data['5. adjusted close'].plot()

# print(len(data))

# plt.show()
