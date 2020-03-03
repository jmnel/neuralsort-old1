import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent / 'libs'))

from db_connectors import SQLite3Connector
from pathlib import Path
from datetime import datetime
from pprint import pprint

import matplotlib
matplotlib.use('Qt5Cairo')
import matplotlib.pyplot as plt

import numpy as np


class AdjRelReturnsBuilder:

    def __init__(self,
                 data_path: Path):

        av_db_path = data_path / 'av.db'
        av_db = SQLite3Connector.connect(av_db_path)

        dates = av_db.select('daily_adjusted', ['date'])
        dates = list([datetime.strptime(d[0], '%Y-%m-%d') for d in dates])

        filter_cls = 'WHERE ts_type == "daily_adj"'
        symbols = av_db.select('symbols_meta', ['symbol'], filter_cls)
        symbols = {s[0] for s in symbols}
        symbols -= {'DOW', 'UHN', 'V'}

        min_dates = list()
        max_dates = list()
        for s in symbols:
            dmin = av_db.select('daily_adjusted', ['min(date)'],
                                'WHERE symbol=="{}"'.format(s))
            dmax = av_db.select('daily_adjusted', ['max(date)'],
                                'WHERE symbol=="{}"'.format(s))

            dmin = datetime.strptime(dmin[0][0], '%Y-%m-%d')
            dmax = datetime.strptime(dmax[0][0], '%Y-%m-%d')

            min_dates.append((dmin, s))
            max_dates.append((dmax, s))

        min_date = max(min_dates)
        max_date = min(max_dates)

        print(min_date)
        print(max_date)

        assert(all(min_dates[0][0] == x[0] for x in min_dates))
        assert(all(max_dates[0][0] == x[0] for x in max_dates))

        min_date = datetime.strftime(min_date[0], '%Y-%m-%d')
        max_date = datetime.strftime(max_date[0], '%Y-%m-%d')

        data = dict()
        n = len(symbols)
        m = 0
        for s in symbols:
            filter_cls = f'WHERE'
            filter_cls += f' symbol == "{s}"'
            filter_cls += f' and date >= "{min_date}"'
            filter_cls += f' and date <= "{max_date}"'
            close_price = av_db.select('daily_adjusted', ['adj_close'],
                                       filter_cls)

            data[s] = list([float(x[0]) for x in close_price])
            m = len(data[s]) - 1

        rel_returns = np.zeros((m, n))
        for j, s in enumerate(symbols):
            for i in range(m):
                r0 = data[s][i]
                r1 = data[s][i + 1]
                rr = np.power(np.abs((r1 - r0) / r0), 1 / 8)
                rel_returns[i, j] = rr

        plt.plot(rel_returns[0:300, 0:8], linewidth=0.5)
#        plt.plot(rel_returns[:, 1], linewidth=0.5)
        plt.show()

        av_db.close()


data_path = Path(__file__).absolute().parent.parent.parent / 'data'
print(data_path)
builder = AdjRelReturnsBuilder(data_path)
