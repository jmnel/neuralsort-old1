from alpha_vantage.timeseries import TimeSeries
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep, time
import json


class AlphaVantageApi:
    """
    API wrapper to allow asyncronous downloading of Alpha Vantage realtime and historical
    data.

    """

    _pool = ThreadPoolExecutor(6)
    _max_calls_per_min = 0
    _max_calls_per_day = 0

    _min_wait = 60 + 5
    _day_wait = 60 * 60 * 24 + 60 * 2

    _time_prev_min = 0
    _time_prev_day = 0

    _lock = Lock()
    _wait_flag = False

    _process_task = None

    _quota_min = 0
    _quota_day = 0

    _av_api = None

    _info_json_file = None

    _load_info_enabled = False
    _save_info_enabled = False

#    _num_retries = 0
#    _short_retry_timeout = 0
#    _long_retry_timeout = 0

    _api_output_format = None

    def __init__(self,
                 api_key: str,
                 max_calls_per_min: int = 5,
                 max_calls_per_day: int = 500,
                 info_json_file: str = None,
                 load_info_enabled: bool = False,
                 save_info_enabled: bool = False,
                 output_format: str = 'pandas'):
        """
        Constructs Alpha Vantage API wrapper.

        Args:
            api_key:                API key
            max_calls_per_min:      limit of number of API calls/minute
            max_calls_per_day:      limit of number of API calls/day
            info_json_file:         location to store API call counts and timers
            load_info_enabled:      turns on info json loading
            save_info_enabled:      turns on info json saving
            output_format:          output format; either 'pandas' or 'csv'

        """

        self._max_calls_per_min = max_calls_per_min
        self._max_calls_per_day = max_calls_per_day

        # Initialize minute and day timers with current time.
        self._time_prev_min = time()
        self._time_prev_day = time()

        # Check that output format is either 'pandas' or 'csv'.
        if output_format not in ('pandas', 'csv'):
            raise ValueError(
                'output_format must be either \'pandas\' or \'csv\'.')

        # Create Alpha Vantage time series object.
        self._av_api = TimeSeries(api_key, output_format=output_format)

        self._load_info_enabled = load_info_enabled
        self._save_info_enabled = save_info_enabled
        self._info_json_file = info_json_file

        self._api_output_format = output_format

        # Load json configuration if enabled.
        if load_info_enabled:
            self._load_info()

    def done(self):
        """
        Done must be called to cleanup resources and save configuration to json file.
        """

        # Save json configuration if enabled.
        if self._save_info_enabled:
            self._save_info()

    # Loads information from json file.
    def _load_info(self):

        # Throw exception if json file path was not provided.
        if self._info_json_file is None:
            raise ValueError(
                'JSON info file path must be provided with load_info enabled.')

        # Try to open information json file and decode.
        try:
            with open(self._info_json_file, 'r') as json_file:
                json_info = json.load(json_file)

        # Catch non-fatal missing file and json decoder exceptions.
        except (FileNotFoundError, json.decoder.JSONDecodeError) as e:
            print(f' JSON file {self._info_json_file} not found or corrupted.')
            print('Not loading JSON info.')

        else:

            # Check that json file has correct format.
            if ('quota_min' not in json_info or
                    'quota_day' not in json_info or
                    'time_prev_min' not in json_info or
                    'time_prev_day' not in json_info):

                raise ValueError('JSON info file currupted.')

            # Store information values from json file to this class.
            self._quota_min = json_info['quota_min']
            self._quota_day = json_info['quota_day']
            self._time_prev_min = json_info['time_prev_min']
            self._time_prev_day = json_info['time_prev_day']

    # Saves the current information to json file.
    def _save_info(self):

        # Ensure that json file path was provided if save is enabled.
        if self._info_json_file is None:
            raise ValueError(
                'JSON info file path must be provided with save_info enabled.')

        # Build dictionary of usage quotas and timers.
        info_data = {'quota_min': self._quota_min,
                     'quota_day': self._quota_day,
                     'time_prev_min': self._time_prev_min,
                     'time_prev_day': self._time_prev_day}

        # Write information to file.
        with open(self._info_json_file, 'w') as json_file:
            json.dump(info_data, json_file)

    def _update_quotas(self):

        time_delta_min = time() - self._time_prev_min
        time_delta_day = time() - self._time_prev_day

        if time_delta_min > self._min_wait:
            with self._lock:
                self._time_prev_min = time()
                self._quota_min = 0

        if time_delta_day > self._day_wait:
            with self._lock:
                self._time_prev_day = time()
                self._quota_day = 0

        with self._lock:
            self._wait_flag = not(self._quota_min < self._max_calls_per_min
                                  and self._quota_day < self._max_calls_per_day)

    def _task(self, task_fn, **kargs):

        while True:
            self._update_quotas()

            if not self._wait_flag:

                with self._lock:
                    self._quota_min += 1
                    self._quota_day += 1

                return task_fn(**kargs)

        sleep(0.1)

    def _daily_adj_api(self, **kargs):

        return self._av_api.get_daily_adjusted(**kargs)

    def get_daily_adj(self,
                      symbol: str,
                      outputsize='full'):

        if symbol in ('', None):
            raise ValueError('symbol must be non-empty string.')

        if outputsize not in ('full', 'compact'):
            raise ValueError(
                'outputsize must be either \'full\' or \'compact\'.')

        t = self._pool.submit(self._task,
                              self._daily_adj_api,
                              symbol=symbol,
                              outputsize=outputsize)

        return t


av = AlphaVantageApi(api_key='5SSRQTB3PVHK28H9',
                     load_info_enabled=True,
                     save_info_enabled=True,
                     info_json_file='alpha_vantage_info.json')

# tasks = [av.get_daily_adj(str(i)) for i in range(50)]

# syms = ['MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'KO']
# syms = ['MMM', 'AXP']

# tasks = [(s, av.get_daily_adj(s, outputsize='compact')) for s in syms]

# while any(not t[1].done() for t in tasks):
#    sleep(5)
#    for i, t in enumerate(tasks):
#        if t[1].done():
#            print(f'{t[0]} is done')
#            tasks.pop(i)
#    if av._wait_flag:
#        print('Waiting for {:.0f} seconds.'.format(
#            65 + av._time_prev_min - time()))
#    sleep(2)


# av.done()
#    print(sum(not t[1].done() for t in tasks))
#    sleep(2)

# data = [t.result()[0] for t in tasks]
# meta = [t.result()[1] for t in tasks]

# for m in meta:
#    print(m)


# syms = ['MMM', 'AXP', 'AAPL', 'BA', 'CAT',
#        'CVX', 'CSCO', 'KO', 'DIS', 'DOW',
#        'XOM', 'GS', 'HD', 'GS', 'HD',
#        'IBM', 'INTC', 'JNJ', 'MCD', 'MRK',
#        'MSFT', 'NKE', 'PFE', 'PG', 'TRV',
#        'UTX', 'UHN', 'VZ', 'V', 'WBA', 'WMT']
