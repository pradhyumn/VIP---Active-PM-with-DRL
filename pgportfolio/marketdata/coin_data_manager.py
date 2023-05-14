# Import necessary libraries
import sqlite3
import logging
import numpy as np
import pandas as pd
from pgportfolio.marketdata.coin_list import CoinList
from pgportfolio.constants import *
from pgportfolio.utils.misc import parse_time, get_volume_forward, \
    get_feature_list
from datetime import datetime

# Define a CoinDataManager class
class CoinDataManager:
    # Constructor method, initializes the object with given parameters
    # If working offline, the coin_list could be None
    # NOTE: return of the sqlite results is a list of tuples,
    # each tuple is a row.
    def __init__(self, coin_number, end, volume_average_days=1,
                 volume_forward=0, online=True, db_directory=None):
        self._storage_period = FIVE_MINUTES  # keep this as 300
        self._coin_number = coin_number
        self._online = online
        if self._online:
            self._coin_list = CoinList(end, volume_average_days, volume_forward)
        self._volume_forward = volume_forward
        self._volume_average_days = volume_average_days
        self._coins = None
        self._db_dir = (db_directory or DATABASE_DIR) + "/data.db"
        self._initialize_db()

    # Returns the coins
    @property
    def coins(self):
        return self._coins

    # This method is used to get coin features like close price, open price, volume, high and low prices.
    def get_coin_features(self, start, end, period=300, features=('close',)):
        """
        Args:
            start/end: Linux timestamp in seconds.
            period: Time interval of each data access point.
            features: Tuple or list of the feature names.

        Returns:
            A ndarray of shape [feature, coin, time].
        """
        from matplotlib import pyplot as plt
        start = int(start - (start % period))
        end = int(end - (end % period))
        coins = self.select_coins(
            start=end - self._volume_forward -
                  self._volume_average_days * DAY,
            end=end - self._volume_forward
        )
        self._coins = coins
        # Error handling in case the length of selected coins is not equal to expected
        if len(coins) != self._coin_number:
            raise ValueError(
                "The length of selected coins %d is not equal to expected %d"
                % (len(coins), self._coin_number))

        logging.info("Feature type list is %s" % str(features))
        self._check_period(period)

        # Initialize a data array with NaNs
        time_num = (end - start) // period + 1
        data = np.full([len(features), len(coins), time_num],
                       np.NAN, dtype=np.float32)
        
        # Open a connection to the SQLite database
        connection = sqlite3.connect(self._db_dir)
        try:
            # Loop through each coin and feature, fetch data from the database
            for coin_num, coin in enumerate(coins):
                for feature_num, feature in enumerate(features):
                    logging.info("Getting feature {} of coin {}".format(
                        feature, coin
                    ))
                    # Different SQL queries for different features
                    # The queries are designed to fetch data for the specified periods
                    # The fetched data is then inserted into the 'data' array at appropriate positions
                    # NOTE: transform the start date to end date
                    if feature == "close":
                        sql = (
                            'SELECT date+300 AS date_norm, close '
                            'FROM History WHERE '
                            'date_norm>={start} and date_norm<={end} '
                            'and date_norm%{period}=0 and coin="{coin}" '
                            'ORDER BY date_norm'
                            .format(start=start, end=end,
                                    period=period, coin=coin)
                        )
                    elif feature == "open":
                        sql = (
                            'SELECT date+{period} AS date_norm, open '
                            'FROM History WHERE '
                            'date_norm>={start} and date_norm<={end}'
                            'and date_norm%{period}=0 and coin="{coin}" '
                            'ORDER BY date_norm'
                            .format(start=start, end=end,
                                    period=period, coin=coin)
                        )
                    elif feature == "volume":
                        sql = (
                            'SELECT date_norm, SUM(volume) '
                            'FROM (SELECT date+{period}-(date%{period}) '
                            'AS date_norm, volume, coin FROM History) '
                            'WHERE date_norm>={start} '
                            'and date_norm<={end} and coin="{coin}" '
                            'GROUP BY date_norm '
                            'ORDER BY date_norm'
                            .format(start=start, end=end,
                                    period=period, coin=coin)
                        )
                    elif feature == "high":
                        sql = (
                            'SELECT date_norm, MAX(high) '
                            'FROM (SELECT date+{period}-(date%{period}) '
                            'AS date_norm, high, coin FROM History) '
                            'WHERE date_norm>={start} '
                            'and date_norm<={end} and coin="{coin}" '
                            'GROUP BY date_norm '
                            'ORDER BY date_norm'
                            .format(start=start, end=end,
                                    period=period, coin=coin)
                        )
                    elif feature == "low":
                        sql = (
                            'SELECT date_norm, MIN(low) '
                            'FROM (SELECT date+{period}-(date%{period}) '
                            'AS date_norm, low, coin FROM History) '
                            'WHERE date_norm>={start} '
                            'and date_norm<={end} and coin="{coin}"'
                            'GROUP BY date_norm '
                            'ORDER BY date_norm'
                            .format(start=start, end=end,
                                    period=period, coin=coin)
                        )
                    else:
                        msg = ("The feature %s is not supported" % feature)
                        logging.error(msg)
                        raise ValueError(msg)
                    serial_data = pd.read_sql_query(sql, con=connection,
                                                    parse_dates=["date_norm"],
                                                    index_col="date_norm")
                    time_index = ((serial_data.index.astype(np.int64) // 10**9
                                   - start) / period).astype(np.int64)
                    data[feature_num, coin_num, time_index] = \
                        serial_data.values.squeeze()
        # Ensure the database connection is closed, even if an error occurs
        finally:
            connection.commit()
            connection.close()

        # Fill NaNs and invalid values in the data
        data = self._fill_nan_and_invalid(data, bound=(0, 1),
                                          forward=True, axis=0)
        # backward fill along the period axis
        data = self._fill_nan_and_invalid(data, bound=(0, 1),
                                          forward=False, axis=2)
        assert not np.any(np.isnan(data)), "Filling nan failed, unknown error."

        # for manual checking
        # for f in range(data.shape[0]):
        #     for c in range(data.shape[1]):
        #         plt.plot(data[f, c])
        #         plt.show()
        return data

    def select_coins(self, start, end):
        """
        This function selects the top coins by volume within a specified time range.

        Args:
            start: The start of the time range as a timestamp in seconds.
            end: The end of the time range as a timestamp in seconds.

        Returns:
            A list of the names of the selected coins.
        """
        # If offline, select coins from local database
        if not self._online:
            # Log the time range for the coin selection
            logging.info(
                "Selecting coins offline from %s to %s" %
                (datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M'),
                 datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M'))
            )
            # Connect to the local database
            connection = sqlite3.connect(self._db_dir)
            print(self._db_dir)
            try:
                cursor = connection.cursor()
                int_start=int(start)
                int_end=int(end)
                # Query the database for the coins with the highest total volume in the time range, limit to the predefined number of coins
                cursor.execute(
                    'SELECT coin,SUM(volume) AS total_volume FROM History WHERE date>=? and date<=? GROUP BY coin ORDER BY total_volume DESC LIMIT ?',
                    (int_start, int_end, self._coin_number)
                )
                print(int(start),"-------------",int(end))
                # Fetch the result of the query
                coins_tuples = cursor.fetchall()
                print(coins_tuples)
                print(self._coin_number)

                # If the number of coins returned is not the expected number, log an error message
                if len(coins_tuples) != self._coin_number:
                    logging.error("The sqlite error happened.") #################################
            finally:
                # Commit any changes and close the connection to the database
                connection.commit()
                connection.close()
            # Extract the coin names from the result tuples
            coins = []
            for tuple in coins_tuples:
                coins.append(tuple[0])
        else:
            # If online, use the CoinList object to select the top coins by volume
            coins = list(
                self._coin_list.top_n_volume(n=self._coin_number).index
            )
        # Log the names of the selected coins
        logging.info("Selected coins are: " + str(coins))
        # Return the list of selected coin names
        return coins

    def _initialize_db(self):
        """
        This function initializes the local database.
        It creates a table named 'History' with the necessary columns if it does not already exist.
        """
        with sqlite3.connect(self._db_dir) as connection:
            cursor = connection.cursor()
            # Create the 'History' table if it does not exist
            cursor.execute('CREATE TABLE IF NOT EXISTS History (date INTEGER,'
                           ' coin varchar(20), high FLOAT, low FLOAT,'
                           ' open FLOAT, close FLOAT, volume FLOAT, '
                           ' quoteVolume FLOAT, weightedAverage FLOAT,'
                           'PRIMARY KEY (date, coin));')
            # Commit any changes to the database
            connection.commit()

    @staticmethod
    def _check_period(period):
        """
        This function checks whether the period specified is one of the predefined constants. 
        Args:
            period: The time period to check.
        Raises:
            ValueError: If the period is not a predefined constant.
        allowed_periods = [FIVE_MINUTES, FIFTEEN_MINUTES, HALF_HOUR, TWO_HOUR, FOUR_HOUR, DAY]
        """
        if period == FIVE_MINUTES:
            return
        elif period == FIFTEEN_MINUTES:
            return
        elif period == HALF_HOUR:
            return
        elif period == TWO_HOUR:
            return
        elif period == FOUR_HOUR:
            return
        elif period == DAY:
            return
        else:
            raise ValueError(
                'peroid has to be 5min, 15min, 30min, 2hr, 4hr, or a day'
            )

    @staticmethod
    def _fill_nan_and_invalid(array, bound=(0, 1), forward=True, axis=-1):
        """
        Forward fill or backward fill nan values.
        See https://stackoverflow.com/questions/41190852
        /most-efficient-way-to-forward-fill-nan-values-in-numpy-array

        Basical idea is finding non-nan indexes, then use maximum.accumulate
        or minimum.accumulate to aggregate them
        """
        # Create a mask of invalid or missing values
        mask = np.logical_or(np.isnan(array),
                             np.logical_or(array < bound[0], array > bound[1]))
        # Create an index array
        index_shape = [1] * mask.ndim
        index_shape[axis] = mask.shape[axis]

        index = np.arange(mask.shape[axis]).reshape(index_shape)
        # Fill in the missing or invalid values
        if forward:
            idx = np.where(~mask, index, 0)
            np.maximum.accumulate(idx, axis=axis, out=idx)
        else:
            idx = np.where(~mask, index, mask.shape[axis] - 1)
            idx = np.flip(
                np.minimum.accumulate(np.flip(idx, axis=axis),
                                      axis=axis, out=idx),
                axis=axis
            )
        return np.take_along_axis(array, idx, axis=axis)

    def _update_data(self, start, end, coin):
        """
        Add new history data into the database. This function updates the data for a specific coin in the database.
        Args:
            start: The start of the time range to update as a timestamp in seconds.
            end: The end of the time range to update as a timestamp in seconds.
            coin: The coin to update the data for.
        """
        # Connect to the database
        connection = sqlite3.connect(self._db_dir)
        try:
            cursor = connection.cursor()
            # Get the minimum and maximum dates currently in the database for the coin
            min_date = \
                cursor.execute('SELECT MIN(date) FROM History WHERE coin=?;',
                               (coin,)).fetchall()[0][0]
            max_date = \
                cursor.execute('SELECT MAX(date) FROM History WHERE coin=?;',
                               (coin,)).fetchall()[0][0]
            # If there is no data for the coin in the database, fill it
            if min_date is None or max_date is None:
                self._fill_data(start, end, coin, cursor)
            else:
                # If the maximum date in the database is more than 10 storage periods before the end of the update range, update the data
                if max_date + 10 * self._storage_period < end:
                    if not self._online:
                        raise Exception("Have to be online")
                    self._fill_data(max_date + self._storage_period,
                                    end,
                                    coin,
                                    cursor)
                # Check if the earliest date in the database is after the requested start date
                if min_date > start and self._online:
                    self._fill_data(start,
                                    min_date - self._storage_period - 1,
                                    coin,
                                    cursor)

            # if there is no data
        finally:
            # Commit any changes and close the database connection
            connection.commit()
            connection.close()

    # This function fills the database with data for a specific coin over a defined time range
    def _fill_data(self, start, end, coin, cursor):
        duration = 7819200  # three months
        bk_start = start
        # Loop over the range in three-month intervals
        for bk_end in range(start + duration - 1, end, duration):
            self._fill_part_data(bk_start, bk_end, coin, cursor)
            bk_start += duration
        # Handle the remaining interval if it is less than three months
        if bk_start < end:
            self._fill_part_data(bk_start, end, coin, cursor)

    # This function gets the historical market data for a specific coin over a specified time range and inserts it into the database
    def _fill_part_data(self, start, end, coin, cursor):
        # Get the historical market data for the coin
        chart = self._coin_list.get_chart_until_success(
            pair=self._coin_list.all_active_coins.at[coin, 'pair'],
            start=start,
            end=end,
            period=self._storage_period
        )
        # Logging the operation
        logging.info(
            "Filling %s data from %s to %s" % (
                coin,
                datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M'),
                datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M')
            ))
        # Loop over the chart data and insert it into the database
        for c in chart:
            if float(c["date"]) > 0:
                # Handle the case where the weighted average is 0
                if float(c['weightedAverage']) == 0:
                    weightedAverage = float(c['close'])
                else:
                    weightedAverage = float(c['weightedAverage'])
                # If the coin is in reversed order, insert the inverse values into the database
                if 'reversed_' in coin:
                    cursor.execute(
                        'INSERT OR IGNORE INTO History VALUES (?,?,?,?,?,?,?,?,?)',
                        (c['date'], coin, 1.0 / float(c['low']), 1.0 / float(c['high']),
                         1.0 / float(c['open']),
                         1.0 / float(c['close']), float(c['quoteVolume']), float(c['volume']),
                         1.0 / weightedAverage))
                # Otherwise, insert the values as they are
                else:
                    cursor.execute(
                        'INSERT OR IGNORE INTO History VALUES (?,?,?,?,?,?,?,?,?)',
                        (c['date'], coin, c['high'], c['low'], c['open'],
                         c['close'], c['volume'], c['quoteVolume'],
                         weightedAverage))

# This is a helper function that initializes a CoinDataManager object and optionally downloads coin features.
def coin_data_manager_init_helper(config, online=True,
                                  download=False, db_directory=None):
    # Extract the input configuration from the provided configuration
    input_config = config["input"]
    # Parse the start and end dates
    start = parse_time(input_config["start_date"])
    end = parse_time(input_config["end_date"])
    # Initialize CoinDataManager object with the extracted configuration
    cdm = CoinDataManager(
        coin_number=input_config["coin_number"],
        end=int(end),
        volume_average_days=input_config["volume_average_days"],
        volume_forward=get_volume_forward(
            int(end) - int(start),
            (input_config["validation_portion"] +
             input_config["test_portion"]),
            input_config["portion_reversed"]
        ),
        online=online,
        db_directory=db_directory
    )
    # If the download argument is set to False, just return the CoinDataManager object
    if not download:
        return cdm
    else:
        # Otherwise, download the features of the coins
        features = cdm.get_coin_features(
            start=start,
            end=end,
            period=input_config["global_period"],
            features=get_feature_list(input_config["feature_number"])
        )
        # And return both the CoinDataManager object and the downloaded features
        return cdm, features