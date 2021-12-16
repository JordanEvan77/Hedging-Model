"""Seattle University, OMSBA 5062, P3, Jordan Gropper

Classes:
TimeSeries - class to hold a date/value series of data
Difference - a time series that is the difference between two other time
             series
Trim - helps split dates and trim them to be proper format
Fred - a time series that is based on a csv file downloaded from
       fred.stlouis.org
Basket - sets up the give portfolio assigning weights, name, and portfolio of values.
USDForex - helps convert foriegn currencies to proper dollar ratio for comparison purposes and
evaluation
USDCommodity - selects the proper values for the dates given and insert them into all dates of
year, referencing last value when missing value is found in data set.
"""

import os
import csv
from P3_Timeseries import TimeSeries
from P3_Timeseries import trim
from P3_Timeseries import Fred
from P3_Timeseries import USDForex
from P3_Timeseries import USDCommodity
from datetime import datetime
from datetime import timedelta
import numpy as np
import math
import itertools

DATA = "C:/Users/jorda/OneDrive/Desktop/PyCharm Community Edition 2021.2.2/5062/P3_FSS"




def regress_one_column(feat_vec, resp_vec, col_num):
    """takes a set of feature vectors, a corresponding set of responses, and a column number to pick
     out of the feature vectors. returns a slope, intercept, and the RSS for using that column
     alone (along with the intercept) to explain the response"""
    x = [[feat_vec[i][col_num], 1.0] for i in range(len(feat_vec))]
    beta, rss = ols(x, resp_vec)
    return beta[1], beta[0], rss  # slope, intercept, rss


def ols(predictors, response):
    """
    Ordinary Least Squares
    :param predictors:  observations (n observations x m predictors)
    :param response:    responses (n observations)
    :return:            coefficients (for m predictors), residual sum of squares

    >>> ols([[1.0, 0.0], [1.0, 1.0]], [10.0, 11.5])
    ([10.0, 1.5], 0.0)
    >>> ols([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]], [10.0, 10.5, 14.0])
    ([9.499999999999998, 2.0], 1.5)
    """
    big_x = np.matrix(predictors)
    # print(big_x)
    bold_y = np.matrix(response).T
    gramian = big_x.T * big_x
    beta_hat = gramian.I * big_x.T * bold_y
    residuals = bold_y - big_x * beta_hat
    return list(beta_hat.A[:, 0]), sum(residuals.A[:, 0] ** 2)


class Basket(object):
    """
    :param portfolio, a list of time series
    :param weights, a list of weights to the corresponding items in portfolio
    """

    def __init__(self, portfolio, weights, first=None, last=None):
        self.portfolio = portfolio
        self.weights = weights
        self.y_vec = []
        dates = portfolio[0].get_dates()
        for i in portfolio:
            dates = [n for n in i.get_dates() if n in dates]

        self.dates = dates

        y_vec = {}
        if first is None:
            first = min(self.dates)

        if last is None:
            last = max(self.dates)

        for i in self.dates:
            if i >= first and i <= last:
                num = 0
                iter = 0
                for j in portfolio:
                    num += j.data[i] * weights[iter]
                    iter += 1

                y_vec[i] = num

        self.y_vec1 = y_vec
        # print(self.y_vec)

    def regression(self, risk_list, first=None, last=None):
        """
        returns a residual sum of squares (RSS) and a set of weights to use in the surrogate basket.
        Also takes an optional start and end date for the data sampling. Returns the results as a
        dictionary
        >>> b = Basket([wti(), copper(), alumninum(), chy(), inr(), krw(), mxn(), myr()], [10485,      172,       1307,  30e6,  57e6, 1.3e6,  94e6, 1.4e9])
        >>> b.regression(['wti', 'copper', 'aluminum', 'chy', 'inr', 'krw', 'mxn', 'myr'])
        {'wti': 10484.999999958447, 'copper': 172.0000000114606, 'aluminum': 1306.999999976957, 'chy': 29999999.997258574, 'inr': 56999999.99397454, 'krw': 1299999.9086068869, 'mxn': 94000000.00018702, 'myr': 1400000000.0003767, 'intercept': 0.0010840147733688354, 'rss': 3.186354965833971e-05, 'start': '1993-11-08', 'end': '2019-12-31'}
        >>> b.regression(['wti'])
        {'wti': -128574.74963008212, 'intercept': 434678111.22619164, 'rss': 3.3982070271169606e+19, 'start': '1993-11-08', 'end': '2019-12-31'}
        >>> b.regression(['aluminum'], first=datetime(2001,1,1), last=datetime(2001,12,31))
        {'aluminum': 664.6905012856951, 'intercept': 384801223.44195986, 'rss': 12115661843637.746, 'start': '2001-01-01', 'end': '2001-12-31'}
        >>> b.regression(['wti', 'copper', 'krw', 'chy'], last=datetime(2001,1,23))
        {'wti': -1895033.4964562245, 'copper': 27372.39341832535, 'krw': 363875831384.4539, 'chy': 553191179.6710736, 'intercept': 42598309.516827166, 'rss': 1.2828976237060792e+18, 'start': '1993-11-08', 'end': '2001-01-23'}
        """
        simple_dates = []
        # print(first, last) # turn on and off
        self.risk_list = risk_list
        if first == None:
            first = min(self.dates)
        if last == None:
            last = max(self.dates)

        # print(first, last) # turn on and off
        for i in self.dates:
            if i >= first and i <= last:
                simple_dates.append(i)
        # print(simple_dates) # turn on and off
        self.y_vec = [self.y_vec1[k] for k in self.y_vec1 if k in simple_dates]
        # print(len(self.y_vec), len(simple_dates), simple_dates[-1]) # turn on and off
        regr1 = []
        b0 = 1 / math.sqrt(len(risk_list))
        for i in range(len(simple_dates)):
            regr1.append([b0])
        lengther = len(regr1)
        # print(regr1[0:10], lengther)  # working
        for i in risk_list:
            for j in self.portfolio:
                if i == j.name:
                    vals = j.get_values(simple_dates)
                    count = 0
                    for l in regr1:
                        l.append(vals[count])
                        # print(l)
                        count += 1
        beta, rss = ols(regr1, self.y_vec)
        dict1 = {risk_list[i]: beta[i + 1] for i in range(len(risk_list))}
        dict1['incercept'] = beta[0]
        dict1['rss'] = rss
        dict1['start'] = first.strftime('%Y-%m-%d')
        dict1['end'] = last.strftime('%Y-%m-%d')
        # print(dict1)
        return dict1

    def best_regression_n(self, n, first=None, last=None):
        """returns a portfolio of components and their weights that is the best and RSS. This
        return is the optimized set of needed assets to hedge against the value of the
        portfolio with the given limit of n. I am calling regression over and over with 4 sets in it
        >>> b.best_regression_n(4, last=datetime(2004,7,1))
        {'inr': 77087677.04121648, 'krw': 1281874355.1213226, 'mxn': 91618011.19604576, 'myr': 1398194975.0349388, 'intercept': 10585749.82807017, 'rss': 202812170884775.06, 'start': '1993-11-08', 'end': '2004-07-01'}
        """
        if first == None:
            first = min(self.dates)
        if last == None:
            last = max(self.dates)
        option_list = self.portfolio
        best_rss = None
        big_list = [i.name for i in self.portfolio]  # worked with Brooke to improve this
        for i in itertools.combinations(big_list, n):
            # print(i)
            a = b.regression(i, first, last)
            if best_rss == None or abs(a['rss']) < best_rss:
                weighted_portfolio = a

        return weighted_portfolio

    def best_regression_backtest(self, n, split_date):
        """performs back testing of a proxy portfolio chosen by the best_regression_n method.
        (Training set is up to split_date, hold-out set is after split_date.) Returns the standard
        deviation of the value of the complete Basket along with the suggested hedges from
        the training set held during the hold-out period."""
        listera = ['wti', 'copper', 'aluminum', 'chy', 'inr', 'krw', 'mxn', 'myr']
        train = b.best_regression_n(n, last=split_date)
        for i in listera:
            if i in train:
                pass
            if i not in train:
                train[i] = 0  # just a dummy
        # print(train)
        c = Basket([wti(), copper(), aluminum(), chy(), inr(), krw(), mxn(), myr()],
                   [(10485 - train['wti']), (172 - train['copper']), (1307 - train['aluminum']),
                    (30e6 - train['chy']), (57e6 - train['inr']), (1.3e6 - train['krw']),
                    (94e6 - train['mxn']),
                    (1.4e9 - train['myr'])])
        hold_out = c.regression(['wti', 'copper', 'aluminum', 'chy', 'inr', 'krw', 'mxn', 'myr'],
                                first=split_date)  # this is the val of whole basket
        mean = (sum(c.y_vec) / len(c.y_vec))  # taking the list of daily vals from above
        var = sum([((x - mean) ** 2) for x in c.y_vec]) / len(c.y_vec)
        # print(mean, var)
        SD = var ** 0.5
        return SD, train  # returning the standard deviation of the value of the complete Basket
        # along with the suggested hedges from the training set.


class myr(USDForex):
    def __init__(self):
        super().__init__('DEXMAUS')  # data column is labeled DEXMAUS in file


class mxn(USDForex):
    def __init__(self):
        super().__init__('DEXMXUS')


class krw(USDForex):
    def __init__(self):
        super().__init__('DEXKOUS')


class inr(USDForex):
    def __init__(self):
        super().__init__('DEXINUS')


class chy(USDForex):
    def __init__(self):
        super().__init__('DEXCHUS')


class wti(USDCommodity):
    def __init__(self):
        super().__init__('wti', 'OIL Commodity', 'USD', 'DCOILWTICO')  # new format


class copper(USDCommodity):
    def __init__(self):
        super().__init__('copper', 'COPPER Commodity', 'USD', 'PCOPPUSDM')  # new format


class aluminum(USDCommodity):
    def __init__(self):
        super().__init__('aluminum', 'Alumninum Commodity', 'USD', 'PALUMUSDM')  # new format


if __name__ == '__main__':
    b = Basket([wti(), copper(), aluminum(), chy(), inr(), krw(), mxn(), myr()],
               [10485, 172, 1307, 30e6, 57e6, 1.3e6, 94e6, 1.4e9])
    print(b.regression(
        ['wti', 'copper', 'aluminum', 'chy', 'inr', 'krw', 'mxn', 'myr']))  # workingell
    print(b.regression(['wti']))
    # -{'wti': -128574.74963008212, 'intercept': 434678111.22619164, 'rss': 3.3982070271169606e+19,
    # - 'start': '1993-11-08', 'end': '2019-12-31'}  # looks good!! how close should the intercept be
    print(b.regression(['aluminum'], first=datetime(2001, 1, 1), last=datetime(2001, 12, 31)))
    # -{'aluminum': 664.6905012856951, 'intercept': 384801223.44195986, 'rss': 12115661843637.746,
    # - 'start': '2001-01-01', 'end': '2001-12-31'} # pretty far off
    print(b.regression(['wti', 'copper', 'krw', 'chy'], last=datetime(2001, 1, 23)))
    # {'wti': -1895033.4964562245, 'copper': 27372.39341832535, 'krw': 363875831384.4539, 'chy': 553191179.6710736,
    # 'intercept': 42598309.516827166, 'rss': 1.2828976237060792e+18, 'start': '1993-11-08', 'end': '2001-01-23'}
    print(b.best_regression_n(4, last=datetime(2004, 7, 1)))  # not the right shape?
    # {'inr': 77087677.04121648, 'krw': 1281874355.1213226,
    # 'mxn': 91618011.19604576, 'myr': 1398194975.0349388,
    # 'intercept': 10585749.82807017, 'rss': 202812170884775.06, 'start': '1993-11-08', 'end': '2004-07-01'}
    print(b.best_regression_backtest(4, datetime(2001, 1, 1)))  # looks to be working!!
    # 4=>1371149.321256426, 2=>1297602.378128659
