#!/Users/jacob/anaconda3/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime

data = pd.read_csv('single.csv', parse_dates=['date'])
data.set_index('date', drop=True, inplace=True)
data['p_diff1'] = data['p'].diff()
data['mu3_diff1'] = data['mu3'].diff()
data['M_shift1'] = data['M'].shift()
data['M_shift2'] = data['M'].shift(2)
data['mu3_shift1'] = data['mu3'].shift()
data['mu2_shift1'] = data['mu2'].shift()
data['mu1_shift1'] = data['mu1'].shift()

data_sz = pd.read_excel("pmodel.xlsx")
data_sz = data_sz[["month", "pp", "zb", "lastp", "avg_p"]]


def estimate_n():
    """
    :return:

    model: N(t) = f( M(t-1), mu3(t-1), p(t-1) )
    """
    drop_diff = 1
    drop_head = 8
    y = data['N'][drop_diff + drop_head:].as_matrix()
    x = data[['N', 'M', 'M_shift1', 'mu3', 'p_diff1']][drop_head:-drop_diff].as_matrix()

    linear_model = sm.OLS(y, x)
    results = linear_model.fit()
    print(results.summary())
    y_hat = results.predict(x)

    plt.plot(y_hat, 'y')
    plt.plot(y)
    plt.show()

    return 10000


def estimate_mu3():
    """
    :return:

    model: mu3(t) = f( mu1(t), mu2(t), mu1(t), mu2(t), mu3(t-1) )
    """
    drop_diff = 1
    drop_head = 1
    y = data['mu3'][drop_diff + drop_head:].as_matrix()
    x = data[['mu1', 'mu2', 'mu1_shift1', 'mu2_shift1', 'mu3_shift1']][drop_head:-drop_diff].as_matrix()

    linear_model = sm.OLS(y, x)
    results = linear_model.fit()
    print(results.summary())
    y_hat = results.predict(x)

    plt.plot(y_hat, 'y')
    plt.plot(y)
    plt.show()

    return 10000


def estimate_p_para_core(m, n, mu, l, de, date):
    """
    :param N: 总参与人数
    :param n: 中标人数
    :param mu: 最后平均价格
    :param l: 中标最低价
    :param de: 中标最低价人数
    :param date: 日期
    :return:

    model:
    x ~ B1*Poison(r) + B2*Beta(a,b) + (1-B1-B2)*Beta(c,d)
    B1 ~ B(p1,1)
    B2 ~ B(p2,1)

    condition:
    mean(x) = mu
    x.cdf(l) = 1-n/N
    x.pdf(l) = d/N
    """
    from gen_model.One_Generator import One_Generator
    game = One_Generator()
    game.cal_para(m, n, mu, l, de, date)


def estimate_p_para():
    tran = data[['N', 'n', 'mu3', 'p', 'call']][5:]
    for k, v in tran.iterrows():
        estimate_p_para_core(*v.tolist(), datetime.strftime(k, "%Y-%m-%d"))


def doc():
    """
    from scipy.stats import beta
    rvs:随机变量
    pdf：概率密度函。
    cdf：累计分布函数
    sf：残存函数（1-CDF）
    ppf：分位点函数（CDF的逆）
    isf：逆残存函数（sf的逆）
    stats:返回均值，方差，（费舍尔）偏态，（费舍尔）峰度。
    moment:分布的非中心矩。
    :return:
    """
    pass


def estimate_p_2(size):
    """
    for i in range(1, 10):
        estimate_p_2(i)
    :param size:
    :return:
    """
    gbdt = GradientBoostingRegressor(
        loss='ls'
        , learning_rate=0.1
        , n_estimators=100
        , subsample=1
        , min_samples_split=2
        , min_samples_leaf=1
        , max_depth=3
        , init=None
        , random_state=None
        , max_features=None
        , alpha=0.9
        , verbose=0
        , max_leaf_nodes=None
        , warm_start=False
    )
    shape_s = data_sz.shape
    x = data_sz.ix[:(shape_s[0] - size), :4]
    y = data_sz.ix[:(shape_s[0] - size), 4:]
    x_m = data_sz.ix[(shape_s[0] - size):, :4]
    y_m = np.reshape(np.array(data_sz.ix[(shape_s[0] - size):, 4:]), size)
    gbdt.fit(x, y)
    print(gbdt.predict(x_m)[0] - y_m[0])


def estimate_p(li, n, mu3):
    m = [0.0102, 0.7565, 71.6482, 30570.8981, 10503.7339, 88772.7527, 13970.2140]
    from gen_model.One_Generator import One_Generator
    game = One_Generator()
    game.setter(*m)
    t = game.gen_game(n)
    plt.hist(t, bins=100)
    plt.show()

    return t


def plot_para():
    para = pd.read_csv('para.csv')
    para = para[['date', 'p1', 'p2', 'lam', 'a', 'b', 'c', 'd']]
    para.plot(subplots=True, figsize=(12, 16))
    plt.show()


def main():
    li = 2900
    n = estimate_n()
    mu3 = estimate_mu3()
    t = estimate_p(li, n, mu3)


if __name__ == '__main__':
    estimate_p(3000, 10000, 20000)
    # estimate_p_para()
    plot_para()
