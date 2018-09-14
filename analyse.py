#!/Users/jacob/anaconda3/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

data = pd.read_csv('single.csv', parse_dates=['date'])

data.set_index('date', drop=True, inplace=True)
data['p_diff1'] = data['p'].diff()
data['mu3_diff1'] = data['mu3'].diff()
data['M_shift1'] = data['M'].shift()
data['M_shift2'] = data['M'].shift(2)


# data[['p', 'mu1', 'mu2', 'mu3']].plot()
# data[['M', 'N', 'n']].plot()
# plt.show()

# model: N(t) = f( M(t-1) + mu3(t-1) + p(t-1) )
drop_diff = 1
drop_head = 8
y = data['N'][drop_diff+drop_head:].as_matrix()
X = data[['N', 'M', 'M_shift1', 'mu3', 'p_diff1']][drop_head:-drop_diff].as_matrix()

linear_model = sm.OLS(y, X)
results = linear_model.fit()
print(results.summary())
y_hat = results.predict(X)

plt.plot((y - y_hat))
plt.show()

