#!/Users/jacob/anaconda3/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd

data = pd.read_csv('data.csv')

data['日期'] = data['日期'] * 100 + 25

person = ['日期', '个人平均报价1', '个人平均报价2', '个人平均成交价', '个人最低成交价',
          '个人有效编码', '个人参与竞价的有效编码', '个人成交编码',
          '个人增量指标', '个人最低成交价的报价人数', '个人最低成交价的成交人数'
          ]
sort_name = ['date', 'mu1', 'mu2', 'mu3', 'p', 'M', 'N', 'n', 'L', 'call', 'deal']
rename_map = dict(zip(person, sort_name))
data = data[person].rename(columns=rename_map)
data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')

data.to_csv('single.csv', index=False)
