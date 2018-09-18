from gen_model.Para_solution import Para_solution
from gen_model.Gen_kernel import gen_kernel
import pandas as pd
import os

cache_file = 'para.csv'


class One_Generator:
    def __init__(self):
        self.p1 = 0
        self.p2 = 0
        self.lam = 0
        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0
        self.vio = 0
        self.init = 0

    def setter(self, p1, p2, lam, a, b, c, d):
        self.p1 = p1
        self.p2 = p2
        self.lam = lam
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def cal_para(self, m, n, mu, l, de, date):
        res = 1e18
        if os.path.exists(cache_file):
            cc = pd.read_csv(cache_file)
            index = cc['date'] == date
            d = cc[['res', 'p1', 'p2', 'lam', 'a', 'b', 'c', 'd']][index]
            if d.size != 0:
                t = d.loc[d['res'].idxmin].tolist()
                (res, self.p1, self.p2, self.lam, self.a, self.b, self.c, self.d) = t
        para_solution = Para_solution(m, n, mu, l, de)
        if self.vio:
            mm = para_solution.violent_solution()
        else:
            mm = para_solution.genetic_solution()
        if mm[0] < res:
            res, self.p1, self.p2, self.lam, self.a, self.b, self.c, self.d = mm
            cc = pd.DataFrame([[date, ] + mm], columns=['date', 'res', 'p1', 'p2', 'lam', 'a', 'b', 'c', 'd'])
            if os.path.exists(cache_file):
                cc.to_csv(cache_file, index=False, float_format='%.4f', mode='a', header=False)
            else:
                cc.to_csv(cache_file, index=False, float_format='%.4f')

    def gen_game(self, n=10000):
        t = []
        m = (self.p1, self.p2, self.lam, self.a, self.b, self.c, self.d)
        for _ in range(n):
            t.append(gen_kernel(m) // 100 * 100)
        return t
