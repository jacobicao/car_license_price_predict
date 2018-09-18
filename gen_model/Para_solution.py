import random
import numpy as np
from gen_model.Gen_kernel import gen_kernel


class Para_solution:
    def __init__(self, N, n, mu, l, de):
        self.N = int(N)
        self.n = int(n)
        self.mu = mu
        self.l = l
        self.de = de
        self.domain = [(0.001, 0.02), (0.7, 0.9), (10, 200),
                       (20000, 70000), (100, 20000),
                       (70000, 120000), (100, 20000)]
        self.step = [0.001, 0.01, 10, 100, 10, 100, 10]

    def loss_fun(self, m):
        t = []
        for q in range(self.N):
            t.append(gen_kernel(m) // 100 * 100)
        t = sorted(t, reverse=True)
        a = np.array(t)
        tn = a[self.n - 1]
        loss = float(
            (a.mean() - self.mu) ** 2 +
            ((a > self.l).sum() - self.n) ** 2 +
            (tn - self.l) ** 2 +
            (((tn - 100 < t) & (t < tn + 100)).sum() - self.de) ** 2
        )
        return [loss, m]

    def violent_solution(self):
        n = 2
        x = [
            ((a + 1) / n, (b + 1) / n, c + 1, (d + 1) * 10 * n, (e + 1) * 100 * n, (f + 1) * 100 * n, (g + 1) * 100 * n)
            for a in range(n)
            for b in range(n)
            for c in range(n)
            for d in range(n)
            for e in range(n)
            for f in range(n)
            for g in range(n)
        ]
        from multiprocessing import Pool, cpu_count
        p = Pool(cpu_count())
        res = p.map(self.loss_fun, x)
        if len(res) == 0:
            print('ERROR: No solution found!')
            return (0.1, 0.1, 0.1, 0.1, 0.1)
        res.sort()
        print('res:%.8f\nsol:%s' % (res[0][0], np.array(res[0][1]).round(4)))
        return [res[0][0]] + res[0][1]

    def genetic_solution(self):
        from multiprocessing import Pool, cpu_count
        cpu_ct = cpu_count()
        p = Pool(cpu_ct)
        res = p.map(self.geneticoptimize, [self.domain, ] * cpu_ct)
        res.sort()
        print('res:%.8f\nsol:%s' % (res[0][0], np.array(res[0][1]).round(4)))
        return [res[0][0]] + res[0][1]

    def geneticoptimize(self, domain, step=0.1, popsize=50, mutrob=0.2, elite=0.2, maxiter=10):
        step = self.step

        def mutate(vec):
            i = random.randint(0, len(domain) - 1)
            res = []
            if random.random() < 0.5 and vec[i] > domain[i][0]:
                res = vec[0:i] + [vec[i] - step[i]] + vec[i + 1:]
            elif vec[i] < domain[i][1]:
                res = vec[0:i] + [vec[i] + step[i]] + vec[i + 1:]
            else:
                res = vec
            return res

        def crossover(r1, r2):
            i = random.randint(0, len(domain) - 1)
            return r1[0:i] + r2[i:]

        pop = []
        for _ in range(popsize):
            vec = [random.uniform(domain[k][0], domain[k][1]) for k in range(len(domain))]
            pop.append(vec)
        topelite = int(elite * popsize)
        for i in range(maxiter):
            if [] in pop:
                print('***')
            try:
                scores = [self.loss_fun(v) for v in pop]
            except Exception as args:
                print(i, 'pop!', args)
            sorted(scores)
            ranked = [v for (s, v) in scores]
            pop = ranked[0:topelite]
            while len(pop) < popsize:
                if random.random() < mutrob:
                    c = random.randint(0, topelite)
                    if len(ranked[c]) != len(domain):
                        continue
                    temp = mutate(ranked[c])
                    if temp == []:
                        print('******', ranked[c])
                    else:
                        pop.append(temp)
                else:
                    c1 = random.randint(0, topelite)
                    c2 = random.randint(0, topelite)
                    pop.append(crossover(ranked[c1], ranked[c2]))
        return scores[0]
