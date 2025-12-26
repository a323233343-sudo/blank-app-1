import random
import numpy as np

# ----------------------------
# NSGA-II class (your version, integrated)
# ----------------------------
class NSGAII_tsp:
    def __init__(self, start_idx, end_idx):
        self.start_idx = start_idx
        self.end_idx = end_idx
        
    def generate_initial_route(self, n):
        nodes = list(range(n))
        # 移除固定起點和終點
        nodes.remove(self.start_idx)
        nodes.remove(self.end_idx)
        random.shuffle(nodes)
        route = [self.start_idx] + nodes + [self.end_idx]
        return self.enforce_order(route)

    def ordered_crossover_fixed(self, p1, p2):
        n = len(p1)
        a, b = sorted(random.sample(range(n), 2))
        child = [-1] * n
        child[a:b + 1] = p1[a:b + 1]
        pos = 0
        for i in range(n):
            idx = (b + 1 + i) % n
            if child[idx] != -1:
                continue
            while p2[pos] in child:
                pos += 1
            child[idx] = p2[pos]
        return child

    def swap_mutation_fixed(self, route, prob=0.2):
        r = route[:]
        if random.random() < prob:
            i, j = random.sample(range(len(r)), 2)
            r[i], r[j] = r[j], r[i]
        return r

    def dominates(self, a, b):
        le = all(x <= y for x, y in zip(a, b))
        lt = any(x < y for x, y in zip(a, b))
        return le and lt

    def fast_non_dominated_sort(self, pop_objs):
        S = [set() for _ in pop_objs]
        n_dom = [0] * len(pop_objs)
        fronts = [[]]
        for p in range(len(pop_objs)):
            for q in range(len(pop_objs)):
                if p == q:
                    continue
                if self.dominates(pop_objs[p], pop_objs[q]):
                    S[p].add(q)
                elif self.dominates(pop_objs[q], pop_objs[p]):
                    n_dom[p] += 1
            if n_dom[p] == 0:
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            nxt = []
            for p in fronts[i]:
                for q in S[p]:
                    n_dom[q] -= 1
                    if n_dom[q] == 0:
                        nxt.append(q)
            i += 1
            fronts.append(nxt)
        if not fronts[-1]:
            fronts.pop()
        return fronts

    def crowding_distance(self, front_objs):
        l = len(front_objs)
        if l == 0:
            return {}
        nobj = len(front_objs[0][1])
        dist = {idx: 0 for idx, _ in front_objs}
        for m in range(nobj):
            sorted_front = sorted(front_objs, key=lambda x: x[1][m])
            minv, maxv = sorted_front[0][1][m], sorted_front[-1][1][m]
            dist[sorted_front[0][0]] = dist[sorted_front[-1][0]] = float('inf')
            if maxv == minv:
                continue
            for i in range(1, l - 1):
                prevv, nextv = sorted_front[i - 1][1][m], sorted_front[i + 1][1][m]
                dist[sorted_front[i][0]] += (nextv - prevv) / (maxv - minv)
        return dist

    def tournament_selection(self, pop):
        a, b = random.sample(pop, 2)
        if a['rank'] < b['rank']:
            return a
        if a['rank'] > b['rank']:
            return b
        return a if a['cd'] > b['cd'] else b

    def enforce_order(self, route):
        # 順序約束：第13點(12) 要在第14點(13)之前
        #bridge_idx = self.D.index[self.D['name'] == '大港橋'][0]
        #park_idx = self.D.index[self.D['name'] == '公園二路(集合)'][0]
        """precedence_rules = [(self.bridge_idx, self.park_idx)]
        for a, b in precedence_rules:
            if a >= len(route) or b >= len(route):
                continue
            # only if both in route
            if a in route and b in route:
                ia, ib = route.index(a), route.index(b)
                if ia > ib:
                    route[ia], route[ib] = route[ib], route[ia]

                # 新增第13必須在倒數第二或倒數第三位
                ia = route.index(a)
                n = len(route)
                if ia < n - 3:
                    elem = route.pop(ia)
                    route.insert(n - 3, elem)"""
        return route

    def nsga2_tsp(self, D, T, coords=None, pop_size=80, gens=200, cx_prob=0.9, mut_prob=0.2, close_loop=False, seed=None):
        self.D = D
        self.T = T
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        n = D.shape[0]

        def obj_distance(r):
            total = sum(D[r[i], r[i + 1]] for i in range(len(r) - 1))
            if close_loop:
                total += D[r[-1], r[0]]
            return total

        def obj_time(r):
            total = sum(T[r[i], r[i + 1]] for i in range(len(r) - 1))
            if close_loop:
                total += T[r[-1], r[0]]
            return total

        population = [{'route': self.enforce_order(random.sample(range(n), n)), 'objs': None} for _ in range(pop_size)]

        def evaluate(pop):
            for ind in pop:
                ind['objs'] = (obj_distance(ind['route']), obj_time(ind['route']))

        evaluate(population)

        for gen in range(gens):
            pop_objs = [ind['objs'] for ind in population]
            fronts = self.fast_non_dominated_sort(pop_objs)
            for i, f in enumerate(fronts):
                for idx in f:
                    population[idx]['rank'] = i
            for f in fronts:
                f_objs = [(idx, population[idx]['objs']) for idx in f]
                cd = self.crowding_distance(f_objs)
                for idx in f:
                    population[idx]['cd'] = cd.get(idx, 0)

            offspring = []
            while len(offspring) < pop_size:
                p1 = self.tournament_selection(population)
                p2 = self.tournament_selection(population)
                child = self.ordered_crossover_fixed(p1['route'], p2['route']) if random.random() < cx_prob else p1['route'][:]
                child = self.swap_mutation_fixed(child, mut_prob)
                child = self.enforce_order(child)
                # enforce start & end
                if child[0] != self.start_idx:
                    child.remove(self.start_idx)
                    child = [self.start_idx] + child
                if child[-1] != self.end_idx:
                    child.remove(self.end_idx)
                    child = child + [self.end_idx]
                offspring.append({'route': child, 'objs': None})
            evaluate(offspring)

            combined = population + offspring
            comb_objs = [ind['objs'] for ind in combined]
            fronts = self.fast_non_dominated_sort(comb_objs)
            new_pop = []
            for f in fronts:
                if len(new_pop) + len(f) <= pop_size:
                    for idx in f:
                        new_pop.append(combined[idx])
                else:
                    f_objs = [(idx, combined[idx]['objs']) for idx in f]
                    cd = self.crowding_distance(f_objs)
                    f_sorted = sorted(f, key=lambda i: cd.get(i, 0), reverse=True)
                    remain = pop_size - len(new_pop)
                    for idx in f_sorted[:remain]:
                        new_pop.append(combined[idx])
                    break
            population = new_pop

            # progress print for debugging - in streamlit we'll show spinner instead
            if (gen + 1) % 50 == 0:
                best_dist = min(ind['objs'][0] for ind in population)
                print(f"Gen {gen + 1}/{gens}: best_distance={best_dist:.2f}")

        # final pareto front (first front)
        fronts = self.fast_non_dominated_sort([ind['objs'] for ind in population])
        pareto = [population[i] for i in fronts[0]]
        return pareto
