import numpy as np
import gurobipy as gurobi

from .problem import Problem

class PC(Problem):
    _ATTRIBUTES = [
        ('cpu', 37),
        ('hd', 10),
        ('manufacturer', 8),
        ('ram', 10),
        ('monitor', 8),
        ('pctype', 3),
    ]

    _ATTR_TO_COSTS = {
        'pctype': [50, 0, 80],
        'manufacturer': [100, 0, 100, 50, 0, 0, 50, 50],
        'cpu' : [
            1.4*100, 1.4*130, 1.1*70, 1.1*90, 1.2*80, 1.2*50, 1.2*60, 1.2*80,
            1.2*90, 1.2*100, 1.2*110, 1.2*120, 1.2*130, 1.2*140, 1.2*170,
            1.5*50, 1.5*60, 1.5*80, 1.5*90, 1.5*100, 1.5*110, 1.5*130, 1.5*150,
            1.5*160, 1.5*170, 1.5*180, 1.5*220, 1.4*27, 1.4*30, 1.4*40, 1.4*45,
            1.4*50, 1.4*55, 1.4*60, 1.4*70, 1.6*70, 1.6*73,
        ],
        'monitor': [
            0.6*100, 0.6*104, 0.6*120, 0.6*133, 0.6*140, 0.6*150, 0.6*170,
            0.6*210
        ],
        'ram': [
            0.8*64, 0.8*128, 0.8*160, 0.8*192, 0.8*256, 0.8*320, 0.8*384,
            0.8*512, 0.8*1024, 0.8*2048
        ],
        'hd': [
            4*8, 4*10, 4*12, 4*15, 4*20, 4*30, 4*40, 4*60, 4*80, 4*120
        ],
    }

    def __init__(self, **kwargs):
        super().__init__(sum(attr[1] for attr in self._ATTRIBUTES))
        self.cost_matrix = np.hstack([
                np.array(self._ATTR_TO_COSTS[attr], dtype=float)
                for attr, _ in self._ATTRIBUTES
            ]).reshape((1, -1)) / 2754.4

    def _add_constraints(self, model, x):
        base, offs = 0, {}
        for attr, size in self._ATTRIBUTES:
            offs[attr] = base
            x_attr = [x[z] for z in range(base, base + size)]
            model.addConstr(gurobi.quicksum(x_attr) == 1)
            base += size

        def implies(head, body):
            # NOTE here we subtract 1 from head and body bits because the bit
            # numbers in the constraints were computed starting from one, to
            # work in MiniZinc, while Gurobi expects them to start from zero
            head = 1 - x[head - 1]
            body = gurobi.quicksum([x[i - 1] for i in body])
            return model.addConstr(head + body >= 1)

        # Manufacturer -> Type
        implies(offs['manufacturer'] + 2, [offs['pctype'] + i for i in [1, 2]])
        implies(offs['manufacturer'] + 4, [offs['pctype'] + 1])
        implies(offs['manufacturer'] + 6, [offs['pctype'] + 2])
        implies(offs['manufacturer'] + 7, [offs['pctype'] + i for i in [1, 3]])

        # Manufacturer -> CPU
        implies(offs['manufacturer'] + 1, [offs['cpu'] + i for i in range(28, 37+1)])
        implies(offs['manufacturer'] + 2, [offs['cpu'] + i for i in list(range(1, 4+1)) + list(range(6, 27+1))])
        implies(offs['manufacturer'] + 7, [offs['cpu'] + i for i in list(range(1, 4+1)) + list(range(6, 27+1))])
        implies(offs['manufacturer'] + 4, [offs['cpu'] + i for i in range(5, 27+1)])
        implies(offs['manufacturer'] + 3, [offs['cpu'] + i for i in range(6, 27+1)])
        implies(offs['manufacturer'] + 5, [offs['cpu'] + i for i in range(6, 27+1)])
        implies(offs['manufacturer'] + 8, [offs['cpu'] + i for i in range(6, 27+1)])
        implies(offs['manufacturer'] + 6, [offs['cpu'] + i for i in range(16, 27+1)])

        # Type -> RAM
        implies(offs['pctype'] + 1, [offs['ram'] + i for i in range(1, 9+1)])
        implies(offs['pctype'] + 2, [offs['ram'] + i for i in [2, 5, 8, 9]])
        implies(offs['pctype'] + 3, [offs['ram'] + i for i in [5, 8, 9, 10]])

        # Type -> HD
        implies(offs['pctype'] + 1, [offs['hd'] + i for i in range(1, 6+1)])
        implies(offs['pctype'] + 2, [offs['hd'] + i for i in range(5, 10+1)])
        implies(offs['pctype'] + 3, [offs['hd'] + i for i in range(5, 10+1)])

        # Type -> Monitor
        implies(offs['pctype'] + 1, [offs['monitor'] + i for i in range(1, 6+1)])
        implies(offs['pctype'] + 2, [offs['monitor'] + i for i in range(6, 8+1)])
        implies(offs['pctype'] + 3, [offs['monitor'] + i for i in range(6, 8+1)])
