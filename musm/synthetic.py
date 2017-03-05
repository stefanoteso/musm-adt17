from .problem import Problem

class Synthetic(Problem):
    def __init__(self, domain_sizes=4, **kwargs):
        try:
            # Assumes a list of integers
            len(domain_sizes)
            self.domain_sizes = domain_sizes
        except TypeError:
            # Assumes an integer
            self.domain_sizes = [domain_sizes] * domain_sizes
        num_attributes = sum(self.domain_sizes)
        super().__init__('', num_attributes)

    def _constraints(self, x):
        constraints = []
        lo = 0
        for domain_size in self.domain_sizes:
            hi = lo + domain_size
            constraints.append('sum([{x}[z] | z in {lo} + 1 .. {hi}]) == 1'.format(**locals()))
            lo = hi
        return constraints
