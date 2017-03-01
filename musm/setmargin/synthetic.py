from .problem import Problem

TEMPLATE = """\
int: DOMAIN_SIZE = {domain_size};
set of int: DOMAIN = 1 .. DOMAIN_SIZE;
"""

class Synthetic(Problem):
    def __init__(self, domain_size=4, **kwargs):
        self.domain_size = domain_size
        super().__init__(TEMPLATE.format(**locals()))

    def _mzn_attributes(self, prefix=''):
        attributes = {}
        for i in range(1, self.domain_size + 1):
            attr = '{}a{}'.format(prefix, i)
            attributes[attr] = 'var DOMAIN'
        return attributes

    def _constraints(self, prefix=''):
        return []
