from pymzn import MiniZincModel, minizinc

class Problem(object):
    def __init__(self, template):
        self.template = template

    @property
    def num_attributes(self):
        if not hasattr(self, '_num_attributes'):
            attributes = self._mzn_attributes()
            self._num_attributes = len(attributes)
        return self._num_attributes

    def _mzn_attributes(self, prefix=''):
        raise NotImplementedError()

    def _constraints(self, prefix=''):
        raise NotImplementedError()

    @staticmethod
    def _fixup(attr, attr_type):
        if attr_type.startswith('array'):
            attr += ' :: output_array([{}])'.format(_indexset(attr_type))
        return attr

    def infer(self, w):
        model = MiniZincModel(self.template)

        model.par('w', w)

        attributes = self._mzn_attributes()
        for attr, attr_type in attributes.items():
            model.var(attr_type, self._fixup(attr, attr_type))

        for constr in self._constraints():
            model.constraint(constr)

        model.var('var float',  'utility', 'sum(i in ATTRIBUTES)(w[i] * x[i])')
        model.solve('maximize utility')

        return minizinc(model, output_vars=attributes.keys())[0]
