import abc

class ElasticityTensor(abc.ABC):
    # Evaluate the elasticity tensor at position x
    # and at index ijkl.
    @abc.abstractmethod
    def evaluate(self, x, i, j, k, l):
        pass

class ConstantIsotropicElasticityTensor(ElasticityTensor):
    def __init__(self, l, mu):
        self.l = l
        self.mu = mu

    def evaluate(self, x, i, j, k, l):
        val = 0.0

        if i == j and k == l:
            val += self.l

        if i == k and j == l:
            val += self.mu

        if i == l and j == k:
            val += self.mu

        return val
