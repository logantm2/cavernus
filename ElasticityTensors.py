import Utils
import mfem.par as mfem
import abc

class ElasticityTensor(abc.ABC):
    # Evaluate the elasticity tensor at position x
    # and at index ijkl.
    @abc.abstractmethod
    def evaluate(self, x, i, j, k, l):
        pass

    # TODO: optimize this. Can probably skip the unflatten->flatten steps
    def calcFlattenedStress(self, x, flattened_elastic_strain):
        elastic_strain = Utils.unflattenSymmetricTensor(flattened_elastic_strain)
        dims = elastic_strain.Height()
        stress = mfem.DenseMatrix(dims)
        stress.Assign(0.0)
        for i in range(dims):
            for j in range(dims):
                for k in range(dims):
                    for l in range(dims):
                        stress[i,j] += self.evaluate(x, i, j, k, l) * elastic_strain[k, l]

        return Utils.flattenSymmetricTensor(stress)

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
