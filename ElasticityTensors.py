import Utils
import mfem.par as mfem
import abc

class ElasticityTensor(abc.ABC):
    # Evaluate the elasticity tensor at position x
    # and at index ijkl.
    @abc.abstractmethod
    def evaluate(self, x, i, j, k, l):
        pass

    def calcContraction(self, x, input_tensor):
        dims = input_tensor.Height()
        contraction = mfem.DenseMatrix(dims)
        contraction.Assign(0.0)
        for i in range(dims):
            for j in range(dims):
                for k in range(dims):
                    for l in range(dims):
                        contraction[i,j] += self.evaluate(x, i, j, k, l) * input_tensor[k, l]

        return contraction

    # Only works with symmetric input tensors.
    # TODO: optimize this. Can probably skip the unflatten->flatten steps
    def calcFlattenedContraction(self, x, flattened_symmetric_input_tensor):
        symmetric_input_tensor = Utils.unflattenSymmetricTensor(flattened_symmetric_input_tensor)
        contraction = self.calcContraction(x, symmetric_input_tensor)
        return Utils.flattenSymmetricTensor(contraction)

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
