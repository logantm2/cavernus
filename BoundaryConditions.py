import mfem.par as mfem
import numpy as np
import abc

class BoundaryCondition(mfem.VectorPyCoefficient, abc.ABC):
    def __init__(
        self,
        vdim,
        boundary_attribute,
        type
    ):
        mfem.VectorPyCoefficient.__init__(self, vdim)
        self.boundary_attribute = boundary_attribute
        self.type = type

    @abc.abstractmethod
    def EvalValue(self, x):
        pass

class ZeroBoundaryCondition(BoundaryCondition):
    def __init__(self, boundary_attribute, type):
        super().__init__(1, boundary_attribute, type)

    def EvalValue(self, x):
        return np.zeros(x.Size())
