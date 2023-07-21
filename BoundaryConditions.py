import mfem.par as mfem
import numpy as np
import abc

class BoundaryCondition(mfem.VectorPyCoefficient, abc.ABC):
    def __init__(
        self,
        boundary_attribute,
        type
    ):
        self.boundary_attribute = boundary_attribute
        self.type = type

    def setVDim(self, vdim):
        mfem.VectorPyCoefficient.__init__(self, vdim)

    @abc.abstractmethod
    def EvalValue(self, x):
        pass

class ZeroBoundaryCondition(BoundaryCondition):
    def __init__(self, boundary_attribute, type):
        super().__init__(boundary_attribute, type)

    def EvalValue(self, x):
        return np.zeros(x.size)

class TestBoundaryCondition(BoundaryCondition):
    def __init__(self, boundary_attribute, type):
        super().__init__(boundary_attribute, type)

    def EvalValue(self, x):
        result = np.zeros(x.size)
        result[0] = 1.0e-5
        return result
