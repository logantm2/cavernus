import mfem.par as mfem
import numpy as np
import abc

# Used to specify the displacement, as a vector, on the boundary.
class DirichletBoundaryCondition(mfem.VectorPyCoefficientT, abc.ABC):
    def __init__(
        self,
        boundary_attribute
    ):
        self.boundary_attribute = boundary_attribute
        self.type = "dirichlet"

    def setVDim(self, vdim):
        mfem.VectorPyCoefficientT.__init__(self, vdim)

    @abc.abstractmethod
    def EvalValue(self, x, t):
        pass

class ZeroDirichletBoundaryCondition(DirichletBoundaryCondition):
    def __init__(self, boundary_attribute):
        super().__init__(boundary_attribute)

    def EvalValue(self, x, t):
        return np.zeros(x.size)

# Used to specify the stress, as a matrix, on the boundary.
class NeumannBoundaryCondition(mfem.VectorPyCoefficientBase, abc.ABC):
    def __init__(
        self,
        boundary_attribute
    ):
        self.boundary_attribute = boundary_attribute
        self.type = "neumann"

    def setVDim(self, vdim):
        mfem.VectorPyCoefficientBase.__init__(self, vdim, 1)

    # Return the stress tensor at this point in time.
    @abc.abstractmethod
    def EvalValue(self, x, t):
        pass

    def Eval(self, V, T, ip):
        vdim = self.GetVDim()
        x = mfem.Vector(3)
        T.Transform(ip, x)
        V.SetSize(vdim)
        V.Assign(0.0)

        stress = self.EvalValue(x.GetDataArray(), self.GetTime())

        # Compute the normal on this face.
        normal = mfem.Vector(vdim)
        T.SetIntPoint(ip)
        mfem.CalcOrtho(T.Jacobian(), normal)
        # Normal is not unit, so normalize it.
        norm_mag = normal.Norml2()
        for i in range(vdim):
            normal[i] = normal[i] / norm_mag

        # Contract normal with stress tensor.
        for i in range(vdim):
            for j in range(vdim):
                V[i] = V[i] + stress[i,j] * normal[j]

class ZeroNeumannBoundaryCondition(NeumannBoundaryCondition):
    def EvalValue(self, x, t):
        return np.zeros((x.size, x.size))
