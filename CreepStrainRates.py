import Utils
import mfem.par as mfem
from scipy.constants import gas_constant
import numpy as np
import abc

class CreepStrainRate(abc.ABC):
    # Evaluate the elasticity tensor at position x
    # and at index ijkl.
    @abc.abstractmethod
    def evaluate(self, displacement, creep_strain, creep_strain_rate):
        pass

class CarterCreepStrainRate(CreepStrainRate):
    def __init__(
        self,
        constant,
        exponent,
        activation_energy,
        temperature,
        elasticity_tensor
    ):
        self.constant = constant
        self.exponent = exponent
        self.activation_energy = activation_energy
        self.temperature = temperature
        self.elasticity_tensor = elasticity_tensor

    def evaluate(
        self,
        displacement_fe,
        creep_strain_fe,
        element_transformation,
        displacement_mat,
        creep_strain_mat,
        creep_strain_rate
    ):
        u_num_dofs = displacement_fe.GetDof()
        u_num_dims = displacement_fe.GetDim()
        e_num_dofs = creep_strain_fe.GetDof()
        e_num_dims = u_num_dims * (u_num_dims+1) // 2

        u_dshape = mfem.DenseMatrix(u_num_dofs, u_num_dims)
        displacement_fe.CalcPhysDShape(element_transformation, u_dshape)
        e_shapef = mfem.Vector(e_num_dofs)
        creep_strain_fe.CalcPhysShape(element_transformation, e_shapef)

        grad_u = mfem.DenseMatrix(u_num_dims)
        mfem.MultAtB(displacement_mat, u_dshape, grad_u)
        grad_u.Symmetrize()
        total_strain = Utils.flattenSymmetricTensor(grad_u)

        creep_strain = mfem.Vector(e_num_dims)
        creep_strain_mat.MultTranspose(e_shapef, creep_strain)

        elastic_strain = mfem.Vector(e_num_dims)
        # mfem subtract doesn't work for some reason
        # mfem.subtract(total_strain, creep_strain, elastic_strain)
        for i in range(e_num_dims):
            elastic_strain[i] = total_strain[i] - creep_strain[i]

        x = element_transformation.Transform(element_transformation.GetIntPoint())

        stress = self.elasticity_tensor.calcFlattenedContraction(x, elastic_strain)
        stress_deviator, hydrostatic_stress = Utils.calcFlattenedDeviator(stress)
        vM_stress = Utils.calcVonMisesStress(stress_deviator, hydrostatic_stress)
        rate_coefficient = 3./2. * self.constant * np.exp(-self.activation_energy/gas_constant/self.temperature) * np.abs(np.power(vM_stress, self.exponent-2)) * vM_stress

        creep_strain_rate.SetSize(e_num_dims)
        creep_strain_rate.Assign(stress_deviator)
        creep_strain_rate *= rate_coefficient
