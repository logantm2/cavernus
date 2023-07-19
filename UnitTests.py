import Utils
import Integrators
import ElasticityTensors
import CreepStrainRates

import mfem.par as mfem
import numpy as np
from scipy.constants import gas_constant
import unittest

class UtilTests(unittest.TestCase):
    def testFlattenThenUnflatten(self):
        for dims in range(1,4):
            original = mfem.DenseMatrix(dims)

            for i in range(dims):
                for j in range(dims):
                    original[i,j] = (i+1)*(j+1)

            original.Symmetrize()

            tested = Utils.unflattenSymmetricTensor(Utils.flattenSymmetricTensor(original))

            for i in range(dims):
                for j in range(dims):
                    self.assertAlmostEqual(
                        original[i,j],
                        tested[i,j],
                        msg=f"Failing at index {i},{j} with dimensions {dims}"
                    )

    def testVonMisesStress(self):
        # There is a quick formula for 2D that we wanna check our general
        # function against.
        # \sigma_{vM} = sqrt( sigma_xx^2 - \sigma_xx \sigma_yy + \sigma_yy^2 + 3 \sigma_xy )
        dims = 2
        original = mfem.DenseMatrix(dims)

        for i in range(dims):
            for j in range(dims):
                original[i,j] = (i+1)*(j+1)

        original.Symmetrize()

        analytic = np.sqrt(np.square(original[0,0]) - original[0,0] * original[1,1] + np.square(original[1,1]) + 3 * np.square(original[0,1]))

        flat_original = Utils.flattenSymmetricTensor(original)
        stress_deviator, hydrostatic_stress = Utils.calcFlattenedDeviator(flat_original)
        vM_stress = Utils.calcVonMisesStress(stress_deviator, hydrostatic_stress)

        self.assertAlmostEqual(analytic, vM_stress)

class IntegratorTests(unittest.TestCase):
    def testElasticIntegrator(self):
        # Test on the triangle
        # x > 0, y < 2, y > x
        # with basis order 1.
        # There are 6 basis functions for this finite element:
        # psi_1 = (1-y/2  , 0      )
        # psi_2 = (x/2    , 0      )
        # psi_3 = (y/2-x/2, 0      )
        # psi_4 = (0      , 1-y/2  )
        # psi_5 = (0      , x/2    )
        # psi_6 = (0      , y/2-x/2)
        # The gradients of the basis functions are:
        # \nabla \psi_1 = (   0,   0)
        #                 (-1/2,   0)
        # \nabla \psi_2 = ( 1/2,   0)
        #                 ( 0  ,   0)
        # \nabla \psi_3 = (-1/2,   0)
        #                 ( 1/2,   0)
        # \nabla \psi_4 = (   0,   0)
        #                 (   0,-1/2)
        # \nabla \psi_5 = (   0, 1/2)
        #                 (   0,   0)
        # \nabla \psi_6 = (   0,-1/2)
        #                 (   0, 1/2)
        # The symmetric gradients are therefore:
        # \nabla^s \psi_1 = (   0,-1/4)
        #                   (-1/4, 0  )
        # \nabla^s \psi_2 = ( 1/2, 0  )
        #                   ( 0  , 0  )
        # \nabla^s \psi_3 = (-1/2, 1/4)
        #                   ( 1/4, 0  )
        # \nabla^s \psi_4 = ( 0  , 0  )
        #                   ( 0  ,-1/2)
        # \nabla^s \psi_5 = ( 0  , 1/4)
        #                   ( 1/4, 0  )
        # \nabla^s \psi_6 = ( 0  ,-1/4)
        #                   (-1/4, 1/2)
        # Use an isotropic, Cartesian elasticity tensor
        # C_{ijkl} = \lambda \delta_{ij} \delta_{kl} + \mu (\delta_{ik} \delta_{jl} + \delta{il} \delta_{kj})
        # This contracted with each of the symmetric gradients give:
        # C \nabla^s \psi_1 = (               0,          -\mu/2)
        #                     (          -\mu/2,               0)
        # C \nabla^s \psi_2 = ( \lambda/2+\mu  ,               0)
        #                     (               0, \lambda/2      )
        # C \nabla^s \psi_3 = (-\lambda/2-\mu  ,           \mu/2)
        #                     (           \mu/2,-\lambda/2      )
        # C \nabla^s \psi_4 = (-\lambda/2      ,               0)
        #                     (               0,-\lambda/2-\mu  )
        # C \nabla^s \psi_5 = (               0,           \mu/2)
        #                     (           \mu/2,               0)
        # C \nabla^s \psi_6 = ( \lambda/2      ,          -\mu/2)
        #                     (          -\mu/2, \lambda/2+\mu  )
        # Finally the Frobenius norms in matrix form are:
        # (           \mu/4,               0,          - \mu/4,               0,          -\mu/4,            \mu/4)
        # (               0, \lambda/4+\mu/2,-\lambda/4- \mu/2,-\lambda/4      ,               0, \lambda/4       )
        # (          -\mu/4,-\lambda/4-\mu/2, \lambda/4+3\mu/4, \lambda/4      ,           \mu/4,-\lambda/4- \mu/4)
        # (               0,-\lambda/4      , \lambda/4       , \lambda/4+\mu/2,               0,-\lambda/4- \mu/2)
        # (          -\mu/4,               0,            \mu/4,               0,           \mu/4,          - \mu/4)
        # (           \mu/4, \lambda/4      ,-\lambda/4- \mu/4,-\lambda/4-\mu/2,          -\mu/4, \lambda/4+3\mu/4)
        # The integrals of these over the triangle is
        # simply these multiplied by -2, since they are constant over the triangle.

        # Make a triangle mesh
        mesh = mfem.Mesh.MakeCartesian2D(
            1,
            1,
            mfem.Element.TRIANGLE,
            True,
            2.0,
            2.0,
            False
        )
        Trans = mesh.GetElementTransformation(0)
        finite_element = mfem.H1_TriangleElement(1)

        # Lame constants \lambda and \mu
        l = 2.0
        mu = 3.0
        elasticity_tensor = ElasticityTensors.ConstantIsotropicElasticityTensor(l, mu)

        analytic_solution = np.zeros((6,6))
        analytic_solution[0,0] =           mu/4.0
        analytic_solution[0,1] = 0.0
        analytic_solution[0,2] =          -mu/4.0
        analytic_solution[0,3] = 0.0
        analytic_solution[0,4] =          -mu/4.0
        analytic_solution[0,5] =           mu/4.0

        analytic_solution[1,0] = 0.0
        analytic_solution[1,1] = l/4.0 +   mu/2.0
        analytic_solution[1,2] =-l/4.0 -   mu/2.0
        analytic_solution[1,3] =-l/4.0
        analytic_solution[1,4] = 0.0
        analytic_solution[1,5] = l/4.0

        analytic_solution[2,0] =          -mu/4.0
        analytic_solution[2,1] =-l/4.0-    mu/2.0
        analytic_solution[2,2] = l/4.0+3.0*mu/4.0
        analytic_solution[2,3] = l/4.0
        analytic_solution[2,4] =           mu/4.0
        analytic_solution[2,5] =-l/4.0-    mu/4.0

        analytic_solution[3,0] = 0.0
        analytic_solution[3,1] =-l/4.0
        analytic_solution[3,2] = l/4.0
        analytic_solution[3,3] = l/4.0+    mu/2.0
        analytic_solution[3,4] = 0.0
        analytic_solution[3,5] =-l/4.0-    mu/2.0

        analytic_solution[4,0] =          -mu/4.0
        analytic_solution[4,1] = 0.0
        analytic_solution[4,2] =           mu/4.0
        analytic_solution[4,3] = 0.0
        analytic_solution[4,4] =           mu/4.0
        analytic_solution[4,5] =          -mu/4.0

        analytic_solution[5,0] =           mu/4.0
        analytic_solution[5,1] = l/4.0
        analytic_solution[5,2] =-l/4.0-    mu/4.0
        analytic_solution[5,3] =-l/4.0-    mu/2.0
        analytic_solution[5,4] =          -mu/4.0
        analytic_solution[5,5] = l/4.0+3.0*mu/4.0

        analytic_solution *= -2.0

        integrator = Integrators.ElasticIntegrator(elasticity_tensor)

        elmat = mfem.DenseMatrix(2)
        integrator.AssembleElementMatrix(finite_element, Trans, elmat)
        self.assertEqual(6, elmat.Height())
        self.assertEqual(6, elmat.Width())

        for i in range(6):
            for j in range(6):
                self.assertAlmostEqual(analytic_solution[i,j], elmat[i,j], msg=f"Failing at index {i},{j}")

    def testInelasticIntegrator(self):
        # Test on the triangle
        # x > 0, y < 2, y > x
        # with basis order 1.
        # There are 3 scalar basis functions for this finite element:
        # psi_1 = 1-y/2
        # psi_2 = x/2
        # psi_3 = y/2-x/2
        # There are 6 test functions:
        # \hat{\psi_{1,2,3}} = ( \psi_{1,2,3} )
        #                      (       0      )
        # \hat{\psi_{4,5,6}} = (       0      )
        #                      ( \psi_{1,2,3} )
        # There are 9 trial functions (using rt2 to denote sqrt(2)):
        # \bar{\psi_{1,2,3}} = ( \psi_{1,2,3}    ,      0           )
        #                      (      0          ,      0           )
        # \bar{\psi_{4,5,6}} = (      0          ,      0           )
        #                      (      0          , \psi_{1,2,3}     )
        # \bar{\psi_{7,8,9}} = (      0          , \psi_{1,2,3}/rt2 )
        #                      ( \psi_{1,2,3}/rt2,      0           )
        # Use an isotropic, Cartesian elasticity tensor
        # C_{ijkl} = \lambda \delta_{ij} \delta_{kl} + \mu (\delta_{ik} \delta_{jl} + \delta{il} \delta_{kj})
        # This contracted with each of the test functions give:
        # C \bar{\psi_{1,2,3}} = ( (\lambda +     2 \mu) \psi_{1,2,3},                                   0)
        #                        (                                  0, \lambda \psi_{1,2,3}               )
        # C \bar{\psi_{4,5,6}} = ( \lambda \psi_{1,2,3}              ,                                   0)
        #                        (                                  0, (\lambda + 2 \mu) \psi_{1,2,3}     )
        # C \bar{\psi_{7,8,9}} = (                                  0, rt2 \mu \psi_{1,2,3}               )
        #                        ( rt2 \mu \psi_{1,2,3}              ,                                   0)
        # Contracting these with the nabla leads to:
        # \nabla C \bar{\psi_1} = (                 0)
        #                         (-\lambda/2        )
        # \nabla C \bar{\psi_2} = ( \lambda/2+\mu    )
        #                         (                 0)
        # \nabla C \bar{\psi_3} = (-\lambda/2-\mu    )
        #                         ( \lambda/2        )
        # \nabla C \bar{\psi_4} = (                 0)
        #                         (-\lambda/2-\mu    )
        # \nabla C \bar{\psi_5} = ( \lambda/2        )
        #                         (                 0)
        # \nabla C \bar{\psi_6} = (-\lambda/2        )
        #                         ( \lambda/2+\mu    )
        # \nabla C \bar{\psi_7} = (          -\mu/rt2)
        #                         (                 0)
        # \nabla C \bar{\psi_8} = (                 0)
        #                         (           \mu/rt2)
        # \nabla C \bar{\psi_9} = (           \mu/rt2)
        #                         (          -\mu/rt2)
        # I'm gonna call these the inelastic stress gradients.
        # The integrals of the basis functions over this triangle are all 2/3.
        # Since the inelastic stress gradients are constant over the triangle,
        # the integrals of these gradients dotted with the test functions
        # are equal to the gradients dotted with the integrals of the test functions.
        # Hence they are, in matrix form,
        # (               0, \lambda/2+\mu  ,-\lambda/2-\mu  ,               0, \lambda/2      ,-\lambda/2      ,          -\mu/rt2,                 0,           \mu/rt2)
        # (               0, \lambda/2+\mu  ,-\lambda/2-\mu  ,               0, \lambda/2      ,-\lambda/2      ,          -\mu/rt2,                 0,           \mu/rt2)
        # (               0, \lambda/2+\mu  ,-\lambda/2-\mu  ,               0, \lambda/2      ,-\lambda/2      ,          -\mu/rt2,                 0,           \mu/rt2)
        # (-\lambda/2      ,               0, \lambda/2      ,-\lambda/2-\mu  ,               0, \lambda/2+\mu  ,                 0,           \mu/rt2,          -\mu/rt2)
        # (-\lambda/2      ,               0, \lambda/2      ,-\lambda/2-\mu  ,               0, \lambda/2+\mu  ,                 0,           \mu/rt2,          -\mu/rt2)
        # (-\lambda/2      ,               0, \lambda/2      ,-\lambda/2-\mu  ,               0, \lambda/2+\mu  ,                 0,           \mu/rt2,          -\mu/rt2)
        # all times 2/3.

        # Make a triangle mesh
        mesh = mfem.Mesh.MakeCartesian2D(
            1,
            1,
            mfem.Element.TRIANGLE,
            True,
            2.0,
            2.0,
            False
        )
        Trans = mesh.GetElementTransformation(0)
        finite_element = mfem.H1_TriangleElement(1)

        # Lame constants \lambda and \mu
        l = 2.0
        mu = 3.0
        elasticity_tensor = ElasticityTensors.ConstantIsotropicElasticityTensor(l, mu)

        sqrt2 = np.sqrt(2.0)

        analytic_solution = np.zeros((6,9))
        for i in range(3):
            analytic_solution[i  ,0] = 0.0
            analytic_solution[i  ,1] = l/2.0 + mu
            analytic_solution[i  ,2] =-l/2.0 - mu
            analytic_solution[i  ,3] = 0.0
            analytic_solution[i  ,4] = l/2.0
            analytic_solution[i  ,5] =-l/2.0
            analytic_solution[i  ,6] =       - mu/sqrt2
            analytic_solution[i  ,7] = 0.0
            analytic_solution[i  ,8] =         mu/sqrt2

            analytic_solution[i+3,0] =-l/2.0
            analytic_solution[i+3,1] = 0.0
            analytic_solution[i+3,2] = l/2.0
            analytic_solution[i+3,3] =-l/2.0 - mu
            analytic_solution[i+3,4] = 0.0
            analytic_solution[i+3,5] = l/2.0 + mu
            analytic_solution[i+3,6] = 0.0
            analytic_solution[i+3,7] =         mu/sqrt2
            analytic_solution[i+3,8] =       - mu/sqrt2

        analytic_solution *= 2.0/3.0

        integrator = Integrators.InelasticIntegrator(elasticity_tensor)

        elmat = mfem.DenseMatrix(2)
        integrator.AssembleElementMatrix2(finite_element, finite_element, Trans, elmat)
        self.assertEqual(6, elmat.Height())
        self.assertEqual(9, elmat.Width())

        for i in range(6):
            for j in range(9):
                self.assertAlmostEqual(analytic_solution[i,j], elmat[i,j], msg=f"Failing at index {i},{j}")

    def testCreepStrainIntegrator(self):
        # Test on the triangle
        # x > 0, y < 2, y > x
        # with basis order 1.
        # There are 3 scalar basis functions for this finite element:
        # psi_1 = 1-y/2
        # psi_2 = x/2
        # psi_3 = y/2-x/2
        # We will define the problem such that the displacement throughout
        # the triangle is given by
        # u(x,y) = [2 - y]
        #          [y - x].
        # This is done by defining the interpolation matrix to be
        # [2, 0, 0]
        # [0, 0, 2].
        # Similarly, the creep strain will be given by
        # \epsilon_cr = [2, 1]
        #               [1, 3]
        # This is done by defining the interpolation matrix to be
        # [  2,   2,   2]
        # [  3,   3,   3]
        # [rt2, rt2, rt2].
        # The gradient of the displacement is
        # \nabla u(x,y) = [ 0,-1]
        #                 [-1, 1].
        # Since this is already symmetric, it is also the total strain.
        # The elastic strain is the total strain minus the creep strain,
        # \epsilon_el = [-2,-2]
        #               [-2,-2].
        # Use an isotropic, Cartesian elasticity tensor
        # C_{ijkl} = \lambda \delta_{ij} \delta_{kl} + \mu (\delta_{ik} \delta_{jl} + \delta{il} \delta_{kj}).
        # The stress is the tensor contraction of the elastic strain with
        # the elasticity tensor, which is
        # \sigma = [-4\lambda -4\mu,          -4\mu]
        #          [          -4\mu,-4\lambda -4\mu].
        # The formula for the von Mises stress in 2D is
        # \sigma_{vM} = sqrt( sigma_xx^2 - \sigma_xx \sigma_yy + \sigma_yy^2 + 3 \sigma_xy^2 )
        # This ends up as
        # sqrt( 16 \lambda^2 + 32 \lambda \mu + 64 \mu^2 ).
        # The hydrostatic stress is the trace of the stress tensor divided by 3,
        # \pi = -8/3(\lambda+\mu).
        # The stress deviator is \sigma - \pi I,
        # s = [-4/3(\lambda+\mu),            -4\mu]
        #     [            -4\mu,-4/3(\lambda+\mu)].
        # The creep strain rate is therefore, with n=3,
        # \dot{\epsilon_cr} = 3/2 a \exp(-Q/RT) \sigma_{vM}^2 s
        # = a \exp(-Q/RT) (24 \lambda^2 + 48 \lambda \mu + 96 \mu^2) [-4/3(\lambda+\mu),            -4\mu]
        #                                                            [            -4\mu,-4/3(\lambda+\mu)].
        # There are 9 test functions (using rt2 to denote sqrt(2)):
        # \bar{\psi_{1,2,3}} = ( \psi_{1,2,3}    ,      0           )
        #                      (      0          ,      0           )
        # \bar{\psi_{4,5,6}} = (      0          ,      0           )
        #                      (      0          , \psi_{1,2,3}     )
        # \bar{\psi_{7,8,9}} = (      0          , \psi_{1,2,3}/rt2 )
        #                      ( \psi_{1,2,3}/rt2,      0           )
        # The operator takes the Frobenius norm of each of these with the creep strain rate
        # and integrates in the triangle to yield
        # \vec{F}_{1-6}   = - a \exp(-Q/RT) (24 \lambda^2 + 48 \lambda \mu + 96 \mu^2) 8/9 (\lambda+\mu)
        # \vec{F}_{7,8,9} = - a \exp(-Q/RT) (24 \lambda^2 + 48 \lambda \mu + 96 \mu^2) rt2 8/3 \mu
        SQRT2 = np.sqrt(2.0)

        # Make a triangle mesh
        mesh = mfem.Mesh.MakeCartesian2D(
            1,
            1,
            mfem.Element.TRIANGLE,
            True,
            2.0,
            2.0,
            False
        )
        Trans = mesh.GetElementTransformation(0)
        space_dims = mesh.Dimension()
        num_symtensor_dims = space_dims * (space_dims+1) // 2

        finite_element = mfem.H1_TriangleElement(1)
        num_dofs = finite_element.GetDof()

        # Lame constants \lambda and \mu
        l = 2.0
        mu = 3.0
        elasticity_tensor = ElasticityTensors.ConstantIsotropicElasticityTensor(l, mu)

        carter_constant = 1.0
        carter_exponent = 3.0
        carter_activation_energy = 3.0
        temperature = 4.0
        creep_strain_rate = CreepStrainRates.CarterCreepStrainRate(
            carter_constant,
            carter_exponent,
            carter_activation_energy,
            temperature,
            elasticity_tensor
        )

        integrator = Integrators.CreepStrainRateIntegrator(creep_strain_rate)

        elfun = mfem.Vector(num_dofs * (space_dims + num_symtensor_dims))
        # The elfun Vector define what the displacement and
        # creep strain will be as functions of space.
        u_elfun = mfem.Vector(elfun, 0, space_dims * num_dofs)
        u_elfun_mat = mfem.DenseMatrix(u_elfun.GetData(), num_dofs, space_dims)
        u_elfun_mat.Assign(0.0)
        u_elfun_mat[0,0] = 2.0
        u_elfun_mat[2,1] = 2.0

        e_elfun = mfem.Vector(elfun, num_dofs * space_dims, num_symtensor_dims * num_dofs)
        e_elfun_mat = mfem.DenseMatrix(e_elfun.GetData(), num_dofs, num_symtensor_dims)
        e_elfun_mat.Assign(0.0)
        for i in range(3):
            e_elfun_mat[i,0] = 2.0
            e_elfun_mat[i,1] = 3.0
            e_elfun_mat[i,2] = SQRT2

        preintegral_constant = - carter_constant * np.exp(-carter_activation_energy/gas_constant/temperature) * (24.0*l**2.0 + 48.0*l*mu + 96.0*mu**2.0)
        analytic = np.zeros(9)
        for i in range(6):
            analytic[i] = preintegral_constant * 8./9. * (l+mu)
        for i in range(6,9):
            analytic[i] = preintegral_constant * SQRT2 * 8./3. * mu

        elvect = mfem.Vector()
        integrator.AssembleElementVector(
            finite_element,
            Trans,
            elfun,
            elvect
        )

        u_elvect = mfem.Vector(elvect, 0, space_dims * num_dofs)
        e_elvect = mfem.Vector(elvect, num_dofs * space_dims, num_symtensor_dims * num_dofs)
        for i in range(6):
            self.assertAlmostEqual(0.0, u_elvect[i], msg=f"u_elvect is nonzero at index {i}")
        for i in range(9):
            self.assertAlmostEqual(analytic[i], e_elvect[i], msg=f"Failing with test function {i}. Analytic = {analytic[i]}, computed = {e_elvect[i]}")

if __name__ == "__main__":
    unittest.main()
