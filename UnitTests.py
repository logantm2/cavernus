import Utils
import Integrators
import ElasticityTensors

import mfem.par as mfem
import numpy as np
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
        # There are 9 trial functions:
        # \bar{\psi_{1,2,3}} = ( \psi_{1,2,3}  ,      0         )
        #                      (      0        ,      0         )
        # \bar{\psi_{4,5,6}} = (      0        ,      0         )
        #                      (      0        , \psi_{1,2,3}   )
        # \bar{\psi_{7,8,9}} = (      0        , \psi_{1,2,3}/2 )
        #                      ( \psi_{1,2,3}/2,      0         )
        # Use an isotropic, Cartesian elasticity tensor
        # C_{ijkl} = \lambda \delta_{ij} \delta_{kl} + \mu (\delta_{ik} \delta_{jl} + \delta{il} \delta_{kj})
        # This contracted with each of the test functions give:
        # C \bar{\psi_{1,2,3}} = ( (\lambda + 2 \mu) \psi_{1,2,3},                               0)
        #                        (                              0, \lambda \psi_{1,2,3}           )
        # C \bar{\psi_{4,5,6}} = ( \lambda \psi_{1,2,3}          ,                               0)
        #                        (                              0, (\lambda + 2 \mu) \psi_{1,2,3} )
        # C \bar{\psi_{7,8,9}} = (                              0, \mu \psi_{1,2,3}               )
        #                        ( \mu \psi_{1,2,3}              ,                               0)
        # Contracting these with the nabla leads to:
        # \nabla C \bar{\psi_1} = (               0)
        #                         (-\lambda/2      )
        # \nabla C \bar{\psi_2} = ( \lambda/2+\mu  )
        #                         (               0)
        # \nabla C \bar{\psi_3} = (-\lambda/2-\mu  )
        #                         ( \lambda/2      )
        # \nabla C \bar{\psi_4} = (               0)
        #                         (-\lambda/2-\mu  )
        # \nabla C \bar{\psi_5} = ( \lambda/2      )
        #                         (               0)
        # \nabla C \bar{\psi_6} = (-\lambda/2      )
        #                         ( \lambda/2+\mu  )
        # \nabla C \bar{\psi_7} = (          -\mu/2)
        #                         (               0)
        # \nabla C \bar{\psi_8} = (               0)
        #                         (           \mu/2)
        # \nabla C \bar{\psi_9} = (           \mu/2)
        #                         (          -\mu/2)
        # I'm gonna call these the inelastic stress gradients.
        # The integrals of the basis functions over this triangle are all 2/3.
        # Since the inelastic stress gradients are constant over the triangle,
        # the integrals of these gradients dotted with the test functions
        # are equal to the gradients dotted with the integrals of the test functions.
        # Hence they are, in matrix form,
        # (               0, \lambda/2+\mu  ,-\lambda/2-\mu  ,               0, \lambda/2      ,-\lambda/2      ,          -\mu/2,               0,           \mu/2)
        # (               0, \lambda/2+\mu  ,-\lambda/2-\mu  ,               0, \lambda/2      ,-\lambda/2      ,          -\mu/2,               0,           \mu/2)
        # (               0, \lambda/2+\mu  ,-\lambda/2-\mu  ,               0, \lambda/2      ,-\lambda/2      ,          -\mu/2,               0,           \mu/2)
        # (-\lambda/2      ,               0, \lambda/2      ,-\lambda/2-\mu  ,               0, \lambda/2+\mu  ,               0,           \mu/2,          -\mu/2)
        # (-\lambda/2      ,               0, \lambda/2      ,-\lambda/2-\mu  ,               0, \lambda/2+\mu  ,               0,           \mu/2,          -\mu/2)
        # (-\lambda/2      ,               0, \lambda/2      ,-\lambda/2-\mu  ,               0, \lambda/2+\mu  ,               0,           \mu/2,          -\mu/2)
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

        analytic_solution = np.zeros((6,9))
        for i in range(3):
            analytic_solution[i  ,0] = 0.0
            analytic_solution[i  ,1] = l/2.0 + mu
            analytic_solution[i  ,2] =-l/2.0 - mu
            analytic_solution[i  ,3] = 0.0
            analytic_solution[i  ,4] = l/2.0
            analytic_solution[i  ,5] =-l/2.0
            analytic_solution[i  ,6] =       - mu/2.0
            analytic_solution[i  ,7] = 0.0
            analytic_solution[i  ,8] =         mu/2.0

            analytic_solution[i+3,0] =-l/2.0
            analytic_solution[i+3,1] = 0.0
            analytic_solution[i+3,2] = l/2.0
            analytic_solution[i+3,3] =-l/2.0 - mu
            analytic_solution[i+3,4] = 0.0
            analytic_solution[i+3,5] = l/2.0 + mu
            analytic_solution[i+3,6] = 0.0
            analytic_solution[i+3,7] =         mu/2.0
            analytic_solution[i+3,8] =       - mu/2.0

        analytic_solution *= 2.0/3.0

        integrator = Integrators.InelasticIntegrator(elasticity_tensor)

        elmat = mfem.DenseMatrix(2)
        integrator.AssembleElementMatrix2(finite_element, finite_element, Trans, elmat)
        self.assertEqual(6, elmat.Height())
        self.assertEqual(9, elmat.Width())

        for i in range(6):
            for j in range(9):
                self.assertAlmostEqual(analytic_solution[i,j], elmat[i,j], msg=f"Failing at index {i},{j}")

if __name__ == "__main__":
    unittest.main()
