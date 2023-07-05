import Integrators
import ElasticityTensors

import mfem.par as mfem
import numpy as np
import unittest

class IntegratorTests(unittest.TestCase):
    def testElasticityOperator(self):
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
                self.assertEqual(analytic_solution[i,j], elmat[i,j], f"Failing at index {i},{j}")

if __name__ == "__main__":
    unittest.main()
