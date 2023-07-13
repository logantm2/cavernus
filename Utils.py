import numpy as np
import mfem.par as mfem

SQRT2 = np.sqrt(2.0)

def logAndContinue(
    level,
    message,
    log_name=""
):
    print(f"[{level}] {log_name}: {message}")

def logAndExit(
    level,
    message,
    log_name="",
    exit_code=1
):
    logAndContinue(level, message, log_name)
    exit(exit_code)

# Symmetric matrices are
# in 1D, a scalar;
# in 2D, [e_xx, e_xy]
#        [e_xy, e_yy];
# in 3D, [e_xx, e_xy, e_xz]
#        [e_xy, e_yy, e_yz]
#        [e_xz, e_yz, e_zz]
# By convention, these get flattened to
# in 1D, just a scalar;
# in 2D, [e_xx, e_yy, sqrt(2)*e_xy];
# in 3D, [e_xx, e_yy, e_zz, sqrt(2)*e_xy, sqrt(2)*e_yz, sqrt(2)*e_xz].
# This means that the Frobenius norm between two symmetric matrices is
# equal to the dot product of their flattened component vectors.
def flattenSymmetricTensor(tensor):
    dims = tensor.Height()
    num_components = dims*(dims+1)//2
    components = mfem.Vector(num_components)

    for i in range(dims):
        components[i] = tensor[i,i]
    if dims == 1:
        pass
    elif dims == 2:
        components[2] = SQRT2 * tensor[0,1]
    elif dims == 3:
        components[3] = SQRT2 * tensor[0,1]
        components[4] = SQRT2 * tensor[1,2]
        components[5] = SQRT2 * tensor[0,2]

    return components

def unflattenSymmetricTensor(components):
    num_components = components.Size()
    if num_components == 1:
        space_dims = 1
    elif num_components == 3:
        space_dims = 2
    elif num_components == 6:
        space_dims = 3
    else:
        logAndExit(
            "error",
            "Invalid number of components!",
            "Utils.unflattenSymmetricTensor"
        )

    if space_dims == 1:
        matrix = mfem.DenseMatrix(1)
        matrix[0,0] = components[0]
    elif space_dims == 2:
        matrix = mfem.DenseMatrix(2)
        matrix[0,0] = components[0]
        matrix[1,1] = components[1]
        matrix[0,1], matrix[1,0] = (components[2]/SQRT2,)*2
    elif space_dims == 3:
        matrix = mfem.DenseMatrix(3)
        for i in range(3):
            matrix[i,i] = components[i]
        matrix[0,1], matrix[1,0] = (components[3]/SQRT2,)*2
        matrix[1,2], matrix[2,1] = (components[4]/SQRT2,)*2
        matrix[0,2], matrix[2,0] = (components[5]/SQRT2,)*2

    return matrix

# Given a flattened symmetric tensor,
# compute the deviator tensor.
# Also computes the mean, which is the trace divided by 3.
# If the given symmetric tensor is a stress tensor,
# the mean is the hydrostatic stress.
def calcFlattenedDeviator(components):
    num_components = components.Size()
    if num_components == 1:
        space_dims = 1
    elif num_components == 3:
        space_dims = 2
    elif num_components == 6:
        space_dims = 3
    else:
        logAndExit(
            "error",
            "Invalid number of components!",
            "Utils.calcFlattenedDeviator"
        )

    mean = 0.0
    for i in range(space_dims):
        mean += components[i]/3.0

    deviator = mfem.Vector(num_components)
    deviator.Assign(components)
    for i in range(space_dims):
        deviator[i] -= mean

    return deviator, mean

# Given a flattened stress deviator,
# compute the von Mises stress, a scalar.
def calcVonMisesStress(stress_deviator_components, hydrostatic_stress):
    num_components = stress_deviator_components.Size()
    if num_components == 1:
        space_dims = 1
    elif num_components == 3:
        space_dims = 2
    elif num_components == 6:
        space_dims = 3
    else:
        logAndExit(
            "error",
            "Invalid number of components!",
            "Utils.calcVonMisesStress"
        )

    vM_stress_sq = stress_deviator_components * stress_deviator_components * 3./2.

    # Even if our stress tensor is in <3 dimensions,
    # the deviator can have a nonzero diagonal all the way down to s_zz.
    # This typically does not impact anything,
    # but will affect the von Mises stress.
    for i in range(3 - space_dims):
        vM_stress_sq += 3./2. * hydrostatic_stress**2.0

    return np.sqrt(vM_stress_sq)
