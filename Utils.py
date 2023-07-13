import numpy as np
import mfem.par as mfem

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

# In 1D, it's just a scalar.
# In 2D, store as [e_xx, e_yy, e_xy].
# In 3D, store as [e_xx, e_yy, e_zz, e_xy, e_yz, e_xz]
# By convention, these get expanded to
# in 1D, a scalar;
# in 2D, [e_xx  , e_xy/2]
#        [e_xy/2, e_yy  ];
# in 3D, [e_xx  , e_xy/2, e_xz/2]
#        [e_xy/2, e_yy  , e_yz/2]
#        [e_xz/2, e_yz/2, e_zz  ]
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
        matrix[0,1], matrix[1,0] = (components[2]/2.0,)*2
    elif space_dims == 3:
        matrix = mfem.DenseMatrix(3)
        for i in range(3):
            matrix[i,i] = components[i]
        matrix[0,1], matrix[1,0] = (components[3]/2.0,)*2
        matrix[1,2], matrix[2,1] = (components[4]/2.0,)*2
        matrix[0,2], matrix[2,0] = (components[5]/2.0,)*2

    return matrix

def flattenSymmetricTensor(tensor):
    dims = tensor.Height()
    num_components = dims*(dims+1)//2
    components = mfem.Vector(num_components)

    for i in range(dims):
        components[i] = tensor[i,i]
    if dims == 1:
        pass
    elif dims == 2:
        # Multiply by 2 because, by convention,
        # the components will be multiplied by 1/2 when unflattened.
        components[2] = 2.0 * tensor[0,1]
    elif dims == 3:
        # Same here.
        components[3] = 2.0 * tensor[0,1]
        components[4] = 2.0 * tensor[1,2]
        components[5] = 2.0 * tensor[0,2]

    return components
