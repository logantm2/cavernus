class BoundaryCondition():
    def __init__(
        self,
        boundary_attribute,
        type
    ):
        self.boundary_attribute = boundary_attribute
        self.type = type

    def eval(self, x, t, values):
        values.Assign(0.0)
