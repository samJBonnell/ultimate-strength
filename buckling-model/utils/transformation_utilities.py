import numpy as np

def homogenous_transform(transformation, point=None):
    """
    Given a point in R3 and a 4x4 homogeneous transformation matrix,
    return the transformed point.

    Parameters
    ----------
    transformation : (4,4) numpy.ndarray
        Homogeneous transformation matrix.
    point : point to be transformed
        Point is length-3 or shape (3,1).

    Returns
    -------
    transformed_point : tuple
        Transformed (x, y, z) coordinates.
    """
    if point is None:
        point = []

    bottom_row = np.array([[1.0]])

    homogenous_point = np.asarray(point).reshape(3, 1)
    homogenous_point = np.vstack([homogenous_point, bottom_row])
    transformed_point = tuple(np.dot(transformation, homogenous_point).flatten()[:3])

    return transformed_point