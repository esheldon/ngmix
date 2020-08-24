import numpy

from .gexceptions import GMixRangeError

ONE_MINUS_EPS = 0.9999999999999999


def shear_reduced(g1, g2, s1, s2):
    """
    addition formula for reduced shear

    parameters
    ----------
    g1,g2: scalar or array
        "reduced shear" shapes
    s1,s2: scalar or array
        "reduced shear" shapes to use as shear

    outputs
    -------
    g1,g2 after shear
    """

    A = 1 + g1 * s1 + g2 * s2
    B = g2 * s1 - g1 * s2
    denom_inv = 1.0 / (A * A + B * B)

    g1o = A * (g1 + s1) + B * (g2 + s2)
    g2o = A * (g2 + s2) - B * (g1 + s1)

    g1o *= denom_inv
    g2o *= denom_inv

    return g1o, g2o


class Shape(object):
    """
    Shape object.  Currently only for reduced shear style shapes

    examples
    --------
    >>> import numpy as np
    >>> from ngmix.shape import Shape
    >>> s = Shape(0.1, 0.2)
    >>> neg_s = -s
    >>> rot_s = s.get_rotated(np.pi/2)
    >>> new_s = s.copy()
    >>> sheared_s = s.get_sheared(-0.05, 0.0)

    parameters
    ----------
    g1,g2: scalar
        "reduced shear" shapes
    """

    def __init__(self, g1, g2):
        self.g1 = g1
        self.g2 = g2

        # can't call the other jitted methods
        g = numpy.sqrt(g1 * g1 + g2 * g2)
        if g >= 1.0:
            raise GMixRangeError("g out of range: %.16g" % g)
        self.g = g

    def set_g1g2(self, g1, g2):
        """
        Set reduced shear style ellipticity

        parameters
        ----------
        g1,g2: scalar
            "reduced shear" shapes
        """
        self.g1 = g1
        self.g2 = g2

        g = numpy.sqrt(g1 * g1 + g2 * g2)
        if g >= 1.0:
            raise GMixRangeError("g out of range: %.16g" % g)
        self.g = g

    def get_sheared(self, s1, s2=None):
        """
        Get a new shape, sheared by the specified amount.

        parameters
        ----------
        s1: scalar or Shape
            The first component of the shape or a Shape instance.
        s2: scalar
            If s1 is given as a scalar, you must send the second component
            of the shape as s2.

        outputs
        -------
        sheared_shape: Shape
            A new shape sheared by (s1, s2).
        """

        if isinstance(s1, Shape):
            sh = s1
            s1 = sh.g1
            s2 = sh.g2
        else:
            if s2 is None:
                raise ValueError("send s1,s2 or a Shape")

        g1, g2 = shear_reduced(self.g1, self.g2, s1, s2)
        return Shape(g1, g2)

    def __neg__(self):
        """
        get Shape(-g1, -g2)
        """
        return Shape(-self.g1, -self.g2)

    def get_rotated(self, theta_radians):
        """
        Rotate the shape by the input angle.

        parameters
        ----------
        theta_radians: scalar
            The rotation angle in radians.

        outputs
        -------
        rot_shape: Shape
            The rotated shape.
        """
        twotheta = 2.0 * theta_radians

        cos2angle = numpy.cos(twotheta)
        sin2angle = numpy.sin(twotheta)
        g1rot = self.g1 * cos2angle + self.g2 * sin2angle
        g2rot = -self.g1 * sin2angle + self.g2 * cos2angle

        return Shape(g1rot, g2rot)

    def rotate(self, theta_radians):
        """
        In-place rotation of the shape by the input angle

        **deprecated, use get_rotated()**

        parameters
        ----------
        theta_radians: scalar
            The rotation angle in radians.
        """
        twotheta = 2.0 * theta_radians

        cos2angle = numpy.cos(twotheta)
        sin2angle = numpy.sin(twotheta)
        g1rot = self.g1 * cos2angle + self.g2 * sin2angle
        g2rot = -self.g1 * sin2angle + self.g2 * cos2angle

        self.set_g1g2(g1rot, g2rot)

    def copy(self):
        """
        Make a new Shape object with the same ellipticity parameters.

        outputs
        -------
        new_shape: Shape
            A copy of the current Shape instance.
        """
        s = Shape(self.g1, self.g2)
        return s

    def __repr__(self):
        return "(%.16g, %.16g)" % (self.g1, self.g2)


def g1g2_to_e1e2(g1, g2):
    """
    convert reduced shear g1,g2 to standard ellipticity
    parameters e1,e2

    uses eta representation but could also use
        e1 = 2*g1/(1 + g1**2 + g2**2)
        e2 = 2*g2/(1 + g1**2 + g2**2)

    parameters
    ----------
    g1,g2: scalars
        Reduced shear space shapes

    outputs
    -------
    e1,e2: tuple of scalars
        shapes in (ixx-iyy)/(ixx+iyy) style space
    """
    g = numpy.sqrt(g1 * g1 + g2 * g2)

    if isinstance(g1, numpy.ndarray):
        (w,) = numpy.where(g >= 1.0)
        if w.size != 0:
            raise GMixRangeError("some g were out of bounds")

        eta = 2 * numpy.arctanh(g)
        e = numpy.tanh(eta)

        numpy.clip(e, 0.0, ONE_MINUS_EPS, e)

        e1 = numpy.zeros(g.size)
        e2 = numpy.zeros(g.size)
        (w,) = numpy.where(g != 0.0)
        if w.size > 0:
            fac = e[w] / g[w]

            e1[w] = fac * g1[w]
            e2[w] = fac * g2[w]

    else:
        if g >= 1.0:
            raise GMixRangeError("g out of bounds: %s" % g)
        if g == 0.0:
            return (0.0, 0.0)

        eta = 2 * numpy.arctanh(g)
        e = numpy.tanh(eta)
        if e >= 1.0:
            e = ONE_MINUS_EPS

        fac = e / g

        e1 = fac * g1
        e2 = fac * g2

    return e1, e2


def e1e2_to_g1g2(e1, e2):
    """
    convert e1,e2 to reduced shear style ellipticity

    parameters
    ----------
    e1,e2: tuple of scalars
        shapes in (ixx-iyy)/(ixx+iyy) style space

    outputs
    -------
    g1,g2: scalars
        Reduced shear space shapes
    """

    e = numpy.sqrt(e1 * e1 + e2 * e2)
    if isinstance(e1, numpy.ndarray):
        (w,) = numpy.where(e >= 1.0)
        if w.size != 0:
            raise GMixRangeError("some e were out of bounds")

        eta = numpy.arctanh(e)
        g = numpy.tanh(0.5 * eta)

        numpy.clip(g, 0.0, ONE_MINUS_EPS, g)

        g1 = numpy.zeros(g.size)
        g2 = numpy.zeros(g.size)
        (w,) = numpy.where(e != 0.0)
        if w.size > 0:
            fac = g[w] / e[w]

            g1[w] = fac * e1[w]
            g2[w] = fac * e2[w]

    else:
        if e >= 1.0:
            raise GMixRangeError("e out of bounds: %s" % e)
        if e == 0.0:
            g1, g2 = 0.0, 0.0

        else:

            eta = numpy.arctanh(e)
            g = numpy.tanh(0.5 * eta)

            if g >= 1.0:
                # round off?
                g = ONE_MINUS_EPS

            fac = g / e

            g1 = fac * e1
            g2 = fac * e2

    return g1, g2


def g1g2_to_eta1eta2(g1, g2):
    """
    convert reduced shear g1,g2 to eta style ellipticity

    parameters
    ----------
    g1,g2: scalars
        Reduced shear space shapes

    outputs
    -------
    eta1,eta2: tuple of scalars
        eta space shapes
    """

    if isinstance(g1, numpy.ndarray):

        g = numpy.sqrt(g1 * g1 + g2 * g2)
        (w,) = numpy.where(g >= 1.0)
        if w.size != 0:
            raise GMixRangeError("some g were out of bounds")

        eta1 = numpy.zeros(g.size)
        eta2 = eta1.copy()

        (w,) = numpy.where(g > 0.0)
        if w.size > 0:

            eta = 2 * numpy.arctanh(g[w])
            fac = eta / g[w]

            eta1[w] = fac * g1[w]
            eta2[w] = fac * g2[w]

    else:
        g = numpy.sqrt(g1 * g1 + g2 * g2)

        if g >= 1.0:
            raise GMixRangeError("g out of bounds: %s converting to eta" % g)

        if g == 0.0:
            eta1, eta2 = 0.0, 0.0
        else:

            eta = 2 * numpy.arctanh(g)

            fac = eta / g

            eta1 = fac * g1
            eta2 = fac * g2

    return eta1, eta2


def e1e2_to_eta1eta2(e1, e2):
    """
    convert reduced shear e1,e2 to eta style ellipticity

    parameters
    ----------
    e1,e2: scalars
        Reduced shear space shapes

    outputs
    -------
    eta1,eta2: tuple of scalars
        eta space shapes
    """

    if not isinstance(e1, numpy.ndarray):
        e1 = numpy.array(e1, ndmin=1, copy=False)
        e2 = numpy.array(e2, ndmin=1, copy=False)
        is_scalar = True
    else:
        is_scalar = False

    e = numpy.sqrt(e1 * e1 + e2 * e2)
    (w,) = numpy.where(e >= 1.0)
    if w.size != 0:
        raise GMixRangeError("some e were out of bounds")

    eta1 = numpy.zeros(e.size)
    eta2 = eta1.copy()

    (w,) = numpy.where(e > 0.0)
    if w.size > 0:

        eta = numpy.arctanh(e)
        fac = eta[w] / e[w]

        eta1[w] = fac * e1[w]
        eta2[w] = fac * e2[w]

    if is_scalar:
        eta1 = eta1[0]
        eta2 = eta2[0]

    return eta1, eta2


def eta1eta2_to_g1g2(eta1, eta2):
    """
    convert eta style shpaes to reduced shear shapes

    parameters
    ----------
    eta1,eta2: tuple of scalars
        eta space shapes

    outputs
    -------
    g1,g2: scalars
        Reduced shear space shapes
    """

    if not isinstance(eta1, numpy.ndarray):
        eta1 = numpy.array(eta1, ndmin=1, copy=False)
        eta2 = numpy.array(eta2, ndmin=1, copy=False)
        is_scalar = True
    else:
        is_scalar = False

    g1 = numpy.zeros(eta1.size)
    g2 = g1.copy()

    eta = numpy.sqrt(eta1 * eta1 + eta2 * eta2)

    g = numpy.tanh(0.5 * eta)

    (w,) = numpy.where(g >= 1.0)
    if w.size != 0:
        raise GMixRangeError("some g were out of bounds")

    (w,) = numpy.where(eta != 0.0)
    if w.size > 0:
        fac = g[w] / eta[w]

        g1[w] = fac * eta1[w]
        g2[w] = fac * eta2[w]

    if is_scalar:
        g1 = g1[0]
        g2 = g2[0]

    return g1, g2


def dgs_by_dgo_jacob(g1, g2, s1, s2):
    """
    jacobian of the transformation
        |dgs/dgo|_{shear}

    parameters
    ----------
    g1,g2: numbers or arrays
        shape pars for "observed" image
    s1,s2: numbers or arrays
        shape pars for shear, applied negative

    outputs
    -------
    jacobian : number or array
        The jacobian of the transformation.
    """

    ssq = s1 * s1 + s2 * s2
    num = (ssq - 1) ** 2
    denom = (
        1 + 2 * g1 * s1 + 2 * g2 * s2 + g1 ** 2 * ssq + g2 ** 2 * ssq
    ) ** 2

    jacob = num / denom
    return jacob


def get_round_factor(g1, g2):
    """
    factor to convert T to round T under shear

    Use by taking T_round = T * get_round_factor(g1, g2)

    parameters
    ----------
    g1,g2: scalars
        Reduced shear space shapes

    outputs
    -------
    f: scalar
        factor to convert T to round T under shear
    """
    gsq = g1 ** 2 + g2 ** 2
    f = (1 - gsq) / (1 + gsq)
    return f


def rotate_shape(g1, g2, theta):
    """
    rotate the shapes by the input angle

    parameters
    ----------
    g1: scalar or array
        Shape to be rotated
    g2: scalar or array
        Shape to be rotated
    theta: scalar or array
        Angle in radians

    outputs
    -------
    g1,g2 after rotation
    """

    twotheta = 2.0 * theta

    cos2angle = numpy.cos(twotheta)
    sin2angle = numpy.sin(twotheta)
    g1rot = g1 * cos2angle + g2 * sin2angle
    g2rot = -g1 * sin2angle + g2 * cos2angle

    return g1rot, g2rot
