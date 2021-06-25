#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2021 Dustin Kleckner
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .vector import dot, dot1, norm, mag, mag1, cross, plus, minus, path_tangents, path_delta, path_sum_delta, angle_between, enforce_shape
import numpy as np
from . import mesh
from .outline import get_outline
from scipy.interpolate import interp1d
π = np.pi


#------------------------------------------------------------------------------
# Path Objects and Descriptiors
#------------------------------------------------------------------------------


# Attributes of paths which are only auto-derived
class DerivedAttr:
    def __init__(self, der_func):
        self.der_func = der_func

    def __set_name__(self, owner, name):
        self.name = name
        self.attr = '_' + name

    def __get__(self, instance, instance_class):
        if not hasattr(instance, self.attr):
            val = self.der_func(instance)
            setattr(instance, self.attr, val)
            return val
        else:
            return getattr(instance, self.attr)

    def __delete__(self, instance):
        if hasattr(instance, self.attr):
            delattr(instance, self.attr)


# Attributes of Path which are only set directly
# Also checks the shape to make sure it matches the path
class SetAttr():
    def __init__(self, set_func=None, elem=1, extra_if_closed=0, extra_if_open=0):
        if set_func is not None:
            self.set_func = set_func
        self.elem = elem
        self.eic = extra_if_closed
        self.eio = extra_if_open

    def __set_name__(self, owner, name):
        self.name = name
        self.attr = '_' + name

    def __set__(self, instance, value):
        value = np.asarray(value)
        extra = self.eic if instance.closed else self.eio
        n = instance.n + extra

        if value.shape == (self.elem,):
            value = np.tile(value, (n, 1))

        if self.elem == 1:
            if value.shape == (n, 1):
                value = value.reshape(n)
            elif value.shape != (n,):
                extra = f"{extra:+d}" if extra else ""
                raise ValueError(f"path attribute '{self.name}' must have shape (n{extra})\n  [where n = len(path.X) = {instance.n}, and the provided array had shape {value.shape}]")
        else:
            if value.shape != (n, self.elem):
                extra = f"{extra:+d}" if extra else ""
                raise ValueError(f"path attribute '{self.name}' must have shape (n{extra}, {self.elem})\n  [where n = len(path.X) = {instance.n}, and the provided array had shape {value.shape}]")

        # if  value.shape[1:] not in self.elem or value.shape[0] != n:
        #     extra = f"{extra:+d}" if extra else ""
        #     valid_elems = ' or '.join('(' + ', '.join(["n" + extra] + list(map(str, e))) + ')' for e in self.elem)
        #     raise ValueError(f"path attribute '{self.name}' must have shape {valid_elems}\n  [where n = len(path.X) = {instance.n}, and the provided array had shape {value.shape}]")

        if hasattr(self, 'set_func'):
            value = self.set_func(instance, value)
        setattr(instance, self.attr, value)
        instance._dir_spec.add(self.name)

    def __get__(self, instance, instance_class):
        if not hasattr(instance, self.attr):
            raise AttributeError(f"'{instance_class.__name__}' instance has not set attribute '{self.name}'")
        else:
            return getattr(instance, self.attr)

    def __delete__(self, instance):
        delattr(instance, self.attr)


# Attributes of path which can either be set directly, or derived (like N)
class OptionalAttr(DerivedAttr, SetAttr):
    def __init__(self, der_func, set_func=None, elem=((),), extra_if_closed=0, extra_if_open=0):
        self.der_func = der_func
        self.set_func = set_func
        self.elem = elem
        self.eic = extra_if_closed
        self.eio = extra_if_open


# Used for color functions
def _as_u1(X):
    if X.dtype in ('f', 'd'):
        return np.clip(X * 255, 0, 255).astype('u1')
    elif X.dtype == ('u1'):
        return X
    else:
        raise ValueError('Color datatypes should be floats or unsigned bytes only!')


class Path:
    _der_order = ('t', 'dXdt', 'T', 'N', 'a', 'color', 'alpha')

    def _der_T(self):
        if hasattr(self, '_dXdt'):
            return norm(self._dXdt)
        else:
            Ts = self.Δ / self.ds.reshape(-1, 1)**2
            if self.closed:
                return norm(Ts + minus(Ts))
            else:
                T = np.empty_like(self.X)
                T[:-1] = Ts
                T[-1] = 0
                T[1:] += Ts
                return norm(T)

    T = OptionalAttr(
        der_func = lambda self: self._der_T(),
        set_func = lambda self, T: norm(T),
        elem=3
    )
    N = OptionalAttr(
        der_func = lambda self: self._parallel_transport_normal()[0],
        set_func = lambda self, N: norm(N - self.T * dot1(self.T, N)),
        elem=3
    )
    B = DerivedAttr(lambda self: cross(self.T, self.N))
    Δ = DerivedAttr(lambda self: path_delta(self.X, self.closed))
    ds = DerivedAttr(lambda self: mag(self.Δ))
    s = DerivedAttr(lambda self: path_sum_delta(self.ds, self.closed))
    L = DerivedAttr(lambda self: self.ds.sum())
    t = SetAttr(extra_if_closed=1)
    a = SetAttr()
    color = SetAttr(elem=3)
    alpha = SetAttr()


    def __init__(self, X, closed=False, **kw):
        f'''Create a path, based on the specified attributes.

        Note the following conventions:
            * capital letters are always vectors, with the last axis having
                size 3
            * lower case letters are always scalars
            * Many attributes can either be directly specified, or derived
                automatically
            * Some attributes are per *segment* rather than per point.  In this
                case they will have (n-1) points for open paths or (n) points
                for closed paths.
            * Attributes can be changed after creation, although in some cases
                this may lead to problems.  For example, if you change T after
                setting N, it is no longer gauanteed that N ⟂ T.  In general,
                you should all atributes at creation time, in which case the
                order is gauranteed to be correct.  (Otherwise, set dependent
                attributes after their dependencies -- in this case N is
                dependent on T due to orthonormality enforcement.)

        Parameters
        ----------
        X : (n, 3) shaped array
            The points on the path

        Keywords
        --------
        closed : bool (default: False)
            If True, the path is closed.  It is expected that the start/end
            point is **not** repeated.
        {", ".join(Path._der_order)} : various types
            See attributes below for a description of each.  These will be
            automatically computed if not directly specified.

        Attributes
        ----------
        n : int
            The number of points in the path
        Δ : (n-1, 3) or (n, 3) shaped array
            The straight line displacement for each segment
        ds : (n-1) or (n) shaped array
            The length of each segment
        s : (n) shaped array
            The straight line path length coordinate for each point.  The first
            element is always 0.  For closed paths, the first point really
            has two coordinates, 0 and L -- if the second is needed, use L
            instead.
        L : float
            The total straight line length of the path (L = ∑s)
        t : (n) or (n+1) shaped array
            Some coordinate along the path.  If not specified, it will be
            identified with the straight-line path arc-length coordinate, and
            computed automatically.  In some cases, it is useful to specify it
            directly, for example if dXdt is given, or if we wish a coordinate
            other than staight-line path length (i.e. time for trajectories in
            space).  If the path is closed, this should have one more length
            than the number of points, and the last point is the maximum
            perioidic value of t.
        dXdt : (n, 3) shaped array
            The derivative of the position with respect to the coordinate t.
            This is *not* normalized, nor should it be: if the path is
            interpolated, it is essential this is correct!
        T : (n, 3) shaped array
            The tangent unit vectors associated with the path.  If not
            specified directly, determined in one of two ways:
                (1) T = norm(dXdt)
                (2) The (1/ds**2) weighted average of the neighboring segment
                vectors, which for closely spaced points is the tangent vector
                of a circle passing through the point and it's neighbors.
            If directly specified, it will be normalized.
        N : (n, 3) shaped array
            The normal vector associated with the path.  If not specified
            directly, determined by parallel transport:
                N[n+1] = norm(N[n] - T[n+1] (T[n+1] ⋅ N[n]))
            Where N[0] is an arbitrary initial vector perpendicular to T[0].
            For closed paths, an additional twist is added to make the normals
            close on themselves. (for more information, see the
            `parallel_transport_normal` method.)
            If directly specified, will be made ⟂ to T and normalized.
        B : (n, 3) shaped array
            Binormal vector: B = T × N
        color : (n, 3) shaped array
            The RGB color of each point on the path, used for exports.  Should
            be 0-1 if floating type, or 0-255 if unsigned char
        alpha : (n) shaped array
            The alpha color of the path
        a : (n) shaped array
            The radius of the path, if present used for extrusions
        '''

        self.X = X
        self.n = len(X)
        self.closed = bool(closed)
        # Keep track of which attributes are derived vs. directly specified
        # (derived attributes are not saved!)
        self._dir_spec = {'X', 'closed'}

        kw = kw.copy()
        for k in self._der_order:
            if k in kw:
                setattr(self, k, kw.pop(k))

        if kw.keys():
            raise ValueError(f'Tried to create path with invalid keywords: {tuple(kw.keys())}')


    def _parallel_transport_normal(self, N0=None, twist_closed=True):
        if N0 is None:
            N0 = np.eye(3, dtype=self.X.dtype)[np.argmin(self.T[0], -1)]

        N = np.zeros_like(self.X)
        T = self.T[0]
        N[0] = norm(N0 - T * dot1(T, N0))

        for i in range(1, self.n):
            T = self.T[i]
            N[i] = norm(N[i-1] - T * dot1(T, N[i-1]))

        if self.closed and twist_closed:
            T = self.T[0]
            Nf = norm(N[-1] - T * dot1(T, N[-1]))
            ϕ = np.arctan2(
                np.clip(dot(cross(N[0], Nf), T), -1, 1),
                np.clip(dot(N[0], Nf), -1, 1)
            )
            # self.twist(self.s / self.L * ϕ, _dir_spec=False)
            # print(self.s.shape, N.shape, self.T.shape)
            N = self._twist(self.s / self.L * ϕ, N)
            return N, ϕ
        else:
            return N, 0

    def parallel_transport_normal(self, N0=None, twist_closed=True):
        '''Construct a parallel transport normal for the path.

        Keywords
        --------
        N0 : (3) shaped array
            The initial normal vector.
        twist_closed : bool (default: True)
            If True, add twist to the path propotional to s, which causes the
            normals to close on themselves.  If False, closed paths will not
            have normal vectors which smoothly close across the periodic
            boundary

        Returns
        -------
        angle : float
            The angle required to twist the path normal closed
        '''
        self.N, ϕ = self._parallel_transport_normal(N0, twist_closed)
        return ϕ

    def _twist(self, twist, N):
        twist = np.asarray(twist).reshape(-1, 1)
        return N * np.cos(twist) - cross(self.T, N) * np.sin(twist)

    def twist(self, twist):
        '''Add twist to the normal/binormal.

        Parameters
        ----------
        twist : float or (n) shaped array of floats
            The twist (in radians) to apply to the normals.  The direction is
            counterclockwise if you are looking in the direction of the
            tangent vector
        '''

        self.N = self._twist(twist, self.N)
        del self.B

    def _check_attr(self, name):
        if not hasattr(self, '_' + name):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def smooth_resample(self, t0=None, t1=None, angle=2*π/50):
        '''Resample the path, producing a new path with a specified angle
        between points.  Internally, a cubic Bezier is used, and derivatives
        are matched used t and dXdt.

        Presently, and error is raised is t and dXdt are not present, as
        methods to derive them have not yet been implemented.

        Keywords
        --------
        t0, t1 : float or None (default: 0, t.max())
            The initial/final value of t in the resampled path.  If directly
            specified, the returned path will *always* be open.  If not
            specified, the whole path is returned, and will be closed if the
            original path was
        angle : float (default: 0.1)
            The target angle between segments, in radians.  The default
            produces 50 per circle.

        Returns
        -------
        smooth_path : Path
            The resampled path.  Note: at present, only t, X, and dXdt are
            present in the resampled path -- other attached attributes are not
            interpolated.  (This will likey change in the future.)
        '''
        # Build the resample Bezier curve and interpolating functions.
        self._check_resample()

        closed = self.closed and (t0 is None) and (t1 is None)

        if t0 is None:
            t0 = 0
        if t1 is None:
            t1 = self.t[-1]

        # We want to space the points evenly in angle, not t!  Transform to
        #  ϕ-space first, create a linspace, and then convert back to
        #  franctional segment number (n)
        ϕ0, ϕ1 = self._t_to_ϕ([t0, t1])
        ϕ = np.linspace(ϕ0, ϕ1, int(np.ceil(abs(ϕ1 - ϕ0) / angle)))
        n = self._ϕ_to_n(ϕ)

        # i is the segment number, x is the fractional distance across the
        #    segment
        i = np.clip(np.floor(n).astype('i'), 0, len(self._P0) - 1)
        x = (n - i).reshape(-1, 1)
        ox = 1 - x
        Xt = ox**3 * self._P0[i] + 3*ox**2*x * self._P1[i] + \
                3*ox*x**2 * self._P2[i] + x**3 * self._P3[i]
        dXt_dt = ox**2 * self._D0[i] + x*ox * self._D1[i] + x**2 * self._D2[i]

        # Here n is the number of points in the output
        n = len(Xt)
        if closed:
            n -= 1

        return Path(X = Xt[:n, :3], t = Xt[:, 3],
                    dXdt = dXt_dt[:n, :3] / dXt_dt[:n, 3:4], closed=closed)


    def _check_resample(self):
        if getattr(self, '_resample_data', False):
            return

        else:
            t = self.t
            n = len(t) # Not the same for self.t for closed paths!

            # Construct points as (x, y, z, t)
            # The idea is that we also resample t for free in the Bezier
            X = np.empty((n, 4))
            X[:self.n, :3] = self.X
            X[:, 3] = t
            # ... and the corresponding velocity = dX/dt
            V = np.empty((n, 4))
            V[:self.n, :3] = self.dXdt
            V[:, 3] = 1 # dt/dt = 1!

            if self.closed:
                # Repeat the first point at the end
                X[-1:, :3] = X[:1, :3]
                V[-1:, :3] = V[:1, :3]

            self._dt = t[1:] - t[:-1]
            # Multiplication factor to go from V to P1/P2
            α = self._dt.reshape(-1, 1) / 3

            self._P0 = X[:-1]
            self._P1 = self._P0 + V[:-1] * α
            self._P3 = X[1:]
            self._P2 = self._P3 - V[1:] * α
            self._D0 = 3 * (self._P1 - self._P0)
            self._D1 = 6 * (self._P2 - self._P1)
            self._D2 = 3 * (self._P3 - self._P2)

            N = norm(V[:, :3])
            Nm = norm(-self._P0[:, :3] - self._P1[:, :3] + self._P2[:, :3] + self._P3[:, :3]) # Midpoint tangent
            dϕ = angle_between(N[:-1], Nm) + angle_between(Nm, N[1:])
            self._ϕ = np.zeros_like(t)
            self._ϕ[1:] = np.cumsum(dϕ)

            self._t_to_ϕ = interp1d(t, self._ϕ, copy=False, fill_value='extrapolate')
            # self._ϕ_to_t = interp1d(self._ϕ, t, copy=False, fill_value='extrapolate')
            # n is the segment # coordinate.
            self._ϕ_to_n = interp1d(self._ϕ, np.arange(n), copy=False, fill_value='extrapolate')

            self._resample_data = True

    def u1_rgba(self):
        '''Get the RGBA color of each point on the path, as a 1 byte unsigned
        int.

        Returns
        -------
        rgba : (n, 4) shaped array or None
            The color and alpha.  If the path has neither color nor alpha
            defined, returns None.  If just the alpha is attached, the color
            is white, and if just the color is attached the alpha channel
            is 255
        '''

        if hasattr(self, 'color'):
            color = np.empty((self.n, 4), dtype='u1')
            color[:, :3] = _as_u1(self.color)

            if hasattr(self, 'alpha'):
                color[:, 3] = _as_u1(self.alpha)
            else:
                color[:, 3] = 255
            # print(color)
            return color

        elif hasattr(self, 'alpha'):
            color = np.empty(self.n, 4, dtype='u1')
            color[:, :3] = 255
            color[:, 3] = _as_u1(self.alpha)
            return color

        else:
            return None


    def colorize(self, c, cmap=None, clim=None):
        '''Color a mesh using matplotlib specifications.

        Parameters
        ----------
        c : string or (n) shaped array
            If c is a string, it is interpreted like a normal matplotlib color
            (e.g., `"r"` for red, or `"0.5"` for 50% gray, or `"#FF8888"`
            for pink).  If type is array, it should have the same length as the
            number of points, and it will be converted to a colormap.
        cmap : string or matplotlib colormap (default: matplotlib default)
            The color map to use.  Only relevant if ``c`` is an array.
        clim : tuple (default: None)
            The color limits.  If None, the max/min of c are used.
        '''

        if isinstance(c, (str, bytes)):
            import matplotlib.colors
            rgba = np.tile(matplotlib.colors.colorConverter.to_rgba(c), (self.n, 1))
        else:
            import matplotlib.cm
            c = enforce_shape(c, (self.n,))
            if clim is None: clim = (c.min(), c.max())
            rgba = matplotlib.cm.get_cmap(cmap)((c - clim[0]) / (clim[1] - clim[0]))

        self.color = rgba[:, :3]
        self.alpha = rgba[:, 3]


    def extrude(self, outline='circle', scale=1., dtype=None):
        '''Extrude a shape about the path.  Typically used to create a 3D tube
        with the shape of the path.

        Keywords
        --------
        outline : string or outline.Outline object (default: "circle")
            The outline to trace.  See `outline` module for more details.
            Alternatively, simple outlines can be specified with a string:
                - "circle": a circle of radius 1, with 20 points.
                - "striped": a circle with a stripe on it, indicating N
                - "arrow": an arrow of length 1
                - "ribbon": a thin ribbon, edge oriented with N
                - "triangle": A triangle (max radius 1)
                - "square": A square (max radius 1), face oriented with N
        scale : float or (n) shaped array of floats (default: 1)
            The scale of the outline at each point.  Generally identified with
            the radius, but depends on the outline.  Note that this is
            multiplied by the `a` attribute, if present on the path.
        dtype : numpy data type spec (default: derived from outline)
            The data type of the vertices -- if exporting to polygon mesh files,
            single precision floats ("f") should be sufficient and will
            roughly half file size, but doubles can be given if needed.

        Returns
        -------
        mesh : mesh.Mesh
            A triangle mesh of the tube.  Can be saved to STL or PLY formats with
            `mesh.save(<filename>)`.
        '''
        if isinstance(outline, str):
            outline = get_outline(outline)

        m = outline.m
        n = self.n - (0 if self.closed else 1)

        if dtype is None:
            dtype = outline.dtype

        X = self.X.reshape(-1, 1, 3).astype(dtype)
        N = self.N.reshape(-1, 1, 3).astype(dtype)
        B = self.B.reshape(-1, 1, 3).astype(dtype)

        scale = np.asarray(scale).reshape(-1, 1, 1).astype(dtype)
        if hasattr(self, '_a'):
            scale = scale * self.a.reshape(-1, 1, 1)

        verts = X + scale * ( outline.outline[:, 0:1] * N + \
                              outline.outline[:, 1:2] * B )

        if hasattr(outline, "T"):
            # First tangent is from outline alone
            T1 = (outline.T[:, 0:1] * N + outline.T[:, 1:2] * B)
            # Second tangent is from the path a specific point on the outline
            #  takes as it traverses the path
            # Because verts still has shape (n, m, 3), path tangents computes
            #  the tangent along the 0-axis.
            T2 = path_tangents(verts, closed=self.closed)
            VN = norm(cross(T1, T2)).reshape(-1, 3)

        verts = verts.reshape(-1, 3)


        tris = np.arange(n).reshape(-1, 1, 1) * m + outline.stitch
        if self.closed:
            tris %= len(verts)

        tris = tris.reshape(-1, 3)

        has_ends = False

        if not self.closed and hasattr(outline, 'end_stitch'):
            tris = np.vstack([
                tris,
                outline.end_stitch + len(verts),
                outline.end_stitch[:, ::-1] + len(verts) + m
            ])

            # Add new points to get a sharp edge
            verts = np.vstack([
                verts,
                verts[:m],
                verts[-m:]
            ])

            # Ditto with normals
            if VN is not None:
                VN = np.vstack([
                    VN,
                    -np.tile(self.T[0], (m, 1)),
                    np.tile(self.T[-1], (m, 1))
                ])

            has_ends = True

        color = self.u1_rgba()
        if color is None:
            if hasattr(outline, 'color'):
                color = _as_u1(outline.color).reshape(1, -1, 4)
                color = np.tile(color, (len(VN)//m, 1, 1)).reshape(-1, 4)
        else:
            color = color.reshape(-1, 1, 4)
            if has_ends:
                color = np.vstack([color, color[:1], color[-1:]])
            color = np.tile(color, (1, m, 1)).reshape(-1, 4)

        return mesh.Mesh(verts, tris, normals=VN, colors=color)


def line(P0, P1, N=20):
    '''Create a straight line path

    Parameters
    --------
    P0 : (3) shaped array-like
    P1 : (3) shaped array-like
        The start and end points

    Keywords
    --------
    N : int (default: 20)
        The number of points in the line.

    Returns
    -------
    path : Path
        A Path object with the specified line.  The "t" axis goes from 0-1.
    '''

    P0 = np.asarray(P0)
    P1 = np.asarray(P1)
    Δ = P1 - P0

    t = np.linspace(0, 1, N)

    return Path(P0 + t.reshape(-1, 1) * Δ, t=t, dXdt=np.tile(Δ, (N, 1)), closed=False)


def torus_knot(p=2, q=3, R=1, a=0.25, N=100):
    '''Make a torus knot path

    Keywords
    --------
    p : int (default: 2)
        The winding number around the major axis
    q : int (default: 3)
        The winding number around the minor axis
    R : float (default: 1)
        The major radius of the torus
    a : float (default: 1)
        The minor radius of the torus
    N : int (default: 100)
        The number of points in the path

    Returns
    -------
    path : Path
        A Path object with the specified knot.  The "t" attribute goes from
        0-2π.  The winding is around the z-axis
    '''

    # t has one more point than "t" to define the periodicity of the path
    # In other words the first point (which is also the last) has two t coordinates
    t = np.linspace(0, 2*π, N+1)
    ϕ = t[:-1]

    sq = np.sin(q * ϕ)
    cq = np.cos(q * ϕ)
    sp = np.sin(p * ϕ)
    cp = np.cos(p * ϕ)

    r = R + a * cq
    z = a * sq

    X = np.empty((N, 3))
    X[:, 0] = r * cp
    X[:, 1] = r * sp
    X[:, 2] = z

    # Compute the path derivative, for resampling, etc.
    dXdt = np.empty((N, 3))
    dXdt[:, 0] = -p * r * sp - a * q * cp * sq
    dXdt[:, 1] =  p * r * cp - a * q * sp * sq
    dXdt[:, 2] = a * q * cq

    return Path(X, t=t, dXdt=dXdt, closed=True)
