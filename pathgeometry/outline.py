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

from .vector import dot, norm, mag1, plus, minus, path_tangents
import numpy as np
π = np.pi

#------------------------------------------------------------------------------
# Outline Objects -- Used for Extruding Paths
#------------------------------------------------------------------------------

# Depreciated: tangents used instead of normals now
# def path_normals(outline, closed=True):
#     T = path_tangents(outline, closed)
#     N = np.empty_like(outline)
#     N[:, 0] = T[:, 1]
#     N[:, 1] = -T[:, 0]
#     return N

def side_stitch(n, m=None, closed=True):
    if m is None: m = n

    if closed:
        i0 = np.arange(n, dtype='i')
        i1 = plus(i0)
    else:
        i0 = np.arange(n-1, dtype='i')
        i1 = i0 + 1

    stitch = np.empty((len(i0), 2, 3), 'i')
    stitch[:, 0, 0] = i0
    stitch[:, 0, 1] = i1
    stitch[:, 0, 2] = i0 + m
    stitch[:, 1, 0] = i1
    stitch[:, 1, 1] = i1 + m
    stitch[:, 1, 2] = i0 + m

    return stitch.reshape(-1, 3)

def face_stitch(n):
    i0 = np.arange(1, n-1)
    stitch = np.empty((n-2, 3), 'i')
    stitch[:, 0] = 0
    stitch[:, 1] = i0+1
    stitch[:, 2] = i0
    return stitch


class Outline:
    def __init__(self, outline, T=True, stitch=None,
        end_stitch=None, color=None, dtype='f'):
        '''Create a 2D outline, used to extrude tubes of a particular shape.

        Note: here 'n' is the number of points in the path to be traced, 'm' is
        the number of vertices per path point and 't' is the number of triangles
        per path point, and 'et' is the number of triangles per end patch.

        Parameters
        ----------
        X : (m, 2) or (n, m, 2) shaped array
            The coordinates of the outline.  The first axis corresponds to N
            on the path, and the second to B.  Winding should be in counter-
            clockwise direction, assuming N (axis 0) is the the right and
            B (axis 1) is up

        Keywords
        --------
        T : True, (m, 2) or (n, m, 2) shaped array (default: True)
            The tangents for the outline, if specified the extruded path will
            have attached tangents (otherwise it will not).  If `True` is
            specified, they are created automatically from the path, assuming
            it has counterclockwise winding.
        stitch : (t, 3) shaped array of ints
            The stitching pattern used for the triangle vertices.  `t` is the number
            of triangles per path point.  Specified for the first point on the
            path.  For example, if our outline has 10 points, and we wish a
            triangle connecting elements 0 and 1 on the first point, and 2 on the
            second point, this would be specified as (0, 1, 12).  If not
            specified, a closed profile is automatically generated.
        end_stitch : (et, 3) shaped array of ints
            The stitching pattern used for the ends of the tube.  If not
            specified, one is automatically created assuming a concave shape
        color : (m, 4) or (n, m, 4) shaped array of floats or u1
            If defined, rgba color per point on the outline.  Note that if the
            path has an attached color, this will override the outline color
        dtype : numpy data type spec (default: "f")
            The data type of the vertices -- if exporting to polygon mesh files,
            single precision floats ("f") should be sufficient and will
            roughly half file size, but doubles can be given if needed.
        '''
        self.outline = outline.astype(dtype)
        self.m = self.outline.shape[-2]
        self.dtype = dtype

        if T is True:
            self.T = path_tangents(self.outline, closed=True)
        elif isinstance(T, (list, tuple, np.ndarray)):
            self.T = norm(np.asarray(T, dtype))

        if stitch is None:
            self.stitch = side_stitch(self.m)
        else:
            self.stitch = np.asarray(stich, dtype='i')

        if end_stitch is None:
            self.end_stitch = face_stitch(self.m)
        else:
            self.end_stitch = np.asarray(end_stitch, dtype='i')

        if color is not None:
            self.color = color


class Circle(Outline):
    def __init__(self, r=1, N=20, T=True):
        '''Create a round outline.

        Keywords
        --------
        r : float (default: 1)
            The circle radius
        N : int (default: 20)
            The number of points per circumference
        T : bool (default: True)
            If True, tangents attached to path
        '''

        ϕ = np.linspace(0, 2*π, N, False)
        super().__init__(r * np.array([np.cos(ϕ), np.sin(ϕ)]).T, T=bool(T))


class SegmentedOutline(Outline):
    def __init__(self, X, corners=None, T=True, dtype='f'):
        '''Create a 2D outline composed of segments, with hard edges between
        them.  A convenience class for defining shapes like squares which
        handles the nitty gritty details of buidling the tangents, etc.

        Parameters
        ----------
        X : (m, 2) shaped array
            The points of the outline

        Keywords
        --------
        corners : list-like of integers
            The indices of the points which correspond to hard corners.  If
            not specified, all points are assumed to be hard corners.
        T : True or (m, 2) shaped array (default: True)
            The tangents for the outline, if specified the extruded path will
            have attached tangents (otherwise it will not).  If `True` is
            specified, they are created automatically from the path, assuming
            it has counterclockwise winding.
        dtype : numpy data type spec (default: "f")
            The data type of the vertices -- if exporting to polygon mesh files,
            single precision floats ("f") should be sufficient and will
            roughly half file size, but doubles can be given if needed.

        Note that tangents and stitching are created automatically.  If you need
        more control (e.g. open outlines), use the Outline class directly.
        '''
        X = np.asarray(X, dtype=dtype)

        if corners is None:
            corners = np.arange(len(X))
        else:
            corners = np.array(corners)
            corners.sort()

        if not(len(corners)):
            super().__init__(X)
        else:
            n_tot = len(X) + len(corners)
            X = np.roll(X, -corners[0])
            X = np.vstack((X, X[:1]))
            corners = np.concatenate((corners[1:] - corners[0], [len(X)-1]))


            outlines = []
            Ts = []
            stitch = []
            end_points = []

            i = 0
            n = 0

            for j in corners:
                m = j+1 - i
                outlines.append(X[i:j+1])

                if T is True:
                    Ts.append(path_tangents(outlines[-1], closed=False))
                elif isinstance(T, (list, tuple, np.ndarray)):
                    Ts.append(T[i:j+1])

                stitch.append(side_stitch(m, n_tot, closed=False) + n)
                end_points.append(np.arange(n, n+m-1, dtype='i'))
                n += m
                i = j


            self.outline = np.vstack(outlines)
            self.m = len(self.outline)
            self.dtype = dtype
            self.T = np.vstack(Ts)
            self.stitch = np.vstack(stitch)
            end_points = np.concatenate(end_points)
            self.end_stitch = end_points[face_stitch(len(end_points))]


class Polygon(SegmentedOutline):
    def __init__(self, r=1, N=3, ϕ0=0):
        '''Create a hard edge polygon outline.

        Keywords
        --------
        r : float (default: 1)
            The circle radius
        N : int (default: 3)
            The number of faces
        ϕ0 : float (default: 0)
            The angle by which to rotate the outline.  By default the first
            vertex will be at (1, 0)
        '''

        ϕ = np.linspace(0, 2*π, N, False) + ϕ0
        super().__init__(r * np.array([np.cos(ϕ), np.sin(ϕ)]).T)


class Arrow(SegmentedOutline):
    def __init__(self, h=1, w=0.5):
        outline = np.array([
            (1, 0),
            (0, 1),
            (0, 0.35),
            (-1, 0.35),
            (-1, -0.35),
            (0, -0.35),
            (0, -1)
        ]) * np.asarray((h, w))

        super().__init__(outline)


class Striped(SegmentedOutline):
    def __init__(self, r=1, N=20, N_stripe=5, T=True, base_color=np.ones(4),
        stripe_color=np.array((0, 0, 1, 1), dtype='d')):
        '''Create a round outline with a stripe on one side, indicating the
        normal direction

        Keywords
        --------
        r : float (default: 1)
            The circle radius
        N : int (default: 20)
            The number of points per circumference
        N_stripe : int (default: 2)
            The number of segments which are colored
        T : bool (default: True)
            If True, tangents attached to path
        base_color : (4,) shaped array-like (default: [1, 1, 1, 1])
            The base color of the circle
        stripe_color : (4,) shaped array-like (default: [1, 0, 0, 1])
            The stripe color
        '''

        ϕ = np.linspace(0, 2*π, N, False) - π * N_stripe / N
        X = r * np.array([np.cos(ϕ), np.sin(ϕ)]).T
        T = np.array([-np.sin(ϕ), np.cos(ϕ)]).T

        base_color = np.asarray(base_color)
        stripe_color = np.asarray(stripe_color)

        super().__init__(X, corners=[0, N_stripe]) #, T=T)

        # Add the colors
        self.color = np.vstack([
            np.tile(stripe_color, (N_stripe + 1, 1)),
            np.tile(base_color, (N - N_stripe + 1, 1))
        ])

        # Fix the end stitching so the color looks good
        self.end_stitch = np.vstack([
            face_stitch(N_stripe + 1),
            face_stitch(N - N_stripe + 1) + N_stripe + 1
        ])


_OUTLINES = dict()

def get_outline(outline):
    if outline not in _OUTLINES:
        if outline == 'circle':
            _OUTLINES[outline] = Circle()
        elif outline == 'striped':
            _OUTLINES[outline] = Striped()
        elif outline == 'triangle':
            _OUTLINES[outline] = Polygon(N=3)
        elif outline == 'square':
            _OUTLINES[outline] = Polygon(N=4, ϕ0 = π/4)
        elif outline == 'arrow':
            _OUTLINES[outline] = Arrow()

        else:
            raise ValueError(f"Unknown outline: '{outline}'")

    return _OUTLINES[outline]
