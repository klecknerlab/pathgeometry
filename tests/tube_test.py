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

import numpy as np
import pathgeometry
import matplotlib.pyplot as plt
π = np.pi

path = pathgeometry.torus_knot(a=0.5, N=200) #.smooth_resample(0, π)

# Put normals in the z-axis.  In practice, they will be as close as possible
#   while still being ⟂ to T, but this is handled automatically.
path.N = (0, 0, 1)

# Put normals pointing out from the z-axis
path.N = path.X * (1, 1, 0)

path.twist(path.t[:path.n] * 2)

# Colorize by the s-coordinate.  This will use a numpy colormap by default
# path.colorize(path.s)

# Set the thickness of the line.
path.a = 0.1 * (1.5 + np.cos(9*path.t[:path.n]))

path.extrude(outline='striped').save('test.ply')
