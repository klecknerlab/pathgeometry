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
π = np.pi

p = 3
q = 5
N = 200
a = 0.25
R = 1

ϕ = np.linspace(0, 2*π, N)

r = R + a * np.cos(q * ϕ)
z = a * np.sin(q * ϕ)

X = np.empty((N, 3))
X[:, 0] = r * np.cos(p * ϕ)
X[:, 1] = r * np.sin(p * ϕ)
X[:, 2] = z

path = pathgeometry.Path(X)
path.a = 0.05 # Tube radius, here constant, but can also vary!
path.colorize(path.X[:, 2], cmap='RdBu') # Color by z coordinate

path.extrude().save('trefoil.ply')
