import os
import numpy as np
import enoki as ek
import mitsuba
import math

# Set the desired mitsuba variant
mitsuba.set_variant('scalar_rgb')

from mitsuba.core import Thread, math, Properties, Frame3f, Float, Vector3f, warp
from mitsuba.core.xml import load_file, load_string
from mitsuba.render import BSDF, BSDFContext, BSDFFlags, BSDFSample3f, SurfaceInteraction3f, \
                           register_bsdf, Texture


def calc_fiberlikeGGX(roughness, omega3):
    roughness2 = roughness * roughness
    sxx = roughness2 * omega3.x * omega3.x + omega3.y * omega3.y + omega3.z * omega3.z
    sxy = roughness2 * omega3.x * omega3.y - omega3.x * omega3.y
    sxy = roughness2 * omega3.x * omega3.y - omega3.x * omega3.y
    sxz = roughness2 * omega3.x * omega3.z - omega3.x * omega3.z
    syy = roughness2 * omega3.y * omega3.y + omega3.x * omega3.x + omega3.z * omega3.z
    szz = roughness2 * omega3.z * omega3.z + omega3.x * omega3.x + omega3.y * omega3.y
    syz = roughness2 * omega3.y * omega3.z - omega3.y * omega3.z
    fiberlikeMatrix = [[sxx, sxy, sxz],
                       [sxy, syy, syz],
                       [sxz, szz, szz]]
    return np.array(fiberlikeMatrix).reshape(3,3)


def sigma(wi, sxx, syy, szz, sxy, sxz, syz):
    sigma_squared = wi.x * wi.x * sxx + wi.y * wi.y * syy + wi.z * wi.z * szz + 2 * (wi.x * wi.y * sxy + wi.x * wi.z * sxz + wi.y * wi.z * syz)
    if sigma_squared > 0:
        return math.sqrt(sigma_squared)
    else:
        return 0

def buildOrthonormalBasis(wi):
    if wi.z < -0.9999999 :
        wk = Vector3f(0, -1, 0)
        wj = Vector3f(-1, 0, 0)
