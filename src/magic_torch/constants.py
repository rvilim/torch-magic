"""Constants matching Fortran's constants.f90."""

import torch
import math

from .precision import DTYPE, CDTYPE, DEVICE

pi = math.pi
sq4pi = math.sqrt(4.0 * pi)
osq4pi = 1.0 / sq4pi

one = 1.0
two = 2.0
three = 3.0
four = 4.0
half = 0.5
third = one / three

zero = complex(0.0, 0.0)
ci = complex(0.0, 1.0)

sin36 = math.sin(36.0 * pi / 180.0)
sin60 = 0.5 * math.sqrt(3.0)
sin72 = math.sin(72.0 * pi / 180.0)
cos36 = math.cos(36.0 * pi / 180.0)
cos72 = math.cos(72.0 * pi / 180.0)
