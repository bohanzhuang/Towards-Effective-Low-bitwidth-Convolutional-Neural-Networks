# from sphere_auto import *
from .sphere_norm_prelu import *
# from sphere_thin_prelu import *
# from sphere_thin import *
# from wc_sphere_prelu import *
# from wc_sphere import *
# from wc_sphere_thin_prelu import *
# from sphere_test import *
from .sphere_nin import *
from .sphere_mobilenet_v2 import *
from .margin_linear import *
"""
sphere_auto: use autograd of pytorch
sphere_norm: define backward function by ourself
sphere_test: use autograd of pytorch, but compute gradient of weight before normalized
"""