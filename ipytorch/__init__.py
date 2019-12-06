from . import checkpoint
from . import dataloader
from . import models
from . import options
from . import trainer
from . import utils
from . import visualization 

from . import prune
from . import quantization

"""
To use this package, you need to add it into the python_path:
1 vi .bashrc
2 insert : export PYTHONPATH="~/../ipytorch/:$PYTHONPATH"
3 source .bashrc
4 testing: type "import ipytorch" on python command line
"""

__authors__ = "ICEORY"
__version__ = "v0.01"
__license__ = "Copyright..."