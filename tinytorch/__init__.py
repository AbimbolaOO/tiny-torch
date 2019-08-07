from .ann import *
from .dataloaders import *
from .data_viewer import *
from .train_loop import *
from .test_model import *


__add__ = (dataloaders.__add__ + train_loop.__add__ + ann.__add__ 
            + data_viewer.__add__ + test_model.__add__)