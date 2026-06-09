import mneflow
from . import models
from . import layers
from . import utils
from . import data
from . import meta
from . import lfcnn
from . import fc_models
from .utils import produce_tfrecords, load_meta
from .data import Dataset
from .models import VARCNN, Deep4, FBCSP_ShallowNet, EEGNet
from .lfcnn import LFCNN
from .meta import MetaData
mneflow.__version__ = '0.5.13dev'

