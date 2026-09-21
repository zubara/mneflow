"""
mneflow: a lightweight framework for building, training, and interpreting
neural network models of neuroimaging (M/EEG) data using TensorFlow.

Exposes the main package API:

- :func:`mneflow.produce_tfrecords` and :func:`mneflow.load_meta` for
  preparing and reloading TFRecords datasets (see ``mneflow.utils``).
- :class:`mneflow.Dataset` for wrapping TFRecords into ``tf.data``
  pipelines (see ``mneflow.data``).
- :class:`mneflow.MetaData` for storing and restoring dataset/model
  metadata (see ``mneflow.meta``).
- Model classes :class:`mneflow.VARCNN`, :class:`mneflow.Deep4`,
  :class:`mneflow.FBCSP_ShallowNet`, :class:`mneflow.EEGNet` (see
  ``mneflow.models``) and :class:`mneflow.LFCNN` (see ``mneflow.lfcnn``).

"""
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

try:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError
    mneflow.__version__ = _pkg_version("mneflow")
except PackageNotFoundError:
    mneflow.__version__ = "0+unknown"
