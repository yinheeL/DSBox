


from .base_strategy import BaseStrategy, register_strategy, get_strategy_class

from . import random #1
from . import LeastConfidence #1
from . import EntropySampling #1
from . import Margin #1
from . import BALDDropout #1
from . import LearningLoss
from . import CoreSet #1
from . import KMeans #loss 10%
from . import ClusterMargin #loss
from . import contrastive
from . import BADGE
from . import BatchBALD
from . import LESS
from . import DatasetQuantization
from . import EGL
from . import STAFF
from . import ZIP
from . import SPUQ
from . import UQ_ICL
from . import BM25  #Compute-constrained data selection ICLR2025