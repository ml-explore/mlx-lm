# Copyright © 2026 Apple Inc.

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs, create_ssm_mask
from .cache import ArraysCache
from .ssm import ssm_update


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    