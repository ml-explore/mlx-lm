# Copyright © 2023 Apple Inc.

import inspect
from dataclasses import dataclass

__all__ = [
    "BaseModelArgs",
]


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
