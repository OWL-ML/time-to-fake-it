from .accumulated_trainer import AccumulatedTrainer
from .der_trainer import DERPPTrainer
from .ewc_trainer import EWCTrainer
from .pina_trainer import PINATrainer
from .soyo_trainer import SOYOTrainer
from .sprompts_trainer import SPromptsTrainer
from .trainer import Trainer

__all__ = [
    "Trainer",
    "DERPPTrainer",
    "EWCTrainer",
    "AccumulatedTrainer",
    "PINATrainer",
    "SOYOTrainer",
    "SPromptsTrainer",
]


def __getattr__(name):
    # Avoid importing GradMatch dependencies (which include global NumPy
    # error-state mutation) unless GradMatchTrainer is explicitly requested.
    if name == "GradMatchTrainer":
        from .gm_trainer import GradMatchTrainer

        return GradMatchTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
