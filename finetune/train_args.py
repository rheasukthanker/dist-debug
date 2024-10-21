from typing import Optional
from dataclasses import dataclass

from litgpt.args import TrainArgs


@dataclass
class FineTuningArgs(TrainArgs):

    learning_rate: Optional[float] = 2e-5
    temperature: Optional[float] = 10.0
    distillation_weight : Optional[float] = 0.5