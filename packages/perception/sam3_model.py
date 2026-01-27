from dataclasses import dataclass

from ultralytics.models.sam import SAM3SemanticPredictor


@dataclass
class SamConfig:
    conf: float
    task: str
    mode: str
    model: str
    half: bool = True
    save: bool = False

    def to_dict(self) -> dict:
        """Convert the config to a dictionary."""
        return {
            "conf": self.conf,
            "task": self.task,
            "mode": self.mode,
            "model": self.model,
            "half": self.half,
            "save": self.save,
        }


sam_config = SamConfig(
    conf=0.25,
    task="segment",
    mode="predict",
    model="sam3.pt",
    half=True,
    save=True,
)

overrides = sam_config.to_dict()

predictor = SAM3SemanticPredictor(overrides=overrides)
predictor.set_image("image.jpg")
results = predictor(text=["person", "bus", "glasses"])

results = predictor(text=["person with red cloth", "person with blue cloth"])

results = predictor(text=["a person"])
