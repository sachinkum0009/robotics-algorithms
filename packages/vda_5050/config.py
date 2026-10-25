from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class MqttConfig:
    mqtt_broker: str
    mqtt_port: int

    @staticmethod
    def load_from_file(file_path: Path) -> "MqttConfig":
        # Implement loading logic here
        if not file_path.exists():
            raise FileNotFoundError(f"Config file {file_path} not found.")

        # check if file is yaml type
        if file_path.suffix != ".yaml" and file_path.suffix != ".yml":
            raise ValueError(f"Config file {file_path} must be a yaml file.")

        with open(file_path) as f:
            config_dict = yaml.safe_load(f)

        return MqttConfig(
            mqtt_broker=config_dict.get("host", "localhost"),
            mqtt_port=config_dict.get("port", 1883),
        )
