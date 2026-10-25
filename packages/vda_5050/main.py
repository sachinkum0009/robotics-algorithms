from pathlib import Path
from vda_5050.config import Config
from vda_5050.utils import create_mqtt_client


def main():
    # config path should first get the current file path, then get the parent path, then get the config path
    config_path = Path(__file__).parent / "config" / "config.yaml"
    config = Config.load_from_file(config_path)

    mqtt_client = create_mqtt_client(config)
    
