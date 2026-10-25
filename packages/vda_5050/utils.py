from paho.mqtt.client import Client

from vda_5050.config import Config
from vda_5050.mqtt_utils import VDA5050MQTTClient


def create_mqtt_client(config: Config) -> Client:
    client = VDA5050MQTTClient(
        broker=config.mqtt_broker, port=config.mqtt_port
    )
    client.connect_to_broker()
    return client
