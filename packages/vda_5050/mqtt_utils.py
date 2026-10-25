import logging

from paho.mqtt.client import Client
from paho.mqtt.enums import CallbackAPIVersion, MQTTErrorCode

logger = logging.getLogger(__name__)


class VDA5050MQTTClient(Client):
    def __init__(self, broker: str, port: int):
        super().__init__(CallbackAPIVersion.VERSION2)
        self.broker = broker
        self.port = port

    def connect_to_broker(self) -> MQTTErrorCode:
        if self.is_connected():
            logger.info("Already connected to MQTT broker.")
            return MQTTErrorCode.MQTT_ERR_SUCCESS
        return self.connect(self.broker, self.port)
    
    def disconnect_from_broker(self) -> MQTTErrorCode:
        if not self.is_connected():
            logger.info("Already disconnected from MQTT broker.")
            return MQTTErrorCode.MQTT_ERR_SUCCESS
        return self.disconnect()
    
