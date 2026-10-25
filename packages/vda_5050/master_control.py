from vda_5050.config import MqttConfig
from vda_5050.mqtt_utils import VDA5050MQTTClient


class MasterControl(VDA5050MQTTClient):
    def __init__(self, config: MqttConfig):
        super().__init__(config.mqtt_broker, config.mqtt_port)
        self.connect_to_broker()
    
    
