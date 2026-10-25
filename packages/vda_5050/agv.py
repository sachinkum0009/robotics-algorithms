import logging

from vda_5050.config import MqttConfig
from vda_5050.mqtt_utils import VDA5050MQTTClient
from vda_5050.protocol.vda_2_0_0.vda_5050_2_0_0_order import Order

logger = logging.getLogger(__name__)


class AGV(VDA5050MQTTClient):
    def __init__(self, config: MqttConfig):
        super().__init__(config.mqtt_broker, config.mqtt_port)
        self.connect_to_broker()

    def send_order(self):
        order = Order("my_interface", "v0.0.1", "apple", "12345", "talk")
        order_json = order.to_json()
        logger.info(f"order data: {order_json}")
