import threading

import pytest
from paho.mqtt.enums import MQTTErrorCode
from vda_5050.mqtt_utils import VDA5050MQTTClient


@pytest.fixture
def mqtt_client():
    client = VDA5050MQTTClient(broker="localhost", port=1883)
    yield client
    client.loop_stop()
    client.disconnect_from_broker()


def test_mqtt_connection(mqtt_client):
    connected_event = threading.Event()

    # def on_connect(client, userdata, connect_flags, reason_code, properties):
    #     if not reason_code.is_failure:
    #         connected_event.set()

    # mqtt_client.on_connect = on_connect

    assert not mqtt_client.is_connected()
    mqtt_client.loop_start()
    err_code = mqtt_client.connect_to_broker()
    assert err_code == MQTTErrorCode.MQTT_ERR_SUCCESS

    connected = connected_event.wait(timeout=5.0)
    print(f"Is connected: {mqtt_client.is_connected()}")
    assert connected, "Timed out waiting for CONNACK"
    assert mqtt_client.is_connected()
