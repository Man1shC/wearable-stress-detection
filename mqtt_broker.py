"""
Local MQTT broker. Stands in for a cloud IoT endpoint (AWS IoT Core, Azure IoT Hub).

Usage:
    python mqtt_broker.py

Leave running. Listens on localhost:1883, anonymous access.
"""

import asyncio
import logging

from amqtt.broker import Broker

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s broker  %(message)s",
                    datefmt="%H:%M:%S")

CONFIG = {
    "listeners": {
        "default": {"type": "tcp", "bind": "127.0.0.1:1883", "max_connections": 10},
    },
    "sys_interval": 0,
    "auth": {"allow-anonymous": True, "plugins": ["auth_anonymous"]},
    "topic-check": {"enabled": False},
}


async def main():
    broker = Broker(CONFIG)
    await broker.start()
    print("broker listening on 127.0.0.1:1883 - ctrl+c to stop")
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        await broker.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nbroker stopped")
