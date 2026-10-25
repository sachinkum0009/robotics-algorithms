import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Order:
    interface_name: str
    major_version: str
    manufacturer: str
    serial_number: str
    topic: str

    def to_json(self) -> str:
        return json.dumps(asdict(self))
