"""
Verify Torch
"""

import asyncio

import control as ct
import onnxruntime as ort
from casadi.casadi import DeserializerBase


class VerifyTorch:
    def __init__(self):
        pass
    
    async def do_something(self):
        print('doing something')
        return 10
    
    async def do_something2(self):
        print('doing somethin2')


async def main():
    verify_torch = VerifyTorch()
    t1 = verify_torch.do_something()
    t2 = verify_torch.do_something2()
    await t2
    await t1


if __name__ == "__main__":
    asyncio.run(main())
