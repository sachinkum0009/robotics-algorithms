from itertools import chain
import pint


# class MyDict(object):
#     def __init__(self):
#         self.data = {}

#     def __setitem__(self, key, value):
#         print(f"Setting {key} to {value}")
#         self.data[key] = value

#     def __getitem__(self, key):
#         print(f"Getting value for {key}")
#         return self.data[key]


# my_dict = MyDict()
# my_dict["a"] = 1
# my_dict["b"] = 2
# print(my_dict)
# print(type(my_dict))

ureg = pint.UnitRegistry()
my_force = 9.8 * ureg.newton
distance = my_force * (0.2 * ureg.second ** 2)
print(my_force)
print(distance)