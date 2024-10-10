import math


def cross_entropy(pre_value, label):
    return -(label * math.log(pre_value) + (1 - label) * math.log(1 - pre_value))

