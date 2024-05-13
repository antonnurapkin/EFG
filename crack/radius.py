from helpers import distance
from params import LAMBDA, DS


def r_diffraction_method(point, crack_top, node):
    s0 = distance(x1=point.x, y1=point.y, x2=node.x, y2=node.y)
    s1 = distance(x1=node.x, y1=node.y, x2=crack_top.x, y2=crack_top.y)
    s2 = distance(x1=point.x, y1=point.y, x2=crack_top.x, y2=crack_top.y)

    return (((s1 + s2) / s0) ** LAMBDA) * s0 / DS


def drdx(point, crack_top, node):
    s0 = distance(x1=point.x, y1=point.y, x2=node.x, y2=node.y)
    s1 = distance(x1=node.x, y1=node.y, x2=crack_top.x, y2=crack_top.y)
    s2 = distance(x1=point.x, y1=point.y, x2=crack_top.x, y2=crack_top.y)

    return (LAMBDA * (((s1 + s2) / s0) ** (LAMBDA - 1)) * ((point.x - crack_top.x) / s2) +
            (1 - LAMBDA) * (((s1 + s2) / s0) ** LAMBDA) * ((point.x - crack_top.x) / s0)) / DS



def drdy(point, crack_top, node):
    s0 = distance(x1=point.x, y1=point.y, x2=node.x, y2=node.y)
    s1 = distance(x1=node.x, y1=node.y, x2=crack_top.x, y2=crack_top.y)
    s2 = distance(x1=point.x, y1=point.y, x2=crack_top.x, y2=crack_top.y)

    return (LAMBDA * (((s1 + s2) / s0) ** (LAMBDA - 1)) * ((point.y - crack_top.y) / s2) +
            (1 - LAMBDA) * (((s1 + s2) / s0) ** LAMBDA) * ((point.y - crack_top.y) / s0)) / DS
