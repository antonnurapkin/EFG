from helpers import distance
from params import LAMBDA, DS


def r_diffraction_method(point, crack_top, node):
    s0 = distance(x1=point.x, y1=point.y, x2=node.x, y2=node.y)
    s1 = distance(x1=node.x, y1=node.y, x2=crack_top.x, y2=crack_top.y)
    s2 = distance(x1=point.x, y1=point.y, x2=crack_top.x, y2=crack_top.y)

    return (((s1 + s2) / s0) ** LAMBDA) * s0 / DS


def drdx(point, crack_top, node):
    s0 = distance(x1=point.x, y1=point.y, x2=node.x, y2=node.y)
    s2 = distance(x1=point.x, y1=point.y, x2=crack_top.x, y2=crack_top.y)

    return ((point.x - crack_top.x) * (s0 ** 2) - (point.x - node.x) * (s2 ** 2)) / ((s0 ** 3) * s2) / DS


def drdy(point, crack_top, node):
    s0 = distance(x1=point.x, y1=point.y, x2=node.x, y2=node.y)
    s2 = distance(x1=point.x, y1=point.y, x2=crack_top.x, y2=crack_top.y)

    return ((point.y - crack_top.y) * (s0 ** 2) - (point.y - node.y) * (s2 ** 2)) / ((s0 ** 3) * s2) / DS

