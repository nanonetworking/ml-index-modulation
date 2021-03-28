# -*- coding: utf-8 -*-
"""
@author: Ozgur Kara
"""
import numpy as np

PI = np.pi


def sph2cart(az, el, r):
    """converting spherical coordinates to cartesian coordinates"""
    rsin_theta = r * np.sin(el)
    y = rsin_theta * np.sin(az)
    x = rsin_theta * np.cos(az)
    z = r * np.cos(el)
    return x, y, z


def between(value, a, b):
    """find and returns the substring which is between the characters of a and b"""
    pos_a = value.find(a)
    if pos_a == -1: 
        return ""
    pos_b = value.rfind(b)
    if pos_b == -1: 
        return ""
    adjusted_pos_a = pos_a + len(a)
    if adjusted_pos_a >= pos_b: 
        return ""
    return value[adjusted_pos_a:pos_b]
       
def az_el_pair(az_num, el_num):
    azimuth = np.linspace(-PI, PI, az_num + 1)
    elevation = np.linspace(0,PI, el_num + 1)
    return azimuth, elevation