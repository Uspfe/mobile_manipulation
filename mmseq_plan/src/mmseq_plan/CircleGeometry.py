#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def find_tangents(p, c, r):
    xp = p[0]
    yp = p[1]
    a = c[0]
    b = c[1]
    
    tempa = r**2 * (xp - a)
    tempb = (xp-a)**2 + (yp-b)**2
    tempc = r*(yp - b)*np.sqrt(tempb- r**2)
    
    x = (tempa + tempc)/tempb + a
    
    tempd = r**2 * (yp - b)
    tempe = r*(xp - a)*np.sqrt(tempb- r**2)
    y = (tempd - tempe)/tempb + b
    
    return [x, y]

def find_intersection(p, c, r):
    p = np.array(p)
    c = np.array(c)
    d = np.linalg.norm(p - c)
    lam = (d - r)/d
    
    return list((1-lam)*p + lam*c)

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    
    p = [-5, 0.5]
    c = [0, 0]
    r = 1
    
    pt = find_tangents(p, c, r)
    pi = find_intersection(p, c, r)
    fig, axis = plt.subplots(1)
    plt.plot([p[0],pt[0]], [p[1],pt[1]], '-',label="tangent")
    plt.plot([p[0],pi[0]], [p[1],pi[1]], '-',label="intersect")
    plt.plot(pi[0], pi[1], 'o')
    circle = plt.Circle((c[0], c[1]), r, color='r')
    axis.add_patch(circle)
    
    plt.legend()
    plt.show()