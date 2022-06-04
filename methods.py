import numpy as np


# Solves for x: f(x) = x
def fpi(f, x0, diff):
    xs = [x0, f(x0)]

    while abs(xs[-1] - xs[-2]) >= diff:
        xs.append(f(xs[-1]))

    return xs


# Solves for x: f(x) = 0
def newton(f, f_grad, x0, diff):
    def step(x): return x - f(x) / f_grad(x)

    return fpi(step, x0, diff)


# Solves for y(x): dy/dx = f(x, y)
def euler(f, xy0, h, iters):
    def adv_x(x): return x + h
    def adv_y(y, x): return y + h * f(x, y)

    xys = [xy0]

    for _ in range(0, iters):
        x, y = xys[-1]
        xys.append((adv_x(x), adv_y(y, x)))

    return xys


# Solves for y(x): dy/dx = f(x, y)
def euler_mod(f, xy0, h, iters=10):
    def adv_x(x): return x + h

    def adv_y(y, x):
        return y + (h/2) * (f(x, y) + f(adv_x(x), y + h * f(x, y)))

    xys = [xy0]

    for _ in range(0, iters):
        x, y = xys[-1]
        xys.append((adv_x(x), adv_y(y, x)))

    return xys
