import matplotlib.pyplot as plt
import numpy as np


def plotter(func, a, b, root):
    arg = np.linspace(a, b)
    values = [func(x) for x in arg]

    plt.plot(arg, values, label='f(x)')
    plt.plot(arg, np.zeros(50), label='x = 0')
    plt.scatter(root, func(root), linewidths=5, label='root')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def f1(x):
    return x ** 3 + 3 * x * x - 3


def f2(x):
    return x ** 3 + x - 1


def f3(x):
    return x ** 3 + 8 * x + 10


def dx_dy(func, x, method='central', h=1e-10):
    if method == 'central':
        return (func(x + h) - func(x - h))/(2*h)
    elif method == 'forward':
        return (func(x + h) - func(x))/h
    elif method == 'backward':
        return (func(x) - func(x - h))/h
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")


def dx2_dy(func, x, method='central', h=1e-5):
    if method == 'central':
        return (dx_dy(func, x + h, method=method, h=h) - dx_dy(func, x-h, method=method, h=h))/(2*h)
    elif method == 'forward':
        return (dx_dy(func, x + h, method=method, h=h) - dx_dy(func, x, method=method, h=h))/h
    elif method == 'backward':
        return (dx_dy(func, x, method=method, h=h) - dx_dy(func, x - h, method=method, h=h))/h
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")


def combo_newton_secant_method(func, a, b, eps):
    if func(a) * func(b) >= 0:
        raise ValueError('Bad interval [a, b]')
    else:
        if func(a) * dx2_dy(func, a) > 0:
            X = [a]
        else:
            X = [b]
        x11 = X[-1] - func(X[-1]) / dx_dy(func, X[0])
        x22 = a - ((b-a) * func(a) / (func(b) - func(a)))
        X.append((x11 + x22) / 2)
        while abs(X[-1] - x11) > eps:
            x22 = x11 - ((x22 - x11) * func(x11) / (func(x22) - func(x11)))
            x11 -= func(x11) / dx_dy(func, x11)
            X.append((x11 + x22) / 2)
        return X[-1]


if __name__ == '__main__':
    eps = 0.001
    a = -2.75
    b = -2.5
    sol = combo_newton_secant_method(f1, a, b, eps)
    print(sol)
    print(f1(sol))
    plotter(f1, a, b, sol)
    print('\n\n')
    # test2 (3)
    eps = 0.001
    a = -2
    b = -1
    sol = combo_newton_secant_method(f3, a, b, eps)
    print(sol)
    print(f3(sol))
    plotter(f3, a, b, sol)
