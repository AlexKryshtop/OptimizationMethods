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


def secant_method(func, a, b, eps):
    if func(b) * dx2_dy(func, b) > 0:
        X = [a]
        temp = (-1) * func(X[-1]) * (b - X[-1]) / (func(b) - func(X[-1]))
        while abs(temp) > eps:
            X.append(X[-1] + temp)
            temp = (-1) * func(X[-1]) * (b - X[-1]) / (func(b) - func(X[-1]))
        return X[-1]
    else:
        X = [b]
        temp = (-1) * func(X[-1]) * (X[-1] - a) / (func(X[-1]) - func(a))
        while abs(temp) > eps:
            X.append(X[-1] + temp)
            temp = (-1) * func(X[-1]) * (X[-1] - a) / (func(X[-1]) - func(a))
        return X[-1]


if __name__ == '__main__':
    # test 1
    eps = 0.001
    a = -2.75
    b = -2.5
    sol = secant_method(f1, a, b, eps)
    print(sol)
    print(f1(sol))
    plotter(f1, a, b, sol)

    # test 2
    a = 0
    b = 1
    sol = secant_method(f2, a, b, eps)
    print(sol)
    print(f2(sol))
    plotter(f2, a, b, sol)