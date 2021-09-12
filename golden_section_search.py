import pandas as pd
import matplotlib.pyplot as plt


class GoldenSectionSearch:
    """
    The GoldenSectionSearch class provides solution to optimization task without subjects.

    Attributes
    ----------
    a (list[float]): Left border of area that contracts in prospect
    b (list[float]): Right border of area that contracts in prospect
    func (function): Coefficients of function(x) that we try to optimize
    opt (str): Either 'min' or 'max'. Determine the way we do calculations. If 'max' we multiply func on (-1)
    esp (float): Precision of method's approach
    y (list[float]): Left Golden dots
    z (list[float]): Right Golden dots
    f_val (list[tuple()]): Values of func in golden dots
    """
    def __init__(self, a, b, func, opt, eps=1e-1):
        """
        The constructor for GoldenSectionSearch class.

        Parameters:
            a: Left border of area that contracts in prospect.
            b: Right border of area that contracts in prospect.
            func: Coefficients of function(x) that we try to optimize
            opt: Either 'min' or 'max'. Determine the way we do calculations. If 'max' we multiply func on (-1)
            eps: Precision of method's approach
        """
        self.a = [a]
        self.b = [b]
        self.func = func
        self.opt = opt
        self.eps = eps
        self.y = []
        self.z = []
        self.f_val = []

    def find_opt(self):
        """
        The main function to find optimum dots
        :return: FloatNumber that are in the middle of contracted area
        """
        k = 0
        # На первой итерации расчитаем две точки, которые произвлдять 2 золотых сечения
        self.y.append(self.a[-1] + 0.38196 * (self.b[-1] - self.a[-1]))
        self.z.append(self.a[-1] + self.b[-1] - self.y[-1])

        # пока расстояние между точками ссуженного отрезка не достаточно малое
        while abs(self.a[-1] - self.b[-1]) > self.eps:
            t = abs(self.a[-1] - self.b[-1])

            # считаем значения функции в точках, которые делают золотые сечения всего отрезка
            self.f_val.append(tuple((self.func(self.y[-1], self.opt), self.func(self.z[-1], self.opt))))

            # если значение функции в такой левой точке меньше или равно значению в правой
            if self.f_val[-1][0] <= self.f_val[-1][1]:
                # то левый конец отрезка остается неизменным, а правый конец принимает значение правой золотой точки
                self.a.append(self.a[-1])
                self.b.append(self.z[-1])

                # происходит смещение золотых точек: правая на новом отрезке становится той,
                # которая была левой на прошлом отрезке [a;b]
                # а левая высчитывается заново по формуле
                self.z.append(self.y[-1])
                # self.y.append(self.a[-1] + self.b[-1] - self.y[-1])
                self.y.append(self.a[-1] + 0.38196 * (self.b[-1] - self.a[-1]))

            # иначе все наооборот
            else:
                self.a.append(self.y[-1])
                self.b.append(self.b[-1])

                self.y.append(self.z[-1])
                # self.z.append(self.a[-1] + self.b[-1] - self.z[-1])
                self.z.append(self.a[-1] + 0.618 * (self.b[-1] - self.a[-1]))

            k += 1

        self.f_val.append(tuple((self.func(self.y[-1], self.opt), self.func(self.z[-1], self.opt))))
        # возращаем значение середины последнего интервала для увеличения точности
        return (self.a[-1] + self.b[-1]) / 2

    def cross_tab(self):
        """
        The function that creates table with calculations during each iteration
        :return: pandas.DataFrame object
        """
        adder = {'Iter': '', 'a': '', 'b': '', '|a - b|': '', 'y': '', 'z': '', 'f(y)': '', 'f(z)': '', 'eps': ''}
        cols = list(adder.keys())
        df = pd.DataFrame(columns=cols)
        for i in range(len(self.a)):
            adder['Iter'] = i
            adder['a'] = float('{:.4f}'.format(self.a[i]))
            adder['b'] = float('{:.4f}'.format(self.b[i]))
            adder['|a - b|'] = float('{:.4f}'.format(abs(self.a[i] - self.b[i])))
            adder['y'] = float('{:.4f}'.format(self.y[i]))
            adder['z'] = float('{:.4f}'.format(self.z[i]))
            adder['f(y)'] = float('{:.4f}'.format(self.f_val[i][0]))
            adder['f(z)'] = float('{:.4f}'.format(self.f_val[i][1]))
            adder['eps'] = self.eps

            df = df.append(adder, ignore_index=True)
        return df.set_index('Iter')

    def visualize(self):
        """
        The function that visualize function and its optima dot
        :return: nothing
        """
        plt.figure(figsize=(12, 8))
        x = np.linspace(self.a[0], self.b[0])
        # всегда считаем функции как для минимизации, чтобы на графике при максимизации функции не умножались на -
        f = np.array([self.func(xi, 'min') for xi in x])
        x_opt = (self.a[-1] + self.b[-1]) / 2
        f_opt = self.func(x_opt, 'min')
        plt.title('gcc for eps = ' + str(self.eps))
        plt.plot(x, f, label='F(x) ->' + self.opt)
        plt.scatter(x_opt, f_opt, color='r', label=self.opt, linewidths=5)
        plt.legend()
        plt.show()
        # plt.savefig('C:\\Users\\Admin\\PycharmProjects\\Polynom\\graphs\\' + str(self.func.__name__) +
                  # '_' + str(self.eps) + '.png')


if __name__ == '__main__':
    import numpy as np
    # задаем функции, на которых будем тестировать

    def f1(x, opt='min'):
        return abs(x) + abs(x + 1) - 1 if opt == 'min' else 1 - abs(x) - abs(x + 1)

    def f2(x, opt='min'):
        return x / (x * x + 1) if opt == 'min' else (-1) * x / (x * x + 1)

    def f3(x, opt='min'):
        return 1 + x - 2.5 * x * x + 0.25 * x**4 if opt == 'min' else -1 - x + 2.5 * x * x - 0.25 * x**4

    def f4(x, opt='min'):
        return np.log(x) - 2 * x * x if opt == 'min' else 2 * x * x - np.log(x)


    # задаем значения точности от 1^-4 и до 1^-1
    eps = np.logspace(-4, -1, 4)
    # задаем соответсвующие параметры, которые будем использовать для автоматического тестирования
    # здесь каждый столбец этих 4 списков принадлежит к одной функции
    f = [f1, f2, f3, f4]
    a = [1, -3/2, 0, 0.25]
    b = [3, 0, 1, 2]
    opt = ['min', 'min', 'max', 'max']
    k = 1

    for v in zip(a, b, f, opt):
        # конструтруем кортеж значений по столбцам
        for e in eps:
            print('\n\n\t\t\tFor function f' + str(k) + '\n')
            # затем к нему добавляем новое возможное значение eps
            temp = v + (e,)
            # проводим оптимизацию и выводим таблицы и графики
            gss = GoldenSectionSearch(*temp)
            sol = gss.find_opt()
            print(gss.cross_tab())
            print('Optimal X = ', sol, '\tOptimal Value = ', v[2](sol), '\n\n')
            gss.visualize()
        k += 1

