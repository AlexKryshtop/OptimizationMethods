import numpy as np
from DFP import DFP
from scipy.optimize import minimize
#from IPython.display import display, clear_output
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

class SUO:
    """
        Sequential Unconstrained Optimization method that transform constraints into
        combined penalty function and perform unconstrained optimization

        Attributes
        ----------
        f (function): function that is optimized
        g (list(function)): list of inequality constraints
        h (list(function)): list of equality constraints
        beta (int): size of step that increases r
        eps (float): precision
        r (list(float)): penalty param
        y (list(tuple)): list of dots that were considered as optimal
        f_val (list(float)): list of f values in y
        n_iter (int): amount of iterations
        penalty_f_val (list(float)): value of penalty function in y
    """
    def __init__(self, f, g, h, beta=4, eps=1e-2):
        self.f = f
        self.g = list(g)
        self.h = list(h)
        self.beta = beta
        self.eps = eps
        self.r = [1.0]
        self.y = []
        self.f_val = []
        self.n_iter = 0
        self.penalty_f_val = []

    def get_penalty_function(self, x):
        """
            Creates penalty functions including each of constraint
        :param x: dot of evaluation
        :return: penalty function
        """
        penalty = 0

        for ineq_constraint in self.g:
            tmp = ineq_constraint(*x)
            if tmp < 0:
                penalty -= self.r[-1] * np.log(-ineq_constraint(*x))

        for eq_constraint in self.h:
            penalty += (1 / (self.r[-1] * 2)) * (eq_constraint(*x)**2)

        return penalty

    def opt_function(self, x):
        """
            Creates P-function that includes base function f and penalty function. Needed to
             perform unconstraied optimization
        :param x: dot
        :return: value of function
        """
        return self.f(*x) + self.get_penalty_function(x)

    def optimize(self, x0):
        """
            Main function to optimize
        :param x0: starting dot
        :return: optimal dot x_opt
        """
        self.y.append(x0)
        self.f_val.append(self.f(*x0))
        self.penalty_f_val.append(self.get_penalty_function(x0))
        opt_x = x0

        # пока занч по модолую штрафной фкнуции превосходит погрешность
        while np.abs(self.get_penalty_function(self.y[-1])) > self.eps:
            # Минимизируем безусловно комбинированую функцию
            dfp = DFP(self.opt_function, self.y[-1])
            # opt_x = dfp.solve()
            opt_x = minimize(self.opt_function, self.y[-1]).x

            # обновляем вес штрафного параметра и переходим дальше
            self.r.append(self.r[-1] / self.beta)
            self.y.append(opt_x)
            self.f_val.append(self.f(*opt_x))
            self.penalty_f_val.append(self.get_penalty_function(opt_x))
            self.n_iter += 1

        return opt_x

    def plot_contour(self):
        """
            Plot contour of main function. Also visualizes constraints
        :return: None
        """
        x1 = np.arange(-100, 100, 10)
        x2 = np.arange(-100, 100, 10)
        x1, x2 = np.meshgrid(x1, x2)

        fs = self.f(x1, x2)
        gs = []
        hs = []
        for ineq in self.g:
            gs.append(ineq(x1, x2))
        for eq in self.h:
            hs.append(eq(x1, x2))

        fig = plt.figure(figsize=(25, 20))
        layout = (2, 2)
        ax1 = plt.subplot2grid(layout, (0, 0), colspan=2)
        ax2 = plt.subplot2grid(layout, (1, 0))
        ax3 = plt.subplot2grid(layout, (1, 1))

        locator = ticker.SymmetricalLogLocator(linthresh=0.1, base=2)
        lin_locator = ticker.LogitLocator()

        picture1 = ax1.contourf(x1, x2, fs, locator=locator)
        for g in gs:
            picture2 = ax1.contourf(x1, x2, g, locator=lin_locator, colors='black')
            picture2 = ax2.contourf(x1, x2, g, locator=lin_locator, colors='black')
        for h in hs:
            picture3 = ax1.contourf(x1, x2, h, locator=lin_locator, colors='red')
            picture3 = ax3.contourf(x1, x2, h, locator=lin_locator, colors='red')

        ax1.set_xlabel('x1', fontsize=20)
        ax1.set_ylabel('x2', fontsize=20)
        ax1.set_title('Main function', fontsize=25)

        ax2.set_xlabel('x1', fontsize=15)
        ax2.set_ylabel('x2', fontsize=15)
        ax2.set_title('Constraint g', fontsize=20)

        ax3.set_xlabel('x1', fontsize=15)
        ax3.set_ylabel('x2', fontsize=15)
        ax3.set_title('Constraint h', fontsize=20)

        plt.show();

    def plot_solution(self):
        """
            Plots contour of main function and its constraints. Also visualizes optimal dot
            and descent for it.
        :return:
        """
        x1 = np.arange(-5, 5, 1)
        x2 = np.arange(-5, 5, 1)
        x1, x2 = np.meshgrid(x1, x2)

        fs = self.f(x1, x2)
        gs = []
        hs = []
        for ineq in self.g:
            gs.append(ineq(x1, x2))
        for eq in self.h:
            hs.append(eq(x1, x2))

        plt.figure(figsize=(25, 20))
        locator = ticker.SymmetricalLogLocator(linthresh=0.1, base=2)
        lin_locator = ticker.LogitLocator()

        picture1 = plt.contourf(x1, x2, fs, locator=locator)
        for g in gs:
            picture2 = plt.contourf(x1, x2, g, locator=lin_locator, colors='yellow')
        for h in hs:
            picture3 = plt.contourf(x1, x2, h, locator=lin_locator, colors='red')

        self.y = np.array(self.y)
        for i in range(1, len(self.y) - 1):
            plt.plot(self.y[i - 1:i + 1, 0], self.y[i - 1:i + 1, 1], color='black')
        plt.plot(self.y[-3:-1, 0], self.y[-3:-1, 1], color='black', label='Descent')

        plt.scatter(self.y[-1, 0], self.y[-1, 1], color='yellow', label='Solution')

        plt.legend(fontsize=14)
        plt.xlabel('x1', fontsize=20)
        plt.ylabel('x2', fontsize=20)
        plt.title('Solution', fontsize=25)
        plt.show();

    def stats(self):
        """
            Creates statistics report for each iteration/
        :return: pandas DataFrame
        """
        adder = {'k': '', 'rk': '', 'xk': '', 'f(xk)': '', 'P(xk, rk)': ''}

        cols = list(adder.keys())
        df = pd.DataFrame(columns=cols)
        for k in range(self.n_iter):

            adder['k'] = k
            adder['rk'] = self.r[k]
            adder['xk'] = self.y[k]
            adder['f(xk)'] = self.f_val[k]
            adder['P(xk, rk)'] = self.penalty_f_val[k]
            df = df.append(adder, ignore_index=True)
        return df

    def notebook_geom_process(self):
        """
            Function that can be called only from .ipynb.
            Demonstrates dynamic process of finding optimal dot
        :return:
        """
        fig = plt.figure(figsize=(20, 15))
        ax = fig.add_subplot(1, 1, 1)

        x1 = np.linspace(-2, 4)
        x2 = np.linspace(-2, 4)
        x1, x2 = np.meshgrid(x1, x2)

        fs = self.f(x1, x2)
        gs = []
        hs = []
        for ineq in self.g:
            gs.append(ineq(x1, x2))
        for eq in self.h:
            hs.append(eq(x1, x2))

        xs = np.array(self.y)

        locator = ticker.SymmetricalLogLocator(linthresh=0.1, base=2)
        lin_locator = ticker.LogitLocator()

        picture1 = plt.contourf(x1, x2, fs, locator=locator)
        for g in gs:
            picture2 = plt.contourf(x1, x2, g, locator=lin_locator, colors='yellow')
        for h in hs:
            picture3 = plt.contourf(x1, x2, h, locator=lin_locator, colors='red')
        ax.scatter(xs[-1, 0], xs[-1, 1], color='red')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        display(fig)

        clear_output(wait=True)
        plt.pause(0.5)

        for i in range(1, self.n_iter):
            ax.set_title('Iter: {}'.format(i), fontsize=25)
            ax.plot(xs[i - 1:i + 1, 0], xs[i - 1:i + 1, 1], color='black', linewidth=4)
            display(fig)

            clear_output(wait=True)
            plt.pause(0.3)


if __name__ == '__main__':
    print('First Problem:\n')
    solver = SUO(lambda x1, x2: x1*x1 + x2*x2,
                 [lambda x1, x2: x1 + x2 - 2],
                 [lambda x1, x2: x1 - 1])
    solver.plot_contour()
    solver.optimize(x0=(-2, 2))
    solver.plot_solution()
    problem1 = solver.stats()
    print(problem1)

    print('\n\nSecond Problem:\n')
    solver = SUO(lambda x1, x2: np.sqrt(x1**2 + x2**2),
                 [lambda x1, x2: -x1],
                 [lambda x1, x2: x1 + x2 - 2])
    solver.plot_contour()
    solver.optimize(x0=(-2, -3))
    solver.plot_solution()
    problem2 = solver.stats()
    print(problem2)

    print('\n\nThird Problem:\n')
    solver = SUO(lambda x1, x2: np.exp(x1) + np.exp(x2),
                 [lambda x1, x2: -x1 - x2 + 1,
                  lambda x1, x2: -x1,
                  lambda x1, x2: -x2],
                 [lambda x1, x2: x1*x1 + x2*x2 - 9])
    solver.plot_contour()
    solver.optimize(x0=(1, 1))
    solver.plot_solution()
    problem3 = solver.stats()
    print(problem3)

