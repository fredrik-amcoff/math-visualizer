import sys
import warnings
import inspect
import ast
import math
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import sympy as sp
import sympy.stats
from sympy import exp, sin, cos, oo, pi, log, S, nsimplify
from sympy.calculus.util import continuous_domain
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sympy import diff, integrate
import pickle
from pathlib import Path
import dill
from dataclasses import dataclass, asdict
import json
import inspect
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from itertools import product
import random
import itertools
import bisect


class GraphWindow(QtWidgets.QWidget):
    def __init__(self, rows=1, cols=1):
        super().__init__()
        self.setWindowTitle("Graph Plot")

        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)

        #self.plotWidget = pg.PlotWidget()
        #layout.addWidget(self.plotWidget)
        self.rows = rows
        self.cols = cols
        self.total_cells = rows * cols

        self.plots = []  # contains actual PlotItem objects
        self.graph_objects = []  # stores your Graph instances

        # Slider panel
        self.slider_window = SliderWindow()
        self.slider_window.show()
        self.slider_panel = self.slider_window.layout

        # Expression window
        self.expression_window = ExpressionWindow()
        self.expression_window.show()
        self.expression_panel = self.expression_window.layout
        self.hbox = None

        # Store where graphs go
        self.cells = [[None for _ in range(cols)] for _ in range(rows)]
        self.graph_objects = []

        self.obj_graph_connections = {}
        self.parameters = {}
        self.parameter_connections = {}
        self.single_values = {}

        self.current_index = 0  # next free cell

    def add_graph(self, graph_obj):
        if self.current_index >= self.rows * self.cols:
            raise ValueError("GraphWindow is full.")

        row = self.current_index // self.cols
        col = self.current_index % self.cols

        # Plot widget
        plot_widget = pg.PlotWidget()
        self.layout.addWidget(plot_widget, row, col)

        # create the plot item for this cell
        #plot_item = self.addPlot(row=row, col=col)

        # assign the PlotItem to the graph
        graph_obj.set_plot_item(plot_widget)
        graph_obj.parent = self

        self.plots.append(graph_obj)
        self.current_index += 1

    def add_parameter(self, name, min_val, max_val, init_val, steps=None, numeric_domain='real'):
        if name in self.parameters:
            raise NameError("Parameter {} already exists".format(name))
        if steps is None:
            steps = (max_val - min_val)*10
        if numeric_domain == 'integer':
            init_val = int(init_val)
        if numeric_domain == 'rational':  # WORK IN PROGRESS
            init_val = nsimplify(init_val)
        if numeric_domain == 'real':
            init_val = float(init_val)

        step_size = (max_val - min_val)/steps

        layout = self.slider_window.main_layout

        container = QtWidgets.QWidget()
        hbox = QtWidgets.QHBoxLayout(container)

        label = QtWidgets.QLabel(f"{name}: {init_val}")
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setFixedWidth(45)
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(int(min_val / step_size), int(max_val / step_size))
        slider.setValue(int(init_val / step_size))
        slider.setSingleStep(1)

        hbox.addWidget(label)
        hbox.addWidget(slider)
        layout.addWidget(container)

        param = Parameter(name, label, slider, min_val, max_val, init_val, steps, numeric_domain)
        slider.valueChanged.connect(lambda _: self._update_param(param))
        self.parameters[name] = param
        self.parameter_connections[name] = []
        self.single_values[name] = ('parameter', init_val)
        return param

    def _update_param(self, param):
        step_size = (param.max_val - param.min_val) / param.step
        decimals = max(0, math.ceil(-np.log10(step_size)) if step_size < 1 else 0)
        param.value = round(param.slider.value() * step_size, decimals)
        if param.numeric_domain == 'integer':
            param.value = int(param.value)
        elif param.numeric_domain == 'rational':
            param.value = nsimplify(param.value)
        elif param.numeric_domain == 'real':
            param.value = float(param.value)
        param.label.setText(f"{param.name}: {param.value}")
        self.single_values[param.name] = ("parameter", param.value)
        param.update_values(param.value)
        for obj in self.parameter_connections[param.name]:
            graph = self.obj_graph_connections[obj]  # The graph the object is in
            if isinstance(obj, Expression):
                updated_value = obj.update_values(self.parameters)
                obj._label.setText(f"{obj.name} = {obj.name}: {updated_value}")
                self.expression_window.update_values(obj)
            else:
                param_updates = {}
                for key, value in obj.param_connections.items():
                    if param.name in value:
                        param_updates[key] = param.value
                obj.update_values(graph.x_view_range, graph.y_view_range, **param_updates)
            # print(param_updates)
        # print(self.points[0].x, self.points[0].y)



class SliderWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Parameters")
        self.resize(250, 600)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)

        self.container = QtWidgets.QWidget()
        self.main_layout = QtWidgets.QVBoxLayout(self.container)
        scroll.setWidget(self.container)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(scroll)


class ExpressionWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Expressions")
        self.resize(500, 600)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)

        self.container = QtWidgets.QWidget()
        self.main_layout = QtWidgets.QVBoxLayout(self.container)
        scroll.setWidget(self.container)

        self.expression_widgets = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(scroll)

    def add_expression(self, expression):
        """Render a Sympy expression in LaTeX and add to layout"""
        container = QtWidgets.QWidget()
        hbox = QtWidgets.QHBoxLayout(container)
        latex_str = sp.latex(expression.expr)
        expression._label = QtWidgets.QLabel(f"${expression.expression_name}={latex_str}={round(expression.value, 5)}$")
        fig = Figure(figsize=(3, 0.5))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f"${expression.expression_name} = {latex_str} = {round(expression.value, 5)}$", ha='center', va='center',
                fontsize=10)
        ax.axis('off')
        hbox.addWidget(canvas)

        self.expression_widgets[expression.expression_name] = (canvas, ax)

        self.main_layout.addWidget(container)

        # Set initial value
        #self.update_value(expression)

    def update_values(self, expression):
        """Re-evaluate and update the LaTeX rendering with numeric value."""
        canvas, ax = self.expression_widgets[expression.expression_name]
        ax.clear()
        ax.axis("off")

        # Evaluate numeric value with current parameter values
        subs = {p.expr: p.value for p in self.params.values()}
        value = expression.expr.subs(subs).evalf()

        # Build LaTeX string including evaluated result
        latex_expr = sp.latex(expression.expr)
        latex_full = f"${expression.expression_name} = {latex_expr} = {sp.latex(round(value, 5))}$"

        ax.text(0.5, 0.5, latex_full, ha="center", va="center", fontsize=10)
        canvas.draw_idle()


class Parameter():
    def __init__(self, name, label, slider, min_val, max_val, value, step, numeric_domain):
        self.name = name
        self.label = label
        self.slider = slider
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.step = step
        self.parameter_dependencies = {self.name: self.value}
        self.expr = sp.symbols(name)
        self.symbol = sp.Symbol(name)
        self.numeric_domain = numeric_domain

    def update_values(self, new_val):
        self.value = new_val

    def save_data(self):
        return ("parameter", {
                "name": self.name,
                "min_val": self.min_val,
                "max_val": self.max_val,
                "init_val": self.value,
                "step": self.step}
                )


    def __add__(self, other, reverse=False):
        if type(other) in (Parameter, Expression):
            other_dependencies = other.parameter_dependencies
            other = other.expr
        else:
            other_dependencies = {}
        other = sp.sympify(other)
        expression = other + self.expr if reverse else self.expr + other
        params = self.parameter_dependencies | other_dependencies
        value = expression.subs(params).evalf()
        return Expression(str(expression), "", value, params, expression)

    def __radd__(self, other):
        return self.__add__(other, reverse=True)

    def __sub__(self, other, reverse=False):
        if type(other) in (Parameter, Expression):
            other_dependencies = other.parameter_dependencies
            other = other.expr
        else:
            other_dependencies = {}
        other = sp.sympify(other)
        expression = other - self.expr if reverse else self.expr - other
        params = self.parameter_dependencies | other_dependencies
        value = expression.subs(params).evalf()
        return Expression(str(expression), "", value, params, expression)

    def __rsub__(self, other):
        return self.__sub__(other, reverse=True)

    def __mul__(self, other, reverse=False):
        if type(other) in (Parameter, Expression):
            other_dependencies = other.parameter_dependencies
            other = other.expr
        else:
            other_dependencies = {}
        other = sp.sympify(other)
        expression = other * self.expr if reverse else self.expr * other
        params = self.parameter_dependencies | other_dependencies
        value = expression.subs(params).evalf()
        return Expression(str(expression), "", value, params, expression)

    def __rmul__(self, other):
        return self.__mul__(other, reverse=True)

    def __truediv__(self, other, reverse=False):
        if type(other) in (Parameter, Expression):
            other_dependencies = other.parameter_dependencies
            other = other.expr
        else:
            other_dependencies = {}
        other = sp.sympify(other)
        expression = other / self.expr if reverse else self.expr / other
        params = self.parameter_dependencies | other_dependencies
        value = expression.subs(params).evalf()
        return Expression(str(expression), "", value, params, expression)

    def __rtruediv__(self, other):
        return self.__truediv__(other, reverse=True)

    def __floordiv__(self, other, reverse=False):
        if type(other) in (Parameter, Expression):
            other_dependencies = other.parameter_dependencies
            other = other.expr
        else:
            other_dependencies = {}
        other = sp.sympify(other)
        expression = other // self.expr if reverse else self.expr // other
        params = self.parameter_dependencies | other_dependencies
        value = expression.subs(params).evalf()
        return Expression(str(expression), "", value, params, expression)

    def __rfloordiv__(self, other):
        return self.__floordiv__(other, reverse=True)

    def __mod__(self, other, reverse=False):
        if type(other) in (Parameter, Expression):
            other_dependencies = other.parameter_dependencies
            other = other.expr
        else:
            other_dependencies = {}
        other = sp.sympify(other)
        expression = other % self.expr if reverse else self.expr % other
        params = self.parameter_dependencies | other_dependencies
        value = expression.subs(params).evalf()
        return Expression(str(expression), "", value, params, expression)

    def __rmod__(self, other):
        return self.__mod__(other, reverse=True)

    def __pow__(self, other, reverse=False):
        if type(other) in (Parameter, Expression):
            other_dependencies = other.parameter_dependencies
            other = other.expr
        else:
            other_dependencies = {}
        other = sp.sympify(other)
        expression = other ** self.expr if reverse else self.expr ** other
        params = self.parameter_dependencies | other_dependencies
        value = expression.subs(params).evalf()
        return Expression(str(expression), "", value, params, expression)

    def __rpow__(self, other):
        return self.__pow__(other, reverse=True)

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)

    def __str__(self):
        return self.name

    def _sympy_(self):
        return self.symbol


class Expression():
    def __init__(self, name, label, value, parameter_dependencies, expr):
        self.name = name
        self.label = label
        self.value = value
        self.parameter_dependencies = parameter_dependencies
        self.param_connections = {'params': self.parameter_dependencies}
        self.expr = expr


    def update_values(self, params_dict):
        subs_dict = {p.expr: p.value for p in params_dict.values()}
        return self.expr.subs(subs_dict)

    def save_data(self):
        return ("expression", {"expression": self.expr, "expression_name": self.name})

    def __add__(self, other, reverse=False):
        if type(other) in (Parameter, Expression):
            other_dependencies = other.parameter_dependencies
            other = other.expr
        else:
            other_dependencies = {}  # possible bug, can be solved by changing to set() instead
        other = sp.sympify(other)
        expression = other + self.expr if reverse else self.expr + other
        params = self.parameter_dependencies | other_dependencies
        value = expression.subs(params).evalf()
        return Expression(str(expression), "", value, params, expression)

    def __radd__(self, other):
        return self.__add__(other, reverse=True)

    def __sub__(self, other, reverse=False):
        if type(other) in (Parameter, Expression):
            other_dependencies = other.parameter_dependencies
            other = other.expr
        else:
            other_dependencies = {}
        other = sp.sympify(other)
        expression = other - self.expr if reverse else self.expr - other
        params = self.parameter_dependencies | other_dependencies
        value = expression.subs(params).evalf()
        return Expression(str(expression), "", value, params, expression)

    def __rsub__(self, other):
        return self.__sub__(other, reverse=True)

    def __mul__(self, other, reverse=False):
        if type(other) in (Parameter, Expression):
            other_dependencies = other.parameter_dependencies
            other = other.expr
        else:
            other_dependencies = {}
        other = sp.sympify(other)
        expression = other * self.expr if reverse else self.expr * other
        params = self.parameter_dependencies | other_dependencies
        value = expression.subs(params).evalf()
        return Expression(str(expression), "", value, params, expression)

    def __rmul__(self, other):
        return self.__mul__(other, reverse=True)

    def __truediv__(self, other, reverse=False):
        if type(other) in (Parameter, Expression):
            other_dependencies = other.parameter_dependencies
            other = other.expr
        else:
            other_dependencies = {}
        other = sp.sympify(other)
        expression = other / self.expr if reverse else self.expr / other
        params = self.parameter_dependencies | other_dependencies
        value = expression.subs(params).evalf()
        return Expression(str(expression), "", value, params, expression)

    def __rtruediv__(self, other):
        return self.__truediv__(other, reverse=True)

    def __floordiv__(self, other, reverse=False):
        if type(other) in (Parameter, Expression):
            other_dependencies = other.parameter_dependencies
            other = other.expr
        else:
            other_dependencies = {}
        other = sp.sympify(other)
        expression = other // self.expr if reverse else self.expr // other
        params = self.parameter_dependencies | other_dependencies
        value = expression.subs(params).evalf()
        return Expression(str(expression), "", value, params, expression)

    def __rfloordiv__(self, other):
        return self.__floordiv__(other, reverse=True)

    def __mod__(self, other, reverse=False):
        if type(other) in (Parameter, Expression):
            other_dependencies = other.parameter_dependencies
            other = other.expr
        else:
            other_dependencies = {}
        other = sp.sympify(other)
        expression = other % self.expr if reverse else self.expr % other
        params = self.parameter_dependencies | other_dependencies
        value = expression.subs(params).evalf()
        return Expression(str(expression), "", value, params, expression)

    def __rmod__(self, other):
        return self.__mod__(other, reverse=True)

    def __pow__(self, other, reverse=False):
        if type(other) in (Parameter, Expression):
            other_dependencies = other.parameter_dependencies
            other = other.expr
        else:
            other_dependencies = {}
        other = sp.sympify(other)
        expression = other ** self.expr if reverse else self.expr ** other
        params = self.parameter_dependencies | other_dependencies
        value = expression.subs(params).evalf()
        return Expression(str(expression), "", value, params, expression)

    def __rpow__(self, other):
        return self.__pow__(other, reverse=True)

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)


class Point():
    def __init__(self, x, y, param_connections, scatter, func=lambda x, y: (x, y), color="r", size=10, plot=True):
        self.x = x
        self.y = y
        self.func = func
        self.transformed_coord = self.func(x, y)
        self.x_transform = self.transformed_coord[0]
        self.y_transform = self.transformed_coord[1]
        self.param_connections = param_connections
        self.scatter = scatter
        self.color = color
        self.size = size
        self.plot = plot

    def update_values(self, x_range, y_range, **kwargs):
        if not self.plot:
            return
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f'{key} not found.')
        x_transform, y_transform = self.func(self.x, self.y)
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.scatter.setData([x_transform], [y_transform])

    def save_data(self):
        return ("point", {"X": self.x, "Y": self.y, "func": self.func, "color": self.color, "size": self.size})


class Function():
    def __init__(self, x_func, y_func, params, param_connections, y_values, x_values, t_space, t_range, num_points,
                 color, width, curve, y_expr, x_expr, y_symbols, x_symbols, type, initial_parametric_resolution):
        self.x_func = x_func
        self.y_func = y_func
        self.params = {}
        for k, v in params.items():
            self.params = {k: str(v)}
        self.param_connections = param_connections
        self.y_values = y_values
        self.param_values = self.y_values
        self.x_values = x_values
        self.t_space = t_space
        self.t_range = t_range
        self.num_points = num_points
        self.color = color
        self.width = width
        self.curve = curve
        # Symbolic expressions
        self.y_expr = y_expr
        self.x_expr = x_expr
        self.y_symbols = y_symbols
        self.x_symbols = x_symbols
        self.type = type
        self.initial_parametric_resolution = int(initial_parametric_resolution)
        self.plot = plot
        self.scatter = scatter

    def update_values(self, x_range, y_range, **kwargs):
        if not self.plot:
            return
        for key, value in kwargs.items():
            if key in self.x_values:
                self.x_values[key] = value
            if key in self.y_values:
                self.y_values[key] = value

        # nsimplify turns floats into rational numbers, might change later
        self.y_num = self.y_expr.subs(self.y_values)
        self.x_num = self.x_expr.subs(self.x_values)
        try:
            y_inherent_domain = continuous_domain(self.y_num, self.y_symbols[0], S.Reals)
        except NotImplementedError:
            y_inherent_domain = sp.Interval(-oo, oo)
        try:
            x_inherent_domain = continuous_domain(self.x_num, self.x_symbols[0], S.Reals)
        except NotImplementedError:
            x_inherent_domain = sp.Interval(-oo, oo)

        base_intervals, base_points = decompose_numeric_set(self.base_domain)
        window_interval = sp.Interval(x_range[0], x_range[1])

        intervals, points = decompose_numeric_set(self.domain.numerical_set)

        mask_func = make_mask(intervals)  # Function for masking intervals

        self.base_domain = self.domain.expr.subs(self.domain.param_values)

        self.curve.setDownsampling(auto=False)
        self.curve.setClipToView(False)
        sliced_points = fast_intersection(y_inherent_domain, x_inherent_domain, window_interval, base_intervals, points)

        if self.type == "x_cartesian":
            # Visible y_interval:
            t_space = np.linspace(y_range[0], y_range[1], self.num_points)
            func = sp.lambdify(self.x_symbols, self.x_expr, modules=["numpy", "scipy"])
            mask = mask_func(t_space)
            y = np.where(mask, t_space, np.nan)
            x = func(y, **self.x_values)
            x = np.where(mask, x, np.nan)
            if len(sliced_points)>0:
                y_points = np.array(sliced_points)
                x_points = func(y_points, **self.x_values)
                self.scatter.setData(x_points, y_points)
            else:
                self.scatter.setData([np.nan], [np.nan])
            self.curve.setData(x.astype(float), y.astype(float))
            return

        if self.type == "y_cartesian":
            # Visible x-interval:
            t_space = np.linspace(x_range[0], x_range[1], self.num_points)
            #mask = self.domain.make_numpy_mask(self.domain.numerical_set)
            func = sp.lambdify(self.y_symbols, self.y_expr, modules=["numpy", "scipy"])
            mask = mask_func(t_space)
            x = np.where(mask, t_space, np.nan)
            y = func(x, **self.y_values)
            if len(sliced_points) > 0:
                x_points = np.array(sliced_points).astype(float)
                y_points = func(x_points, **self.y_values)
                self.scatter.setData(x_points, y_points)
            else:
                self.scatter.setData([np.nan], [np.nan])

            self.curve.setData(x.astype(float), y.astype(float))  # Change to float before plotting
            return
        else:
            t_space = np.linspace(self.t_range[0], self.t_range[1], self.initial_parametric_resolution)
            mask = mask_func(t_space)

        x = sp.lambdify(self.x_symbols, self.x_expr, modules=["numpy", "scipy"])(t_space, **self.x_values)
        x = np.where(mask, x, np.nan)
        y = sp.lambdify(self.y_symbols, self.y_expr, modules=["numpy", "scipy"])(t_space, **self.y_values)
        y = np.where(mask, y, np.nan)

        if self.type == "parametric":
            visible_mask = (
                    (x >= x_range[0]) & (x <= x_range[1]) &
                    (y >= y_range[0]) & (y <= y_range[1])
            )
            visible_part = np.mean(visible_mask)  # percentage of parametric curve inside the viewing window

            if not np.any(visible_mask):
                # Nothing visible
                self.curve.setData([], [])
                return
            visible_indices = np.where(visible_mask)[0]
            groups = np.split(visible_indices, np.where(np.diff(visible_indices) > 1)[0] + 1)
            x_segments, y_segments = [], []

            if len(groups[0]) > 0:  # If curve is inside viewing window
                for g in groups:
                    t_min, t_max = t_space[g[0]], t_space[g[-1]]  # t_range for group g
                    # Group range is the percentage of the visible part of the curve in group g
                    group_range = ((t_max - t_min) / (self.t_range[1] - self.t_range[0])) / visible_part

                    # Each group gets a minimum of 5 rendered points, otherwise the corresponding percentage of the
                    # total points
                    # The sum of num_visible_points for all groups roughly add up to self.num_points
                    num_visible_points = max(5, int(self.num_points * group_range))  # number of points per g

                    t_dense = np.linspace(t_min, t_max, num_visible_points)
                    x_dense = sp.lambdify(self.x_symbols, self.x_expr, modules=["numpy", "scipy"])(t_dense,
                                                                                                   **self.x_values)
                    y_dense = sp.lambdify(self.y_symbols, self.y_expr, modules=["numpy", "scipy"])(t_dense,
                                                                                                   **self.y_values)
                    x_segments.append(x_dense)
                    y_segments.append(y_dense)

                x_parts, y_parts = [], []
                for x_seg, y_seg in zip(x_segments, y_segments):
                    x_parts.append(x_seg)
                    y_parts.append(y_seg)

                    # NaN is inserted at all points where a group ends to indicate discontinuity
                    x_parts.append(np.array([np.nan]))
                    y_parts.append(np.array([np.nan]))

                x = np.concatenate(x_parts)
                y = np.concatenate(y_parts)

                self.curve.setData(x.astype(float), y.astype(float))



    def save_data(self):
        return ("function", {
                "x_func": self.x_func,
                "y_func": self.y_func,
                "params": self.params,
                "t_range": self.t_range,
                "num_points": self.num_points,
                "color": self.color,
                "width": self.width})


class Vector:
    def __init__(self, line, start, vec, params, param_values, color, width):
        self.line = line
        self.start = start
        self.vec = vec
        for k, v in params.items():
            self.params = {k: str(v)}
        self.color = color
        self.width = width
        self.param_values = param_values

    def update_values(self, **kwargs):
        for key, value in kwargs.items():
            self.param_values[key] = value

        self.line.setData(x, y)

    def save_data(self):
        return ("vector", {
                "vec": self.vec,
                "params": self.params,
                "start": self.start,
                "color": self.color,
                "width": self.width})


class Grid():
    def __init__(self, x_lines, y_lines, x_transformed, y_transformed, x_vals, y_vals, grid_plot, params,
                 param_connections, x_func, y_func, x_expr, y_expr, x_symbols, y_symbols, x_space, y_space, x_range,
                 y_range, num_lines, num_points, color, width, alpha, plot=True):
        self.x_lines = x_lines
        self.y_lines = y_lines
        self.x_transformed = x_transformed
        self.y_transformed = y_transformed
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.param_values = dict(y_vals, **x_vals)
        self.grid_plot = grid_plot
        self.params = params
        self.x_func = x_func
        self.y_func = y_func
        self.x_expr = x_expr
        self.y_expr = y_expr
        self.x_symbols = x_symbols
        self.y_symbols = y_symbols
        self.x_space = x_space
        self.y_space = y_space
        self.param_connections = param_connections
        self.x_range = x_range
        self.y_range = y_range
        self.num_lines = num_lines
        self.num_points = num_points
        self.color = color
        self.width = width
        self.alpha = alpha
        self.plot = plot

    def update_values(self, x_range, y_range, **kwargs):
        if not self.plot:
            return
        for line in self.x_transformed:
            self.grid_plot.removeItem(line)
        for line in self.y_transformed:
            self.grid_plot.removeItem(line)

        for key, value in kwargs.items():
            if key in self.x_vals:
                self.x_vals[key] = value
            if key in self.y_vals:
                self.y_vals[key] = value
            self.param_values[key] = value

        for line in self.x_lines:
            line = np.repeat(line, self.num_points)
            y = self.x_func(line, self.y_space, **self.x_vals)
            x = self.y_func(line, self.y_space, **self.y_vals)
            x_transform = self.grid_plot.plot(x, y, pen=pg.mkPen(self.color, width=self.width))
            x_transform.setAlpha(self.alpha, False)
            self.x_transformed.append(x_transform)

        for line in self.y_lines:
            line = np.repeat(line, self.num_points)
            y = self.x_func(self.x_space, line, **self.x_vals)
            x = self.y_func(self.x_space, line, **self.y_vals)
            y_transform = self.grid_plot.plot(x, y, pen=pg.mkPen(self.color, width=self.width))
            y_transform.setAlpha(self.alpha, False)
            self.y_transformed.append(y_transform)


    def save_data(self):
        return ("grid", {
                "x_range": self.x_range,
                "y_range": self.y_range,
                "num_lines": self.num_lines,
                "x_func": self.x_func,
                "y_func": self.y_func,
                "params": self.params,
                "num_points": self.num_points,
                "color": self.color,
                "width": self.width,
                "alpha": self.alpha})


def decompose_numeric_set(S, universal=None):
    """
    Recursively decompose a numeric 1D SymPy set into intervals and points,
    handling Interval, FiniteSet, Range, Union, Intersection, and Complement.

    Args:
        S: SymPy set
        universal: optional Interval defining bounds for complements

    Returns:
        intervals: list of Interval objects
        points: list of numeric points
    """
    intervals = []
    points = []

    # Base cases
    if S.is_empty:
        return [], []

    elif isinstance(S, sp.Interval):
        intervals.append(S)
        return intervals, points

    elif isinstance(S, sp.FiniteSet):
        points.extend([float(p) for p in S])
        return intervals, points

    elif isinstance(S, sp.Range):
        points.extend(list(S))
        return intervals, points

    elif isinstance(S, sp.Union):
        for part in S.args:
            ivals, pts = decompose_numeric_set(part, universal)
            intervals.extend(ivals)
            points.extend(pts)
        return intervals, points

    elif isinstance(S, sp.Intersection):
        # recursively intersect all arguments
        if not S.args:
            return [], []
        ivals, pts = decompose_numeric_set(S.args[0], universal)
        for part in S.args[1:]:
            ivals2, pts2 = decompose_numeric_set(part, universal)
            # intersect intervals
            if ivals and ivals2:
                new_intervals = []
                for I1 in ivals:
                    for I2 in ivals2:
                        inter = I1.intersect(I2)
                        if not inter.is_empty:
                            new_intervals.append(inter)
                ivals = new_intervals
            else:
                ivals = ivals or ivals2  # if one empty, take the other
            # intersect points
            pts = list(set(pts) & set(pts2))
        return ivals, pts

    elif isinstance(S, sp.Complement):
        # Decompose complement: A - B = A ∩ (universal - B)
        A, B = S.args
        # define universal if not given
        if universal is None:
            # numeric universal can be the smallest/largest interval from A
            # fallback: [-1e9, 1e9]
            universal = sp.Interval(-1e9, 1e9)
        # compute B complement relative to universal
        B_complement = universal - B
        return decompose_numeric_set(A & B_complement, universal)

    else:
        # fallback: treat as single numeric point
        try:
            points.append(float(S))
            return intervals, points
        except:
            raise TypeError(f"Cannot handle set: {S}")


def make_mask(intervals):

    def mask(xs):
        # vectorized boolean array
        m = np.zeros(xs.shape, dtype=bool)

        # apply interval masks
        for I in intervals:
            if I.left_open:
                left_cond = xs > float(I.start)
            else:
                left_cond = xs >= float(I.start)

            if I.right_open:
                right_cond = xs < float(I.end)
            else:
                right_cond = xs <= float(I.end)

            m |= (left_cond & right_cond)

        return m

    return mask


def fast_intersection(y_inherent_domain, x_inherent_domain, window_interval, base_intervals, points):
    # (A ∩ B) \ C = all points in inherent domains and window interval minus all points in the base intervals
    if not base_intervals:
        base_intervals = [sp.EmptySet]
    interval_intersection = (sp.Intersection(y_inherent_domain, x_inherent_domain, window_interval) -
                             sp.Intersection(*base_intervals))
    # Normalize to a list of intervals or empty
    #print(sp.Intersection(y_inherent_domain, x_inherent_domain, window_interval))
    if interval_intersection.is_empty:
        parts = []
    elif isinstance(interval_intersection, sp.Interval):
        parts = [interval_intersection]
    elif isinstance(interval_intersection, sp.Union):
        parts = list(interval_intersection.args)
    else:
        parts = [interval_intersection]

    points.sort()  # sort points

    sliced_points = []
    print(parts)
    for interval in parts:
        start, end = interval.start, interval.end
        # find left index
        if interval.left_open:
            left = bisect.bisect_right(points, start)
        else:
            left = bisect.bisect_left(points, start)

        # find right index
        if interval.right_open:
            right = bisect.bisect_left(points, end)
        else:
            right = bisect.bisect_right(points, end)

        # select the points inside current interval
        sliced_points.extend(points[left:right])
    return sliced_points


    #for part in interval_intersection:
    #    print(f"Part {part} {type(part)}")


class NumSet:
    def __init__(self, sp_set, params, param_values, parameter_connections, plot):
        self.expr = sp_set  # Symbolic expression (sp.Set)
        self.numerical_set = sp_set.subs(param_values)  # Numerical expression (sp.Set)
        self.params = params
        self.param_values = param_values
        self.param_connections = parameter_connections
        self.plot = plot

    def update_values(self, x_range, y_range, **kwargs):
        if not self.plot:
            return
        for key, value in kwargs.items():
            self.param_values[key] = value

        self.numerical_set = self.expr.subs(self.param_values)


class GenericSampler:
    def __init__(self, var, pdf, domain):
        """
        var     = SymPy symbol (x)
        pdf     = SymPy expression for pdf(x) or pmf(x)
        domain  = Interval(a, b) for continuous, or FiniteSet for discrete
        """
        self.var = var
        self.pdf = pdf
        self.domain = domain

        # Convert pdf to numerical function
        self.f = sp.lambdify(var, pdf, 'math')

        # Detect discrete or continuous
        self.is_discrete = isinstance(domain, sp.FiniteSet)

        if self.is_discrete:
            # Precompute discrete probabilities
            points = list(domain)
            probs = [float(pdf.subs(var, p)) for p in points]
            Z = sum(probs)
            self.points = points
            self.probs = [p/Z for p in probs]
        else:
            # For rejection sampling on continuous bounded interval
            if domain.is_Interval and domain.start.is_finite and domain.end.is_finite:
                # Find max of pdf on domain (symbolically)
                x = var
                d = sp.diff(pdf, x)
                crit = [domain.start, domain.end]  # endpoints
                crit += list(sp.solve(sp.Eq(d, 0), x))  # interior critical points
                # Filter valid points
                crit = [c for c in crit if c.is_real and c >= domain.start and c <= domain.end]
                # Evaluate pdf
                self.M = max(float(pdf.subs(x, c)) for c in crit)
            else:
                raise NotImplementedError("Unbounded or symbolic domains need special handling")

    def sample(self, n=1):
        if self.is_discrete:
            # Use random.choices for PMFs
            return random.choices(self.points, weights=self.probs, k=n)

        # Continuous: rejection sampling
        a, b = float(self.domain.start), float(self.domain.end)
        samples = []
        while len(samples) < n:
            x_try = random.uniform(a, b)
            y_try = random.uniform(0, self.M)
            if y_try < self.f(x_try):
                samples.append(x_try)
        return samples


class Graph(QtWidgets.QWidget):
    def __init__(self, name=None, xmin=-10, xmax=10, ymin=-10, ymax=10, bg_color="w", left_color="k", bottom_color="k", x_axis=True, y_axis=True, axis_color="k", axis_width=2):
        super().__init__()
        self.name = name
        self.parent = None  # GraphWindow object, gets added when Graph is added to GraphWindow

        # Plot widget
        self.plotWidget = None

        # Basic config
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.bg_color = bg_color
        self.left_color = left_color
        self.bottom_color = bottom_color
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.axis_color = axis_color
        self.axis_width = axis_width
        self.x_view_range = (self.xmin, self.xmax)
        self.y_view_range = (self.ymin, self.ymax)

        # References
        self.parameters = {}
        self.parameter_connections = {}
        self.expressions = {}
        self.functions = []
        self.points = []
        self.objects = []
        self.plot_objects = []

    def set_plot_item(self, plot_widget):
        """Assign a PlotItem. Must be called once the Graph is added to a GraphWindow."""
        # Configure plot
        self.plotWidget = plot_widget
        self.plotWidget.setBackground(self.bg_color)
        self.plotWidget.showGrid(x=self.x_axis, y=self.y_axis, alpha=0.3)
        self.plotWidget.setMouseEnabled(x=self.x_axis, y=self.y_axis)
        self.plotWidget.setRange(xRange=[self.xmin, self.xmax], yRange=[self.ymin, self.ymax])
        self.plotWidget.getAxis("left").setPen(self.left_color)
        self.plotWidget.getAxis("bottom").setPen(self.bottom_color)
        self.plotWidget.setAspectLocked(False)
        self.plotWidget.getViewBox().sigRangeChanged.connect(self._update_range)

        # Main axes
        axis_pen = pg.mkPen(self.axis_color, width=self.axis_width)
        if self.x_axis:
            self.plotWidget.addItem(pg.InfiniteLine(angle=0, pen=axis_pen))  # X-axis
        if self.y_axis:
            self.plotWidget.addItem(pg.InfiniteLine(angle=90, pen=axis_pen))  # Y-axis

    def _get_param_values(self, params):
        """Helper function for getting parameter values from list of parameters"""
        return {key: value[1] for key, value in self.parent.single_values.items() if key in params}

    def _update_range(self):
        vb = self.plotWidget.getViewBox()
        x_range, y_range = vb.viewRange()
        self.x_view_range = x_range
        self.y_view_range = y_range

        for obj in self.plot_objects:
            param_values = obj.param_values
            #x_range = (np.max(x_range[0], func.x_range[0]), np.max(x_range[1], func.x_range[1]))
            obj.update_values(x_range, y_range, **param_values)

    def save_config(self, filename, filepath="saves"):
        data = []
        for obj in self.objects:
            data.append(obj.save_data())
        path = Path(f"{filepath}/{filename}.pkl")
        with path.open("wb") as f:
            dill.dump(data, f, protocol=dill.HIGHEST_PROTOCOL)

    def load_config(self, filename, filepath="saves"):
        object_operations_map = {
            "parameter": self.add_parameter,
            "expression": self.add_expression,
            "point": self.add_point,
            "function": self.add_function,
            "grid": self.add_grid,
            "vector": self.add_vector
        }
        path = Path(f"{filepath}/{filename}.pkl")
        with path.open("rb") as f:
            loaded_data = dill.load(f)
        for obj in loaded_data:
            type, kwargs = obj
            object_operations_map[type](**kwargs)

    def add_parameter(self, name, min_val, max_val, init_val, steps=None):
        if name in self.parameters:
            raise NameError("Parameter {} already exists".format(name))
        if steps is None:
            steps = (max_val - min_val)*10
        step_size = (max_val - min_val)/steps

        layout = self.slider_window.main_layout

        container = QtWidgets.QWidget()
        hbox = QtWidgets.QHBoxLayout(container)

        label = QtWidgets.QLabel(f"{name}: {init_val}")
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setFixedWidth(45)
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(int(min_val / step_size), int(max_val / step_size))
        slider.setValue(int(init_val / step_size))
        slider.setSingleStep(1)

        hbox.addWidget(label)
        hbox.addWidget(slider)
        layout.addWidget(container)

        param = Parameter(name, label, slider, min_val, max_val, init_val, steps)
        slider.valueChanged.connect(lambda _: self._update_param(param))
        self.parameters[name] = param
        self.parameter_connections[name] = []
        return param

    def _format_param_dict(self, dict):
        """Helper function to reformat parameter dictionaries where the parameters are entered as strings"""
        for key, value in dict.items():
            if isinstance(value, str):
                dict[key] = self.parameters[value]
        return dict

    def _get_single_values(self, var_list, params, skip_first=0):
        pvals = {}
        pdict = {}
        for key in list(var_list)[skip_first:]:
            key = str(key)  # Key could be sp symbol, in that case convert to str.
            try:
                pvals[key] = params[key].value
                pdict[key] = params[key]
            except KeyError:  # If param not specified
                if key in self.parent.single_values:
                    type, value = self.parent.single_values[key]
                    pvals[key] = value
                    pdict[key] = self.parent.parameters[key]
                    warnings.warn(f"Variable {key} not specified, assumes added {type} {key}.")
                else:
                    raise KeyError(f"Variable {key} not defined.")
        return pvals, pdict

    def add_expression(self, expression, expression_name: str):
        param_values = {p.expr: p.value for p in self.parameters.values()}
        expression = expression.expr
        parameter_dependencies = expression.free_symbols
        expression_evaluation = expression.subs(param_values).evalf()
        expression = Expression(expression_name, "", expression_evaluation, parameter_dependencies, expression)
        expression.expression_name = expression_name


        self.expressions[expression_name] = expression
        self.objects.append(expression)
        self.parent.single_values[expression_name] = ("expression" ,expression_evaluation)
        self.expression_window.params = self.parameters

        self.expression_window.add_expression(expression)
        dependencies = expression.parameter_dependencies
        for dependency in self.parameter_connections.keys():
            if sp.sympify(dependency) in dependencies:
                self.parameter_connections[dependency].append(expression)

        return expression

    def add_point(self, X, Y, func=lambda x, y: (x, y), color="r", size=10):
        if not isinstance(X, (int, float)) or not isinstance(Y, (int, float)):
            if str(X) not in self.parameters.keys() or str(Y) not in self.parameters.keys():
                warnings.warn("Points defined from expressions are not supported. The initial value will be correct "
                              "\nbut any updates to the expression through the parameters will be incorrect. Instead, "
                              "\nenter pure parameters as inputs and define the transformation in the func argument.")
        param_values = {p.expr: p.value for p in self.parent.parameters.values()}
        scatter = pg.ScatterPlotItem(size=size, brush=pg.mkBrush(color))
        self.plotWidget.addItem(scatter)
        X = X.expr if type(X) in (Parameter, Expression) else X
        Y = Y.expr if type(Y) in (Parameter, Expression) else Y
        X = sp.sympify(X)
        Y = sp.sympify(Y)
        params = list(X.free_symbols) + list(Y.free_symbols)
        x_eval = X.subs(param_values).evalf()
        y_eval = Y.subs(param_values).evalf()
        parameter_connections = {'x': [symbol.name for symbol in X.free_symbols], 'y': [symbol.name for symbol in Y.free_symbols]}
        point = Point(X, Y, param_connections=parameter_connections, scatter=scatter, func=func,
                      color=color, size=size)
        for param in set(params):
            self.parent.parameter_connections[str(param)].append(point)
        self.points.append(point)
        self.objects.append(point)
        self.plot_objects.append(point)
        self.parent.obj_graph_connections[point] = self
        x, y = func(x_eval, y_eval)
        scatter.setData([x], [y])

    def add_function(self, y_func, x_func=None, params=None, domain=None, t_range=(-100, 100), num_points=1000,
                     initial_parametric_resolution=int(10000), main_variables=1, plot=True, color="b", width=2,
                     point_size=5):
        """
        Default is cartesian y-function (f(x)). By adding a x_func, it is also possible to create a parametric function.
        :param y_func: function of x
        :param x_func: function of y
        :param params: dictionary of parameters
        :param t_range: domain for cartesian functions, range of t for parametric functions
        :param num_points: number of total rendered points at each state
        :param initial_parametric_resolution: ?
        :param main_variables: first n variables will be treated as main variables
        :param color: color
        :param width: width of plot
        :return: Function object
        """

        if params is None:
            params = {}
        if x_func is None:
            x_func = lambda t: t

        params = self._format_param_dict(params)
        curve = self.plotWidget.plot(pen=pg.mkPen(color=color, width=width))
        curve.setClipToView(True)
        t = np.linspace(t_range[0], t_range[1], num_points)
        parameter_connections = {}

        y_vars = inspect.signature(y_func)
        x_vars = inspect.signature(x_func)

        y_symbols = sp.symbols(list(y_vars.parameters))
        x_symbols = sp.symbols(list(x_vars.parameters))

        y_vals, y_params = self._get_single_values(y_vars.parameters, params, skip_first=main_variables)
        x_vals, x_params = self._get_single_values(x_vars.parameters, params, skip_first=main_variables)

        y_expr = y_func(*y_symbols)
        x_expr = x_func(*x_symbols)

        params = dict(y_params, **x_params)

        for k, v in y_params.items():
            parameter_connections[k] = [v.name]
        for k, v in x_params.items():
            parameter_connections[k] = [v.name]

        x = sp.lambdify(x_symbols, x_expr, modules=["numpy", "scipy"])(t, **x_vals) * np.ones_like(t, dtype=float)
        y = sp.lambdify(y_symbols, y_expr, modules=["numpy", "scipy"])(t, **y_vals) * np.ones_like(t, dtype=float)

        if np.array_equal(y, t):
            func_type = "x_cartesian"  # x = f(y)
        elif np.array_equal(x, t):
            func_type = "y_cartesian"  # y = f(x)
        else:
            func_type = "parametric"  # x = f(t), y = g(t)


        function = Function(x_func, y_func, params, parameter_connections, y_vals, x_vals, t, t_range, num_points,
                            color, width, curve, y_expr, x_expr, y_symbols, x_symbols, func_type,
                            initial_parametric_resolution)

        for param in dict(y_vals, **x_vals):
            param_name = params[param].name
            self.parent.parameter_connections[param_name].append(function)

        curve.setData(x, y)
        self.objects.append(function)
        self.functions.append(function)
        self.plot_objects.append(function)
        self.parent.obj_graph_connections[function] = self
        return function

    def add_grid(self, x_range=(-10, 10), y_range=(-10, 10), num_lines=21, x_func=lambda x, y: x, y_func=lambda x, y: y, params=None, num_points=1000, color="grey", width=2, alpha=0.9):
        if params is None:
            params = {}

        y_vars = inspect.signature(y_func)
        x_vars = inspect.signature(x_func)

        y_symbols = sp.symbols(list(y_vars.parameters))
        x_symbols = sp.symbols(list(x_vars.parameters))

        y_vals, y_params = self._get_single_values(y_vars.parameters, params, skip_first=2)
        x_vals, x_params = self._get_single_values(x_vars.parameters, params, skip_first=2)

        y_expr = y_func(*y_symbols)
        x_expr = x_func(*x_symbols)

        params = dict(y_params, **x_params)

        # Meshgrid data
        x_lines = np.linspace(x_range[0], x_range[1], num_lines)
        y_lines = np.linspace(y_range[0], y_range[1], num_lines)
        x_space = np.linspace(x_range[0], x_range[1], num_points)
        y_space = np.linspace(y_range[0], y_range[1], num_points)

        x_func = sp.lambdify(x_symbols, x_expr, modules=["numpy", "scipy"])
        y_func = sp.lambdify(y_symbols, y_expr, modules=["numpy", "scipy"])

        x_transform_lines = []
        y_transform_lines = []
        grid_plot = self.plotWidget

        for line in x_lines:
            line = np.repeat(line, num_points)
            y = x_func(line, y_space, **x_vals)
            x = y_func(line, y_space, **y_vals)
            x_transform = grid_plot.plot(x, y, pen=pg.mkPen(color, width=width))
            x_transform.setAlpha(alpha, False)
            x_transform_lines.append(x_transform)

        for line in y_lines:
            line = np.repeat(line, num_points)
            y = x_func(x_space, line, **x_vals)
            x = y_func(x_space, line, **y_vals)
            y_transform = grid_plot.plot(x, y, pen=pg.mkPen(color, width=width))
            y_transform.setAlpha(alpha, False)
            y_transform_lines.append(y_transform)

        # Add to object parameter connections
        parameter_connections = {}
        for k, v in params.items():
            parameter_connections[k] = [v.name]

        grid = Grid(x_lines, y_lines, x_transform_lines, y_transform_lines, x_vals, y_vals, grid_plot, params,
                    parameter_connections, x_func, y_func, x_expr, y_expr, x_symbols, y_symbols, x_space, y_space,
                    x_range, y_range, num_lines, num_points, color, width, alpha)

        for param in dict(y_vals, **x_vals):
            param_name = params[param].name
            self.parent.parameter_connections[param_name].append(grid)

        self.objects.append(grid)
        self.plot_objects.append(grid)
        self.parent.obj_graph_connections[grid] = self

        return grid

    def add_vector(self, vec, params=None, start=(0, 0), color="b", width=2):
        """Work in progress"""
        if params is None:
            params = {}
        line = self.plotWidget.plot(pen=pg.mkPen(color), width=width)
        parameter_connections = {}
        pvals = {key: param.value for key, param in params.items()}
        for k, v in params.items():
            parameter_connections[k] = [v.name]

        vector = Vector(line, start, vec, params, pvals, color, width)
        for param in params.values():
            self.parameter_connections[param.name].append(vector)

        line.setClipToView(True)
        line.setData([start[0], start[0] + vec[0]], [start[1], start[1] + vec[1]])

        arrow = pg.ArrowItem(
            pos=[start[0] + vec[0], start[1] + vec[1]],
            angle=np.degrees(np.arctan2(vec[1], -vec[0])),
            brush=color,
            headLen=15
        )

        self.objects.append(arrow)

        #self.plotWidget.addItem(arrow)







app = QtWidgets.QApplication(sys.argv)
viewer = Graph(xmin=-1, xmax=1, ymin=-1, ymax=1)
viewer.resize(900, 900)

a = viewer.add_parameter("a", 0, 1, 1, 50)
b = viewer.add_parameter("b", 0, 100, 15, 100)
e = viewer.add_parameter("e", 0, 10, 5, 100)
f = viewer.add_parameter("f", 0, 10, 5, 100)
R = 6
r = 1
d = 1
#viewer.add_grid(transform_func=lambda x, y, c, d, e, f: (e+c*sp.sin(x), f+d*c*sp.sin(y)), params={"c": a, "d": b})
viewer.add_function(lambda x, b: (b-r)*cos(x) + d*cos(((b-r)/r)*x), x_func=lambda x, b: (b-r)*sin(x) + d*sin(((b-r)/r)*x), num_points=1000, t_range=(0, 2*np.pi), width=1)
f = viewer.add_function(lambda x, a: sin(a*x), t_range=(-10, 10), num_points=100)
f = viewer.add_function(y_func=lambda x, a: x, x_func=lambda x, a: cos(a*x), t_range=(-10, 10), num_points=100)


#viewer.save_config("test")
#viewer.load_config("test")

viewer.show()
sys.exit(app.exec_())