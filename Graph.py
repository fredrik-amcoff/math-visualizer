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
from sympy import exp, sin, cos, oo
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
from itertools import product


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
    def __init__(self, name, label, slider, min_val, max_val, value, step):
        self.name = name
        self.label = label
        self.slider = slider
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.step = step
        self.parameter_dependencies = {self.name: self.value}
        self.expr = sp.symbols(name)

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
    def __init__(self, x, y, param_connections, scatter, func=lambda x, y: (x, y), color="r", size=10):
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

    def update_values(self, x_range, y_range, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f'{key} not found.')
        x_transform, y_transform = self.func(self.x, self.y)
        self.x_transform = x_transform
        self.y_transform = y_transform
        print(x_transform, y_transform)
        print(self.x, self.y)
        self.scatter.setData([x_transform], [y_transform])

    def save_data(self):
        return ("point", {"X": self.x, "Y": self.y, "func": self.func, "color": self.color, "size": self.size})


class Function():
    def __init__(self, x_func, y_func, params, param_connections, param_values, t_space, t_range, num_points, color, width, curve, expr):
        self.x_func = x_func
        self.y_func = y_func
        self.params = {}
        for k, v in params.items():
            self.params = {k: str(v)}
        self.param_connections = param_connections
        self.param_values = param_values
        self.t_space = t_space
        self.t_range = t_range
        self.num_points = num_points
        self.color = color
        self.width = width
        self.curve = curve
        self.expr = expr

    def update_values(self, **kwargs):
        for key, value in kwargs.items():
            self.param_values[key] = value
        x = self.x_func(self.t_space, **self.param_values)
        y = self.y_func(self.t_space, **self.param_values)
        self.curve.setData(x, y)

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
    def __init__(self, x, y, lines, params, param_values, transform_func, param_connections, grid_plot, x_range, y_range, num_points, color, width):
        self.x = x
        self.y = y
        self.lines = lines
        for k, v in params.items():
            self.params = {k: str(v)}
        self.param_values = param_values
        self.transform_func = transform_func
        self.param_connections = param_connections
        self.grid_plot = grid_plot
        self.x_range = x_range
        self.y_range = y_range
        self.num_points = num_points
        self.color = color
        self.width = width

    def update_values(self, **kwargs):
        for line in self.lines:
            self.grid_plot.removeItem(line)
        self.lines = []

        for key, value in kwargs.items():
            self.param_values[key] = value
        x_transform, y_transform = self.transform_func(self.x, self.y, *self.param_values.values())
        for i in range(x_transform.shape[0]):
            line = self.grid_plot.plot(x_transform[i, :], y_transform[i, :], pen=pg.mkPen(self.color))
            self.lines.append(line)

        for j in range(y_transform.shape[1]):
            line = self.grid_plot.plot(x_transform[:, j], y_transform[:, j], pen=pg.mkPen(self.color))
            self.lines.append(line)

    def save_data(self):
        return ("grid", {
                "x_range": self.x_range,
                "y_range": self.y_range,
                "num_points": self.num_points,
                "transform_func": self.transform_func,
                "params": self.params,
                "color": self.color,
                "width": self.width})


class Graph(QtWidgets.QWidget):
    def __init__(self, xmin=-10, xmax=10, ymin=-10, ymax=10):
        super().__init__()
        self.setWindowTitle("Graph Plot")

        layout = QtWidgets.QVBoxLayout(self)

        # Plot widget
        self.plotWidget = pg.PlotWidget()
        layout.addWidget(self.plotWidget)

        # Slider panel
        self.slider_window = SliderWindow()
        self.slider_window.show()
        self.slider_panel = self.slider_window.layout

        # Expression window
        self.expression_window = ExpressionWindow()
        self.expression_window.show()
        self.expression_panel = self.expression_window.layout
        self.hbox = None

        # Configure plot
        self.plotWidget.setBackground("k")
        self.plotWidget.showGrid(x=True, y=True, alpha=0.3)
        self.plotWidget.setMouseEnabled(x=True, y=True)
        self.plotWidget.setRange(xRange=[xmin, xmax], yRange=[ymin, ymax])
        self.plotWidget.getAxis("left").setPen("w")
        self.plotWidget.getAxis("bottom").setPen("w")
        self.plotWidget.setAspectLocked(False)

        # Main axes
        axis_pen = pg.mkPen("w", width=2)
        self.plotWidget.addItem(pg.InfiniteLine(angle=0, pen=axis_pen))  # X-axis
        self.plotWidget.addItem(pg.InfiniteLine(angle=90, pen=axis_pen))  # Y-axis

        self.unit_square_orig = self.plotWidget.plot(pen=pg.mkPen("w", width=2))
        self.unit_square_trans = self.plotWidget.plot(pen=pg.mkPen("orange", width=2))

        # References
        self.parameters = {}
        self.parameter_values = {}
        self.parameter_connections = {}
        self.expressions = {}
        self.points = []
        self.objects = []

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
            print(type, kwargs)
            object_operations_map[type](**kwargs)

    def add_parameter(self, name, min_val, max_val, init_val, step=1.0):
        if name in self.parameters:
            raise NameError("Parameter {} already exists".format(name))
        layout = self.slider_window.main_layout

        container = QtWidgets.QWidget()
        hbox = QtWidgets.QHBoxLayout(container)

        label = QtWidgets.QLabel(f"{name}: {init_val}")
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(int(min_val / step), int(max_val / step))
        slider.setValue(int(init_val / step))
        slider.setSingleStep(1)

        hbox.addWidget(label)
        hbox.addWidget(slider)
        layout.addWidget(container)

        param = Parameter(name, label, slider, min_val, max_val, init_val, step)
        slider.valueChanged.connect(lambda _: self._update_param(param))
        self.parameters[name] = param
        self.parameter_values[name] = init_val
        self.parameter_connections[name] = []
        self.objects.append(param)
        return param

    def _update_param(self, param):
        param_vals = param.__dict__
        decimals = max(0, math.ceil(-np.log10(param_vals["step"])) if param_vals["step"] < 1 else 0)
        param_vals["value"] = round(param_vals["slider"].value() * param_vals["step"], decimals)
        param_vals["label"].setText(f"{param_vals['name']}: {param_vals['value']}")
        self.parameter_values[param_vals['name']] = param_vals['value']
        param.update_values(param_vals["value"])
        for obj in self.parameter_connections[param_vals['name']]:
            if isinstance(obj, Expression):
                updated_value = obj.update_values(self.parameters)
                obj._label.setText(f"{obj.name} = {obj.name}: {updated_value}")
                self.expression_window.update_values(obj)
            else:
                param_updates = {}
                for key, value in obj.param_connections.items():
                    if param_vals["name"] in value:
                        param_updates[key] = param_vals["value"]
                obj.update_values(**param_updates)
            #print(param_updates)
        #print(self.points[0].x, self.points[0].y)




        # ADD REDRAW

    def _format_param_dict(self, dict):
        """Helper function to reformat parameter dictionaries where the parameters are entered as strings"""
        for key, value in dict.items():
            if isinstance(value, str):
                dict[key] = self.parameters[value]
        return dict

    def add_expression(self, expression, expression_name: str):
        param_values = {p.expr: p.value for p in self.parameters.values()}
        expression = expression.expr
        parameter_dependencies = expression.free_symbols
        expression_evaluation = expression.subs(param_values).evalf()
        expression = Expression(expression_name, "", expression_evaluation, parameter_dependencies, expression)
        expression.expression_name = expression_name


        self.expressions[expression_name] = expression
        self.objects.append(expression)
        self.expression_window.params = self.parameters

        self.expression_window.add_expression(expression)
        dependencies = expression.parameter_dependencies
        for dependency in self.parameter_connections.keys():
            if sp.sympify(dependency) in dependencies:
                self.parameter_connections[dependency].append(expression)

    def add_point(self, X, Y, func=lambda x, y: (x, y), color="r", size=10):
        if not isinstance(X, (int, float)) or not isinstance(Y, (int, float)):
            if str(X) not in self.parameters.keys() or str(Y) not in self.parameters.keys():
                warnings.warn("Points defined from expressions are not supported. The initial value will be correct "
                              "\nbut any updates to the expression through the parameters will be incorrect. Instead, "
                              "\nenter pure parameters as inputs and define the transformation in the func argument.")
        param_values = {p.expr: p.value for p in self.parameters.values()}
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
        point = Point(x_eval, y_eval, param_connections=parameter_connections, scatter=scatter, func=func,
                      color=color, size=size)
        for param in set(params):
            self.parameter_connections[str(param)].append(point)
        self.points.append(point)
        self.objects.append(point)
        x, y = func(x_eval, y_eval)
        scatter.setData([x], [y])

    def add_function(self, y_func, x_func=lambda t: t, params=None, t_range=(-10, 10), num_points=1000, color="b", width=2):
        if params is None:
            params = {}
        params = self._format_param_dict(params)
        curve = self.plotWidget.plot(pen=pg.mkPen(color=color, width=width))
        curve.setClipToView(True)
        t = np.linspace(t_range[0], t_range[1], num_points)
        parameter_connections = {}
        pvals = {key: param.value for key, param in params.items()}
        for k, v in params.items():
            parameter_connections[k] = [v.name]

        ### ÄNDRA None SENARE
        function = Function(x_func, y_func, params, parameter_connections, pvals, t, t_range, num_points, color, width, curve, None)
        for param in params.values():
            self.parameter_connections[param.name].append(function)
        x = x_func(t, **pvals)
        y = y_func(t, **pvals)
        curve.setData(x, y)
        self.objects.append(function)
        return function

    def add_grid(self, x_range=(-100, 100), y_range=(-100, 100), num_points=201, transform_func=lambda x, y: (x, y), params=None, color="grey", width=5):
        if params is None:
            params = {}
        params = self._format_param_dict(params)
        param_values = {p.expr: p.value for p in self.parameters.values()}
        x_space = np.linspace(x_range[0], x_range[1], num_points)
        y_space = np.linspace(y_range[0], y_range[1], num_points)
        param_evals = {param_key: param_value.value for param_key, param_value in params.items()}
        X, Y = np.meshgrid(x_space, y_space)
        x_transform, y_transform = transform_func(X, Y, *param_evals.values())
        grid_lines = []

        parameter_connections = {}
        for k, v in params.items():
            parameter_connections[k] = [v.name]

        grid_plot = self.plotWidget

        for i in range(x_transform.shape[0]):
            line = grid_plot.plot(x_transform[i, :], y_transform[i, :], pen=pg.mkPen(color))
            grid_lines.append(line)

        for j in range(y_transform.shape[1]):
            line = grid_plot.plot(x_transform[:, j], y_transform[:, j], pen=pg.mkPen(color))
            grid_lines.append(line)

        grid = Grid(X, Y, grid_lines, params, param_evals, transform_func, parameter_connections, grid_plot, x_range, y_range, num_points, color, width)

        for param in params.values():
            self.parameter_connections[param.name].append(grid)

        self.objects.append(grid)

        return grid

    def add_vector(self, vec, params=None, start=(0, 0), color="b", width=2):
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
viewer = Graph()
viewer.resize(900, 700)
a = viewer.add_parameter("a", -10, 10, 5, step=0.1)
b = viewer.add_parameter("b", -10, 10, 6, step=1)
c = viewer.add_parameter("c", -10, 10, 7, step=1)
d = viewer.add_parameter("d", 0, 1, 0, step=0.01)
e = viewer.add_parameter("e", 0, 1, 0.4, step=0.05)


#u = a + b + (a+2) + 2**(a+b) - b/2

def normal(x, mu, sigma):
    return (1/(sigma*(np.sqrt(2*np.pi))))*np.exp((-1/2)*((x-mu)/sigma)**2)

u = b*(a + 1 + b)/a + sp.exp(a.expr)

viewer.add_expression(u, "u")
viewer.add_grid(transform_func=lambda x, y, d, e: ((1-e)*x + e*(np.cos(x) - np.sin(y)), (1-e)*y + e*(np.sin(x) + np.cos(y))), params={"d": d, "e": e}, color="grey", x_range=(-10, 10), y_range=(-10,10), num_points=50)
#
#linspace = np.linspace(-10, 10, 100)
#
#viewer.add_point(a, b, lambda x, y: (x*sp.sin(y), y*x), color='b')
#viewer.add_point(5, 5, lambda x, y: (x*sp.sin(y), y*x), color='b')
#viewer.add_point(5, -5, lambda x, y: (x*sp.sin(y), y*x), color='r')
#viewer.add_point(-5, -5, lambda x, y: (x*sp.sin(y), y*x), color='g')
#viewer.add_point(-5, 5, lambda x, y: (x*sp.sin(y), y*x), color='y')
f = viewer.add_function(lambda t, e: (1-e)*t + e*(np.cos(t) - np.sin(-t)), lambda t, e: (1-e)*(-t) + e*(np.sin(t) + np.cos(-t)), params={"e": e}, t_range=(-10, 10), num_points=1000, color="b", width=2)
f_2 = viewer.add_function(lambda t, e: (1-e)*t + e*(np.cos(t) - np.sin(t)), lambda t, e: (1-e)*t + e*(np.sin(t) + np.cos(t)), params={"e": e}, t_range=(-10, 10), num_points=1000, color="red", width=2)

linspace = np.linspace(-10, 10, 100)

viewer.add_point(a, b, lambda x, y: (x*y, y), color='b')
f = viewer.add_function(lambda x, mu, sigma: (x-mu)/sigma, "(x-mu)/sigma",{"mu": a, "sigma": b})
#viewer.add_point(1, 2, func=lambda x, y: (x, np.exp(x)))
#viewer.add_point(5, 5)
#for i in range(10):
#    for j in range(10):
#        #viewer.add_point(i*3 , j *2)
#        viewer.add_point("test", "test", func=lambda x, y, i=i + 1, j=j + 1: (i, np.exp(x)), color="b", size=10)

#print(point.__dict__['func'] for point in viewer.points)
#for point in viewer.points:
#    print(point.__dict__['func'](5,5))
#for connection in viewer.parameter_connections['test']:
#    print(connection.__dict__['func'](5,5))
viewer.show()
sys.exit(app.exec_())