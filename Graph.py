import sys
import warnings
import inspect
import ast
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import sympy as sp
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sympy import diff, integrate


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
    def __init__(self, name, label, slider, min_val, max_val, value, step=1):
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

    def update_values(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f'{key} not found.')
        x_transform, y_transform = self.func(self.x, self.y)
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.scatter.setData([x_transform], [y_transform])


class Function():
    def __init__(self, func, param_connections, param_values, x_space, x_range, num_points, curve, expr):
        self.func = func
        self.param_connections = param_connections
        self.param_values = param_values
        self.x_space = x_space
        self.x_range = x_range
        self.num_points = num_points
        self.curve = curve
        self.expr = expr

    def update_values(self, **kwargs):
        for key, value in kwargs.items():
            self.param_values[key] = value
        y = self.func(self.x_space, **self.param_values)
        self.curve.setData(self.x_space, y)


class Grid():
    def __init__(self, x, y, lines, param_values, transform_func, param_connections, grid_plot, x_range, y_range, num_points, color, width):
        self.x = x
        self.y = y
        self.lines = lines
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

class Grid(QtWidgets.QWidget):
    def __init__(self, xmin=-10, xmax=10, ymin=-10, ymax=10):
        super().__init__()
        self.setWindowTitle("Grid Plot")

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

    def add_parameter(self, name, min_val, max_val, init_val, step=1):
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
        return param

    def _update_param(self, param):
        param_vals = param.__dict__
        decimals = max(0, int(-np.log10(param_vals["step"])) if param_vals["step"] < 1 else 0)
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

    def add_expression(self, expression, expression_name: str):
        param_values = {p.expr: p.value for p in self.parameters.values()}
        expression = expression.expr
        parameter_dependencies = expression.free_symbols
        expression_evaluation = expression.subs(param_values).evalf()
        expression = Expression(expression_name, "", expression_evaluation, parameter_dependencies, expression)
        expression.expression_name = expression_name


        self.expressions[expression_name] = expression
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
        x, y = func(x_eval, y_eval)
        scatter.setData([x], [y])

    def add_function(self, func, params, x_range=(-10, 10), num_points=1000, color="b", width=2):
        """
        The x variable should be the first argument in the function.
        :param func:
        :param params:
        :param x_range:
        :param num_points:
        :param color:
        :param width:
        :return:
        """
        curve = self.plotWidget.plot(pen=pg.mkPen(color=color, width=width))
        curve.setClipToView(True)
        x = np.linspace(x_range[0], x_range[1], num_points)
        func_arguments = func.__code__.co_varnames
        symbols = {}
        for arg in func_arguments:
            symbols[arg] = sp.Symbol(arg)
        print(symbols)
        parameter_connections = {}
        pvals = {key: param.value for key, param in params.items()}
        for k, v in params.items():
            parameter_connections[k] = [v.name]
        function = Function(func, parameter_connections, pvals, x, x_range, num_points, curve)
        for param in params.values():
            self.parameter_connections[param.name].append(function)
        y = func(x, **pvals)
        curve.setData(x, y)



app = QtWidgets.QApplication(sys.argv)
viewer = Grid()
viewer.resize(900, 700)
test = viewer.add_parameter("test", -10, 10, 5, step=0.1)
test2 = viewer.add_parameter("test2", -10, 10, 5, step=1)
print(test)
print(viewer.parameters.values())
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