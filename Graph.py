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


class Parameter():
    def __init__(self, name, label, slider, min_val, max_val, value, step=1):
        self.name = name
        self.label = label
        self.slider = slider
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.step = step

    def update_values(self, new_val):
        self.init_val = new_val


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


class Grid(QtWidgets.QWidget):
    def __init__(self):
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

        # Configure plot
        self.plotWidget.setBackground("k")
        self.plotWidget.showGrid(x=True, y=True, alpha=0.3)
        self.plotWidget.setMouseEnabled(x=True, y=True)
        self.plotWidget.setRange(xRange=[-10, 10], yRange=[-10, 10])
        self.plotWidget.getAxis("left").setPen("w")
        self.plotWidget.getAxis("bottom").setPen("w")
        self.plotWidget.setAspectLocked(True)

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
        #print(self.parameters[param_vals['name']].value)
        for obj in self.parameter_connections[param_vals['name']]:
            param_updates = {}

            for key, value in obj.param_connections.items():
                if param_vals["name"] in value:
                    param_updates[key] = param_vals["value"]
            obj.update_values(**param_updates)
            #print(param_updates)
        #print(self.points[0].x, self.points[0].y)




        # ADD REDRAW

    def add_point(self, X, Y, func=lambda x, y: (x, y), color="r", size=10):
        scatter = pg.ScatterPlotItem(size=size, brush=pg.mkBrush(color))
        self.plotWidget.addItem(scatter)
        values = []
        params = []
        parameter_connections = {}
        for value, var in [(X, "x"), (Y, "y")]:
            parameter_connections[var] = []
            if type(value) == Parameter:
                if value in self.parameters.values():
                    values.append(value.value)
                    params.append(value)
                    parameter_connections[var].append(value.name)
                else:
                    raise NameError("Parameter {} is not defined.".format(value.name))
            else:
                values.append(value)
        point = Point(values[0], values[1], param_connections=parameter_connections, scatter=scatter, func=func,
                      color=color, size=size)
        #print(point.__dict__['func'](5,5))
        #print(point.param_connections)
        for param in set(params):
            self.parameter_connections[param.name].append(point)
        self.points.append(point)
        x, y = func(*values)
        scatter.setData([x], [y])





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