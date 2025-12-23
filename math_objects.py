import numpy as np
import pyqtgraph as pg
import sympy as sp
from PyQt5.uic.properties import QtGui
from pyqtgraph.Qt import QtGui, mkQApp
from sympy import exp, sin, cos, oo, pi, log, S, nsimplify
from sympy.calculus.util import continuous_domain
import random
import bisect
import matplotlib.pyplot as plt
from skimage.measure import find_contours


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

    def __add__(self, other):
        return sp.sympify(self) + sp.sympify(other)

    def __radd__(self, other):
        return sp.sympify(other) + sp.sympify(self)

    def __mul__(self, other):
        return sp.sympify(self) * sp.sympify(other)

    def __rmul__(self, other):
        return sp.sympify(other) * sp.sympify(self)

    def __truediv__(self, other):
        return sp.sympify(self) / sp.sympify(other)

    def __rtruediv__(self, other):
        return sp.sympify(self) / sp.sympify(other)

    def __sub__(self, other):
        return sp.sympify(self) - sp.sympify(other)

    def __rsub__(self, other):
        return sp.sympify(other) - sp.sympify(self)

    def __pow__(self, other):
        return sp.sympify(self) ** sp.sympify(other)

    def __rpow__(self, other):
        return sp.sympify(other) ** sp.sympify(self)

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)

    def __str__(self):
        return str(self.symbol)

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

    def __add__(self, other):
        return sp.sympify(self) + sp.sympify(other)

    def __radd__(self, other):
        return sp.sympify(other) + sp.sympify(self)

    def __mul__(self, other):
        return sp.sympify(self) * sp.sympify(other)

    def __rmul__(self, other):
        return sp.sympify(other) * sp.sympify(self)

    def __truediv__(self, other):
        return sp.sympify(self) / sp.sympify(other)

    def __rtruediv__(self, other):
        return sp.sympify(self) / sp.sympify(other)

    def __sub__(self, other):
        return sp.sympify(self) - sp.sympify(other)

    def __rsub__(self, other):
        return sp.sympify(other) - sp.sympify(self)

    def __pow__(self, other):
        return sp.sympify(self) ** sp.sympify(other)

    def __rpow__(self, other):
        return sp.sympify(other) ** sp.sympify(self)

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)


class Point():
    def __init__(self, x_expr, y_expr, x_func, y_func, x_num, y_num, x_symbols, y_symbols, x_values, y_values, params,
                 param_connections, scatter, color, size, plot):
        self.x = x_expr
        self.y = y_expr
        self.x_func = x_func
        self.y_func = y_func
        self.x_num = x_num
        self.y_num = y_num
        self.x_symbols = x_symbols
        self.y_symbols = y_symbols
        self.x_values = x_values
        self.y_values = y_values
        self.param_values = y_values | x_values
        self.x_point = x_func(**x_values)
        self.y_point = y_func(**y_values)
        self.param_connections = param_connections
        self.scatter = scatter
        self.color = color
        self.size = size
        self.plot = plot

    def update_values(self, x_range, y_range, **kwargs):
        if not self.plot:
            return
        for key, value in kwargs.items():
            if key in self.x_values:
                self.x_values[key] = value
            if key in self.y_values:
                self.y_values[key] = value
            if key in self.param_values:
                self.param_values[key] = value

        self.x_point = self.x_func(**self.x_values)
        self.y_point = self.y_func(**self.y_values)

        self.scatter.setData([self.x_point], [self.y_point])

    def save_data(self):
        return ("point", {"X": self.x, "Y": self.y, "func": self.func, "color": self.color, "size": self.size})


class Function():
    def __init__(self, x_func, y_func, x_num, y_num, params, param_connections, y_values, x_values, domain, base_domain,
                 t_space, t_range, num_points, color, width, curve, scatter, y_expr, x_expr, y_symbols, x_symbols, type,
                 initial_parametric_resolution, plot, parent):
        # Lambda functions
        self.x_func = x_func
        self.y_func = y_func
        # Numerical expressions
        self.x_num = x_num
        self.y_num = y_num
        self.params = {}
        for k, v in params.items():
            self.params = {k: str(v)}
        self.param_connections = param_connections
        self.y_values = y_values
        self.x_values = x_values
        self.param_values = y_values | x_values
        self.domain = domain  # set obj
        self.base_domain = base_domain  # sp Set, numerical evaluation of original domain
        self.t_space = t_space  # ta bort senare
        self.t_range = t_range  # ta bort senare
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
        self.parent = parent

    def update_values(self, x_range, y_range, **kwargs):
        if not self.plot:
            return
        for key, value in kwargs.items():
            if key in self.x_values:
                self.x_values[key] = value
            if key in self.y_values:
                self.y_values[key] = value
            if key in self.param_values:
                self.param_values[key] = value

        self.y_num = self.y_expr.subs(self.y_values)
        self.x_num = self.x_expr.subs(self.x_values)
        try:
            y_inherent_domain = continuous_domain(self.y_num, self.parent.x, S.Reals)  # parent = Graph object
        except NotImplementedError:
            y_inherent_domain = sp.Interval(-oo, oo)
        try:
            x_inherent_domain = continuous_domain(self.x_num, self.parent.y, S.Reals)  # parent = Graph object
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
            func = self.x_func
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
            func = self.y_func
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
        x = self.x_func(t_space, **self.x_values)
        x = np.where(mask, x, np.nan)
        y = self.y_func(t_space, **self.y_values)
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
                    x_dense = self.x_func(t_dense, **self.x_values)
                    y_dense = self.y_func(t_dense, **self.y_values)
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