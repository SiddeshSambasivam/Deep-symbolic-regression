import typing as T
from abc import abstractmethod, ABC
import numpy as np
import sympy as sp

import torch

NumpyArrayType = T.Type[np.ndarray]

class Function(ABC):
    """
    Abstract base class for a function.
    """

    def __init__(self, norm=1) -> None:
        self.norm = norm

    @abstractmethod
    def torch(self):
        """Returns the torch implementation for the function"""

    @abstractmethod
    def sp(self):
        """Returns the sympy implementation of the function"""

    @property
    def name(self):
        """Returns the name of the function."""
        return str(self.sp)
    
    def np(self, x):
        """Returns the numpy of sympy"""
        z = sp.symbols('z')
        return sp.utilities.lambdify(z, self.sp(z), 'numpy')(x)


class Constant(Function):
    def torch(self, x):
        return torch.ones_like(x)

    def sp(self, x):
        return 1

    def np(self, x) -> NumpyArrayType:
        return np.ones_like(x)


class Identity(Function):
    def torch(self, x):
        return x / self.norm  

    def sp(self, x):
        return x / self.norm

    def np(self, x) -> NumpyArrayType:
        return np.array(x) / self.norm

class Square(Function):
    def torch(self, x):
        return torch.square(x) / self.norm  

    def sp(self, x):
        return x**2 / self.norm

    def np(self, x) -> NumpyArrayType:
        return np.square(x) / self.norm

class Pow(Function):
    def __init__(self, power, norm=1):
        Function.__init__(self, norm=norm)
        self.power = power

    def torch(self, x):
        return torch.pow(x, self.power) / self.norm

    def sp(self, x):
        return x**self.power / self.norm

class Sin(Function):
    def torch(self, x):
        return torch.sin(x) / self.norm  

    def sp(self, x):
        return sp.sin(x) / self.norm

class Sigmoid(Function):
    def torch(self, x):
        return torch.sigmoid(x) / self.norm  

    def sp(self, x):
        return (1/ (1 + sp.exp(-1*x))) / self.norm

    def np(self, x) -> NumpyArrayType:
        return (1/ (1 + np.exp(-1*x))) / self.norm

    @property
    def name(self):
        return "sigmoid(x)"

class Exp(Function):
    def __init__(self, norm=np.e) -> None:
        super().__init__(norm)

    def torch(self, x):
        return torch.exp(x) / self.norm  

    def sp(self, x):
        return sp.exp(x) / self.norm

    def np(self, x) -> NumpyArrayType:
        return np.array(x) / self.norm

class Log(Function):
    def torch(self, x):
        return torch.log(x) / self.norm  

    def sp(self, x):
        return sp.log(sp.Abs(x)) / self.norm

    def np(self, x) -> NumpyArrayType:
        return np.array(x) / self.norm

class BinaryInpFunction(ABC):

    def __init__(self, norm=1.):
        self.norm = norm

    @abstractmethod
    def sp(self, x, y):
        """Returns the sympy implementation of the function"""

    @abstractmethod    
    def torch(self, x, y):
        """Returns the torch implementation of the function"""

    def np(self, x, y) -> NumpyArrayType:
        """Returns the numpy implementatino of the function"""
        a, b = sp.symbols('a b')
        return sp.utilities.lambdify([a, b], self.sp(a, b), 'numpy')(x, y)

    @property
    def name(self):
        return str(self.sp)

class Product(BinaryInpFunction):

    def torch(self, x, y):
        return x*y/self.norm
    
    def sp(self, x, y):
        return x*y / self.norm

def count_singular_fns(funcs) -> int:
    count = 0
    for func in funcs:
        if isinstance(func, Function) and not isinstance(func, BinaryInpFunction):
            count += 1

    return count

def count_total_inputs(funcs):
    i = 0
    for func in funcs:
        if isinstance(func, Function):
            i += 1
        elif isinstance(func, BinaryInpFunction):
            i += 2
    return i


def count_double_inputs(funcs):
    i = 0
    for func in funcs:
        if isinstance(func, BinaryInpFunction):
            i += 1
    return i

# NOTE: Need to manually make sure that all the single input functions comes first
default_funcs = [
    *[Constant()] * 2,
    *[Identity()] * 4,
    *[Square()] * 4,
    *[Sin()] * 2,
    *[Exp()] * 2,
    *[Sigmoid()] * 2,
    *[Product()] * 2,
]