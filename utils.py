import copy
from typing import List, Union
import sympy as sp

from functions import Function, BinaryInpFunction, count_double_inputs

class IncorrectMatrixDimension(Exception):
    def __init__(self, expected:tuple, received:tuple) -> None:
        super().__init__(
            f"Expected matrix of shape {expected} but received shape {received}"
        )

def convert_to_symbols(var_names: List[str]) -> None:
    vars = []
    for var in var_names:
        if isinstance(var, str):
            vars.append(sp.Symbol(var))
        else:
            vars.append(var)
    
    expr = sp.Matrix(vars).T

    return expr

def filter_weights(w, threshold:float):
    m,n = w.shape
    for i in range(m):
        for j in range(n):
            if abs(w[i, j]) < threshold:
                w[i, j] = 0
    return w

def apply_activation_fns(w, funcs, n_double):

    w = sp.Matrix(w)
    n_single = len(funcs) - n_double
    if n_double == 0:
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                w[i, j] = funcs[j](w[i, j])
    else:
        w_new = copy.deepcopy(w)
        for i in range(w.shape[0]):
            fn_idx = 0 # tracks the function to apply
            idx = 0

            while fn_idx < n_single:
                w_new[i, fn_idx] = funcs[fn_idx](w[i, idx])
                idx += 1
                fn_idx += 1

            while fn_idx < len(funcs):
                w_new[i, fn_idx] = funcs[fn_idx](w[i, idx], w[i, idx+1])
                fn_idx += 1
                idx + 2
            
        for i in range(n_double):
            w_new.col_del(-1)
        w = w_new

    return w

def print_symbolic_equation(
    weights:list, 
    funcs:List[Union[Function, BinaryInpFunction]], 
    var_names:List[str], 
    n_double:int, 
    threshold:float=0.01):
    
    expr = convert_to_symbols(var_names)

    for w in weights:
        w = filter_weights(sp.Matrix(w), threshold)
        expr = expr * w
        expr = apply_activation_fns(expr, funcs, n_double)
    
    return expr

def print_network(weights, funcs, var_names, threshold=0.01):

    n_double = count_double_inputs(funcs)
    funcs = [func.sp for func in funcs]

    expr = print_symbolic_equation(weights[:-1], funcs, var_names, n_double, threshold)
    expr = expr * filter_weights(weights[-1],  threshold)

    return expr[0,0]
