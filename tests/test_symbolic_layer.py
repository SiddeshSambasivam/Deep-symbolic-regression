import unittest
import torch

from model import SymbolicLayer
from functions import default_funcs

class TestSymbolicLayer(unittest.TestCase):

    def test_output_dimension(self):
        """Test to check the dimension of the output of a symbolic layer"""
        dim = (100, 4)
        input_dim = dim[1]

        n_funcs = len(default_funcs)
        x = torch.randn(dim)

        sym_layer_1 = SymbolicLayer(input_size=input_dim, funcs=default_funcs)
        out = sym_layer_1(x)

        assert out.shape == (100, n_funcs)