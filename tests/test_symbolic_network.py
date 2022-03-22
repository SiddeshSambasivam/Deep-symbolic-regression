import unittest
import torch

from model import SymbolicNN
from functions import default_funcs

class TestSymbolicNN(unittest.TestCase):

    def test_output_dimension(self):
        """Test to check the dimension of the output of a symbolic layer"""
        dim = (100, 1)
        input_dim = dim[1]

        n_funcs = len(default_funcs)
        x = torch.randn(dim)

        nn = SymbolicNN(2, default_funcs)
        out = nn(x)

        assert out.shape == (100, 1)

    def test_symbolic_print(self):
        pass