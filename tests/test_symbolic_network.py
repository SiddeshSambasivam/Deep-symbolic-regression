import unittest
import src as P
import torch

class TestSymbolicNN(unittest.TestCase):

    def test_output_dimension(self):
        """Test to check the dimension of the output of a symbolic layer"""
        dim = (100, 1)
        input_dim = dim[1]

        n_funcs = len(P.default_funcs)
        x = torch.randn(dim)

        nn = P.SymbolicNN(2, P.default_funcs)
        out = nn(x)

        assert out.shape == (100, 1)

    def test_symbolic_print(self):
        pass