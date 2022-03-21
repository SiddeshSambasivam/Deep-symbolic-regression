import unittest
import src as P
import torch

class TestSymbolicLayer(unittest.TestCase):

    def test_output_dimension(self):
        """Test to check the dimension of the output of a symbolic layer"""
        dim = (100, 4)
        input_dim = dim[1]

        n_funcs = len(P.default_funcs)
        x = torch.randn(dim)

        sym_layer_1 = P.SymbolicLayer(input_size=input_dim, funcs=P.default_funcs)
        out = sym_layer_1(x)

        assert out.shape == (100, n_funcs)