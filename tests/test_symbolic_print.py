import unittest
import numpy as np
from pytest import skip
from functions import *
from utils import print_network  

class TestSymbolicPrint(unittest.TestCase):

    @unittest.skip('Skipped')
    def test_print_symbolic(self):

        funcs = [Constant(), Square(), Product()]
        w1 = np.array([[1]*4, [2]*4])
        w2 = 0.1 * np.array([*[*w1]*2])
        w3 = 0.1 * np.array([[2], [2], [2], [2]])
        print(w1.shape, w2.shape, w3.shape)
        
        expr = print_network([w1, w2, w3], funcs, ['a', 'b'])
        print(expr)


