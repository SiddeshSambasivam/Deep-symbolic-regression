import unittest

import numpy as np
import src as P

class TestSymbolicPrint(unittest.TestCase):

    def test_print_symbolic(self):

        funcs = [P.Constant(), P.Square(), P.Product()]
        w1 = np.array([[1]*4, [2]*4])
        w2 = 0.1 * np.array([*[*w1]*2])
        w3 = 0.1 * np.array([[2], [2], [2], [2]])
        print(w1.shape, w2.shape, w3.shape)
        
        expr = P.print_network([w1, w2, w3], funcs, ['a', 'b'])
        assert str(expr) == '0.04*a + 0.08*b + 0.06*(a + 2*b)**2 + 0.064*(0.5*a + b + 0.75*(a + 2*b)**2 + 0.25)**2 + 0.22'
        print(expr)


