import torch
import unittest
import src as P


class TestFunctions(unittest.TestCase):

    def test_square(self):
        x = torch.randint(0, 3, torch.Size([3,1]))
        sq = P.Square()
        torch.testing.assert_allclose(torch.square(x), sq.torch(x))

    def test_sigmoid(self):
        x = torch.randint(0, 3, torch.Size([3,1]))
        sig = P.Sigmoid()
        torch.testing.assert_allclose(torch.sigmoid(x), sig.torch(x))
        print(sig.name)

        