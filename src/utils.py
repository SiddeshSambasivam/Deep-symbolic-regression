
class IncorrectMatrixDimension(Exception):
    def __init__(self, expected:tuple, received:tuple) -> None:
        super().__init__(
            f"Expected matrix of shape {expected} but received shape {received}"
        )

