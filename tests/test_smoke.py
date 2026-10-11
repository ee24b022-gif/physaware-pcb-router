import pcbroute
from pcbroute.board import Board, Pad


def test_version():
    assert pcbroute.__version__


def test_board_nets():
    b = Board(pads=[Pad("Q1", "1", "VIN", "F.Cu", 0, 0, 1, 1)])
    assert b.nets() == {"VIN"}
