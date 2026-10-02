"""A_cone of the end-of-think state in the U-Space."""
from src import uspace

NEEDS = {"forward", "basis"}


def score(cell):
    return uspace.a_cone(cell.forward["eot_state"], cell.basis)
