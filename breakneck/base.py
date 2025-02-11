from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Coords2D:
    x: int
    y: int

    def __post_init__(self):
        object.__setattr__(self, "x", int(self.x))
        object.__setattr__(self, "y", int(self.y))

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __getitem__(self, idx):
        if idx == 0:
            return self.x
        elif idx == 1:
            return self.y
        else:
            raise IndexError("Index out of range")

    def __iter__(self):
        yield self.x
        yield self.y

    def __len__(self):
        return 2

    def __repr__(self):
        return f"Coords({self.x}, {self.y})"

    def as_tol(self, tol_nm):
        return Coords2D(round(self.x / tol_nm), round(self.y / tol_nm))

    def as_tuple(self):
        return self.x, self.y


class Sides(str, Enum):
    front = "front"
    back = "back"
    both = "both"
