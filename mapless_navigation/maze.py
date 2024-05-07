from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple

import torch
import numpy as np


class Direction(Enum):
    VERTICAL = 1
    HORIZONTAL = 2


@dataclass
class Room:
    w: float
    h: float
    resolution: int
    offset_x: float
    offset_y: float
    # If has a division
    region_1: Optional[Room] = None
    region_2: Optional[Room] = None
    division_offset: Optional[float] = None
    division_direction: Optional[Direction] = None
    opening_offset: Optional[float] = None
    opening_length: Optional[float] = None
    cut_division_start: bool = False
    cut_division_end: bool = False

    def __post__init__(self):
        # Intialisation checks
        assert self.w > 0, "Width cannot be negative"
        assert self.h > 0, "Height cannot be negative"
        assert (
            self.region_1 is None
            and self.region_1 is None
            or not self.region_1 is None
            and not self.region_2 is None
        ), "Must be a complete binary tree"
        # Extra helper variables
        self.is_leaf = self.region_1 and self.region_2

    def can_divide(self, minimum_room_length: float, minimum_gap_size: float) -> bool:
        if self.resolution == 0:
            return False
        if self.w < minimum_room_length or self.h < minimum_room_length:
            return False
        if max(self.w, self.h) <= minimum_gap_size:
            return False
        return True

    def division(
        self,
        division_direction: Direction,
        division_offset: float,
        region_1: Room,
        region_2: Room,
    ):
        self.division_direction = division_direction
        self.division_offset = division_offset
        self.region_1 = region_1
        self.region_2 = region_2

    @property
    def area(self):
        return self.w * self.h

    @property
    def division_length(self):
        return self.w if self.has_horizontal_division else self.h

    @property
    def has_division(self):
        return not self.division_direction is None

    @property
    def has_horizontal_division(self):
        return self.division_direction == Direction.HORIZONTAL

    @property
    def has_vertical_division(self):
        return self.division_direction == Direction.VERTICAL

    def get_wall(self, minimum_gap_size: float):
        if not self.has_division:
            return None, None

        # Start wall
        wall_length = self.opening_offset
        x = self.offset_x
        y = self.offset_y
        if self.cut_division_start:
            wall_length = wall_length - minimum_gap_size
            if self.has_vertical_division:
                y += minimum_gap_size
            else:
                x += minimum_gap_size
        if self.has_horizontal_division:
            y += self.division_offset
        else:
            x += self.division_offset
        wall_1 = (x, y, self.division_direction, wall_length)

        # End wall
        wall_length = self.w if self.has_horizontal_division else self.h
        wall_length = wall_length - self.opening_offset - self.opening_length
        if self.cut_division_end:
            wall_length -= minimum_gap_size
        x = self.offset_x
        y = self.offset_y
        if self.has_horizontal_division:
            y += self.division_offset
            x += self.opening_offset + self.opening_length
        else:
            x += self.division_offset
            y += self.opening_offset + self.opening_length

        wall_2 = (x, y, self.division_direction, wall_length)

        return wall_1, wall_2


class Maze:
    """Generates a continuous maze using an adaptation of the recursive division method"""

    def __init__(
        self,
        w: float = 3,
        h: float = 3,
        resolution: int = 1,
        minimum_room_length: float = 0.1,
        minimum_gap_size: float = 0.05,
    ):
        self.w = w
        self.h = h
        assert resolution >= 0, "Expected positive resolution"
        self._random(cache_size=max(1, 2**resolution * 2))
        # Generates the rooms
        self.root = Room(w, h, resolution, 0, 0)
        self.leaves: List[Room] = []  # Can spawn entities inside
        self.rooms: List[Room] = []
        execution_stack = [self.root]

        # Generate divisions
        while len(execution_stack) > 0:
            room = execution_stack.pop()
            self.rooms.append(room)
            if not room.can_divide(minimum_room_length, minimum_gap_size):
                self.leaves.append(room)
                continue
            if room.w >= room.h:
                division_direction = Direction.VERTICAL
                division = (
                    self._random() * (room.w - minimum_gap_size * 2) + minimum_gap_size
                )
                w_1 = division
                w_2 = room.w - division
                h_1 = h_2 = room.h
                offset_x_2 = room.offset_x + division
                offset_y_2 = room.offset_y
            else:
                division_direction = Direction.HORIZONTAL
                division = (
                    self._random() * (room.h - minimum_gap_size * 2) + minimum_gap_size
                )
                h_1 = division
                h_2 = room.h - division
                w_1 = w_2 = room.w
                offset_x_2 = room.offset_x
                offset_y_2 = room.offset_y + division

            room.division(
                division_direction,
                division,
                Room(
                    w_1,
                    h_1,
                    resolution=room.resolution - 1,
                    offset_x=room.offset_x,
                    offset_y=room.offset_y,
                ),
                Room(
                    w_2,
                    h_2,
                    resolution=room.resolution - 1,
                    offset_x=offset_x_2,
                    offset_y=offset_y_2,
                ),
            )
            execution_stack.append(room.region_1)
            execution_stack.append(room.region_2)

        # Cut openings
        execution_stack = [self.root]
        while len(execution_stack) > 0:
            room = execution_stack.pop()
            if not room.has_division:
                continue
            room.opening_offset = self._random()

            if room.has_horizontal_division:
                room.opening_offset = room.opening_offset * (room.w - minimum_gap_size)
            else:
                room.opening_offset = room.opening_offset * (room.h - minimum_gap_size)
            room.opening_length = max(
                minimum_gap_size,
                self._random() * (room.division_length - room.opening_offset),
            )
            # Edge case: We need to cut out inner walls if it blocks our opening
            room.cut_division_start = False
            room.cut_division_end = False
            for region in [room.region_1, room.region_2]:
                if not (
                    region.has_division
                    and region.division_direction != room.division_direction
                ):
                    continue
                if (
                    room.opening_offset <= room.division_offset
                    and room.division_offset
                    <= room.opening_offset + room.opening_length
                ):
                    # We need to cut
                    if region == room.region_2:
                        region.cut_division_start = True
                    else:
                        region.cut_division_end = True

            execution_stack.append(room.region_1)
            execution_stack.append(room.region_2)

        # Generate all walls
        self.walls: List[Tuple[float, float, Direction, float]] = [
            (0, h, Direction.HORIZONTAL, w),
            (0, 0, Direction.HORIZONTAL, w),
            (0, 0, Direction.VERTICAL, h),
            (w, 0, Direction.VERTICAL, h),
        ]

        for room in self.rooms:
            if room.has_division:
                w1, w2 = room.get_wall(minimum_gap_size)
                self.walls.append(w1)
                self.walls.append(w2)

    def _random(self, cache_size: Optional[int] = None) -> float:
        """Use torch to generate list of random numbers to avoid repeated calls

        Returns:
            float: random number between 0 and 1
        """
        if cache_size != None:
            self._random_numbers = torch.rand(cache_size)
            self._random_number_index = 0
        result = self._random_numbers[self._random_number_index]
        self._random_number_index += 1
        if self._random_number_index == len(self._random_numbers):
            self._random_number_index = 0
            self._random_numbers = torch.rand(len(self._random_numbers))
        return result.item()

    def visualise(self) -> np.ndarray:
        image = np.zeros((self.h + 1, self.w + 1))
        for x, y, dir, l in self.walls:
            if dir == Direction.HORIZONTAL:
                for j in range(int(l)):
                    image[int(y), int(x + j)] = 1
            else:
                for i in range(int(l)):
                    image[int(y + i), int(x)] = 1
        return image
