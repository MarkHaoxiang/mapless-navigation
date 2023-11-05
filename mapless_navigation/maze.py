import random

import numpy as np

class Maze:
    """ Generates a continuous maze using an adaptation of the recursive division method
    """
    def __init__(self,
                 w: float,
                 h: float,
                 resolution: int = 1,
                 offset_x: float = 0,
                 offset_y: float = 0,
                 minimum_room_length: float = 4,
                 minimum_gap_size: float = 2 ):
        
        self.resolution = resolution
        self.w = w
        self.h = h
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.division_direction = True
        self.minimum_gap_size = minimum_gap_size
        self.minimum_room_lengt = minimum_room_length


        # Note
        # In my mental model when designing this 
        # Y
        # D
        # O
        # W
        # N
        # x RIGHT

        # Base case
        if self.resolution <= 0 or \
            self.w < minimum_room_length or self.h < minimum_room_length or \
            max(self.w, self.h) <= minimum_gap_size * 2: # Not big enough for a division
            self.resolution = 0
            return
        if w > h or w==h and random.random() < 0.5:
            # Vertical division
                # Also ensure everywhere is traversable, so leave at least gap size
            division = random.random() * (w-minimum_gap_size*2) + minimum_gap_size 
            w_1 = division
            w_2 = w-division
            h_1 = h_2 = h
            offset_x_2 = self.offset_x + division
            offset_y_2 = self.offset_y 
            self.division_direction = True
        else:
            # Horizontal division
            division = random.random() * (h-minimum_gap_size*2) + minimum_gap_size 
            h_1 = division
            h_2 = h - division
            w_1 = w_2 = w
            offset_x_2 = self.offset_x 
            offset_y_2 = self.offset_y + division 
            self.division_direction = False

        self.region_1 = Maze(
            w_1,
            h_1,
            resolution=self.resolution-1,
            offset_x=self.offset_x,
            offset_y=self.offset_y,
            minimum_room_length=minimum_room_length,
            minimum_gap_size=minimum_gap_size
        )
        self.region_2 = Maze(
            w_2, 
            h_2,
            resolution=self.resolution-1,
            offset_x=offset_x_2,
            offset_y=offset_y_2,
            minimum_room_length=minimum_room_length,
            minimum_gap_size=minimum_gap_size
        )

        self.division_offset = division
        self.division_length = self.w if self.has_horizontal_division else self.h

        # Opening within division
        self.opening_offset = random.random()
        if self.has_horizontal_division:
            self.opening_offset = self.opening_offset * (self.w - minimum_gap_size)
        else:
            self.opening_offset = self.opening_offset * (self.h - minimum_gap_size)
        self.opening_length = max(
            minimum_gap_size,
            random.random()*(self.division_length-self.opening_offset)
        )
            # Edge case: We need to cut out inner walls if it blocks our opening
        self.cut_division_start = False
        self.cut_division_end = False
        for region in [self.region_1, self.region_2]:
            if not(region.has_division and region.division_direction != self.division_direction):
                continue
            if self.opening_offset <= region.division_offset and region.division_offset <= self.opening_offset+self.opening_length:
                # We need to cut
                if region == self.region_2:
                    region.cut_division_start = True
                else:
                    region.cut_division_end = True
    @property
    def has_division(self):
        return self.resolution > 0

    @property
    def has_horizontal_division(self):
        return self.has_division and not self.division_direction
    
    @property
    def has_vertical_division(self):
        return self.has_division and self.division_direction

    def visualize(self, image: np.ndarray=None):
        if image is None:
            image = np.zeros((self.h, self.w))

        if self.has_horizontal_division:
            # Draw wall
            i = int(self.division_offset + self.offset_y)
            for j in range(int(self.minimum_gap_size if self.cut_division_start else 0),
                           int(self.w-self.minimum_gap_size if self.cut_division_end else self.w)):
                image[i,int(j+self.offset_x)] = 1
            # Draw opening
            for j in range(int(self.opening_length)):
                image[i, int(self.offset_x+self.opening_offset+j)] = 0
        elif self.has_vertical_division:
            # Draw wall
            j = int(self.division_offset + self.offset_x)
            for i in range(int(self.minimum_gap_size if self.cut_division_start else 0),
                           int(self.h-self.minimum_gap_size if self.cut_division_end else self.h)):
                image[int(i+self.offset_y),j] = 1
            # Draw opening
            for i in range(int(self.opening_length)):
                image[int(self.offset_x+self.opening_offset+i), j] = 0
        if self.has_division:
            self.region_1.visualize(image=image)
            self.region_2.visualize(image=image)
        return image
