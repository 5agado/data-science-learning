import numpy as np
from scipy.spatial import distance
from typing import List


class Flock:
    NB_DIMENSIONS = 2  # defines world/vectors number of dimensions, typically 2 or 3
    VISIBILITY_RADIUS = 100  # unit visibility radius (capacity to identify neighbors)
    CLOSENESS = 20  # distance for which a unit is considered too close (separation rule)
    ATTRACTOR_CONFINE = 300  # distance at which the attractor is enabled

    # sub-behaviors influence factor
    COHESION_FACTOR = 1 / 50
    ALIGNMENT_FACTOR = 1 / 10
    SEPARATION_FACTOR = 1 / 2
    ATTRACTOR_FACTOR = 1 / 5
    VELOCITY_FACTOR = 2

    # whether to apply the given sub-behavior during flock update
    COHESION = True
    ALIGNMENT = True
    SEPARATION = True
    ATTRACTOR = True

    class Unit:
        """Atomic unit of a flock"""

        def __init__(self, pos: np.array, vel: np.array, is_leader=False):
            self.pos = pos
            self.vel = vel
            self.is_leader = is_leader

    def __init__(self, size: int, canvas_size: int, canvas_shift: tuple = None, seed: int = None):
        """
        Defines a flock components and behavior.
        :param size: number of units in the flock
        :param canvas_size: size of the space in which units are randomly generated (same size for each dimension)
        :param canvas_shift: shift random positions by this amount (should be one value for each dimension)
        :param seed: random generation seed
        """

        # Generate random position and velocity for flock units
        # use seed for reproducible results if given
        if seed:
            np.random.seed(seed)
        units_pos = np.random.randint(0, canvas_size, size=(size, Flock.NB_DIMENSIONS))
        if canvas_shift:
            units_pos += np.array(canvas_shift)
        units_vel = np.random.random(size=(size, Flock.NB_DIMENSIONS)) - 0.5

        # Instantiate flock units
        self.units = [Flock.Unit(pos=units_pos[i], vel=units_vel[i]) for i in range(size)]
        self.attractor_pos = np.zeros(Flock.NB_DIMENSIONS) + canvas_size//2

    def update(self):
        """Update flock state. For each unit compute and set new position"""
        # Should first compute new pos for all units, and just then set
        for unit in self.units:
            neighbors = self.get_neighbors(unit.pos)

            # Cohesion/Center
            if Flock.COHESION:
                Flock.apply_cohesion(unit, neighbors)

            # Alignment/Imitation
            if Flock.ALIGNMENT:
                Flock.apply_alignment(unit, neighbors)

            # Separation/Avoidance
            if Flock.SEPARATION:
                Flock.apply_separation(unit, neighbors)

            # Attractor
            # Keep Flock close to an attractor/interest point
            if Flock.ATTRACTOR:
                self.apply_attractor(unit)

            # normalize velocity vector
            unit.vel /= np.linalg.norm(unit.vel)
            # set final position
            unit.pos = unit.pos + unit.vel * Flock.VELOCITY_FACTOR

    @staticmethod
    # TODO should be weighted by group size, e.g. big group should not steer for single unit
    # apply cohesion rule to given unit based on given neighbors
    def apply_cohesion(unit: Unit, neighbors: List[Unit]):
        if neighbors and not unit.is_leader:
            # get neighbors average pos
            neigh_avg_pos = np.mean([n.pos for n in neighbors], axis=0)
            # steer vector is the one separating the unit current pos from
            # neighbors average pos
            steer_vec = neigh_avg_pos - unit.pos
            # make it proportional to its magnitude
            steer_vec *= Flock.COHESION_FACTOR #* np.linalg.norm(steer_vec)
            # steer unit to neighbors by changing velocity vector
            unit.vel += np.mean([unit.vel, steer_vec], axis=0)

    @staticmethod
    # apply alignment rule to given unit based on given neighbors
    def apply_alignment(unit: Unit, neighbors: List[Unit]):
        if neighbors and not unit.is_leader:
            # get neighbors average velocity vector
            neigh_avg_vel = np.mean([n.vel for n in neighbors], axis=0)
            # steer vector is a fraction of the neighbors average velocity/heading vector
            steer_vec = neigh_avg_vel * Flock.ALIGNMENT_FACTOR
            # adjust velocity vector of unit by averaging it with the steer vector
            unit.vel = np.mean([unit.vel, steer_vec], axis=0)

    @staticmethod
    # apply separation rule to given unit based on given neighbors
    def apply_separation(unit: Unit, neighbors: List[Unit]):
        for n in neighbors:
            dist = distance.euclidean(n.pos, unit.pos)
            # move the unit from neighbors that are too close
            # by updating its velocity vector
            if dist < Flock.CLOSENESS:
                avoid_vel = n.pos - unit.pos
                # also should be stronger when really close, and weak otherwise
                unit.vel -= avoid_vel * Flock.SEPARATION_FACTOR

    # apply attractor rule to given unit
    def apply_attractor(self, unit: Unit):
        dist = distance.euclidean(self.attractor_pos, unit.pos)
        # if attractor too far, steer unit towards it
        if dist > Flock.ATTRACTOR_CONFINE:
            converge_vel = self.attractor_pos - unit.pos
            # also should be stronger when really far, and weak otherwise
            unit.vel += converge_vel * Flock.ATTRACTOR_FACTOR

    # consider neighbors all those unit closer enough to the target
    # consequentially a unit is considered to have complete spherical visibility
    def get_neighbors(self, pos: np.array):
        neighbors = []
        for u in self.units:
            dist = distance.euclidean(u.pos, pos)
            if dist != 0. and dist < Flock.VISIBILITY_RADIUS:
                neighbors.append(u)
        return neighbors
