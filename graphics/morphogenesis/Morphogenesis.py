import numpy as np
import sys
from scipy.spatial import distance

from rtree import index

MORPHOGENESIS_BASE_CONFIG = {
    'VISIBILITY_RADIUS': 0.4,
    'REPULSION_FAC': 1/20,
    'ATTRACTION_FAC': 1/20,
    'SPLIT_DIST_THRESHOLD': 0.2,
    'RAND_OPTIMIZATION_FAC': 0,
    'SUBDIVISION_METHOD': 'BY_DISTANCE',
}


class Morphogenesis:
    def __init__(self, nodes, closed: bool, config: dict):
        self.nodes = nodes
        self.config = config

        self.idx_2d = None

        self.VISIBILITY_RADIUS = config['VISIBILITY_RADIUS']
        self.REPULSION_FAC = config['REPULSION_FAC']
        self.ATTRACTION_FAC = config['ATTRACTION_FAC']
        self.SPLIT_DIST_THRESHOLD = config['SPLIT_DIST_THRESHOLD']
        self.RAND_OPTIMIZATION_FAC = config['RAND_OPTIMIZATION_FAC']
        self.SUBDIVISION_METHOD = config['SUBDIVISION_METHOD']

        self.CLOSED = closed

    def update(self, draw_force=None, draw_segment=None):
        # Reset index before subdivision
        self.idx_2d = index.Index()
        new_nodes = self._adaptive_subdivision()

        if draw_segment:
            draw_segment(new_nodes, 3)

        optimized_nodes = self._nodes_optimization(new_nodes, draw_force)

        self.nodes = optimized_nodes

    def _adaptive_subdivision(self):
        # Nodes Insertion
        new_nodes = []

        for i, n in enumerate(self.nodes):
            # add new node between this and previous, if growth conditions are met
            if new_nodes:
                new_node = self._subdivision(new_nodes[-1], n)
                if new_node is not None:
                    new_nodes.append(new_node)
                    self._add_node_to_index(len(new_nodes) - 1, new_node)

            # append current node
            new_nodes.append(n)
            self._add_node_to_index(len(new_nodes) - 1, n)

        # If closed shape, add node between last and first nodes
        if self.CLOSED:
            new_node = self._subdivision(new_nodes[-1], new_nodes[0])
            if new_node is not None:
                new_nodes.append(new_node)
                self._add_node_to_index(len(new_nodes) - 1, new_node)

        return new_nodes

    def _nodes_optimization(self, new_nodes, draw_force=None):
        # Nodes Optimization
        optimized_nodes = []
        for i, n in enumerate(new_nodes):
            # Attraction

            # first and last nodes are not subject to attraction forces if not a closed line
            if not self.CLOSED and (i==0 or i==len(new_nodes)-1):
                attraction_vec = np.array((0, 0, 0))
            else:
                # TODO attraction works even if connected are further than visibility dist
                attraction_vec = ((new_nodes[i-1] + new_nodes[(i+1) % len(new_nodes)])/2) - n
                # normalize
                attraction_norm = np.linalg.norm(attraction_vec)
                if attraction_norm != 0:
                    attraction_vec /= attraction_norm

            # Repulsion

            neighbors_nodes = self._get_neighbors(new_nodes, n)
            # without neighbors there is no repulsion
            repulsion_vec = np.array((0, 0, 0))
            if len(neighbors_nodes) > 1:
                repulsion_vec = np.sum([neigh - n for neigh in neighbors_nodes], axis=0)
                # normalize
                repulsion_norm = np.linalg.norm(repulsion_vec)
                if repulsion_norm != 0:
                    repulsion_vec /= repulsion_norm
                # negate
                repulsion_vec = -repulsion_vec

            if draw_force:
                draw_force(n, n+(repulsion_vec * self.REPULSION_FAC), 2)
                draw_force(n, n+(attraction_vec * self.ATTRACTION_FAC), 1)

            # compute new node optimized position
            new_node = n + (repulsion_vec * self.REPULSION_FAC) \
                         + (attraction_vec * self.ATTRACTION_FAC)

            # if set, add some random noise to node position
            if self.RAND_OPTIMIZATION_FAC > 0:
                new_node += (0.5 - np.random.rand(len(new_node))) * self.RAND_OPTIMIZATION_FAC

            optimized_nodes.append(new_node)

        return optimized_nodes

    def _subdivision(self, from_node, to_node):
        if self.SUBDIVISION_METHOD == "BY_DISTANCE":
            dist = Morphogenesis._get_dist(from_node, to_node)
            if dist > self.SPLIT_DIST_THRESHOLD:
                # new node is halfway between the two connected ones
                new_node = (from_node + to_node) / 2
                return new_node
            else:
                return None
        else:
            print("No such subdivision method: {}. Exiting".format(self.SUBDIVISION_METHOD))
            sys.exit(1)

    def _add_node_to_index(self, node_idx, node):
        self.idx_2d.insert(node_idx, (node[0], node[1], node[0], node[1]))

    def _get_neighbors(self, nodes, pos: np.array):
        left, bottom, right, top = (pos[0] - self.VISIBILITY_RADIUS,
                                    pos[1] - self.VISIBILITY_RADIUS,
                                    pos[0] + self.VISIBILITY_RADIUS,
                                    pos[1] + self.VISIBILITY_RADIUS)
        neighbors_nodes = np.array(nodes)[list(self.idx_2d.intersection((left, bottom, right, top)))]
        # neighbors_nodes = np.array(new_nodes)[list(idx_2d.nearest((n[0], n[1], n[0], n[1]), 10))]

        return neighbors_nodes

    @staticmethod
    def _get_dist(node_a: np.array, node_b: np.array) -> float:
        dist = distance.euclidean(node_a, node_b)
        return dist
