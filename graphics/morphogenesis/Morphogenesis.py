import numpy as np
import sys
from scipy.spatial import distance

from rtree import index

MORPHOGENESIS_BASE_CONFIG = {
    'VISIBILITY_RADIUS': 0.4,
    'REPULSION_FAC': 1/20,
    'ATTRACTION_FAC': 1/20,
    'SPLIT_DIST_THRESHOLD': 0.2,
    'SIMPLIFICATION_DIST_THRESHOLD': 0.05,
    'SPLIT_CROWD_THRESHOLD': 5,
    'RAND_OPTIMIZATION_FAC': 0,
    'SUBDIVISION_METHOD': 'BY_DISTANCE',  # also BY_CROWDEDNESS
    'ATTRACTION': True,
    'SIMPLIFICATION': False,
    'DIMENSIONS': 2
}

# TODO more elegant handling of container
# currently would influence also subdivision, for example if BY_CROWDEDNESS


class Morphogenesis:
    def __init__(self, nodes, closed: bool, config: dict, container=None):
        """
        :param nodes: initial set of nodes (list of 2/3D coordinates)
        :param closed: whether nodes form a close line
        :param config: morphogenesis parameters
        :param container: optional list of nodes that constrain the growth (via influence on repulsion forces)
        """
        self.nodes = nodes
        self.config = config

        # Init rtree index
        self.index_props = index.Property()
        self.index_props.dimension = config['DIMENSIONS']
        self.index = None
        self.container = container

        self.VISIBILITY_RADIUS = config['VISIBILITY_RADIUS']
        self.REPULSION_FAC = config['REPULSION_FAC']
        self.ATTRACTION_FAC = config['ATTRACTION_FAC']
        self.SPLIT_DIST_THRESHOLD = config['SPLIT_DIST_THRESHOLD']
        self.SIMPLIFICATION_DIST_THRESHOLD = config['SIMPLIFICATION_DIST_THRESHOLD']
        self.SPLIT_CROWD_THRESHOLD = config['SPLIT_CROWD_THRESHOLD']
        self.RAND_OPTIMIZATION_FAC = config['RAND_OPTIMIZATION_FAC']
        self.ATTRACTION = config['ATTRACTION']
        self.SIMPLIFICATION = config['SIMPLIFICATION']
        self.SUBDIVISION_METHOD = config['SUBDIVISION_METHOD']

        self.CLOSED = closed

    def update(self, draw_force=None, draw_segment=None):
        """
        Update system status (run one growth epoch)
        :param draw_force: optional rendering function, takes two points and a material index
        :param draw_segment: optional rendering function, takes N points and a material index
        """
        # Reset index before subdivision
        self.index = index.Index(properties=self.index_props)

        # subdivision
        new_nodes = self._adaptive_subdivision()

        # if we have a container, append its nodes to the index
        if self.container:
            for i, node in enumerate(self.container):
                self._add_node_to_index(len(new_nodes)+i, node)

        if draw_segment:
            draw_segment(new_nodes, 3)

        optimized_nodes = self._nodes_optimization(new_nodes, draw_force)

        self.nodes = optimized_nodes

    def _adaptive_subdivision(self):
        # start with first node
        new_nodes = [self.nodes[0]]
        self._add_node_to_index(0, new_nodes[0])

        # If closed shape, allow to add node between last and first nodes
        if self.CLOSED:
            self.nodes.append(self.nodes[0])

        for i, n in enumerate(self.nodes[1:]):
            # add new node between this and previous, if growth conditions are met
            new_node = self._subdivision(new_nodes[-1], n, new_nodes)
            if new_node is not None:
                self._add_node_to_index(len(new_nodes), new_node)
                new_nodes.append(new_node)

            # simplification
            # avoid appending current node is simplification enabled, and too close to previous one
            if self.SIMPLIFICATION:
                dist = Morphogenesis._get_dist(n, new_nodes[-1])
                if dist < self.SIMPLIFICATION_DIST_THRESHOLD:
                    continue

            # append current node
            if self.CLOSED and i >= (len(self.nodes)-2):
                continue
            else:
                self._add_node_to_index(len(new_nodes), n)
                new_nodes.append(n)

        return new_nodes

    def _nodes_optimization(self, new_nodes, draw_force=None):
        # Nodes Optimization
        optimized_nodes = []
        for i, n in enumerate(new_nodes):

            # Attraction
            if self.ATTRACTION:
                # first and last nodes are not subject to attraction forces if not a closed line
                if not self.CLOSED and (i == 0 or i == len(new_nodes)-1):
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
                if self.ATTRACTION:
                    draw_force(n, n+(attraction_vec * self.ATTRACTION_FAC), 1)

            # compute new node optimized position
            new_node = n + (repulsion_vec * self.REPULSION_FAC)

            if self.ATTRACTION:
                new_node = new_node + (attraction_vec * self.ATTRACTION_FAC)

            # if set, add some random noise to node position
            if self.RAND_OPTIMIZATION_FAC > 0:
                new_node += (0.5 - np.random.rand(len(new_node))) * self.RAND_OPTIMIZATION_FAC

            optimized_nodes.append(new_node)

        return optimized_nodes

    def _subdivision(self, from_node, to_node, nodes):
        new_node = None

        if self.SUBDIVISION_METHOD == "BY_DISTANCE":
            dist = Morphogenesis._get_dist(from_node, to_node)
            if dist > self.SPLIT_DIST_THRESHOLD:
                # new node is halfway between the two connected ones
                new_node = (from_node + to_node) / 2
        elif self.SUBDIVISION_METHOD == "BY_CROWDEDNESS":
            neighbors_nodes = self._get_neighbors(nodes, from_node)
            if len(neighbors_nodes) < self.SPLIT_CROWD_THRESHOLD:
                # new node is halfway between the two connected ones
                new_node = (from_node + to_node) / 2
        else:
            print("No such subdivision method: {}. Exiting".format(self.SUBDIVISION_METHOD))
            sys.exit(1)

        return new_node

    def _add_node_to_index(self, node_idx, node):
        if self.index_props.dimension == 2:
            self.index.insert(node_idx, (node[0], node[1], node[0], node[1]))
        else:
            self.index.insert(node_idx, (node[0], node[1], node[2], node[0], node[1], node[2]))

    def _get_neighbors(self, nodes, pos: np.array):
        if self.container is not None:
            nodes = nodes + self.container

        left, bottom, back, right, top, front = (pos[0] - self.VISIBILITY_RADIUS,
                                                 pos[1] - self.VISIBILITY_RADIUS,
                                                 pos[2] - self.VISIBILITY_RADIUS,
                                                 pos[0] + self.VISIBILITY_RADIUS,
                                                 pos[1] + self.VISIBILITY_RADIUS,
                                                 pos[2] + self.VISIBILITY_RADIUS
                                                 )
        if self.index_props.dimension == 2:
            neighbors_nodes = np.array(nodes)[list(self.index.intersection((left, bottom, right, top)))]
        else:
            neighbors_nodes = np.array(nodes)[list(self.index.intersection((left, bottom, back, right, top, front)))]
        # neighbors_nodes = np.array(new_nodes)[list(index.nearest((n[0], n[1], n[0], n[1]), 10))]

        return neighbors_nodes

    @staticmethod
    def _get_dist(node_a: np.array, node_b: np.array) -> float:
        dist = distance.euclidean(node_a, node_b)
        return dist
