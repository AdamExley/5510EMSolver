import re
import numpy as np

from typing import List, Tuple, Union
import logging
import scipy.sparse.linalg as spla

logging.basicConfig(level=logging.INFO)

######################################################## Regular Expressions ########################################################
TRIANGLE_REGEX = r'(?P<index>\d+\s*):\s*\(\s*(?P<a>\d+)\s*,\s*(?P<b>\d+)\s*,\s*(?P<c>\d+)\s*\)\s*(\s*\((?P<eps>[+-]?([0-9]*[.])?[0-9]+)\)\s*)?'
POINT_REGEX = r"(?P<index>\d+\s*):\s*\(\s*(?P<x>[+-]?([0-9]*[.])?[0-9]+)\s*,\s*(?P<y>[+-]?([0-9]*[.])?[0-9]+)\s*\)"
PEC_REGEX = r'Edge\((\d+),(\d+)\)'

MU_0 = 4 * np.pi * 10**-7
EPS_0 = 8.854 * 10**-12

K0_SQ = MU_0 * EPS_0


############################################################# Constants #############################################################

A0 = np.ones((3, 3))

Q0 = np.array([[3, -1, -1],[-1, 1,  1], [-1,  1, 1]])
Q1 = np.array([[1, -1,  1],[-1, 3, -1], [ 1, -1, 1]])
Q2 = np.array([[1,  1, -1],[ 1, 1, -1], [-1, -1, 3]])

G0 = np.array([[ 0, -2, 2], [0, 1, -1], [ 0, 1, -1]])
G1 = np.array([[-1,  0, 1], [2, 0, -2], [-1, 0,  1]])
G2 = np.array([[ 1, -1, 0], [1, -1, 0], [-2, -2, 0]])

D0 = np.array([[ 0,  0,  0], [ 0, 1, -1], [0, -1, 1]])
D1 = np.array([[ 1,  0, -1], [ 0, 0,  0], [-1, 0, 1]])
D2 = np.array([[ 1, -1,  0], [-1, 1,  0], [ 0, 0, 0]])

E0 = np.ones((3, 3)) + np.eye(3)

############################################################# Classes #############################################################




# class Point():
#     def __init__(self, coords: np.ndarray):
#         self.coords = np.array(coords)

# class PointSet():
#     def __init__(self, points: List[Point] = []):
#         self.points = points
    
#     def __len__(self) -> int:
#         return len(self.points)



class Edge():
    def __init__(self, idx_from: int, idx_to: int):
        self.idx_from = idx_from
        self.idx_to = idx_to

    def __eq__(self, other: 'Edge') -> bool:
        return self.idx_from == other.idx_from and self.idx_to == other.idx_to
    
    def __neg__(self) -> 'Edge':
        return Edge(self.idx_to, self.idx_from)
    

class EdgeSet():
    def __init__(self, edges: List[Edge] = []):
        self.edges = edges

    def __len__(self) -> int:
        return len(self.edges)
    
    def __getitem__(self, idx: int) -> Edge:
        return self.edges[idx]
    
    def __iter__(self):
        return iter(self.edges)
    
    def __contains__(self, edge: Edge) -> bool:
        return edge in self.edges or -edge in self.edges

    def check(self, edge: Edge) -> Tuple[int, int]:
        # Return index, sign of the edge
        if edge in self.edges:
            return self.edges.index(edge), 1
        elif -edge in self.edges:
            return self.edges.index(-edge), -1
        else:
            return None, None

    def add(self, edge: Edge) -> Tuple[int, int]:
        """
        Parameters
        ----------
        edge : Edge
            Edge to add

        Returns
        -------
        tuple[int, int]
            Index, sign of the edge
        """

        retv = self.check(edge)

        if retv[0] is not None:
            return retv
        
        self.edges.append(edge)

        return len(self.edges) - 1, 1
    
    @property
    def includedPoints(self) -> set:
        return set([edge.idx_from for edge in self.edges] + [edge.idx_to for edge in self.edges])
    



class UnknownSet():
    def __init__(self, edge_set: EdgeSet, point_set: List[int]):
        self.edge_start_idx = 0
        self.point_start_idx = len(edge_set)

        self.edge_set = edge_set
        self.point_set = point_set

    def __len__(self) -> int:
        return len(self.edge_set) + len(self.point_set)
    
    def __getitem__(self, idx: int) -> Union[Edge, int]:
        if idx < len(self.edge_set):
            return self.edge_set[idx]
        else:
            return self.point_set[idx - len(self.edge_set)]
        
    def __contains__(self, item: Union[Edge, int]) -> bool:
        if isinstance(item, Edge):
            return item in self.edge_set
        else:
            return item in self.point_set
        
    def index_sign(self, item: Union[Edge, int]) -> int:
        if item not in self:
            return None, 0
        elif isinstance(item, Edge):
            return self.edge_set.check(item)
        else:
            return (self.point_set.index(item) + len(self.edge_set)), 1




class Triangle():
    def __init__(self, a, b, c, eps=1.0):
        self.pts = [a, b, c]
        self.eps = eps
        self.mu = 1

        # Create edges
        self.edges = [
            Edge(b,c),
            Edge(c,a),
            Edge(a,b)
        ]

        self.ele_mtxs = None

    @staticmethod
    def from_string(string) -> List['Triangle']:
        # Create a list of all triangle matches
        tri_iter = re.finditer(TRIANGLE_REGEX, string)
        
        # Create a list of Triangle objects
        triangles = []

        for match in tri_iter:
            a = int(match.group('a'))
            b = int(match.group('b'))
            c = int(match.group('c'))
            eps = float(match.group('eps')) if match.group('eps') else 1.0
            triangles.append(Triangle(a, b, c, eps))

        return triangles
    
    
    @staticmethod
    def uniqueEdges(triangles: 'List[Triangle]') -> EdgeSet:
        edges = EdgeSet()
        for triangle in triangles:
            for edge in triangle.edges:
                edges.add(edge)
        return edges
    
    def __repr__(self):
        return f'Triangle({self.pts[0]}, {self.pts[1]}, {self.pts[2]}) eps_r = {self.eps}'
    
    def compute_element_matrices(self, point_coords):

        # Get coordinates for each point
        a = point_coords[self.pts[0]]
        b = point_coords[self.pts[1]]
        c = point_coords[self.pts[2]]

        # Calculate the area of the triangle
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))

        # Compute cotangent of each angle
        cot_a = np.dot(b - a, c - a) / (2 * area)
        cot_b = np.dot(a - b, c - b) / (2 * area)
        cot_c = np.dot(a - c, b - c) / (2 * area)

        # Compute the element matrices
        A = (1/area) * A0
        B = (area / 12) * (cot_a * Q0 + cot_b * Q1 + cot_c * Q2)
        C = (area / 6 ) * (cot_a * G0 + cot_b * G1 + cot_c * G2)
        D = (area / 2 ) * (cot_a * D0 + cot_b * D1 + cot_c * D2)
        E = (area / 12) * E0

        # Add in scaling factors
        A = (1 / self.mu) * A - K0_SQ * self.eps * B
        B = (1 / self.mu) * B
        C = (1 / self.mu) * C
        D = (1 / self.mu) * D - K0_SQ * self.eps * E

        self.ele_mtxs = A, B, C, D

        return self.ele_mtxs
    
    def print_element_matrices(self):
        if self.ele_mtxs is None:
            print('Element matrices not yet computed')
        else:
            A, B, C, D = self.ele_mtxs
            print(f'A = {A}')
            print(f'B = {B}')
            print(f'C = {C}')
            print(f'D = {D}')


    def global_edge_info(self, global_unknowns: UnknownSet):
        return [global_unknowns.index_sign(edge) for edge in self.edges]
    
    def global_point_info(self, global_unknowns: UnknownSet):
        return [global_unknowns.index_sign(point)[0] for point in self.pts]

############################################################# Functions #############################################################



def parse_file(filename):
    with open(filename, 'r') as file:
        content = file.read()



    # Unpack points
    pts_itr = re.finditer(POINT_REGEX, content)
    points = [(int(match.group('index')), (float(match.group('x')), float(match.group('y'))) ) for match in pts_itr]
    points = sorted(points, key=lambda x: x[0])

    # Create numpy array
    pts = np.zeros((points[-1][0] + 1, 2), dtype=float)
    for index, (x, y) in points:
        pts[index] = [x, y]


    # Get all triangles
    triangles = Triangle.from_string(content)

    # Find all PEC edges
    pec_edges = re.findall(PEC_REGEX, content)
    pec_edges = EdgeSet([Edge(int(a), int(b)) for a, b in pec_edges])

    return pts, triangles, pec_edges










if __name__ == '__main__':
    # Parse the file
    points, triangles, pec_edges = parse_file('input.txt')
    logging.info(f"Parsed {len(points)} points, {len(triangles)} triangles, and {len(pec_edges)} PEC edges")

    # Create a list of all edges in the mesh
    edges = Triangle.uniqueEdges(triangles)
    logging.info(f"Found {len(edges)} unique edges")

    # Get PEC points
    pec_points = pec_edges.includedPoints
    logging.info(f"{len(pec_points)} points are included in PEC edges")


    unknown_edges : EdgeSet = EdgeSet([edge  for edge  in edges  if edge  not in pec_edges ])
    unknown_points : List[int] = [x for x in range(len(points)) if x not in pec_points]
    logging.info(f"Found {len(unknown_edges)} unknown edges and {len(unknown_points)} unknown points")


    global_unknowns = UnknownSet(unknown_edges, unknown_points)
    num_unknowns = len(global_unknowns)
    logging.info(f"{len(global_unknowns)} global unknowns")


    # Compute the element matrices for each triangle
    for triangle in triangles:
        triangle.compute_element_matrices(points)


    A_mat   = np.zeros((num_unknowns,num_unknowns))
    BCD_mat = np.zeros((num_unknowns,num_unknowns))

    for tri in triangles:
        edge_mapping = tri.global_edge_info(global_unknowns)
        point_mapping = tri.global_point_info(global_unknowns)


        # Add the A matrix
        for i, (idx, sign_i) in enumerate(edge_mapping):
            for j, (jdx, sign_j) in enumerate(edge_mapping):
                A_mat[idx, jdx] += sign_i * sign_j * tri.ele_mtxs[0][i,j]

        # Add the B matrix
        for i, (idx, sign_i) in enumerate(edge_mapping):
            for j, (jdx, sign_j) in enumerate(edge_mapping):
                BCD_mat[idx, jdx] += sign_i * sign_j * tri.ele_mtxs[1][i,j]

        # Add the C and C^T matrices
        for i, (idx, sign) in enumerate(edge_mapping):
            for j, (jdx) in enumerate(point_mapping):
                BCD_mat[idx, jdx] += sign * tri.ele_mtxs[2][i,j]
                BCD_mat[jdx, idx] += sign * tri.ele_mtxs[2][j,i]

        # Add the D matrix
        for i, (idx) in enumerate(point_mapping):
            for j, (jdx) in enumerate(point_mapping):
                BCD_mat[idx, jdx] += tri.ele_mtxs[3][i,j]


    # Suppress degenerate solutions
    eps_max = max([tri.eps for tri in triangles])
    mu_max = max([tri.mu for tri in triangles])

    theta_bound = K0_SQ * eps_max * mu_max

    P_mat = BCD_mat
    Q_mat = BCD_mat + A_mat / theta_bound

    # use Lanczos to solve the generalized eigenvalue problem
    eigvals, eigvecs = spla.eigsh(P_mat, k=1, M=Q_mat, which='LM')

    # Convert eigenvectors to wave numbers
    betas = theta_bound - theta_bound / eigvals

    print(eigvals)










