import re
import numpy as np

from typing import List, Tuple, Union
import logging
import scipy.sparse.linalg as spla
import scipy.linalg as la


logging.basicConfig(level=logging.INFO)

######################################################## Regular Expressions ########################################################
TRIANGLE_REGEX = r'(?P<index>\d+\s*):\s*\(\s*(?P<a>\d+)\s*,\s*(?P<b>\d+)\s*,\s*(?P<c>\d+)\s*\)\s*(\s*\((?P<eps>[+-]?([0-9]*[.])?[0-9]+)\)\s*)?'
POINT_REGEX = r"(?P<index>\d+\s*):\s*\(\s*(?P<x>[+-]?([0-9]*[.])?[0-9]+)\s*,\s*(?P<y>[+-]?([0-9]*[.])?[0-9]+)\s*\)"
PEC_REGEX = r'Edge\((\d+),(\d+)\)'


FREQ = 1e9

OMEGA = 2 * np.pi * FREQ

MU_0 = 4 * np.pi * 10**-7
EPS_0 = 8.854 * 10**-12

K0_SQ = MU_0 * EPS_0 * OMEGA**2


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

        self.coords = None
        self.vector = None

    def __eq__(self, other: 'Edge') -> bool:
        return self.idx_from == other.idx_from and self.idx_to == other.idx_to
    
    def __neg__(self) -> 'Edge':
        return Edge(self.idx_to, self.idx_from)
    
    def apply_coords(self, coords: np.ndarray):
        self.coords = np.array([coords[self.idx_from], coords[self.idx_to]])
        self.vector = self.coords[1] - self.coords[0]
    

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
    
    def apply_coords(self, coords: np.ndarray):
        for edge in self.edges:
            edge.apply_coords(coords)
    



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
            if item in self.point_set:
                return self.point_set.index(item) + len(self.edge_set), 1
            else:
                return None, 0




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
        self.coords = None
        self.d_zeta = None
        self.area = None

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
    
    def apply_coords(self, point_coords):
        self.coords = [point_coords[pt] for pt in self.pts]

        for edge in self.edges:
            edge.apply_coords(point_coords)

        # Get the area of the triangle
        self.area = 0.5 * np.linalg.norm(np.cross(self.coords[1] - self.coords[0], self.coords[2] - self.coords[0]))

        # For each edge calculate the inward normal
        self.d_zeta = np.zeros((3, 2))

        for i, edge in enumerate(self.edges):
            a, b = edge.coords
            # Get inward-pointing unit normal
            n = np.array([b[1] - a[1], a[0] - b[0]])
            n /= np.linalg.norm(n)

            self.d_zeta[i] = n * np.linalg.norm(edge.vector) / (2 * self.area)


    def compute_element_matrices(self):

        # Get coordinates for each point
        p0 = self.coords[0]
        p1 = self.coords[1]
        p2 = self.coords[2]

        # Compute cotangent of each angle
        cot_0 = np.dot(p1 - p0, p2 - p0) / (2 * self.area)
        cot_1 = np.dot(p0 - p1, p2 - p1) / (2 * self.area)
        cot_2 = np.dot(p0 - p2, p1 - p2) / (2 * self.area)

        # Compute the element matrices
        A = (1/self.area) * A0
        B = (1 / 12) * (cot_0 * Q0 + cot_1 * Q1 + cot_2 * Q2)
        C = (1 / 6 ) * (cot_0 * G0 + cot_1 * G1 + cot_2 * G2)
        D = (1 / 2 ) * (cot_0 * D0 + cot_1 * D1 + cot_2 * D2)
        E = (self.area / 12) * E0

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
    
    def barycentric(self, point):
        # Get the barycentric coordinates of a point by taking the sub-triangle areas
        bary_point = np.zeros(3)

        # Compute the area of the sub-triangles
        for i in range(3):
            a = point
            b = self.coords[(i + 1) % 3]
            c = self.coords[(i + 2) % 3]

            # Compute the area of the sub-triangle
            area_point = 0.5 * np.linalg.norm(np.cross(b - a, c - a))

            bary_point[i] = area_point / self.area


        return bary_point
    
    def edge_interp(self, barycentric_point):
        basis_out = np.zeros((3, 2))
        zeta = barycentric_point

        for i in range(3):
            basis_out[i] = zeta[(i+1)%3] * self.d_zeta[(i+2)%3] - zeta[(i+2)%3] * self.d_zeta[(i+1)%3]

        return basis_out
        






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
    points, triangles, pec_edges = parse_file('input2.txt')
    logging.info(f"Parsed {len(points)} points, {len(triangles)} triangles, and {len(pec_edges)} PEC edges")

    # Create a list of all edges in the mesh
    edges = Triangle.uniqueEdges(triangles)
    edges.apply_coords(points)
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
        triangle.apply_coords(points)
        triangle.compute_element_matrices()


    A_mat   = np.zeros((num_unknowns,num_unknowns))
    BCD_mat = np.zeros((num_unknowns,num_unknowns))

    for tri in triangles:
        edge_mapping = tri.global_edge_info(global_unknowns)
        point_mapping = tri.global_point_info(global_unknowns)


        # Add the A matrix
        for i, (idx, sign_i) in enumerate(edge_mapping):
            for j, (jdx, sign_j) in enumerate(edge_mapping):
                if idx is None or jdx is None:
                    continue
                A_mat[idx, jdx] += sign_i * sign_j * tri.ele_mtxs[0][i,j]
                logging.info(f"Adding {sign_i * sign_j * tri.ele_mtxs[0][i,j]} to A_mat[{idx},{jdx}]")

        # Add the B matrix
        for i, (idx, sign_i) in enumerate(edge_mapping):
            for j, (jdx, sign_j) in enumerate(edge_mapping):
                if idx is None or jdx is None:
                    continue
                BCD_mat[idx, jdx] += sign_i * sign_j * tri.ele_mtxs[1][i,j]
                logging.info(f"Adding {sign_i * sign_j * tri.ele_mtxs[1][i,j]} to BCD_mat[{idx},{jdx}]")

        # Add the C and C^T matrices
        for i, (idx, sign) in enumerate(edge_mapping):
            for j, (jdx) in enumerate(point_mapping):
                if idx is None or jdx is None:
                    continue
                BCD_mat[idx, jdx] += sign * tri.ele_mtxs[2][i,j]
                BCD_mat[jdx, idx] += sign * tri.ele_mtxs[2][j,i]
                logging.info(f"Adding {sign * tri.ele_mtxs[2][i,j]} to BCD_mat[{idx},{jdx}]")

        # Add the D matrix
        for i, (idx) in enumerate(point_mapping):
            for j, (jdx) in enumerate(point_mapping):
                if idx is None or jdx is None:
                    continue
                BCD_mat[idx, jdx] += tri.ele_mtxs[3][i,j]
                logging.info(f"Adding {tri.ele_mtxs[3][i,j]} to BCD_mat[{idx},{jdx}]")


    # Suppress degenerate solutions
    eps_max = max([tri.eps for tri in triangles])
    mu_max = max([tri.mu for tri in triangles])

    theta_bound = K0_SQ * eps_max * mu_max
    logging.info(f"Theta bound = {theta_bound}")

    P_mat = BCD_mat
    Q_mat = BCD_mat + A_mat / theta_bound

    # Solve eigenvalue problem
    try:
        eigvals, eigvecs = spla.eigsh(P_mat, k=2, M=Q_mat, which='LM')  
        betas = theta_bound - theta_bound / eigvals
    except Exception as e:  
        eigvals, eigvecs = spla.eigsh(A_mat, k=2, M=BCD_mat, which='LM', sigma=theta_bound)
        betas = eigvals

    # Convert eigenvectors to wave numbers
    print(betas)

    # Use matplotlib to plot the results
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # Plot PEC edges
    for edge in pec_edges:
        a, b = points[edge.idx_from], points[edge.idx_to]
        ax.plot([a[0], b[0]], [a[1], b[1]], 'k', lw=2)

    # Plot unknown edges
    for edge in unknown_edges:
        a, b = points[edge.idx_from], points[edge.idx_to]
        # ax.plot([a[0], b[0]], [a[1], b[1]], 'k:')
        ax.arrow(a[0], a[1], b[0] - a[0], b[1] - a[1],length_includes_head=True, width=.0001, head_width=.0005, head_length=.001, ls="")

    
    xs = np.linspace(points[:,0].min(), points[:,0].max(), 25)
    ys = np.linspace(points[:,1].min(), points[:,1].max(), 25)

    xp, yp = np.meshgrid(xs, ys)

    mag = np.zeros_like(xp)

    x_dir = np.zeros_like(xp)
    y_dir = np.zeros_like(yp)


    # Choose a single eigenvector to plot
    print(eigvecs.shape)
    eignv = eigvecs[:,0]


    # Go through the grid and calculate the field at each point
    for i, x in enumerate(xs):

        for j, y in enumerate(ys):
            point = np.array([x, y])

            # Find a triangle that contains the point
            for tri in triangles:
                u, v, w = tri.barycentric(point)
                if u < 0 or v < 0 or w < 0:
                    continue

                if u > 1 or v > 1 or w > 1:
                    continue

                if u + v + w > 1:
                    continue

                # Add the field from each edge
                field = np.zeros(2)

                edge_basis = tri.edge_interp([u, v, w])

                # print(edge_basis)

                for k, edge in enumerate(tri.edges):
                    # if edge in pec_edges:
                    #     continue

                    # Get the global index of the edge
                    idx, sign = global_unknowns.index_sign(edge)

                    if idx is None:
                        continue

                    # Construct the field
                    field += edge_basis[k] * eignv[idx] * sign


                mag[j,i] = np.linalg.norm(field)

                x_dir[j,i] = field[0]
                y_dir[j,i] = field[1]
                break

    # Plot the field
    # ax.quiver(xp, yp, x_dir, y_dir, mag, cmap='viridis', pivot='mid')

    # Add a colorbar
    cbar = plt.colorbar(ax.quiver(xp, yp, x_dir, y_dir, np.log10(mag + 0.001), cmap='viridis', pivot='mid'))

    # Change cbar
    cbar_min = np.floor(cbar.get_ticks().min())
    cbar_max = np.ceil(cbar.get_ticks().max()) + 0.5

    cbar.set_ticks(np.arange(cbar_min, cbar_max, 0.5))
    cbar.set_ticklabels(['$10^{'+f"{i}"+'}$' for i in np.arange(cbar_min, cbar_max, 0.5)])

    cbar.set_label('Electric Field Magnitude (V/m)')


    plt.show()












