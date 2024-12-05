import re
import numpy as np

from typing import List, Tuple, Union
import logging
import scipy.sparse.linalg as spla
import scipy.linalg as la

from engineering_notation import EngNumber
from functools import lru_cache

import numpy as np
from scipy.linalg import solve
from scipy.linalg import eigh_tridiagonal
import matplotlib.pyplot as plt

# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)
# plt.ion()




######################################################## Config ########################################################

DEFAULT_NUM_SOLUTIONS = 5
NUM_RESAMPLES = 0
FREQ = 1e9


DEBUG_PRE_ELEMATS = True



######################################################## Regular Expressions ########################################################
TRIANGLE_REGEX = r'(?P<index>\d+\s*):\s*\(\s*(?P<a>\d+)\s*,\s*(?P<b>\d+)\s*,\s*(?P<c>\d+)\s*\)\s*(\s*\((?P<eps>[+-]?([0-9]*[.])?[0-9]+)\)\s*)?'
POINT_REGEX = r"(?P<index>\d+\s*):\s*\(\s*(?P<x>[+-]?([0-9]*[.])?[0-9]+)\s*,\s*(?P<y>[+-]?([0-9]*[.])?[0-9]+)\s*\)"
PEC_REGEX = r'Edge\((\d+),(\d+)\)'




############################################################# Constants #############################################################

MU_0 = 4 * np.pi * 10**-7
EPS_0 = 8.854 * 10**-12

A0 = np.ones((3, 3))

Q0 = np.array([[3, -1, -1],[-1, 1,  1], [-1,  1, 1]])
Q1 = np.array([[1, -1,  1],[-1, 3, -1], [ 1, -1, 1]])
Q2 = np.array([[1,  1, -1],[ 1, 1, -1], [-1, -1, 3]])

G0 = np.array([[ 0, -2, 2], [0, 1, -1], [ 0, 1, -1]])
G1 = np.array([[-1,  0, 1], [2, 0, -2], [-1, 0,  1]])
G2 = np.array([[ 1, -1, 0], [1, -1, 0], [-2, 2, 0]])

D0 = np.array([[ 0,  0,  0], [ 0, 1, -1], [0, -1, 1]])
D1 = np.array([[ 1,  0, -1], [ 0, 0,  0], [-1, 0, 1]])
D2 = np.array([[ 1, -1,  0], [-1, 1,  0], [ 0, 0, 0]])

E0 = np.ones((3, 3)) + np.eye(3)

############################################################# Classes #############################################################


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
    
    def remove(self, edge: Edge) -> None:
        if edge in self.edges:
            self.edges.remove(edge)
        elif -edge in self.edges:
            self.edges.remove(-edge)



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
            
    def printToFile(self, filename):
        with open(filename, 'w') as file:
            for i in range(len(self)):
                if isinstance(self[i], Edge):
                    file.write(f'{i:02d}: Edge({self[i].idx_from},{self[i].idx_to})\n')
                else:
                    file.write(f'{i:02d}: Point({self[i]})\n')




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
        logging.debug(f"Applying coordinates to triangle: {self}")
        self.coords = [point_coords[pt] for pt in self.pts]

        for edge in self.edges:
            edge.apply_coords(point_coords)

        # Get the area of the triangle
        self.area = 0.5 * np.linalg.norm(np.cross(self.coords[1] - self.coords[0], self.coords[2] - self.coords[0]))
        logging.debug(f"Area of triangle: {self.area}")

        # For each edge calculate the inward normal
        self.d_zeta = np.zeros((3, 2))

        for i, edge in enumerate(self.edges):
            a, b = edge.coords
            # Get inward-pointing unit normal
            n = np.array([b[1] - a[1], a[0] - b[0]])
            n /= np.linalg.norm(n)

            self.d_zeta[i] = n * np.linalg.norm(edge.vector) / (2 * self.area)

        logging.debug(f"Inward Normals: {self.d_zeta}")

        # Compute cotangent of each angle
        p0 = self.coords[0]
        p1 = self.coords[1]
        p2 = self.coords[2]

        # Compute cotangent of each angle
        cot_0 = np.dot(p1 - p0, p2 - p0) / (2 * self.area)
        cot_1 = np.dot(p0 - p1, p2 - p1) / (2 * self.area)
        cot_2 = np.dot(p0 - p2, p1 - p2) / (2 * self.area)

        self.cots = [cot_0, cot_1, cot_2]
        logging.debug(f"Cotangents: {self.cots}")

        # Precompute some element matrices
        self._elemat_preA = (1 / self.area) * A0
        self._elemat_preB = (1 / 12) * (self.cots[0] * Q0 + self.cots[1] * Q1 + self.cots[2] * Q2)
        self._elemat_preC = (1 / 6 ) * (self.cots[0] * G0 + self.cots[1] * G1 + self.cots[2] * G2)
        self._elemat_preD = (1 / 2 ) * (self.cots[0] * D0 + self.cots[1] * D1 + self.cots[2] * D2)
        self._elemat_preE = (self.area / 12) * E0

        if DEBUG_PRE_ELEMATS:
            logging.debug(f"Pre-Element Matrices:")
            logging.debug(f"A = {self._elemat_preA}")
            logging.debug(f"B = {self._elemat_preB}")
            logging.debug(f"C = {self._elemat_preC}")
            logging.debug(f"D = {self._elemat_preD}")
            logging.debug(f"E = {self._elemat_preE}")

        



    def compute_element_matrices(self, k_sq) -> None:
        """Compute the element matrices for the triangle

        Parameters
        ----------
        k_sq : Free Space Wavenumber (Squared)
        """

        # Add in scaling factors
        A = (1 / self.mu) * self._elemat_preA - k_sq * self.eps * self._elemat_preB
        B = (1 / self.mu) * self._elemat_preB
        C = (1 / (2*self.mu)) * self._elemat_preC
        D = (1 / self.mu) * self._elemat_preD - k_sq * self.eps * self._elemat_preE

        self.ele_mtxs = A, B, C, D

        if DEBUG_PRE_ELEMATS:
            logging.debug(f"Element Matrices:")
            logging.debug(f"A = {A}")
            logging.debug(f"B = {B}")
            logging.debug(f"C = {C}")
            logging.debug(f"D = {D}")

        return self.ele_mtxs
    

    def global_edge_info(self, global_unknowns: UnknownSet):
        return [global_unknowns.index_sign(edge) for edge in self.edges]
    
    def global_point_info(self, global_unknowns: UnknownSet):
        return [global_unknowns.index_sign(point)[0] for point in self.pts]
            

    @lru_cache(maxsize=8)
    def _barycentric(self, point: Tuple[float, float, float]) -> np.ndarray:
        """Get the barycentric coordinates of a point "in" the triangle

        Parameters
        ----------
        point : np.ndarray
            Point to calculate barycentric coordinates for

        Returns
        -------
        np.ndarray
            Barycentric coordinates
        """
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
    
    def barycentric(self, point: np.ndarray) -> np.ndarray:
        return self._barycentric(tuple(point.tolist()))

    
    def point_inside(self, point: np.ndarray) -> bool:
        """Check if a point is inside the triangle

        Parameters
        ----------
        point : np.ndarray
            Point to check

        Returns
        -------
        bool
            True if the point is inside the triangle
        """
        u, v, w = self.barycentric(point)

        mag_too_big = (u + v + w >= 1) and not np.isclose(u + v + w, 1)

        return u >= 0 and v >= 0 and w >= 0 and max(u, v, w) <= 1 and not mag_too_big
    
    def edge_interp(self, barycentric_point):
        basis_out = np.zeros((3, 2))
        zeta = barycentric_point

        for i in range(3):
            basis_out[i] = zeta[(i+1)%3] * self.d_zeta[(i+2)%3] - zeta[(i+2)%3] * self.d_zeta[(i+1)%3]

        return basis_out
    

    
    def print_element_matrices(self):
        if self.ele_mtxs is None:
            print('Element matrices not yet computed')
        else:
            A, B, C, D = self.ele_mtxs
            print(f'A = {A}')
            print(f'B = {B}')
            print(f'C = {C}')
            print(f'D = {D}')
        






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



class Solver():
    def __init__(self, points : np.ndarray, triangles : List[Triangle], pec_edges : EdgeSet) -> None:
        self.points = points
        self.triangles = triangles
        self.pec_edges = pec_edges

        # Create a list of all edges in the mesh
        self.edges = Triangle.uniqueEdges(self.triangles)
        self.edges.apply_coords(self.points)
        logging.info(f"Found {len(self.edges)} unique edges")

        # Get PEC points
        self.pec_points = self.pec_edges.includedPoints
        logging.info(f"{len(self.pec_points)} points are included in PEC edges")

        # Extract unknown edges and points
        self.unknown_edges : EdgeSet = EdgeSet([edge  for edge  in self.edges  if edge  not in self.pec_edges ])
        self.unknown_points : List[int] = [x for x in range(len(points)) if x not in self.pec_points]
        logging.info(f"Found {len(self.unknown_edges)} unknown edges and {len(self.unknown_points)} unknown points")

        # Get global unknown set
        self.global_unknowns = UnknownSet(self.unknown_edges, self.unknown_points)
        self.num_unknowns = len(self.global_unknowns)
        self.global_unknowns.printToFile('unknowns.log')
        logging.info(f"{len(self.global_unknowns)} global unknowns (saved to unknowns.log)")


        # Apply coordinates to triangles
        for triangle in self.triangles:
            triangle.apply_coords(self.points)



    def lanczos_generalized(self, A, B, num_eigenvalues):
        # n = A.shape[0]
        # Q = np.zeros((n, num_eigenvalues + 1))
        # T = np.zeros((num_eigenvalues, num_eigenvalues))
        
        # Initial vector (random or chosen based on specific conditions)
        q = np.zeros((self.num_unknowns))
        q[:len(self.unknown_edges)] = 1
        q /= np.linalg.norm(q)
        
        # Q[:, 0] = q
        
        # alpha = np.zeros(num_eigenvalues)
        # beta = np.zeros(num_eigenvalues)
        
        # for k in range(num_eigenvalues):
        #     w = A @ Q[:, k] - beta[k-1] * Q[:, k-1] if k > 0 else A @ Q[:, k]
            
        #     alpha[k] = Q[:, k].T @ w
        #     w -= alpha[k] * Q[:, k]
            
        #     for j in range(k):
        #         w -= (Q[:, j].T @ w) * Q[:, j]
            
        #     beta[k] = np.linalg.norm(w)
        #     if beta[k] < 1e-10:
        #         break
            
        #     Q[:, k+1] = w / beta[k]
            
        #     T[k, k] = alpha[k]
        #     if k < num_eigenvalues - 1:
        #         T[k, k+1] = T[k+1, k] = beta[k]
        
        # # Solve the tridiagonal eigenvalue problem
        # eigenvalues, eigenvectors = eigh_tridiagonal(T.diagonal(), T.diagonal(1))
        
        # # Construct Ritz vectors
        # ritz_vectors = Q[:, :num_eigenvalues] @ eigenvectors
        
        # return eigenvalues, ritz_vectors




    # def modified_lanczos(self, P, Q, num_eigenvalues):

    #     converged_eigenvectors = []

    #     v_vecs = []

    #     # Find a vector orthogonal to the null space of Q
    #     # Choose all edges to be 1 and all points to be 0
    #     q_vec = np.zeros((self.num_unknowns,1))
    #     q_vec[:len(self.unknown_edges)] = 1

    #     print(q_vec.shape)

    #     # Normalize the vector against converged eigenvectors
    #     for vec in converged_eigenvectors:
    #         q_vec -= vec @ q_vec @ vec.T

    #     m = 0
    #     H_mat = None

    #     print(q_vec)


    #     # Solve Q @ z_vec = q_vec for z_vec
    #     z_vec = solve(Q, q_vec)
    #     v_vecs.append(z_vec / np.linalg.norm(z_vec))
        

    #     while True:
    #         m += 1
    #         print(m)

    #         # Increase the size of the H matrix
    #         if H_mat is None:
    #             H_mat = np.zeros((2, 2))
    #         else:
    #             H_mat = np.pad(H_mat, ((0, 1), (0, 1)), mode='constant')

    #         def H_m_m():
    #             num = v_vecs[-1].T @ P @ v_vecs[-1]
    #             den = v_vecs[-1].T @ Q @ v_vecs[-1]
    #             return num / den
            
    #         def H_m1_m():
    #             if m == 1:
    #                 return 0
    #             num = v_vecs[-2].T @ P @ v_vecs[-1]
    #             den = v_vecs[-2].T @ Q @ v_vecs[-2]
    #             return num / den
            
    #         print(H_mat)

    #         # Update the H matrix
    #         H_mat[m-1, m-1] = H_m_m()
    #         if m > 1:
    #             H_mat[m-2, m] = H_m1_m()

        
    #         # Construct f vector
    #         f_vec = P @ v_vecs[-1] - H_m_m() * Q @ v_vecs[-1]
    #         if m > 1:
    #             f_vec -= H_m1_m() * v_vecs[-2]
            
    #         # Solve Q @ omega_vec = f_vec for omega_vec
    #         omega_vec = solve(Q, f_vec)

    #         # Add component to H matrix
    #         H_mat[m, m-1] = np.linalg.norm(omega_vec)

    #         # Compute the dominant eigenpair of the H matrix
    #         eigvals, eigvecs = la.eig(H_mat)
    #         dom_idx = np.argmax(np.abs(eigvals))
    #         dom_eigval = eigvals[dom_idx]
    #         dom_eigvec = eigvecs[:, dom_idx]

    #         # Check the residual norm for convergence
    #         residual = np.linalg.norm(f_vec - H_mat[m, m-1] * np.abs(dom_eigval))

    #         logging.debug(f"Residual: {residual}")

    #         if residual < 1e-10:
    #             # This converged
    #             converged_eigenvectors.append(dom_eigvec)
    #             break

                
    #     if len(converged_eigenvectors) == num_eigenvalues:
    #         return converged_eigenvectors

            






    #     # # Assign f_vector
    #     # f_vec = P @ v_vecs[-1]




    def modified_lanczos(self, P, Q, num_eigenvalues):

        NUM_ITS = 20

        # Create a matrix for the vectors
        v_vecs = np.zeros((NUM_ITS+1, self.num_unknowns))
        H_mat = np.zeros((NUM_ITS+1, NUM_ITS+1))
        
        converged_eigenvectors = []
        converged_eigenvalues = []

        # Find a vector orthogonal to the null space of Q
        # Choose all edges to be 1 and all points to be 0
        q_vec = np.zeros((self.num_unknowns,1))
        q_vec[:len(self.unknown_edges)] = 1

        print(q_vec.shape)

        # Normalize the vector against converged eigenvectors
        for vec in converged_eigenvectors:
            q_vec -= vec @ q_vec @ vec.T

        m = 0
        print(q_vec)


        # Solve Q @ phi_vec = q_vec for phi_vec
        phi_vec = solve(Q, q_vec)

        v_vecs[0] = (phi_vec / np.linalg.norm(phi_vec))[:,0]

        print(v_vecs[0])

        residuals = []
        

        for m in range(NUM_ITS):

            
            
            num = v_vecs[m].T @ P @ v_vecs[m]
            den = v_vecs[m].T @ Q @ v_vecs[m]
            H_mat[m, m] = num / den
            
            if m != 0:
                num = v_vecs[m-1].T @ P @ v_vecs[m]
                den = v_vecs[m-1].T @ Q @ v_vecs[m-1]
                H_mat[m-1, m] =  num / den
            
            # plt.imshow(H_mat != 0, cmap='viridis', interpolation='nearest')
            # plt.colorbar()
            # plt.title('Heatmap of Random Data')
            # plt.show()


        
            # Construct f vector
            f_vec = P @ v_vecs[m] - H_mat[m, m] * Q @ v_vecs[m]
            if m > 0:
                f_vec -= H_mat[m-1, m] * Q @ v_vecs[m-1]

            
            # Solve Q @ omega_vec = f_vec for omega_vec
            omega_vec = solve(Q, f_vec)

            # Add component to H matrix
            H_mat[m+1, m] = np.linalg.norm(omega_vec)

            # Compute the dominant eigenpair of the H matrix
            eigvals, eigvecs = la.eig(H_mat[:m+2, :m+2])
            dom_idx = np.argmax(np.abs(eigvals))
            dom_eigval = eigvals[dom_idx]
            dom_eigvec = eigvecs[:, dom_idx]


            print(dom_eigval)

            # Check the residual norm for convergence
            # print()
            # residual = np.linalg.norm(H_mat[:m+2, :m+2] - H_mat[m+1, m] * np.abs(dom_eigvec))


            # print((P -  dom_eigval * Q))

            # v_vecs[m+1] = f_vec - H_mat[m+1, m] * np.abs(dom_eigval)
            # print(f_vec)

            # v_vecs[m+1] = v_vecs[m+1] / np.linalg.norm(v_vecs[m+1])


            # print(omega_vec)
            # print(v_vecs[m])
            res_vec = omega_vec

            for i in range(m):
                res_vec -= (v_vecs[i].T @ res_vec) * v_vecs[i]
            

            res_vec = res_vec / np.linalg.norm(res_vec)
            # print(res_vec)


            # v_vecs[m+1] = omega_vec / np.linalg.norm(omega_vec)
            # v_vecs[m+1] = v_vecs[m] - dom_eigval * omega_vec

            v_vecs[m+1] = res_vec

            # logging.debug(f"Residual: {residual}")
            # print(f"Residual: {residual}")
            # residuals.append(residual)

            # if residual < 1e-10:
            #     # This converged
            #     converged_eigenvectors.append(dom_eigvec)
            #     break
            # else:
            #     m += 1
            if m == NUM_ITS - 1:
                dom_eigval = np.real(dom_eigval)
                converged_eigenvalues.append(dom_eigval)

                # Calculate the eigenvector associateed with the eigenvalue
                # eigenvector = P - dom_eigval * Q
                boi = P - dom_eigval * Q
                U, S, Vt = np.linalg.svd(boi)
            
                converged_eigenvectors.append(Vt[-1])

        # plt.plot(residuals)
        # plt.show()

        # converged_eigenvectors = np.array(converged_eigenvectors)

        # num_eigenvalues = 1

        # print(Q[:, :num_eigenvalues].shape)
        # print(converged_eigenvectors.shape)

        # ritz_vectors = Q[:, :num_eigenvalues] @ converged_eigenvectors
        return np.array(converged_eigenvalues), np.array(converged_eigenvectors).T
                
            
            




    def solve(self, freq : float, n_solutions : int = DEFAULT_NUM_SOLUTIONS) -> Tuple[np.ndarray, np.ndarray]:
        """Solve at a specific frequency

        Parameters
        ----------
        freq : float
            Frequency to solve at

        Returns
        -------
        np.ndarray, np.ndarray
            Propagation constants and eigenvectors
        """

        # Get the wavenumber
        ang_freq = 2 * np.pi * freq
        k_sq = MU_0 * EPS_0 * ang_freq**2

        # Compute the element matrices for each triangle
        for triangle in self.triangles:
            triangle.compute_element_matrices(k_sq)


        # Allocate matrices                
        A_mat   = np.zeros((self.num_unknowns,self.num_unknowns))
        BCD_mat = np.zeros((self.num_unknowns,self.num_unknowns))

        # Map triangle element matrices to global matrices
        for tri in self.triangles:
            edge_mapping = tri.global_edge_info(self.global_unknowns)
            point_mapping = tri.global_point_info(self.global_unknowns)

            print("\n\n")
            print(edge_mapping)
            print(point_mapping)

            # Add the A matrix
            for i, (idx, sign_i) in enumerate(edge_mapping):
                for j, (jdx, sign_j) in enumerate(edge_mapping):
                    if idx is None or jdx is None:
                        continue
                    A_mat[idx, jdx] += sign_i * sign_j * tri.ele_mtxs[0][i,j]
                    logging.debug(f"Adding {sign_i * sign_j * tri.ele_mtxs[0][i,j]} to A_mat[{idx},{jdx}]")

            # Add the B matrix
            for i, (idx, sign_i) in enumerate(edge_mapping):
                for j, (jdx, sign_j) in enumerate(edge_mapping):
                    if idx is None or jdx is None:
                        continue
                    BCD_mat[idx, jdx] += sign_i * sign_j * tri.ele_mtxs[1][i,j]
                    logging.debug(f"Adding {sign_i * sign_j * tri.ele_mtxs[1][i,j]} to BCD_mat[{idx},{jdx}]")

            # Add the C and C^T matrices
            for i, (idx, sign) in enumerate(edge_mapping):
                for j, (jdx) in enumerate(point_mapping):
                    if idx is None or jdx is None:
                        continue

                    # # idx is the edge, jdx is the point

                    BCD_mat[idx, jdx] += sign * tri.ele_mtxs[2][i,j]
                    BCD_mat[jdx, idx] += sign * tri.ele_mtxs[2][j,i]

                    # BCD_mat[idx, jdx] += sign * tri.ele_mtxs[2][j,i]
                    # BCD_mat[jdx, idx] += sign * tri.ele_mtxs[2][i,j]
                    logging.debug(f"Adding {sign * tri.ele_mtxs[2][i,j]} to BCD_mat[{idx},{jdx}]")

            # Add the D matrix
            for i, (idx) in enumerate(point_mapping):
                for j, (jdx) in enumerate(point_mapping):
                    if idx is None or jdx is None:
                        continue
                    BCD_mat[idx, jdx] += tri.ele_mtxs[3][i,j]
                    logging.debug(f"Adding {tri.ele_mtxs[3][i,j]} to BCD_mat[{idx},{jdx}]")


        print(A_mat)
        print(BCD_mat)

        # Suppression of degenerate solutions
        eps_max = max([tri.eps for tri in triangles])
        mu_max = max([tri.mu for tri in triangles])
        theta_bound = k_sq * eps_max * mu_max
        logging.info(f"Theta bound^2 = {theta_bound} sqrt=({np.sqrt(theta_bound)})")


        # Try to solve problem by suppressing degenerate solutions
        # try:

        # Modified matricies to solve for
        P_mat = BCD_mat
        Q_mat = BCD_mat + A_mat / theta_bound


        # plt.imshow(np.log10(np.abs(BCD_mat)), cmap='viridis', interpolation='nearest')
        # plt.colorbar()
        # plt.show()



        # self.modified_lanczos(P_mat, Q_mat, n_solutions)
        eigvals, eigvecs = self.modified_lanczos(P_mat, Q_mat, n_solutions)

        # eigvals, eigvecs = self.lanczos_generalized(P_mat, Q_mat, n_solutions)

        # q_vec = np.zeros((self.num_unknowns,1))
        # q_vec[:len(self.unknown_edges)] = 1
        # q_vec /= np.linalg.norm(q_vec)
        # phi_vec = solve(Q_mat, q_vec)
        # phi_vec /= np.linalg.norm(phi_vec)


        # eigvals, eigvecs = spla.eigsh(P_mat, k=n_solutions, M=Q_mat, which='LM')#, sigma=100)
        # eigvals, eigvecs = spla.eigsh(P_mat, k=2, M=Q_mat, which='LA')#, sigma=100)
        # eigvals, eigvecs = la.eigh(P_mat, Q_mat)

        # eigvals, eigvecs = spla.eigsh(A_mat, k=2, M=BCD_mat, which='LA', sigma=25)

        

        print(eigvecs.shape)




            
        print(eigvals)
        # betas = np.sqrt(theta_bound - theta_bound / eigvals)
        betas = np.sqrt(theta_bound * (eigvals - 1 ) / eigvals)
        print(betas)
        # except Exception as e:  

        #     try:
        #         # If the above fails, try to solve without the suppression.
        #         eigvals, eigvecs = spla.eigsh(A_mat, k=n_solutions, M=BCD_mat, which='LM')#, sigma=theta_bound)
        #         betas = np.sqrt(eigvals)
        #         logging.error(f"Failed to solve with suppression @ {EngNumber(freq)}Hz")
        #     except Exception as e:
        #         logging.error(f"Failed to solve @ {EngNumber(freq)}Hz")
        #         return None, None

        # logging.info(f"Solved @ {EngNumber(freq)}Hz: {betas}  (Free Space: {np.sqrt(k_sq)})")

        self.betas = betas
        self.eigvecs = eigvecs

        return betas, eigvecs





class Program():
    def __init__(self, solver : Solver):
        self.solver = solver
        self.is_setup = False
        self.has_run_once = False


    def setup(self, grid_size : int = 25):
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')

        points = self.solver.points

        # Plot PEC edges
        for edge in self.solver.pec_edges:
            a, b = points[edge.idx_from], points[edge.idx_to]
            self.ax.plot([a[0], b[0]], [a[1], b[1]], 'k', lw=2)

        # Plot unknown edges
        for edge in self.solver.unknown_edges:
            a, b = points[edge.idx_from], points[edge.idx_to]
            self.ax.arrow(a[0], a[1], b[0] - a[0], b[1] - a[1],length_includes_head=True, width=.0001, head_width=.0005, head_length=.001, ls="")

        
        self.xs = np.linspace(points[:,0].min(), points[:,0].max(), grid_size)
        self.ys = np.linspace(points[:,1].min(), points[:,1].max(), grid_size)

        self.xp, self.yp = np.meshgrid(self.xs, self.ys)

        self.mag = np.zeros_like(self.xp)
        self.x_dir = np.zeros_like(self.xp)
        self.y_dir = np.zeros_like(self.yp)

        self.is_setup = True

    def run(self, freq : float):
        if not self.is_setup:
            self.setup()

        betas, eigvecs = self.solver.solve(freq)

            
        # Choose a single eigenvector to plot
        eignv = eigvecs[:,0]


        # Go through the grid and calculate the field at each point
        for i, x in enumerate(self.xs):
            for j, y in enumerate(self.ys):
                point = np.array([x, y])

                tri_found = False

                # Find a triangle that contains the point
                for tri in self.solver.triangles:

                    if not tri.point_inside(point):
                        continue

                    # Add the field from each edge
                    field = np.zeros(2)
                    edge_basis = tri.edge_interp(tri.barycentric(point))

                    for k, edge in enumerate(tri.edges):
                        # Get the global index of the edge
                        idx, sign = solver.global_unknowns.index_sign(edge)

                        # Skip if the edge is not in the unknown set
                        if idx is None:
                            continue

                        # Construct the field
                        field += edge_basis[k] * eignv[idx] * sign


                    self.mag[j,i] = np.linalg.norm(field)

                    # Size arrow according to log10 of magnitude
                    field_show = np.log10(self.mag[j,i]) * field / self.mag[j,i]

                    self.x_dir[j,i] = field_show[0]
                    self.y_dir[j,i] = field_show[1]
                    tri_found = True
                    break
                
                if not tri_found:
                    logging.warning(f"Triangle assignment not found for {point}")
                    for tri in self.solver.triangles:
                        bary = tri.barycentric(point)

                        bary_sum = np.sum(bary)

                        print(f"Triangle {tri} Barycentric: {bary} Sum: {bary_sum}")



        if not self.has_run_once:
            # Plot for the first time
                    
            # Plot and add a colorbar
            self.plot = self.ax.quiver(self.xp, self.yp, self.x_dir, self.y_dir, np.log10(self.mag + 0.001), cmap='viridis', pivot='mid')
            self.cbar = plt.colorbar(self.plot)

            # Change cbar
            self.cbar_min = np.floor(self.cbar.get_ticks().min())
            self.cbar_max = np.ceil(self.cbar.get_ticks().max()) + 0.5

            self.cbar.set_ticks(np.arange(self.cbar_min, self.cbar_max, 0.5))
            self.cbar.set_ticklabels(['$10^{'+f"{i}"+'}$' for i in np.arange(self.cbar_min, self.cbar_max, 0.5)])

            self.cbar.set_label('Electric Field Magnitude (V/m)')

            # Make a custom legend
            self.ax.legend()

            # Custom legend 
            legend_elements = [
                plt.Line2D([0], [0], color='black', lw=2, label='PEC'),
                plt.Line2D([0], [0], color='blue', lw=1, label='Edge') 
                ]
            
            self.ax.legend(handles=legend_elements, loc='upper right')

            self.has_run_once = True

        else:
            # Update plot
            self.plot.set_UVC(self.x_dir, self.y_dir, np.log10(self.mag + 0.001))
            self.cbar.set_ticks(np.arange(self.cbar_min, self.cbar_max, 0.5))
            self.cbar.set_ticklabels(['$10^{'+f"{i}"+'}$' for i in np.arange(self.cbar_min, self.cbar_max, 0.5)])

        plt.show()


    



def resample(points: np.ndarray, triangles : List[Triangle], pec_edges):
    new_triangles = []

    # Apply coordinates to edges


    for tri in triangles:
        tri.apply_coords(points)

        # Find the longest edge of the triangle
        longest_edge_idx = np.argmax([np.linalg.norm(x.vector) for x in tri.edges])

        longest_edge : Edge = tri.edges[longest_edge_idx]
        is_pec = longest_edge in pec_edges

        # Get a point that bisects the longest edge
        new_point = (points[longest_edge.idx_from] + points[longest_edge.idx_to]) / 2

        # Check if the new point already exists
        new_idx = None
        for i, point in enumerate(points):
            if np.allclose(point, new_point):
                new_idx = i
                break

        # If the point doesn't exist, add it
        if new_idx is None:
            points = np.vstack([points, new_point])
            new_idx = len(points) - 1

        # Create new triangles, making sure to obey right hand wrapping order
        new_triangles.append(Triangle(tri.pts[longest_edge_idx], tri.pts[(longest_edge_idx + 1) % 3], new_idx, tri.eps))
        new_triangles.append(Triangle(tri.pts[longest_edge_idx], new_idx, tri.pts[(longest_edge_idx + 2) % 3], tri.eps))

        # Modify PEC edges
        if is_pec:
            pec_edges.remove(longest_edge)
            pec_edges.add(Edge(longest_edge.idx_from, new_idx))
            pec_edges.add(Edge(new_idx, longest_edge.idx_to))
  
    
    return points, new_triangles, pec_edges

    






if __name__ == '__main__':
    # Parse the file
    points, triangles, pec_edges = parse_file('debug_input.txt')
    # points, triangles, pec_edges = parse_file('input.txt')
    logging.info(f"Parsed {len(points)} points, {len(triangles)} triangles, and {len(pec_edges)} PEC edges")

    for i in range(NUM_RESAMPLES):
        points, triangles, pec_edges = resample(points, triangles, pec_edges)


    target_ksq = 6
    freq = np.sqrt(target_ksq / (MU_0 * EPS_0)) / (2 * np.pi)

    solver = Solver(points, triangles, pec_edges)
    # betas, eigvecs = solver.solve(1e9)
    betas, eigvecs = solver.solve(freq)

    # p = Program(solver)
    # p.run(FREQ)
   












