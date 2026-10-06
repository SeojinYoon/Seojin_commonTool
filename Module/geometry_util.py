import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Set
import networkx as nx
from shapely.geometry import Polygon, Point, MultiPoint
import shapely.prepared
import numpy as np
import trimesh

class Plane:
    def __init__(self, origin, normal):
        self.origin = np.array(origin, dtype=float)
        normal = np.array(normal, dtype=float)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            raise ValueError("Normal vector cannot be a zero vector.")
        self.normal = normal / norm

    def distance_to_point(self, point):
        """Signed perpendicular distance from point to the plane."""
        return np.dot(np.array(point) - self.origin, self.normal)

    def project_point(self, point):
        """Orthogonal projection of a point onto the plane."""
        p = np.array(point)
        return p - self.distance_to_point(p) * self.normal

    def intersect_segment(self, p1, p2, tol=1e-7):
        """
        Calculates the intersection between line segment p1-p2 and the plane.
        Returns the 3D intersection point, or None if parallel or disjoint.
        """
        v = p2 - p1
        denom = np.dot(self.normal, v)
        if np.abs(denom) < tol:
            return None
        
        t = np.dot(self.normal, self.origin - p1) / denom
        if 0.0 <= t <= 1.0:
            return p1 + t * v
        return None

    def intersect_box(self, box_min, box_max, tol=1e-5):
        """
        Finds the intersection polygon vertices between an Axis-Aligned Bounding Box (AABB) and the plane.
        
        :param box_min: [xmin, ymin, zmin]
        :param box_max: [xmax, ymax, zmax]
        :param tol: Numerical tolerance for deduplicating vertices
        :return: Ordered 3D vertices forming the polygon in planar winding order (N, 3), 
                 or empty array (0, 3) if no intersection exists.
        """
        xmin, ymin, zmin = box_min
        xmax, ymax, zmax = box_max

        # 8 bounding box corners
        corners = np.array([
            [xmin, ymin, zmin], [xmax, ymin, zmin],
            [xmax, ymax, zmin], [xmin, ymax, zmin],
            [xmin, ymin, zmax], [xmax, ymin, zmax],
            [xmax, ymax, zmax], [xmin, ymax, zmax]
        ])

        # 12 edge pairs connecting the corners
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical pillars
        ]

        # 1. Collect all segment-plane intersections
        raw_intersections = []
        for i1, i2 in edges:
            pt = self.intersect_segment(corners[i1], corners[i2])
            if pt is not None:
                raw_intersections.append(pt)

        if not raw_intersections:
            return np.empty((0, 3))

        # 2. Deduplicate vertices (e.g., when the plane precisely touches a box corner)
        unique_pts = []
        for pt in raw_intersections:
            if not any(np.allclose(pt, u, atol=tol) for u in unique_pts):
                unique_pts.append(pt)
        unique_pts = np.array(unique_pts)

        if len(unique_pts) < 3:
            return unique_pts

        # 3. Sort vertices circularly around the polygon centroid on the plane
        center = unique_pts.mean(axis=0)

        # Construct an orthonormal planar basis (u, v)
        u_axis = unique_pts[0] - center
        u_axis /= np.linalg.norm(u_axis)
        v_axis = np.cross(self.normal, u_axis)

        # Compute polar angles and sort indices
        coords_u = np.dot(unique_pts - center, u_axis)
        coords_v = np.dot(unique_pts - center, v_axis)
        angles = np.arctan2(coords_v, coords_u)

        return unique_pts[np.argsort(angles)]

def _to_trimesh(mesh: Any) -> trimesh.Trimesh:
    """Convert input to a trimesh.Trimesh object if not already one."""
    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    if hasattr(mesh, "vertices") and hasattr(mesh, "faces") and not hasattr(mesh, "data_vars"):
        return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    # Check if xarray Dataset from make_mesh_ds
    if hasattr(mesh, "data_vars") and "vertices" in mesh and "faces" in mesh.attrs:
        v = mesh["vertices"].values
        if v.ndim == 3:
            v = v[0]
        f = mesh.attrs["faces"]
        return trimesh.Trimesh(vertices=v, faces=f, process=False)
    raise TypeError(f"Unsupported mesh type: {type(mesh)}")


def slice_mesh_by_cutting_edges(
    mesh: Any,
    cutting_edges: Union[List, np.ndarray],
    target_pos: Optional[Union[List[float], np.ndarray]] = None,
    plane_normal: Optional[Union[List[float], np.ndarray]] = None,
    return_split: bool = False,
) -> Union[trimesh.Trimesh, Tuple[trimesh.Trimesh, trimesh.Trimesh]]:
    """Slice a 3D mesh locally along user-configured cutting_edges (e.g. bounded quad/polygon),
    preventing far-away body parts (e.g. legs, feet, torso) from being cut off by an infinite plane.

    :param mesh: Input trimesh.Trimesh, or xarray.Dataset (from make_mesh_ds).
    :param cutting_edges: (N, 3) array of 3D vertices defining the cutting polygon/quad (e.g. cut_edges).
    :param target_pos: Optional 3D point (e.g. elbow_pos) inside the target segment to extract.
                       If provided, the mesh component containing this point is selected.
    :param plane_normal: Optional normal vector for the cutting plane. If None, it is estimated
                         from cutting_edges, and oriented towards target_pos if target_pos is given.
    :param return_split: If True, returns (target_mesh, remaining_mesh). If False, returns target_mesh.
    :return: Sliced trimesh.Trimesh (or tuple of trimesh.Trimesh if return_split=True).
    """
    mesh = _to_trimesh(mesh)
    cutting_edges = np.asanyarray(cutting_edges, dtype=np.float64)
    origin = cutting_edges.mean(axis=0)

    # Determine plane normal
    if plane_normal is None:
        centered = cutting_edges - origin
        _, _, vh = np.linalg.svd(centered)
        normal = vh[-1]
    else:
        normal = np.asanyarray(plane_normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)

    # If target_pos is provided, ensure normal points towards target_pos
    if target_pos is not None:
        target_pos = np.asanyarray(target_pos, dtype=np.float64)
        if np.dot(normal, target_pos - origin) < 0:
            normal = -normal

    # Find intersection of mesh with the plane
    lines, face_indices = trimesh.intersections.mesh_plane(
        mesh, plane_normal=normal, plane_origin=origin, return_faces=True
    )
    if len(lines) == 0:
        if return_split:
            return mesh.copy(), trimesh.Trimesh()
        return mesh.copy()

    # Create 2D coordinate system on the cutting plane
    if abs(normal[0]) < 0.9:
        u = np.cross(normal, [1.0, 0.0, 0.0])
    else:
        u = np.cross(normal, [0.0, 1.0, 0.0])
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    # Project cutting_edges to 2D polygon
    poly_2d_pts = [(np.dot(p - origin, u), np.dot(p - origin, v)) for p in cutting_edges]
    poly_2d = Polygon(poly_2d_pts)
    if not poly_2d.is_valid:
        poly_2d = MultiPoint(poly_2d_pts).convex_hull
    # Add a small buffer for numerical tolerance
    poly_2d_buffered = poly_2d.buffer(1e-4)
    prep_poly = shapely.prepared.prep(poly_2d_buffered)

    # Filter intersection lines that fall inside the polygon
    midpoints = lines.mean(axis=1)
    in_poly_mask = np.zeros(len(lines), dtype=bool)
    for i, m in enumerate(midpoints):
        pt_2d = Point(np.dot(m - origin, u), np.dot(m - origin, v))
        if prep_poly.contains(pt_2d):
            in_poly_mask[i] = True

    target_face_indices = np.unique(face_indices[in_poly_mask])
    if len(target_face_indices) == 0:
        if return_split:
            return mesh.copy(), trimesh.Trimesh()
        return mesh.copy()

    # Clip triangles in target_face_indices against the plane
    def clip_triangle_plane(v0, v1, v2, p_orig, p_norm):
        verts = [v0, v1, v2]
        dots = [np.dot(vert - p_orig, p_norm) for vert in verts]
        pos_idx = [i for i, d in enumerate(dots) if d >= 0]
        neg_idx = [i for i, d in enumerate(dots) if d < 0]
        if len(pos_idx) == 3: return [np.array([v0, v1, v2])], []
        if len(neg_idx) == 3: return [], [np.array([v0, v1, v2])]
        def intersect(va, vb, da, db):
            t = da / (da - db)
            return va + t * (vb - va)
        if len(pos_idx) == 1:
            i0, i1, i2 = pos_idx[0], (pos_idx[0] + 1) % 3, (pos_idx[0] + 2) % 3
            p01 = intersect(verts[i0], verts[i1], dots[i0], dots[i1])
            p02 = intersect(verts[i0], verts[i2], dots[i0], dots[i2])
            return [np.array([verts[i0], p01, p02])], [np.array([p01, verts[i1], verts[i2]]), np.array([p01, verts[i2], p02])]
        else:
            i0, i1, i2 = neg_idx[0], (neg_idx[0] + 1) % 3, (neg_idx[0] + 2) % 3
            p01 = intersect(verts[i0], verts[i1], dots[i0], dots[i1])
            p02 = intersect(verts[i0], verts[i2], dots[i0], dots[i2])
            return [np.array([p01, verts[i1], verts[i2]]), np.array([p01, verts[i2], p02])], [np.array([verts[i0], p01, p02])]

    pos_cut_tris = []
    neg_cut_tris = []
    for fi in target_face_indices:
        f = mesh.faces[fi]
        v0, v1, v2 = mesh.vertices[f[0]], mesh.vertices[f[1]], mesh.vertices[f[2]]
        ptris, ntris = clip_triangle_plane(v0, v1, v2, origin, normal)
        pos_cut_tris.extend(ptris)
        neg_cut_tris.extend(ntris)

    # Face adjacency graph excluding cut faces to partition the untouched faces
    adj_faces = mesh.face_adjacency
    mask = ~np.isin(adj_faces[:, 0], target_face_indices) & ~np.isin(adj_faces[:, 1], target_face_indices)
    valid_edges = adj_faces[mask]

    g = nx.Graph()
    g.add_nodes_from(range(len(mesh.faces)))
    g.add_edges_from(valid_edges)
    for f in target_face_indices:
        g.remove_node(f)

    # Identify target component
    if target_pos is not None:
        face_centers = mesh.triangles_center
        dist = np.linalg.norm(face_centers - target_pos, axis=1)
        seed_face = np.argmin(dist)
        target_connected_faces = set(nx.node_connected_component(g, seed_face))
    else:
        # Without target_pos, find components on positive normal side
        target_connected_faces = set()
        for comp in nx.connected_components(g):
            comp_list = list(comp)
            centers = mesh.triangles_center[comp_list]
            if np.mean(np.dot(centers - origin, normal)) > 0:
                target_connected_faces.update(comp_list)

    target_untouched_tris = mesh.vertices[mesh.faces[list(target_connected_faces)]]
    all_target_tris = np.vstack([target_untouched_tris, np.array(pos_cut_tris)])
    target_mesh = trimesh.Trimesh(**trimesh.triangles.to_kwargs(all_target_tris), process=True)

    if not return_split:
        return target_mesh

    # Remaining mesh
    remaining_faces = [i for i in range(len(mesh.faces)) if i not in target_face_indices and i not in target_connected_faces]
    remaining_untouched_tris = mesh.vertices[mesh.faces[remaining_faces]]
    all_remaining_tris = np.vstack([remaining_untouched_tris, np.array(neg_cut_tris)])
    remaining_mesh = trimesh.Trimesh(**trimesh.triangles.to_kwargs(all_remaining_tris), process=True)

    return target_mesh, remaining_mesh


# Alias
slice_mesh_by_polygon = slice_mesh_by_cutting_edges


if __name__ == "__main__":
    plane = Plane(origin=[0.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0])
    
    box_min = [-1.0, -1.0, -1.0]
    box_max = [1.0, 1.0, 1.0]
    
    polygon = plane.intersect_box(box_min, box_max)
    
    print(f"#polygon: {len(polygon)}")
    print("coord:\n", polygon)
    

def find_boundary_loops(faces: np.ndarray) -> List[List[int]]:
    """Find all ordered boundary loops of a triangle mesh.
    
    :param faces: (F, 3) triangle array.
    :return: List of lists, each containing ordered vertex indices along a boundary loop.
    """
    edge_counts = {}
    for f in faces:
        for i in range(3):
            u, v = f[i], f[(i + 1) % 3]
            edge_key = (min(u, v), max(u, v))
            edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1
    boundary_edges = {k for k, count in edge_counts.items() if count == 1}
    bnd_graph = nx.DiGraph()
    for f in faces:
        for i in range(3):
            u, v = f[i], f[(i + 1) % 3]
            if (min(u, v), max(u, v)) in boundary_edges:
                bnd_graph.add_edge(u, v)

    loops = []
    visited = set()
    for node in list(bnd_graph.nodes()):
        if node not in visited:
            loop = []
            curr = node
            while curr not in visited and curr in bnd_graph:
                visited.add(curr)
                loop.append(curr)
                succ = list(bnd_graph.successors(curr))
                if not succ or succ[0] in loop:
                    break
                curr = succ[0]
            if len(loop) > 2:
                loops.append(loop)
    return loops


def find_inner_arm_seam(
    mesh: trimesh.Trimesh,
    start_v: Optional[int] = None,
    end_v: Optional[int] = None
) -> List[int]:
    """Find the inner arm seam path connecting shoulder boundary to wrist boundary.
    
    :param mesh: Cylindrical sleeve arm mesh (with 2 boundary loops: shoulder and wrist).
    :param start_v: Optional index of the starting vertex on the shoulder loop (defaults to max X / medial).
    :param end_v: Optional index of the ending vertex on the wrist loop (defaults to max X / medial).
    :return: List of vertex indices forming the seam path from shoulder to wrist.
    """
    verts = mesh.vertices
    faces = mesh.faces
    loops = find_boundary_loops(faces)
    if len(loops) < 2:
        raise ValueError(f'Expected 2 boundary loops for arm cylinder, found {len(loops)}')
    
    # Sort loops by Z coordinate (Shoulder loop higher Z, Wrist loop lower Z)
    loops_by_z = sorted(loops, key=lambda lp: verts[lp, 2].mean(), reverse=True)
    shoulder_loop, wrist_loop = loops_by_z[0], loops_by_z[1]
    
    if start_v is None:
        # Inner shoulder vertex (maximum X, closest to chest/torso)
        start_v = max(shoulder_loop, key=lambda v: verts[v, 0])
    if end_v is None:
        # Inner wrist vertex (maximum X, closest to torso)
        end_v = max(wrist_loop, key=lambda v: verts[v, 0])
        
    G = nx.Graph()
    for f in faces:
        for i in range(3):
            u, v = f[i], f[(i + 1) % 3]
            G.add_edge(u, v, weight=np.linalg.norm(verts[u] - verts[v]))
            
    # Block all other boundary vertices so the shortest path travels internally
    all_bnd = set(shoulder_loop) | set(wrist_loop)
    G_int = G.copy()
    G_int.remove_nodes_from(all_bnd - {start_v, end_v})
    
    seam_path = nx.shortest_path(G_int, source=start_v, target=end_v, weight='weight')
    return seam_path


def cut_mesh_by_seam(
    mesh: trimesh.Trimesh,
    seam_path: List[int]
) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
    """Cut open a cylinder / sleeve mesh along a seam path by duplicating vertices along the cut.
    
    This merges the two boundary loops into a single boundary loop (topological disk).
    :param mesh: Input triangle mesh.
    :param seam_path: Ordered list of vertex indices forming the seam cut.
    :return: Tuple of (cut_mesh, seam_info)
    """
    verts = mesh.vertices.copy()
    faces = mesh.faces.copy()
    
    seam_edges = set()
    for i in range(len(seam_path) - 1):
        u, v = seam_path[i], seam_path[i+1]
        seam_edges.add((min(u, v), max(u, v)))
        
    new_verts = list(verts)
    new_faces = faces.copy()
    v_map_dup = {}
    
    for i, v in enumerate(seam_path):
        inc_faces = [f_i for f_i, f in enumerate(faces) if v in f]
        G_v = nx.Graph()
        for f_i in inc_faces:
            G_v.add_node(f_i)
        for a in range(len(inc_faces)):
            fa = faces[inc_faces[a]]
            for b in range(a + 1, len(inc_faces)):
                fb = faces[inc_faces[b]]
                common = set(fa) & set(fb)
                if len(common) == 2 and v in common:
                    other = list(common - {v})[0]
                    if (min(v, other), max(v, other)) not in seam_edges:
                        G_v.add_edge(inc_faces[a], inc_faces[b])
        comps = list(nx.connected_components(G_v))
        if len(comps) != 2:
            raise RuntimeError(f'Vertex {v} 1-ring did not split into 2 components (got {len(comps)})')
            
        if i < len(seam_path) - 1:
            u_next = seam_path[i+1]
            left_face = [f_i for f_i in inc_faces if faces[f_i][(list(faces[f_i]).index(v) + 1) % 3] == u_next][0]
        else:
            u_prev = seam_path[i-1]
            left_face = [f_i for f_i in inc_faces if faces[f_i][(list(faces[f_i]).index(u_prev) + 1) % 3] == v][0]
        left_comp = [c for c in comps if left_face in c][0]
        
        new_v_idx = len(new_verts)
        v_map_dup[v] = new_v_idx
        new_verts.append(verts[v])
        
        for f_i in left_comp:
            for k in range(3):
                if new_faces[f_i, k] == v:
                    new_faces[f_i, k] = new_v_idx
                    
    cut_mesh = trimesh.Trimesh(vertices=np.array(new_verts), faces=new_faces, process=False)
    info = {
        'seam_path': seam_path,
        'v_map_dup': v_map_dup,
        'start_v': seam_path[0],
        'end_v': seam_path[-1],
        'start_v_dup': v_map_dup[seam_path[0]],
        'end_v_dup': v_map_dup[seam_path[-1]],
    }
    return cut_mesh, info


def tutte_uv_parameterization(
    vertices: np.ndarray,
    faces: np.ndarray,
    boundary_shape: str = 'rectangle',
    seam_info: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Compute Tutte's harmonic 2D parameterization (Bijective UV embedding).
    
    :param vertices: (N, 3) coordinates.
    :param faces: (F, 3) triangle faces.
    :param boundary_shape: 'rectangle', 'square', or 'circle'.
    :param seam_info: Optional dict returned by cut_mesh_by_seam for rectangular sleeve unwrapping.
    :return: (N, 2) 2D UV coordinates in [0, 1] range.
    """
    n_verts = len(vertices)
    loops = find_boundary_loops(faces)
    if not loops:
        raise ValueError('Mesh has no boundary! Slice or cut the mesh along a seam first.')
    bnd_loop = max(loops, key=len)
    
    uv = np.zeros((n_verts, 2))
    
    if boundary_shape == 'rectangle' and seam_info is not None:
        c_start_orig = seam_info['start_v']
        c_start_dup = seam_info['start_v_dup']
        c_end_orig = seam_info['end_v']
        c_end_dup = seam_info['end_v_dup']
        
        # Start the loop at c_end_dup (wrist corner on duplicated seam side)
        idx0 = bnd_loop.index(c_end_dup)
        bnd_loop = bnd_loop[idx0:] + bnd_loop[:idx0]
        
        p1 = bnd_loop.index(c_end_orig)
        p2 = bnd_loop.index(c_start_orig)
        p3 = bnd_loop.index(c_start_dup)
        
        idx_corners = [0, p1, p2, p3, len(bnd_loop)]
        target_pts = [
            (np.array([0.0, 0.0]), np.array([1.0, 0.0])), # Wrist
            (np.array([1.0, 0.0]), np.array([1.0, 1.0])), # Seam A
            (np.array([1.0, 1.0]), np.array([0.0, 1.0])), # Shoulder
            (np.array([0.0, 1.0]), np.array([0.0, 0.0]))  # Seam B
        ]
        
        for seg_i in range(4):
            start_i = idx_corners[seg_i]
            end_i = idx_corners[seg_i + 1]
            seg_indices = bnd_loop[start_i:] + [bnd_loop[0]] if seg_i == 3 else bnd_loop[start_i:end_i + 1]
            pts = vertices[seg_indices]
            lens = [np.linalg.norm(pts[k+1] - pts[k]) for k in range(len(seg_indices) - 1)]
            cum = np.cumsum([0] + lens)
            total = cum[-1] if cum[-1] > 0 else 1.0
            t = cum / total
            pt_a, pt_b = target_pts[seg_i]
            for k, v_idx in enumerate(seg_indices):
                uv[v_idx] = pt_a + t[k] * (pt_b - pt_a)
    else:
        bnd_lens = [
            np.linalg.norm(vertices[bnd_loop[(i + 1) % len(bnd_loop)]] - vertices[bnd_loop[i]])
            for i in range(len(bnd_loop))
        ]
        total_len = sum(bnd_lens)
        cum_lens = np.cumsum([0] + bnd_lens[:-1]) / (total_len if total_len > 0 else 1.0)
        if boundary_shape == 'circle':
            theta = 2.0 * np.pi * cum_lens
            for i, idx in enumerate(bnd_loop):
                uv[idx] = [0.5 + 0.5 * np.cos(theta[i]), 0.5 + 0.5 * np.sin(theta[i])]
        else: # square
            for i, idx in enumerate(bnd_loop):
                t = cum_lens[i] * 4.0
                if t < 1.0:
                    uv[idx] = [t, 0.0]
                elif t < 2.0:
                    uv[idx] = [1.0, t - 1.0]
                elif t < 3.0:
                    uv[idx] = [3.0 - t, 1.0]
                else:
                    uv[idx] = [0.0, 4.0 - t]
                    
    bnd_set = set(bnd_loop)
    int_verts = [i for i in range(n_verts) if i not in bnd_set]
    int_map = {idx: i for i, idx in enumerate(int_verts)}
    n_int = len(int_verts)
    
    if n_int > 0:
        adj = {i: set() for i in range(n_verts)}
        for f in faces:
            for i in range(3):
                u, v = f[i], f[(i + 1) % 3]
                adj[u].add(v)
                adj[v].add(u)
                
        rows, cols, data = [], [], []
        rhs_u = np.zeros(n_int)
        rhs_v = np.zeros(n_int)
        
        for i_int, u_idx in enumerate(int_verts):
            deg = len(adj[u_idx])
            rows.append(i_int)
            cols.append(i_int)
            data.append(float(deg))
            
            for v_idx in adj[u_idx]:
                if v_idx in int_map:
                    rows.append(i_int)
                    cols.append(int_map[v_idx])
                    data.append(-1.0)
                else:
                    rhs_u[i_int] += uv[v_idx, 0]
                    rhs_v[i_int] += uv[v_idx, 1]
                    
        L_int = sp.csr_matrix((data, (rows, cols)), shape=(n_int, n_int))
        u_sol = spla.spsolve(L_int, rhs_u)
        v_sol = spla.spsolve(L_int, rhs_v)
        
        for i_int, u_idx in enumerate(int_verts):
            uv[u_idx, 0] = u_sol[i_int]
            uv[u_idx, 1] = v_sol[i_int]
            
    return uv


def lscm_parameterization(vertices: np.ndarray,
                          faces: np.ndarray,
                          pinned_verts: Optional[Dict[int, np.ndarray]] = None) -> np.ndarray:
    """
    Compute Least Squares Conformal Maps (LSCM) 2D parameterization with free boundary.
    
    Unfolds the cut mesh naturally onto 2D while minimizing angle distortion,
    without forcing boundary vertices into an artificial square or circle.
    
    :param vertices: (N, 3) vertex coordinates.
    :param faces: (F, 3) triangle face indices.
    :param pinned_verts: Optional dict of {vertex_idx: np.ndarray([u, v])} to fix rigid motion (needs at least 2 vertices).
    :return: (N, 2) 2D UV coordinates.
    """
    n_verts = len(vertices)
    n_faces = len(faces)
    
    if pinned_verts is None:
        loops = find_boundary_loops(faces)
        bnd = loops[0] if loops else list(range(min(n_verts, 100)))
        pts = vertices[bnd]
        dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        i, j = np.unravel_index(np.argmax(dists), dists.shape)
        p1, p2 = bnd[i], bnd[j]
        pinned_verts = {p1: np.array([0.0, 0.0]), p2: np.array([0.0, dists[i, j]])}

    rows_re, cols_re, vals_re = [], [], []
    rows_im, cols_im, vals_im = [], [], []
    
    for f_idx, (i, j, k) in enumerate(faces):
        p0, p1, p2 = vertices[i], vertices[j], vertices[k]
        e0 = p1 - p0
        e0_len = np.linalg.norm(e0)
        e0_unit = e0 / (e0_len if e0_len > 1e-12 else 1.0)
        e1 = p2 - p0
        normal = np.cross(e0, e1)
        area_2 = np.linalg.norm(normal)
        if area_2 < 1e-12:
            continue
        sqrt_area = np.sqrt(0.5 * area_2)
        
        x0, y0 = 0.0, 0.0
        x1, y1 = e0_len, 0.0
        x2 = np.dot(e1, e0_unit)
        y2 = area_2 / (e0_len if e0_len > 1e-12 else 1.0)
        
        factor = 1.0 / (2.0 * sqrt_area)
        w = [
            ((x1 - x2) * factor, (y1 - y2) * factor),
            ((x2 - x0) * factor, (y2 - y0) * factor),
            ((x0 - x1) * factor, (y0 - y1) * factor)
        ]
        
        v_indices = [i, j, k]
        for m in range(3):
            vm = v_indices[m]
            re_m, im_m = w[m]
            rows_re.extend([2 * f_idx, 2 * f_idx])
            cols_re.extend([vm, n_verts + vm])
            vals_re.extend([re_m, -im_m])
            
            rows_im.extend([2 * f_idx + 1, 2 * f_idx + 1])
            cols_im.extend([vm, n_verts + vm])
            vals_im.extend([im_m, re_m])

    rows = rows_re + rows_im
    cols = cols_re + cols_im
    vals = vals_re + vals_im
    A = sp.csr_matrix((vals, (rows, cols)), shape=(2 * n_faces, 2 * n_verts))
    
    pinned_keys = list(pinned_verts.keys())
    free_vars = [idx for idx in range(2 * n_verts) if (idx % n_verts) not in pinned_keys]
    
    rhs_fixed = np.zeros(2 * n_faces)
    for p_idx, uv_val in pinned_verts.items():
        rhs_fixed += A[:, p_idx].toarray().flatten() * uv_val[0]
        rhs_fixed += A[:, n_verts + p_idx].toarray().flatten() * uv_val[1]
        
    A_free = A[:, free_vars]
    AtA = A_free.T @ A_free
    Atr = -A_free.T @ rhs_fixed
    
    sol = spla.spsolve(AtA, Atr)
    
    full_uv = np.zeros(2 * n_verts)
    for p_idx, uv_val in pinned_verts.items():
        full_uv[p_idx] = uv_val[0]
        full_uv[n_verts + p_idx] = uv_val[1]
        
    for idx_free, v_id in enumerate(free_vars):
        full_uv[v_id] = sol[idx_free]
        
    u = full_uv[:n_verts]
    v = full_uv[n_verts:]
    return np.column_stack([u, v])


def cylindrical_uv_parameterization(
    vertices: np.ndarray,
    faces: np.ndarray,
    seam_info: Dict[str, Any],
) -> np.ndarray:
    """
    Compute Cylindrical Geodesic Unrolling (Alternative A) 2D UV parameterization.
    
    Unrolls a cut cylindrical tubular mesh (e.g. arm sleeve) onto 2D space while preserving
    metric physical distances (arc-lengths along circumference and seam axis in meters).
    Boundary vertices are mapped according to true 3D geodesic/arc lengths, and interior
    vertices are computed using Floater's Mean Value Coordinates (MVC, 2003) to guarantee
    a bijective, fold-free planar embedding.

    :param vertices: (N, 3) 3D vertex coordinates.
    :param faces: (F, 3) triangle face indices.
    :param seam_info: Dict returned by  containing:
                      - 'start_v': Index of proximal (shoulder) seam corner (Side A)
                      - 'start_v_dup': Index of proximal (shoulder) seam corner (Side B)
                      - 'end_v': Index of distal (wrist) seam corner (Side A)
                      - 'end_v_dup': Index of distal (wrist) seam corner (Side B)
                      - 'v_map_dup': Mapping from original cut vertex to duplicated vertex
    :return: (N, 2) 2D UV coordinates in metric units (meters).
    """
    n_verts = len(vertices)
    
    # 1. Identify boundary loop and 4 corner vertices
    loops = find_boundary_loops(faces)
    if not loops:
        raise ValueError("Mesh has no boundary loops; cannot perform boundary-constrained unrolling.")
    
    bnd_loop = list(loops[0])
    
    c_start_orig = seam_info["start_v"]
    c_start_dup = seam_info["start_v_dup"]
    c_end_orig = seam_info["end_v"]
    c_end_dup = seam_info["end_v_dup"]
    
    # Rotate boundary loop to start at c_end_dup (distal Side B)
    if c_end_dup not in bnd_loop:
        raise ValueError(f"Corner vertex {c_end_dup} not found in boundary loop.")
    
    idx0 = bnd_loop.index(c_end_dup)
    bnd_loop = bnd_loop[idx0:] + bnd_loop[:idx0]
    
    # Check winding orientation: expected sequence is c_end_dup -> c_end_orig -> c_start_orig -> c_start_dup
    p1 = bnd_loop.index(c_end_orig)
    p3 = bnd_loop.index(c_start_dup)
    if p1 > p3:
        bnd_loop = [bnd_loop[0]] + bnd_loop[:0:-1]
        p1 = bnd_loop.index(c_end_orig)
        p3 = bnd_loop.index(c_start_dup)
        
    p2 = bnd_loop.index(c_start_orig)
    
    # 4 boundary segments:
    # Seg 1: Wrist loop (end_v_dup -> end_v)
    # Seg 2: Seam Side A (end_v -> start_v)
    # Seg 3: Shoulder loop (start_v -> start_v_dup)
    # Seg 4: Seam Side B (start_v_dup -> end_v_dup)
    seg1 = bnd_loop[0:p1+1]
    seg2 = bnd_loop[p1:p2+1]
    seg3 = bnd_loop[p2:p3+1]
    seg4 = bnd_loop[p3:] + [bnd_loop[0]]
    
    def seg_cum(s):
        pts = vertices[s]
        diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        return np.cumsum(np.insert(diffs, 0, 0.0))
        
    c1 = seg_cum(seg1)
    c2 = seg_cum(seg2)
    c3 = seg_cum(seg3)
    c4 = seg_cum(seg4)
    
    c_wrist = c1[-1]
    l_seam = c2[-1]
    c_shoulder = c3[-1]
    l_seam_b = c4[-1]
    
    uv_bnd: Dict[int, np.ndarray] = {}
    
    # Segment 1: Wrist loop (v = 0, u runs from -c_wrist to 0)
    for k, v in enumerate(seg1):
        u_val = -c_wrist * (1.0 - c1[k] / c_wrist) if c_wrist > 1e-12 else 0.0
        uv_bnd[v] = np.array([u_val, 0.0])
        
    # Segment 2: Seam Side A (u = 0, v runs from 0 to l_seam)
    for k, v in enumerate(seg2):
        uv_bnd[v] = np.array([0.0, c2[k]])
        
    # Segment 3: Shoulder loop (v = l_seam, u runs from 0 to -c_shoulder)
    for k, v in enumerate(seg3):
        u_val = -c_shoulder * (c3[k] / c_shoulder) if c_shoulder > 1e-12 else 0.0
        uv_bnd[v] = np.array([u_val, l_seam])
        
    # Segment 4: Seam Side B (u = -circumference(v), v runs from l_seam down to 0)
    for k, v in enumerate(seg4):
        t = 1.0 - (c4[k] / l_seam_b) if l_seam_b > 1e-12 else 0.0
        c_loc = c_wrist + t * (c_shoulder - c_wrist)
        uv_bnd[v] = np.array([-c_loc, t * l_seam])
        
    # 2. Mean Value Coordinates (MVC) parameterization for interior vertices
    bnd_set = set(uv_bnd.keys())
    free_vars = [i for i in range(n_verts) if i not in bnd_set]
    n_free = len(free_vars)
    
    if n_free == 0:
        uv = np.zeros((n_verts, 2))
        for idx, pt in uv_bnd.items():
            uv[idx] = pt
        return uv
        
    free_map = {idx: i for i, idx in enumerate(free_vars)}
    
    v_faces: Dict[int, List[int]] = {i: [] for i in range(n_verts)}
    for f_idx, f in enumerate(faces):
        for k in range(3):
            v_faces[f[k]].append(f_idx)
            
    rows, cols, data = [], [], []
    rhs_u = np.zeros(n_free)
    rhs_v = np.zeros(n_free)
    
    for i_free, v_i in enumerate(free_vars):
        p_i = vertices[v_i]
        inc_f = faces[v_faces[v_i]]
        
        edges_opp = []
        for f in inc_f:
            idx_in_f = list(f).index(v_i)
            v_next = f[(idx_in_f + 1) % 3]
            v_prev = f[(idx_in_f + 2) % 3]
            edges_opp.append((v_next, v_prev))
            
        curr = edges_opp[0][0]
        ordered_nbrs = [curr]
        while len(ordered_nbrs) < len(edges_opp):
            found = False
            for e in edges_opp:
                if e[0] == curr:
                    curr = e[1]
                    ordered_nbrs.append(curr)
                    found = True
                    break
            if not found:
                ordered_nbrs = list(set([e[0] for e in edges_opp] + [e[1] for e in edges_opp]))
                break
                
        K = len(ordered_nbrs)
        v_vecs = [vertices[nbr] - p_i for nbr in ordered_nbrs]
        dists = [np.linalg.norm(vec) + 1e-12 for vec in v_vecs]
        units = [vec / d for vec, d in zip(v_vecs, dists)]
        
        angles = []
        for k in range(K):
            u1 = units[k]
            u2 = units[(k + 1) % K]
            cos_a = np.clip(np.dot(u1, u2), -1.0, 1.0)
            angles.append(np.arccos(cos_a))
            
        weights = []
        for k in range(K):
            a_prev = angles[(k - 1) % K]
            a_curr = angles[k]
            w = (np.tan(a_prev / 2.0) + np.tan(a_curr / 2.0)) / dists[k]
            weights.append(max(w, 1e-6))
            
        w_sum = sum(weights)
        for k, nbr in enumerate(ordered_nbrs):
            w = weights[k]
            if nbr in free_map:
                rows.append(i_free)
                cols.append(free_map[nbr])
                data.append(-w)
            else:
                rhs_u[i_free] += w * uv_bnd[nbr][0]
                rhs_v[i_free] += w * uv_bnd[nbr][1]
                
        rows.append(i_free)
        cols.append(i_free)
        data.append(w_sum)
        
    L = sp.csr_matrix((data, (rows, cols)), shape=(n_free, n_free))
    sol_u = spla.spsolve(L, rhs_u)
    sol_v = spla.spsolve(L, rhs_v)
    
    uv = np.zeros((n_verts, 2))
    for idx, pt in uv_bnd.items():
        uv[idx] = pt
    for i_free, idx in enumerate(free_vars):
        uv[idx] = [sol_u[i_free], sol_v[i_free]]
        
    return uv
