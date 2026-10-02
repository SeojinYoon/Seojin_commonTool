from typing import List, Optional, Sequence, Union
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

    def slice_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Slice a 3D mesh using this plane, keeping the half on the normal side."""
        return slice_mesh_by_plane(mesh, self)



def slice_mesh_by_plane(
    mesh: trimesh.Trimesh,
    plane_origin: Union[List[float], np.ndarray, Plane],
    plane_normal: Optional[Union[List[float], np.ndarray]] = None,
) -> trimesh.Trimesh:
    """Slice a 3D mesh by a cross-section plane, keeping the half on the normal side.

    :param mesh: Input trimesh.Trimesh.
    :param plane_origin: 3D point on the cutting plane, or a Plane object.
    :param plane_normal: 3D normal vector of the cutting plane (omitted if Plane is passed).
    :return: Sliced trimesh.Trimesh with open boundary.
    """
    if isinstance(plane_origin, Plane):
        plane_normal = plane_origin.normal
        plane_origin = plane_origin.origin
    elif plane_normal is None:
        raise ValueError("plane_normal must be provided when plane_origin is not a Plane instance.")

    sliced = mesh.slice_plane(plane_origin=plane_origin, plane_normal=plane_normal)
    sliced.remove_unreferenced_vertices()
    return sliced


def slice_mesh_between_planes(
    mesh: trimesh.Trimesh,
    plane1_origin: Union[List[float], np.ndarray, Plane],
    plane1_normal: Optional[Union[List[float], np.ndarray, Plane]] = None,
    plane2_origin: Optional[Union[List[float], np.ndarray, Plane]] = None,
    plane2_normal: Optional[Union[List[float], np.ndarray]] = None,
) -> trimesh.Trimesh:
    """Slice a mesh section between two cross-section planes (e.g. upper arm segment).

    Can be called either as:
      slice_mesh_between_planes(mesh, plane1, plane2)  # where plane1, plane2 are Plane objects
    or:
      slice_mesh_between_planes(mesh, p1_origin, p1_normal, p2_origin, p2_normal)
    """
    if isinstance(plane1_origin, Plane) and isinstance(plane1_normal, Plane):
        p1_o, p1_n = plane1_origin.origin, plane1_origin.normal
        p2_o, p2_n = plane1_normal.origin, plane1_normal.normal
    elif isinstance(plane1_origin, Plane) and plane2_origin is not None:
        p1_o, p1_n = plane1_origin.origin, plane1_origin.normal
        if isinstance(plane2_origin, Plane):
            p2_o, p2_n = plane2_origin.origin, plane2_origin.normal
        else:
            p2_o, p2_n = plane2_origin, plane2_normal
    else:
        p1_o, p1_n = plane1_origin, plane1_normal
        p2_o, p2_n = plane2_origin, plane2_normal

    section = mesh.slice_plane(plane_origin=p1_o, plane_normal=p1_n)
    section = section.slice_plane(plane_origin=p2_o, plane_normal=p2_n)
    section.remove_unreferenced_vertices()
    return section

if __name__ == "__main__":
    plane = Plane(origin=[0.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0])
    
    box_min = [-1.0, -1.0, -1.0]
    box_max = [1.0, 1.0, 1.0]
    
    polygon = plane.intersect_box(box_min, box_max)
    
    print(f"#polygon: {len(polygon)}")
    print("coord:\n", polygon)
    