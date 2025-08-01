import numpy as np
import open3d as o3d


def find_starting_point(points, edges, direction):
    # Flatten the list of included indices
    included_indices_flat = [idx for indices in edges for idx in indices]

    # Filter points that are included in the list of indices
    filtered_points = points[np.isin(np.arange(len(points)), included_indices_flat)]

    # Find the index of the point with the smallest/largest x/y-coordinate among the filtered points
    if direction == 'min_y':
        index = np.argmin(filtered_points[:, 1])
    elif direction == 'max_y':
        index = np.argmax(filtered_points[:, 1])
    elif direction == 'min_x':
        index = np.argmin(filtered_points[:, 0])
    else:
        index = np.argmax(filtered_points[:, 0])

    start_point = filtered_points[index]
    return start_point


def filtering_loop(edges, start_index):
    filtered_edges = []
    query_index = start_index
    tmp_edges = edges.copy()

    while True:
        found_edge = False
        for edge in tmp_edges:
            if edge[0] == query_index:
                found_edge = True
                filtered_edges.append(edge)
                query_index = edge[1]
                tmp_edges.remove(edge)
                break
            elif edge[1] == query_index:
                found_edge = True
                filtered_edges.append((edge[1], edge[0]))
                query_index = edge[0]
                tmp_edges.remove(edge)
                break

        if not found_edge or query_index == start_index:
            break

    return filtered_edges


def remove_redundant_indices(edges, mesh_points):
    for i1, (s1, e1) in enumerate(edges):
        for i2, (s2, e2) in enumerate(edges[1:]):
            if s1 == s2 and e1 == e2:
                continue
            i2 += 1

            if s1 == s2:
                to_be_removed = i2 - i1
                for r in range(to_be_removed):
                    del edges[i1]

    polygon = [mesh_points[index] for index in np.asarray(edges)[:,0]]
    return edges, polygon


def filter_edges(edges, mesh):
    mesh_points = np.asarray(mesh.vertices)
    min_y_point = find_starting_point(mesh_points, edges, direction='min_y')
    min_y_index = np.where(mesh_points == min_y_point)[0][0]

    filtered_edges = filtering_loop(edges, start_index=min_y_index)

    if len(filtered_edges) < 10:
        for start in ['max_y', 'min_x', 'max_x']:
            start_point = find_starting_point(mesh_points, edges, direction=start)
            start_point_index = np.where(mesh_points == start_point)[0][0]
            filtered_edges = filtering_loop(edges, start_index=start_point_index)
            if len(filtered_edges) > 10:
                break

    filtered_edges, polygon = remove_redundant_indices(filtered_edges, mesh_points)

    return list(filtered_edges), polygon


def find_outer_edges(mesh):
    mesh.compute_vertex_normals()
    # mesh_points = mesh.vertices
    # Identify non-manifold edges
    non_manifold_edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)

    # Create a list to store the non-manifold edge lines for visualization
    lines = []
    for edge in non_manifold_edges:
        lines.append((edge[0], edge[1]))
        # lines.append((mesh_points[edge[0]], mesh_points[edge[1]]))

    return lines


def create_lineset(mesh, outer_edges):
    lines = []  # List to store line segments

    # Add line segments for outer edges
    for edge in outer_edges:
        lines.append(edge)

    # Create LineSet geometry
    lineset = o3d.geometry.LineSet()
    lineset.points = mesh.vertices
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lineset.lines))])

    return lineset
