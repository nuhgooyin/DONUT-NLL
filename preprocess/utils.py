# Copyright (c) 2023, Zikang Zhou. All rights reserved.
# Modified by Markus Knoche, 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import torch


def side_to_directed_lineseg(
    query_point: torch.Tensor, start_point: torch.Tensor, end_point: torch.Tensor
) -> str:
    cond = (end_point[0] - start_point[0]) * (query_point[1] - start_point[1]) - (
        end_point[1] - start_point[1]
    ) * (query_point[0] - start_point[0])
    if cond > 0:
        return 'LEFT'
    elif cond < 0:
        return 'RIGHT'
    else:
        return 'CENTER'


def compute_crosswalk_boundaries(crosswalk_polygon):
    """
    Compute the centerline, start and end positions, left boundary, and right boundary
    of a crosswalk polygon.

    Args:
        crosswalk_polygon (torch.Tensor): Tensor of shape (N, 3) containing polygon points [x, y, z].

    Returns:
        start_position (torch.Tensor): Start position of the centerline.
        end_position (torch.Tensor): End position of the centerline.
        left_boundary (torch.Tensor): Left boundary points of the crosswalk.
        right_boundary (torch.Tensor): Right boundary points of the crosswalk.
        centerline (torch.Tensor): Centerline points of the crosswalk.
    """
    # Convert torch.Tensor to numpy array if needed
    if isinstance(crosswalk_polygon, torch.Tensor):
        points = crosswalk_polygon.cpu().numpy()
    else:
        points = np.array(crosswalk_polygon)

    # Classify edges into width and length edges
    edges = extract_edges(points)
    edge_groups = group_edges(edges, angle_threshold=30)
    merged_sides = [merge_edges(group) for group in edge_groups]

    # Calculate total length of each merged side
    side_lengths = []
    for side in merged_sides:
        length = 0
        for i in range(len(side) - 1):
            length += edge_length(np.array([side[i], side[i + 1]]))
        side_lengths.append(length)

    # Sort sides by length (ascending)
    sides_with_lengths = list(zip(merged_sides, side_lengths))
    sides_with_lengths.sort(key=lambda x: x[1])

    # Extract width edges (shortest two sides) and length edges (longest two sides)
    width_edges = (
        [side for side, _ in sides_with_lengths[:2]]
        if len(sides_with_lengths) >= 2
        else []
    )
    length_edges = (
        [side for side, _ in sides_with_lengths[-2:]]
        if len(sides_with_lengths) >= 2
        else []
    )

    # Compute midpoints of width edges (start and end of centerline)
    if len(width_edges) >= 2:
        start_midpoint = np.mean(width_edges[0], axis=0)
        end_midpoint = np.mean(width_edges[1], axis=0)
    else:
        # Fallback: Use first and last points if width edges are missing
        start_midpoint = points[0]
        end_midpoint = points[-1]

    # Assign left and right boundaries (length edges)
    if len(length_edges) >= 2:
        # Sort length edges by x-coordinate to determine left/right
        length_edges_sorted = sorted(
            length_edges, key=lambda edge: np.mean([p[0] for p in edge])
        )
        left_boundary = length_edges_sorted[0]
        right_boundary = length_edges_sorted[1]
    else:
        left_boundary = points[: len(points) // 2]
        right_boundary = points[len(points) // 2 :]

    # Reverse the order of right boundary points
    right_boundary = np.flip(right_boundary, axis=0).copy()

    # Compute arc lengths
    left_arc = get_polyline_arc_length(left_boundary[:, :2])
    right_arc = get_polyline_arc_length(right_boundary[:, :2])

    # Choose number of centerline points
    num_points = min(len(left_boundary), len(right_boundary))
    steps = np.linspace(0, min(left_arc[-1], right_arc[-1]), num_points)

    # Interpolate boundaries
    left_interp = interpolate_lane(left_boundary[:, :2], left_arc, steps)
    right_interp = interpolate_lane(right_boundary[:, :2], right_arc, steps)

    # Interpolate Z (height) linearly if present
    if left_boundary.shape[1] == 3:
        left_z = np.interp(steps, left_arc, left_boundary[:, 2])
        right_z = np.interp(steps, right_arc, right_boundary[:, 2])
        xy_avg = (left_interp + right_interp) / 2
        z_avg = ((left_z + right_z) / 2)[:, None]
        centerline_np = np.concatenate([xy_avg, z_avg], axis=1)
    else:
        centerline_np = (left_interp + right_interp) / 2

    # Convert back to torch.Tensor
    start_position = torch.tensor(start_midpoint, dtype=torch.float32)
    end_position = torch.tensor(end_midpoint, dtype=torch.float32)
    left_boundary = torch.tensor(left_boundary, dtype=torch.float32)
    right_boundary = torch.tensor(right_boundary, dtype=torch.float32)
    centerline = torch.tensor(centerline_np, dtype=torch.float32)

    return start_position, end_position, left_boundary, right_boundary, centerline


def get_polyline_arc_length(xy: np.ndarray) -> np.ndarray:
    """Get the arc length of each point in a polyline"""
    diff = xy[1:] - xy[:-1]
    displacement = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
    arc_length = np.cumsum(displacement)
    return np.concatenate((np.zeros(1), arc_length), axis=0)


def interpolate_lane(xy: np.ndarray, arc_length: np.ndarray, steps: np.ndarray):
    xy_inter = np.empty((steps.shape[0], 2), dtype=xy.dtype)
    xy_inter[:, 0] = np.interp(steps, xp=arc_length, fp=xy[:, 0])
    xy_inter[:, 1] = np.interp(steps, xp=arc_length, fp=xy[:, 1])
    return xy_inter


def extract_edges(points):
    # Ensure the polygon is closed
    points = np.array(points)
    if not np.array_equal(points[0], points[-1]):
        points = np.vstack([points, points[0]])

    # Extract edges (pairs of consecutive points)
    edges = []
    for i in range(len(points) - 1):
        edges.append(np.array([points[i], points[i + 1]]))

    return edges


def edge_length(edge):
    return np.linalg.norm(edge[1] - edge[0])


def group_edges(edges, angle_threshold=30):
    if not edges:
        return []

    # Compute angles (in degrees) of edges
    angles = []
    for edge in edges:
        vector = edge[1] - edge[0]
        angle = np.degrees(np.arctan2(vector[1], vector[0])) % 180
        angles.append(angle)

    # Group edges based on angle similarity and connectivity
    groups = []
    current_group = [edges[0]]

    for i in range(1, len(edges)):
        prev_edge = edges[i - 1]
        current_edge = edges[i]

        # Check if current edge is connected to the previous
        connected = np.allclose(prev_edge[1], current_edge[0])

        # Compute angle difference (handling wrap-around)
        angle_diff = abs(angles[i] - angles[i - 1])
        angle_diff = min(angle_diff, 180 - angle_diff)

        if connected and angle_diff < angle_threshold:
            current_group.append(current_edge)
        else:
            groups.append(current_group)
            current_group = [current_edge]

    if current_group:
        groups.append(current_group)

    # Check if the last group is connected to the first and angle difference is small
    if len(groups) > 1:
        last_group = groups[-1]
        first_group = groups[0]
        last_edge = last_group[-1]
        first_edge = first_group[0]

        connected = np.allclose(last_edge[1], first_edge[0])
        angle_diff = abs(angles[-1] - angles[0])
        angle_diff = min(angle_diff, 180 - angle_diff)

        if connected and angle_diff < angle_threshold:
            groups[0] = last_group + first_group
            groups.pop(-1)

    return groups


def merge_edges(edge_group):
    if not edge_group:
        return None
    merged_edge = np.vstack([edge[0] for edge in edge_group] + [edge_group[-1][1]])
    return merged_edge


def orientation_on_lane(line, pos_xyz):
    """
    lane_id: int
    pos_xyz: torch.tensor([x, y, z])
    returns scalar orientation (float)
    """
    if line.shape[0] < 2:
        return torch.tensor(0.0)

    # distances to centerline points in XY
    diff = line[:, :2] - pos_xyz[:2]
    d2 = (diff**2).sum(dim=-1)
    idx = int(torch.argmin(d2).item())

    # segments: cl[i] -> cl[i+1], orientation per segment
    vecs = line[1:] - line[:-1]  # (N-1, 3)
    if idx == 0:
        seg_idx = 0
    else:
        seg_idx = idx - 1
    v = vecs[seg_idx]
    return torch.atan2(v[1], v[0])


def extract_relevant_boundary(boundary_polyline, lane_polyline):
    """
    Given a long boundary polyline and a lane centerline polyline,
    extracts the relevant segment of the boundary corresponding to the lane.

    Args:
        boundary_polyline (torch.Tensor): The full boundary polyline from map_features (N x 3).
        lane_polyline (torch.Tensor): The lane's centerline polyline (M x 3).

    Returns:
        torch.Tensor: The cropped boundary polyline matching the lane segment.
    """
    if boundary_polyline.shape[0] == 0 or lane_polyline.shape[0] == 0:
        # Return empty if either polyline is missing
        return torch.empty((0, 3))

    # Convert to numpy for fast nearest neighbor search
    # boundary_np = boundary_polyline.numpy()
    # lane_np = lane_polyline.numpy()

    # Create KDTree for fast nearest neighbor search
    # boundary_tree = cKDTree(boundary_np[:, :2])  # Use only (x, y) coordinates

    # Find the closest boundary point for the start and end of the lane
    # start_idx = boundary_tree.query(lane_np[0, :2])[1]  # Closest to lane start
    # end_idx = boundary_tree.query(lane_np[-1, :2])[1]  # Closest to lane end

    def nearest_idx(poly_xy, point_xy):
        d = poly_xy - point_xy
        return np.argmin(np.einsum('ij,ij->i', d, d))

    boundary_xy = boundary_polyline[:, :2].numpy()
    lane_xy = lane_polyline[:, :2].numpy()

    start_idx = nearest_idx(boundary_xy, lane_xy[0])
    end_idx = nearest_idx(boundary_xy, lane_xy[-1])

    # Ensure start_idx < end_idx
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    # Extract relevant segment
    extracted_boundary = boundary_polyline[start_idx : end_idx + 1]

    return extracted_boundary
