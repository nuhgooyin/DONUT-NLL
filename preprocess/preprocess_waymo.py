# Copyright (c) 2026, Markus Knoche. All rights reserved.
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
import argparse
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import List, Optional

# silence tensorflow's information output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# improves runtime
os.environ['OMP_NUM_THREADS'] = '1'

import tensorflow as tf
import torch
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2

from utils import compute_crosswalk_boundaries
from utils import extract_relevant_boundary
from utils import orientation_on_lane
from utils import side_to_directed_lineseg

RAW_DIR_NAMES = {
    'test': 'testing',
    'val': 'validation',
    'train': 'training',
}

NUM_SAMPLES = {
    'train': 486995,
    'val': 44097,
    'test': 44920,
}

TRAFFIC_SIGNAL_TYPES = [
    'LANE_STATE_UNKNOWN',
    'LANE_STATE_ARROW_STOP',
    'LANE_STATE_ARROW_CAUTION',
    'LANE_STATE_ARROW_GO',
    'LANE_STATE_STOP',
    'LANE_STATE_CAUTION',
    'LANE_STATE_GO',
    'LANE_STATE_FLASHING_STOP',
    'LANE_STATE_FLASHING_CAUTION',
]

POLYGON_TYPES = [
    'TYPE_UNDEFINED',
    'TYPE_FREEWAY',
    'TYPE_SURFACE_STREET',
    'TYPE_BIKE_LANE',
    'TYPE_CROSSWALK',
    'TYPE_DRIVEWAY',
    'TYPE_SPEED_BUMP',
    'TYPE_STOP_SIGN',
    'TYPE_TRAFFIC_LIGHT',
]

POINT_TYPES = [
    'UNKNOWN',
    'BROKEN_SINGLE_WHITE',
    'SOLID_SINGLE_WHITE',
    'SOLID_DOUBLE_WHITE',
    'BROKEN_SINGLE_YELLOW',
    'BROKEN_DOUBLE_YELLOW',
    'SOLID_SINGLE_YELLOW',
    'SOLID_DOUBLE_YELLOW',
    'PASSING_DOUBLE_YELLOW',
    'CENTERLINE',
    'ROAD_EDGE',
    'CROSSWALK',
    'DRIVEWAY',
    'SPEED_BUMP',
    'STOP_SIGN',
] + TRAFFIC_SIGNAL_TYPES

POINT_SIDES = ['LEFT', 'RIGHT', 'CENTER']

POLYGON_TO_POLYGON_TYPES = ['NONE', 'PRED', 'SUCC', 'LEFT', 'RIGHT']

POINT_TYPE_IDX = {name: i for i, name in enumerate(POINT_TYPES)}
POINT_SIDE_IDX = {name: i for i, name in enumerate(POINT_SIDES)}
POLYGON_TYPE_IDX = {name: i for i, name in enumerate(POLYGON_TYPES)}
POLYGON_TO_POLYGON_TYPE_IDX = {
    name: i for i, name in enumerate(POLYGON_TO_POLYGON_TYPES)
}
TRAFFIC_SIGNAL_TYPE_IDX = {name: i for i, name in enumerate(TRAFFIC_SIGNAL_TYPES)}


def extract_scenarios_from_tfrecord(files):
    filenames = tf.io.matching_files(files)
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.AUTOTUNE)
    dataset_iterator = dataset.as_numpy_iterator()

    for bytes_example in dataset_iterator:
        yield scenario_pb2.Scenario.FromString(bytes_example)


def get_scenario_id(scenario):
    return scenario.scenario_id if scenario.HasField('scenario_id') else None


def get_agent_features(
    scenario, num_historical_steps, num_steps, dim, predict_unseen_agents
):
    if predict_unseen_agents:
        agent_ids = [track.id for track in scenario.tracks]
    else:
        agent_ids = {
            track.id
            for track in scenario.tracks
            if any(state.valid for state in track.states[:num_historical_steps])
        }

    num_agents = len(agent_ids)
    valid_mask = torch.zeros(num_agents, num_steps, dtype=torch.bool)
    current_valid_mask = torch.zeros(num_agents, dtype=torch.bool)
    predict_mask = torch.zeros(num_agents, num_steps, dtype=torch.bool)
    agent_id: List[Optional[str]] = [None] * num_agents
    agent_type = torch.zeros(num_agents, dtype=torch.uint8)
    agent_category = torch.zeros(num_agents, dtype=torch.uint8)
    position = torch.zeros(num_agents, num_steps, dim, dtype=torch.float)
    heading = torch.zeros(num_agents, num_steps, dtype=torch.float)
    velocity = torch.zeros(num_agents, num_steps, dim, dtype=torch.float)
    dimensions = torch.zeros(num_agents, num_steps, 3, dtype=torch.float)
    interesting_mask = torch.zeros(num_agents, dtype=torch.uint8)

    predicted_track_ids = {
        scenario.tracks[idx.track_index].id for idx in scenario.tracks_to_predict
    }
    interesting_track_ids = {idx for idx in scenario.objects_of_interest}

    av_index = (
        scenario.sdc_track_index if scenario.HasField('sdc_track_index') else None
    )

    agent_idx = 0
    for track in scenario.tracks:
        if track.id not in agent_ids:
            continue

        agent_id[agent_idx] = str(track.id)
        agent_type[agent_idx] = track.object_type
        agent_category[agent_idx] = 1 if track.id in predicted_track_ids else 0
        interesting_mask[agent_idx] = 1 if track.id in interesting_track_ids else 0

        for timestep_idx, state in enumerate(track.states):
            valid_mask[agent_idx, timestep_idx] = state.valid

            if state.valid:
                position[agent_idx, timestep_idx, 0] = state.center_x
                position[agent_idx, timestep_idx, 1] = state.center_y
                if dim == 3:
                    position[agent_idx, timestep_idx, 2] = state.center_z

                velocity[agent_idx, timestep_idx, 0] = state.velocity_x
                velocity[agent_idx, timestep_idx, 1] = state.velocity_y

                heading[agent_idx, timestep_idx] = state.heading

                dimensions[agent_idx, timestep_idx, 0] = state.length
                dimensions[agent_idx, timestep_idx, 1] = state.width
                dimensions[agent_idx, timestep_idx, 2] = state.height
                if timestep_idx >= num_historical_steps:
                    predict_mask[agent_idx, timestep_idx] = True

        current_valid_mask[agent_idx] = valid_mask[agent_idx, num_historical_steps - 1]
        agent_idx += 1

    return {
        'num_nodes': num_agents,
        'av_index': av_index,
        'valid_mask': valid_mask,
        'predict_mask': predict_mask,
        'interesting_mask': interesting_mask,
        'id': agent_id,
        'type': agent_type,
        'category': agent_category,
        'position': position,
        'heading': heading,
        'velocity': velocity,
        'dimensions': dimensions,
    }


def get_map_features(scenario, dim):
    map_features = scenario.map_features
    map_feature_dict = {feature.id: idx for idx, feature in enumerate(map_features)}

    lane_segment_ids = [f.id for f in map_features if f.HasField('lane')]
    lane_segment_id_map = {id: idx for idx, id in enumerate(lane_segment_ids)}

    cds_ids = [
        f.id
        for f in map_features
        if f.HasField('crosswalk') or f.HasField('driveway') or f.HasField('speed_bump')
    ]
    polygon_id_map = {id: idx for idx, id in enumerate(cds_ids)}

    num_lanes = len(lane_segment_ids)
    num_cds_polygons = len(cds_ids) * 2

    stop_sign_features = [f for f in map_features if f.HasField('stop_sign')]
    num_stop_signs = len(stop_sign_features)

    if len(scenario.dynamic_map_states) > 0:
        hist_idx = scenario.current_time_index
        lane_states_at_hist = scenario.dynamic_map_states[hist_idx].lane_states
        num_traffic_lights = len(lane_states_at_hist)
    else:
        lane_states_at_hist = []
        num_traffic_lights = 0

    num_polygons = num_lanes + num_cds_polygons + num_stop_signs + num_traffic_lights

    lane_base = 0
    cds_base = lane_base + num_lanes
    stop_base = cds_base + num_cds_polygons
    tl_base = stop_base + num_stop_signs

    polygon_position = torch.zeros(num_polygons, dim, dtype=torch.float)
    polygon_orientation = torch.zeros(num_polygons, dtype=torch.float)
    polygon_height = torch.zeros(num_polygons, dtype=torch.float)
    polygon_type = torch.zeros(num_polygons, dtype=torch.uint8)
    polygon_is_interpolating = torch.zeros(num_polygons, dtype=torch.uint8)
    point_position: List[Optional[torch.Tensor]] = [None] * num_polygons
    point_orientation: List[Optional[torch.Tensor]] = [None] * num_polygons
    point_magnitude: List[Optional[torch.Tensor]] = [None] * num_polygons
    point_height: List[Optional[torch.Tensor]] = [None] * num_polygons
    point_type: List[Optional[torch.Tensor]] = [None] * num_polygons
    point_side: List[Optional[torch.Tensor]] = [None] * num_polygons

    lane_centerlines = {}
    stop_sign_counter = 0

    for feature in map_features:
        feature_id = feature.id
        if feature.HasField('lane'):
            lane_segment_idx = lane_segment_id_map[feature_id]
            centerline = feature.lane.polyline
            centerline = torch.tensor(
                [[point.x, point.y, point.z] for point in centerline],
                dtype=torch.float,
            )
            lane_centerlines[feature.id] = centerline
            polygon_position[lane_segment_idx] = centerline[0, :dim]
            if centerline.shape[0] > 1:
                polygon_orientation[lane_segment_idx] = torch.atan2(
                    centerline[1, 1] - centerline[0, 1],
                    centerline[1, 0] - centerline[0, 0],
                )
                polygon_height[lane_segment_idx] = centerline[1, 2] - centerline[0, 2]
            else:
                polygon_orientation[lane_segment_idx] = torch.tensor(0.0)
                polygon_height[lane_segment_idx] = torch.tensor(0.0)
            polygon_type[lane_segment_idx] = feature.lane.type
            polygon_is_interpolating[lane_segment_idx] = feature.lane.interpolating

            left_boundaries = sorted(
                feature.lane.left_boundaries, key=lambda b: b.lane_start_index
            )
            left_boundaries_polyline = []
            left_boundaries_types = []

            for left_boundary in left_boundaries:
                left_boundary_idx = left_boundary.boundary_feature_id
                left_boundary_idx = map_feature_dict.get(left_boundary_idx)
                left_boundary_feature_map = map_features[left_boundary_idx]

                if left_boundary_feature_map.HasField('road_edge'):
                    left_boundary_feature = left_boundary_feature_map.road_edge
                    boundary_type = POINT_TYPE_IDX['ROAD_EDGE']
                else:
                    left_boundary_feature = left_boundary_feature_map.road_line
                    boundary_type = left_boundary.boundary_type

                full_left_boundary_polyline = torch.tensor(
                    [
                        [point.x, point.y, point.z]
                        for point in left_boundary_feature.polyline
                    ],
                    dtype=torch.float,
                )
                cropped_left_boundary_polyline = extract_relevant_boundary(
                    full_left_boundary_polyline, centerline
                )

                if cropped_left_boundary_polyline.shape[0] > 0:
                    left_boundaries_polyline.append(cropped_left_boundary_polyline)
                    left_boundaries_types.append(
                        torch.full(
                            (len(cropped_left_boundary_polyline),),
                            boundary_type,
                            dtype=torch.long,
                        )
                    )

            left_boundaries_polyline = (
                torch.cat(left_boundaries_polyline, dim=0)
                if left_boundaries_polyline
                else torch.empty((0, 3))
            )
            left_boundaries_types = (
                torch.cat(left_boundaries_types)
                if left_boundaries_types
                else torch.empty((0,), dtype=torch.long)
            )

            right_boundaries = sorted(
                feature.lane.right_boundaries, key=lambda b: b.lane_start_index
            )
            right_boundaries_polyline = []
            right_boundaries_types = []

            for right_boundary in right_boundaries:
                right_boundary_idx = right_boundary.boundary_feature_id
                right_boundary_idx = map_feature_dict.get(right_boundary_idx)
                right_boundary_feature_map = map_features[right_boundary_idx]

                if right_boundary_feature_map.HasField('road_edge'):
                    right_boundary_feature = right_boundary_feature_map.road_edge
                    boundary_type = POINT_TYPE_IDX['ROAD_EDGE']
                else:
                    right_boundary_feature = right_boundary_feature_map.road_line
                    boundary_type = right_boundary.boundary_type

                full_right_boundary_polyline = torch.tensor(
                    [
                        [point.x, point.y, point.z]
                        for point in right_boundary_feature.polyline
                    ],
                    dtype=torch.float,
                )
                cropped_right_boundary_polyline = extract_relevant_boundary(
                    full_right_boundary_polyline, centerline
                )

                if cropped_right_boundary_polyline.shape[0] > 0:
                    right_boundaries_polyline.append(cropped_right_boundary_polyline)
                    right_boundaries_types.append(
                        torch.full(
                            (len(cropped_right_boundary_polyline),),
                            boundary_type,
                            dtype=torch.long,
                        )
                    )

            right_boundaries_polyline = (
                torch.cat(right_boundaries_polyline, dim=0)
                if right_boundaries_polyline
                else torch.empty((0, 3))
            )
            right_boundaries_types = (
                torch.cat(right_boundaries_types)
                if right_boundaries_types
                else torch.empty((0,), dtype=torch.long)
            )

            point_data = []
            orientation_data = []
            magnitude_data = []
            height_data = []
            type_data = []
            side_data = []

            if left_boundaries_polyline.shape[0] > 1:
                left_vectors = (
                    left_boundaries_polyline[1:] - left_boundaries_polyline[:-1]
                )
                point_data.append(left_boundaries_polyline[:-1, :dim])
                orientation_data.append(
                    torch.atan2(left_vectors[:, 1], left_vectors[:, 0])
                )
                magnitude_data.append(torch.norm(left_vectors[:, :2], p=2, dim=-1))
                height_data.append(left_vectors[:, 2])
                type_data.append(
                    left_boundaries_types[:-1].clone().detach().to(dtype=torch.uint8)
                )
                side_data.append(
                    torch.full(
                        (len(left_vectors),),
                        POINT_SIDE_IDX['LEFT'],
                        dtype=torch.uint8,
                    )
                )

            if right_boundaries_polyline.shape[0] > 1:
                right_vectors = (
                    right_boundaries_polyline[1:] - right_boundaries_polyline[:-1]
                )
                point_data.append(right_boundaries_polyline[:-1, :dim])
                orientation_data.append(
                    torch.atan2(right_vectors[:, 1], right_vectors[:, 0])
                )
                magnitude_data.append(torch.norm(right_vectors[:, :2], p=2, dim=-1))
                height_data.append(right_vectors[:, 2])
                type_data.append(
                    right_boundaries_types[:-1].clone().detach().to(dtype=torch.uint8)
                )
                side_data.append(
                    torch.full(
                        (len(right_vectors),),
                        POINT_SIDE_IDX['RIGHT'],
                        dtype=torch.uint8,
                    )
                )

            if centerline.shape[0] == 1:
                centerline = torch.cat(
                    [centerline, centerline + torch.tensor([[0.1, 0.1, 0.0]])],
                    dim=0,
                )
            center_vectors = centerline[1:] - centerline[:-1]
            point_data.append(centerline[1:, :dim])
            orientation_data.append(
                torch.atan2(center_vectors[:, 1], center_vectors[:, 0])
            )
            magnitude_data.append(torch.norm(center_vectors[:, :2], p=2, dim=-1))
            height_data.append(center_vectors[:, 2])
            type_data.append(
                torch.full(
                    (len(center_vectors),),
                    POINT_TYPE_IDX['CENTERLINE'],
                    dtype=torch.uint8,
                )
            )
            side_data.append(
                torch.full(
                    (len(center_vectors),),
                    POINT_SIDE_IDX['CENTER'],
                    dtype=torch.uint8,
                )
            )

            if point_data:
                point_position[lane_segment_idx] = torch.cat(point_data, dim=0)
                point_orientation[lane_segment_idx] = torch.cat(orientation_data, dim=0)
                point_magnitude[lane_segment_idx] = torch.cat(magnitude_data, dim=0)
                point_height[lane_segment_idx] = torch.cat(height_data, dim=0)
                point_type[lane_segment_idx] = torch.cat(type_data, dim=0)
                point_side[lane_segment_idx] = torch.cat(side_data, dim=0)

        if (
            feature.HasField('crosswalk')
            or feature.HasField('driveway')
            or feature.HasField('speed_bump')
        ):
            is_crosswalk = feature.HasField('crosswalk')
            is_driveway = feature.HasField('driveway')

            if is_crosswalk:
                polygon_type_name = 'TYPE_CROSSWALK'
                point_type_name = 'CROSSWALK'
                polygon = feature.crosswalk.polygon
            elif is_driveway:
                polygon_type_name = 'TYPE_DRIVEWAY'
                point_type_name = 'DRIVEWAY'
                polygon = feature.driveway.polygon
            else:
                polygon_type_name = 'TYPE_SPEED_BUMP'
                point_type_name = 'SPEED_BUMP'
                polygon = feature.speed_bump.polygon

            polygon_tensor = torch.tensor(
                [[point.x, point.y, point.z] for point in polygon],
                dtype=torch.float,
            )

            polygon_idx = polygon_id_map[feature.id] + len(lane_segment_ids)

            (
                start_position,
                end_position,
                left_boundary,
                right_boundary,
                centerline,
            ) = compute_crosswalk_boundaries(polygon_tensor)

            polygon_position[polygon_idx] = start_position[:dim]
            polygon_position[polygon_idx + len(polygon_id_map)] = end_position[:dim]
            polygon_orientation[polygon_idx] = torch.atan2(
                (end_position - start_position)[1],
                (end_position - start_position)[0],
            )
            polygon_orientation[polygon_idx + len(polygon_id_map)] = torch.atan2(
                (start_position - end_position)[1],
                (start_position - end_position)[0],
            )
            polygon_height[polygon_idx] = end_position[2] - start_position[2]
            polygon_height[polygon_idx + len(polygon_id_map)] = (
                start_position[2] - end_position[2]
            )

            polygon_type[polygon_idx] = POLYGON_TYPE_IDX[polygon_type_name]
            polygon_type[polygon_idx + len(polygon_id_map)] = POLYGON_TYPE_IDX[
                polygon_type_name
            ]

            polygon_is_interpolating[polygon_idx] = False
            polygon_is_interpolating[polygon_idx + len(polygon_id_map)] = False

            if (
                side_to_directed_lineseg(
                    (left_boundary[0] + right_boundary[-1]) / 2,
                    start_position,
                    end_position,
                )
                != 'LEFT'
            ):
                left_boundary, right_boundary = right_boundary, left_boundary

            point_position[polygon_idx] = torch.cat(
                [
                    left_boundary[:-1, :dim],
                    right_boundary[:-1, :dim],
                    centerline[:-1, :dim],
                ],
                dim=0,
            )
            point_position[polygon_idx + len(polygon_id_map)] = torch.cat(
                [
                    right_boundary.flip(dims=[0])[:-1, :dim],
                    left_boundary.flip(dims=[0])[:-1, :dim],
                    centerline.flip(dims=[0])[:-1, :dim],
                ],
                dim=0,
            )

            left_vectors = left_boundary[1:] - left_boundary[:-1]
            right_vectors = right_boundary[1:] - right_boundary[:-1]
            center_vectors = centerline[1:] - centerline[:-1]

            point_orientation[polygon_idx] = torch.cat(
                [
                    torch.atan2(left_vectors[:, 1], left_vectors[:, 0]),
                    torch.atan2(right_vectors[:, 1], right_vectors[:, 0]),
                    torch.atan2(center_vectors[:, 1], center_vectors[:, 0]),
                ],
                dim=0,
            )

            point_orientation[polygon_idx + len(polygon_id_map)] = torch.cat(
                [
                    torch.atan2(
                        -right_vectors.flip(dims=[0])[:, 1],
                        -right_vectors.flip(dims=[0])[:, 0],
                    ),
                    torch.atan2(
                        -left_vectors.flip(dims=[0])[:, 1],
                        -left_vectors.flip(dims=[0])[:, 0],
                    ),
                    torch.atan2(
                        -center_vectors.flip(dims=[0])[:, 1],
                        -center_vectors.flip(dims=[0])[:, 0],
                    ),
                ],
                dim=0,
            )

            point_magnitude[polygon_idx] = torch.norm(
                torch.cat(
                    [
                        left_vectors[:, :2],
                        right_vectors[:, :2],
                        center_vectors[:, :2],
                    ],
                    dim=0,
                ),
                p=2,
                dim=-1,
            )
            point_magnitude[polygon_idx + len(polygon_id_map)] = torch.norm(
                torch.cat(
                    [
                        -right_vectors.flip(dims=[0])[:, :2],
                        -left_vectors.flip(dims=[0])[:, :2],
                        -center_vectors.flip(dims=[0])[:, :2],
                    ],
                    dim=0,
                ),
                p=2,
                dim=-1,
            )

            point_height[polygon_idx] = torch.cat(
                [left_vectors[:, 2], right_vectors[:, 2], center_vectors[:, 2]],
                dim=0,
            )
            point_height[polygon_idx + len(polygon_id_map)] = torch.cat(
                [
                    -right_vectors.flip(dims=[0])[:, 2],
                    -left_vectors.flip(dims=[0])[:, 2],
                    -center_vectors.flip(dims=[0])[:, 2],
                ],
                dim=0,
            )

            polygon_type_idx = POINT_TYPE_IDX[point_type_name]
            center_type = POINT_TYPE_IDX['CENTERLINE']
            point_type[polygon_idx] = torch.cat(
                [
                    torch.full(
                        (len(left_vectors),), polygon_type_idx, dtype=torch.uint8
                    ),
                    torch.full(
                        (len(right_vectors),), polygon_type_idx, dtype=torch.uint8
                    ),
                    torch.full((len(center_vectors),), center_type, dtype=torch.uint8),
                ],
                dim=0,
            )

            point_type[polygon_idx + len(polygon_id_map)] = torch.cat(
                [
                    torch.full(
                        (len(right_vectors),), polygon_type_idx, dtype=torch.uint8
                    ),
                    torch.full(
                        (len(left_vectors),), polygon_type_idx, dtype=torch.uint8
                    ),
                    torch.full((len(center_vectors),), center_type, dtype=torch.uint8),
                ],
                dim=0,
            )

            point_side[polygon_idx] = torch.cat(
                [
                    torch.full(
                        (len(left_vectors),),
                        POINT_SIDE_IDX['LEFT'],
                        dtype=torch.uint8,
                    ),
                    torch.full(
                        (len(right_vectors),),
                        POINT_SIDE_IDX['RIGHT'],
                        dtype=torch.uint8,
                    ),
                    torch.full(
                        (len(center_vectors),),
                        POINT_SIDE_IDX['CENTER'],
                        dtype=torch.uint8,
                    ),
                ],
                dim=0,
            )

            point_side[polygon_idx + len(polygon_id_map)] = torch.cat(
                [
                    torch.full(
                        (len(right_vectors),),
                        POINT_SIDE_IDX['LEFT'],
                        dtype=torch.uint8,
                    ),
                    torch.full(
                        (len(left_vectors),),
                        POINT_SIDE_IDX['RIGHT'],
                        dtype=torch.uint8,
                    ),
                    torch.full(
                        (len(center_vectors),),
                        POINT_SIDE_IDX['CENTER'],
                        dtype=torch.uint8,
                    ),
                ],
                dim=0,
            )

        if feature.HasField('stop_sign'):
            polygon_idx = stop_base + stop_sign_counter
            stop_sign_counter += 1

            ss = feature.stop_sign
            pos = ss.position
            pos_tensor = torch.tensor([pos.x, pos.y, pos.z], dtype=torch.float)

            if len(ss.lane) > 0:
                lane_id = ss.lane[0]
                ori = orientation_on_lane(lane_centerlines[lane_id], pos_tensor)
            else:
                ori = torch.tensor(0.0)

            polygon_position[polygon_idx] = pos_tensor[:dim]
            polygon_orientation[polygon_idx] = ori
            if dim == 3:
                polygon_height[polygon_idx] = 0.0
            polygon_type[polygon_idx] = POLYGON_TYPE_IDX['TYPE_STOP_SIGN']
            polygon_is_interpolating[polygon_idx] = 0

            point_position[polygon_idx] = pos_tensor[:dim].unsqueeze(0)
            point_orientation[polygon_idx] = ori.unsqueeze(0)
            point_magnitude[polygon_idx] = torch.tensor([0.0])
            if dim == 3:
                point_height[polygon_idx] = torch.tensor([0.0])
            point_type[polygon_idx] = torch.tensor(
                [POINT_TYPE_IDX['STOP_SIGN']], dtype=torch.uint8
            )
            point_side[polygon_idx] = torch.tensor(
                [POINT_SIDE_IDX['CENTER']], dtype=torch.uint8
            )

    tl_counter = 0
    for lane_state in lane_states_at_hist:
        polygon_idx = tl_base + tl_counter
        tl_counter += 1

        lane_id = lane_state.lane
        sp = lane_state.stop_point
        pos_tensor = torch.tensor([sp.x, sp.y, sp.z], dtype=torch.float)

        ori = orientation_on_lane(lane_centerlines[lane_id], pos_tensor)

        polygon_position[polygon_idx] = pos_tensor[:dim]
        polygon_orientation[polygon_idx] = ori
        if dim == 3:
            polygon_height[polygon_idx] = 0.0
        polygon_type[polygon_idx] = POLYGON_TYPE_IDX['TYPE_TRAFFIC_LIGHT']
        polygon_is_interpolating[polygon_idx] = 0

        state_name = TRAFFIC_SIGNAL_TYPES[lane_state.state]
        state_idx = POINT_TYPE_IDX[state_name]

        point_position[polygon_idx] = pos_tensor[:dim].unsqueeze(0)
        point_orientation[polygon_idx] = ori.unsqueeze(0)
        point_magnitude[polygon_idx] = torch.tensor([0.0])
        if dim == 3:
            point_height[polygon_idx] = torch.tensor([0.0])
        point_type[polygon_idx] = torch.tensor([state_idx], dtype=torch.uint8)
        point_side[polygon_idx] = torch.tensor(
            [POINT_SIDE_IDX['CENTER']], dtype=torch.uint8
        )

    num_points = torch.tensor(
        [point.size(0) for point in point_position], dtype=torch.long
    )
    point_to_polygon_edge_index = torch.stack(
        [
            torch.arange(num_points.sum(), dtype=torch.long),
            torch.arange(num_polygons, dtype=torch.long).repeat_interleave(num_points),
        ],
        dim=0,
    )
    polygon_to_polygon_edge_index = []
    polygon_to_polygon_type = []
    for lane_segment in lane_segment_ids:
        lane_segment_feature_idx = map_feature_dict.get(lane_segment)
        lane_segment_feature = map_features[lane_segment_feature_idx].lane
        lane_segment_idx = lane_segment_id_map[lane_segment]
        pred_inds = []
        for pred in lane_segment_feature.entry_lanes:
            pred_id = lane_segment_id_map[pred]
            pred_inds.append(pred_id)
        if len(pred_inds) > 0:
            polygon_to_polygon_edge_index.append(
                torch.stack(
                    [
                        torch.tensor(pred_inds, dtype=torch.long),
                        torch.full(
                            (len(pred_inds),), lane_segment_idx, dtype=torch.long
                        ),
                    ],
                    dim=0,
                )
            )
            polygon_to_polygon_type.append(
                torch.full(
                    (len(pred_inds),),
                    POLYGON_TO_POLYGON_TYPE_IDX['PRED'],
                    dtype=torch.uint8,
                )
            )
        succ_inds = []
        for succ in lane_segment_feature.exit_lanes:
            succ_id = lane_segment_id_map[succ]
            succ_inds.append(succ_id)
        if len(succ_inds) > 0:
            polygon_to_polygon_edge_index.append(
                torch.stack(
                    [
                        torch.tensor(succ_inds, dtype=torch.long),
                        torch.full(
                            (len(succ_inds),), lane_segment_idx, dtype=torch.long
                        ),
                    ],
                    dim=0,
                )
            )
            polygon_to_polygon_type.append(
                torch.full(
                    (len(succ_inds),),
                    POLYGON_TO_POLYGON_TYPE_IDX['SUCC'],
                    dtype=torch.uint8,
                )
            )
        left_neighbors = lane_segment_feature.left_neighbors
        for left_neighbor in left_neighbors:
            left_idx = lane_segment_id_map[left_neighbor.feature_id]
            polygon_to_polygon_edge_index.append(
                torch.tensor([[left_idx], [lane_segment_idx]], dtype=torch.long)
            )
            polygon_to_polygon_type.append(
                torch.tensor([POLYGON_TO_POLYGON_TYPE_IDX['LEFT']], dtype=torch.uint8)
            )
        right_neighbors = lane_segment_feature.right_neighbors
        for right_neighbor in right_neighbors:
            right_idx = lane_segment_id_map[right_neighbor.feature_id]
            polygon_to_polygon_edge_index.append(
                torch.tensor([[right_idx], [lane_segment_idx]], dtype=torch.long)
            )
            polygon_to_polygon_type.append(
                torch.tensor([POLYGON_TO_POLYGON_TYPE_IDX['RIGHT']], dtype=torch.uint8)
            )
    if len(polygon_to_polygon_edge_index) != 0:
        polygon_to_polygon_edge_index = torch.cat(polygon_to_polygon_edge_index, dim=1)
        polygon_to_polygon_type = torch.cat(polygon_to_polygon_type, dim=0)
    else:
        polygon_to_polygon_edge_index = torch.tensor([[], []], dtype=torch.long)
        polygon_to_polygon_type = torch.tensor([], dtype=torch.uint8)

    map_data = {
        'map_polygon': {},
        'map_point': {},
        ('map_point', 'to', 'map_polygon'): {},
        ('map_polygon', 'to', 'map_polygon'): {},
    }
    map_data['map_polygon']['num_nodes'] = num_polygons
    map_data['map_polygon']['position'] = polygon_position
    map_data['map_polygon']['orientation'] = polygon_orientation
    if dim == 3:
        map_data['map_polygon']['height'] = polygon_height
    map_data['map_polygon']['type'] = polygon_type
    map_data['map_polygon']['is_interpolating'] = polygon_is_interpolating
    if len(num_points) == 0:
        map_data['map_point']['num_nodes'] = 0
        map_data['map_point']['position'] = torch.tensor([], dtype=torch.float)
        map_data['map_point']['orientation'] = torch.tensor([], dtype=torch.float)
        map_data['map_point']['magnitude'] = torch.tensor([], dtype=torch.float)
        if dim == 3:
            map_data['map_point']['height'] = torch.tensor([], dtype=torch.float)
        map_data['map_point']['type'] = torch.tensor([], dtype=torch.uint8)
        map_data['map_point']['side'] = torch.tensor([], dtype=torch.uint8)
    else:
        map_data['map_point']['num_nodes'] = num_points.sum().item()
        map_data['map_point']['position'] = torch.cat(point_position, dim=0)
        map_data['map_point']['orientation'] = torch.cat(point_orientation, dim=0)
        map_data['map_point']['magnitude'] = torch.cat(point_magnitude, dim=0)
        if dim == 3:
            map_data['map_point']['height'] = torch.cat(point_height, dim=0)
        map_data['map_point']['type'] = torch.cat(point_type, dim=0)
        map_data['map_point']['side'] = torch.cat(point_side, dim=0)
    map_data['map_point', 'to', 'map_polygon']['edge_index'] = (
        point_to_polygon_edge_index
    )
    map_data['map_polygon', 'to', 'map_polygon']['edge_index'] = (
        polygon_to_polygon_edge_index
    )
    map_data['map_polygon', 'to', 'map_polygon']['type'] = polygon_to_polygon_type

    return map_data


def process_scenario_file(
    raw_file_name,
    raw_dir,
    processed_dir,
    dim,
    num_historical_steps,
    num_steps,
    predict_unseen_agents,
):
    files = os.path.join(raw_dir, raw_file_name)
    scenarios = extract_scenarios_from_tfrecord(files)

    for scenario in scenarios:
        scenario_id = get_scenario_id(scenario)
        data = dict()
        data['scenario_id'] = scenario_id
        data['agent'] = get_agent_features(
            scenario,
            num_historical_steps,
            num_steps,
            dim,
            predict_unseen_agents,
        )
        data.update(get_map_features(scenario, dim))

        with open(
            os.path.join(processed_dir, f'scenario_{scenario_id}.pkl'), 'wb'
        ) as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def preprocess(
    root,
    split,
    raw_dir=None,
    processed_dir=None,
    dim=3,
    num_historical_steps=11,
    num_future_steps=80,
    predict_unseen_agents=False,
    num_workers=None,
    overwrite=False,
):
    root = os.path.expanduser(os.path.normpath(root))

    if split not in ('train', 'val', 'test'):
        raise ValueError(f'{split} is not a valid split')

    if raw_dir is None:
        raw_dir = os.path.join(root, 'waymo', 'raw', RAW_DIR_NAMES[split])
    else:
        raw_dir = os.path.expanduser(os.path.normpath(raw_dir))

    print(f'Loading raw data from {raw_dir}')

    if processed_dir is None:
        processed_dir = os.path.join(root, 'waymo', 'processed', split)
    else:
        processed_dir = os.path.expanduser(os.path.normpath(processed_dir))

    print(f'Writing processed data to {processed_dir}')

    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(raw_dir)

    os.makedirs(processed_dir, exist_ok=True)

    processed_file_names = [
        entry.name
        for entry in os.scandir(processed_dir)
        if entry.is_file() and entry.name.endswith('.pkl')
    ]

    if not overwrite and len(processed_file_names) == NUM_SAMPLES[split]:
        print(
            'Processed files are already available. Skipping processing.',
            file=sys.stderr,
        )
        return

    if overwrite:
        for name in os.listdir(processed_dir):
            if name.endswith(('pkl', 'pickle')):
                os.remove(os.path.join(processed_dir, name))

    raw_file_names = sorted(
        entry.name for entry in os.scandir(raw_dir) if entry.is_file()
    )

    num_steps = num_historical_steps + num_future_steps

    worker = partial(
        process_scenario_file,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        dim=dim,
        num_historical_steps=num_historical_steps,
        num_steps=num_steps,
        predict_unseen_agents=predict_unseen_agents,
    )

    print('Note: Due to multiprocessing, the progress bar will move irregularly.')

    if num_workers == 1:
        list(
            tqdm(
                map(worker, raw_file_names),
                total=len(raw_file_names),
                smoothing=0,
            )
        )
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(
                tqdm(
                    executor.map(worker, raw_file_names),
                    total=len(raw_file_names),
                    smoothing=0,
                )
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument('--raw_dir', type=str, default=None)
    parser.add_argument('--processed_dir', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    for split in ['train', 'val']:
        print(f'Preprocessing {split}')
        preprocess(
            root=args.data_root,
            split=split,
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            num_workers=args.num_workers,
            overwrite=args.overwrite,
        )


if __name__ == '__main__':
    main()
