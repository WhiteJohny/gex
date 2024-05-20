from __future__ import annotations

import math
import random
import numpy as np
import plotly.express as px

from itertools import product

N = 5

field = np.array(list(product(np.arange(N), repeat=3)))
data = np.array([list(field[i]) + [-1] for i in range(N ** 3)])


def set_data(coords: list[int, int, int], info: int) -> None:
    x, y, z = coords
    idx = x * N ** 2 + y * N + z
    data[idx][-1] = info


set_data([2, 2, 2], 0)

set_data([2, 3, 1], 1)
set_data([2, 1, 3], 1)
set_data([1, 2, 3], 1)
set_data([3, 2, 1], 1)
set_data([1, 3, 2], 1)
set_data([3, 1, 2], 1)


def set_rel_data(coords: list[int, int, int], info: int) -> None:
    x, y, z = np.array(coords) + N // 2
    idx = x * N ** 2 + y * N + z
    data[idx][-1] = info


set_rel_data([0, 2, -2], 2)
set_rel_data([1, 1, -2], 2)
set_rel_data([2, 0, -2], 2)
set_rel_data([2, -1, -1], 2)
set_rel_data([2, -2, 0], 2)
set_rel_data([1, -2, 1], 2)
set_rel_data([0, -2, 2], 2)
set_rel_data([-1, -1, 2], 2)
set_rel_data([-2, 0, 2], 2)
set_rel_data([-2, 1, 1], 2)
set_rel_data([-2, 2, 0], 2)
set_rel_data([-1, 2, -1], 2)

fig = px.scatter_3d(
    data,
    x=data[:, 0],
    y=data[:, 1],
    z=data[:, 2],
    color=data[:, 3]
)
fig.show()

data_to_write = data[data[:, 3] != -1]
fig = px.scatter_3d(
    data_to_write,
    x=data_to_write[:, 0],
    y=data_to_write[:, 1],
    z=data_to_write[:, 2],
    color=data_to_write[:, 3]
)


fig.show()


def create_map(max_dist: int):
    N = max_dist * 2 + 1
    field = np.array(list(product(np.arange(N), repeat=3)))
    data = np.array([list(field[i]) + [1] for i in range(N ** 3)])
    return data


MAX_DIST = 20
map_data = create_map(MAX_DIST)


def get_hex_idxs(coords_map):
    return np.array(np.where(coords_map[:, 0] + coords_map[:, 1] + coords_map[:, 2] == 3 * MAX_DIST))[0]


hexagon_idxs = get_hex_idxs(map_data)
print(hexagon_idxs)

hexagon_data = map_data[hexagon_idxs]
for hexagon in hexagon_data:
    x, y, z, d = hexagon
    hexagon[-1] = max(abs(x) - MAX_DIST, abs(y) - MAX_DIST, abs(z) - MAX_DIST)

fig = px.scatter_3d(
    hexagon_data,
    x=hexagon_data[:, 0],
    y=hexagon_data[:, 1],
    z=hexagon_data[:, 2],
    color=hexagon_data[:, 3]
)
fig.show()

new_map = create_map(MAX_DIST)
hex_idxs = get_hex_idxs(new_map)
hex_data = new_map[hex_idxs]

fig = px.scatter_3d(
    hex_data,
    x=hex_data[:, 0],
    y=hex_data[:, 1],
    z=hex_data[:, 2],
    color=hex_data[:, 3]
)
fig.show()

ALL_MOVEMENTS = np.array([
    [0, 1, -1],
    [1, 0, -1],
    [1, -1, 0],
    [0, -1, 1],
    [-1, 0, 1],
    [-1, 1, 0]
])

river_movements = ALL_MOVEMENTS[0:2]

RIVERS_COUNT = 40
MAX_RIVER_LENGTH = 7
RIVER_TYPE = 10

np.random.seed(42)
random.seed(42)

river_map = np.copy(new_map)
N = MAX_DIST * 2 + 1

for _ in range(RIVERS_COUNT):
    river_length = random.randint(0, MAX_RIVER_LENGTH)
    hex_idx = np.random.choice(hex_idxs)
    possible_steps = river_movements
    counter = 0

    while counter < river_length:
        if len(possible_steps) == 0:
            break

        river_map[hex_idx][-1] = RIVER_TYPE
        step = possible_steps[random.randint(0, len(possible_steps) - 1)]
        coords = river_map[hex_idx][:-1] + step

        new_hex_idx = coords[0] * N ** 2 + coords[1] * N + coords[2]
        if new_hex_idx not in hex_idxs:
            new_possible_steps = []
            for item in possible_steps:
                if list(item) != list(step):
                    new_possible_steps.append(item)
            possible_steps = np.array(new_possible_steps)
            continue
        else:
            possible_steps = river_movements

        hex = river_map[new_hex_idx]
        if hex[-1] == RIVER_TYPE:
            hex_idx = new_hex_idx
        else:
            counter += 1
            hex_idx = new_hex_idx

hex_data = river_map[hex_idxs]

fig = px.scatter_3d(
    hex_data,
    x=hex_data[:, 0],
    y=hex_data[:, 1],
    z=hex_data[:, 2],
    color=hex_data[:, 3]
)
fig.show()

hills_map = np.copy(river_map)

HILLS_COUNT = 10
HILLS_TYPE = 100

for _ in range(HILLS_COUNT):
    hex_idx = np.random.choice(hex_idxs)
    while True:
        hex = river_map[hex_idx]
        if 0 in hex[:-1] or 40 in hex[:-1]:
            hex_idx = np.random.choice(hex_idxs)
        else:
            break

    hills_map[hex_idx][-1] = HILLS_TYPE

    for step in ALL_MOVEMENTS:
        coords = hills_map[hex_idx][:-1] + step
        new_hex_idx = coords[0] * N ** 2 + coords[1] * N + coords[2]
        if new_hex_idx not in hex_idxs:
            continue
        hills_map[new_hex_idx][-1] = HILLS_TYPE

PLAIN_TYPE = 1

types_map = {
    PLAIN_TYPE: "Plain",
    HILLS_TYPE: "Hill",
    RIVER_TYPE: "River",
}

vfunc = np.vectorize(types_map.get)

hex_data = hills_map[hex_idxs]
colors = vfunc(hex_data[:, 3])

fig = px.scatter_3d(
    hex_data,
    x=hex_data[:, 0],
    y=hex_data[:, 1],
    z=hex_data[:, 2],
    hover_name=hex_idxs,
    color=colors,
    color_discrete_sequence=["#387C44", "#3EA99F", "gray"],
)
fig.show()

ROUTE_TYPE = 1000


def visualize_solve(route_path):
    route_map = np.copy(hills_map)

    for cell_idx in route_path:
        route_map[cell_idx][-1] = ROUTE_TYPE

    hex_data = route_map[hex_idxs]
    vfunc = np.vectorize(types_map.get)

    colors = vfunc(hex_data[:, 3])

    fig = px.scatter_3d(
        hex_data,
        x=hex_data[:, 0],
        y=hex_data[:, 1],
        z=hex_data[:, 2],
        hover_name=hex_idxs,
        color=colors,
        color_discrete_sequence=["#387C44", "#3EA99F", "gray", "maroon"],
    )
    fig.show()


visualize_solve([59_460, 57_780, 57_740, 56_060, 54_380])

back_from_idx = 52_940
back_to_idx = 45_860


def count_dist(a_idx, b_idx):
    a_data = hills_map[a_idx][:3]
    b_data = hills_map[b_idx][:3]
    return math.sqrt(sum((a_data - b_data) ** 2))


print(count_dist(back_from_idx, back_to_idx))

init_dist = count_dist(back_from_idx, back_to_idx)


def backtracking(curr_idx, path, prev_dist, prev_movement_idx):
    pass


greedy_path = [back_from_idx]
curr_idx = back_from_idx
init_dist = count_dist(back_from_idx, back_to_idx)

curr_dist = init_dist
curr_cords = hills_map[curr_idx][:-1]

while curr_dist > 0:
    for step in ALL_MOVEMENTS:
        new_cords = curr_cords + step
        new_idx = new_cords[0] * N ** 2 + new_cords[1] * N + new_cords[2]
        dist = count_dist(new_idx, back_to_idx)
        if curr_dist > dist:
            curr_dist = dist
            new_curr_idx = new_idx
            new_curr_cords = hills_map[new_curr_idx][:-1]

    curr_cords = new_curr_cords
    greedy_path.append(new_curr_idx)

print(greedy_path)
visualize_solve(greedy_path)


def count_greedy_path(idx_from, idx_to):
    pass


part_idxs = hex_idxs[hex_idxs > 51_000]
part_hex_data = hills_map[part_idxs]
colors = vfunc(part_hex_data[:, 3])

fig = px.scatter_3d(
    hex_data,
    x=part_hex_data[:, 0],
    y=part_hex_data[:, 1],
    z=part_hex_data[:, 2],
    hover_name=part_idxs,
    color=colors,
    color_discrete_sequence=["#387C44", "#3EA99F", "gray"],
)
fig.show()

from_idx = 64_620
to_idx = 52_140
visualize_solve([from_idx, to_idx])

possible_movements = np.array([
    [1, 0, -1],
    [0, 1, -1],
])

part_n = 8
part_m = 19
weights = np.full((part_n, part_m), PLAIN_TYPE)

to_coords = hills_map[to_idx][:-1]

for i in range(part_n):
    row_coords = to_coords + (possible_movements[0] * i)
    for j in range(part_m):
        cell_coords = row_coords + (possible_movements[1] * j)
        cell_idx = cell_coords[0] * N ** 2 + cell_coords[1] * N + cell_coords[2]
        cell = hills_map[cell_idx]
        cell_d = cell[-1]

        elems_to_compare = []

        if i > 0:
            elems_to_compare.append(weights[i - 1][j])

        if j > 0:
            elems_to_compare.append(weights[i][j - 1])

        weights[i][j] = cell_d

        if len(elems_to_compare):
            weights[i][j] += min(elems_to_compare)

path = []
curr_i, curr_j = np.array(weights.shape) - 1

while curr_i > 0 or curr_j > 0:
    cell_coords = to_coords + (possible_movements[0] * curr_i) + (possible_movements[1] * curr_j)
    cell_idx = cell_coords[0] * N ** 2 + cell_coords[1] * N + cell_coords[2]

    path.append(cell_idx)

    left_value = weights[curr_i][curr_j - 1] if curr_j > 0 else 1000
    top_value = weights[curr_i - 1][curr_j] if curr_i > 0 else 1000

    if left_value <= top_value:
        curr_j -= 1
    else:
        curr_i -= 1

path.append(to_idx)
path = np.array(path)
print(path)
visualize_solve(path)

new_from_idx = 42_140
new_to_idx = 64_300


def get_diff(idx_from, idx_to):
    return hills_map[idx_from][:3] - hills_map[idx_to][:3]


def get_movements(idx_from, idx_to):
    positive, negative = [], []
    diff = get_diff(idx_from, idx_to)
    for i in range(3):
        if diff[i] > 0:
            positive.append(i)
        else:
            negative.append(i)

    movement_idxs = np.where(ALL_MOVEMENTS[:, negative[0]] == 1)[0] if len(positive) > len(negative) else np.where(ALL_MOVEMENTS[:, positive[0]] == 1)[0]
    return ALL_MOVEMENTS[movement_idxs], negative[0] if len(positive) > len(negative) else positive[0]


def get_size(movements, fixed_var_idx, diff):
    arr = []
    for move in movements:
        for i in range(3):
            if i != fixed_var_idx and move[i]:
                arr.append(abs(diff[i]) + 1)
    return arr


diff = get_diff(new_from_idx, new_to_idx)
movements, fixed_var_idx = get_movements(new_from_idx, new_to_idx)


def solve_turtle_task(idx_from, idx_to):
    diff = get_diff(idx_from, idx_to)
    possible_movements, fixed_var_idx = get_movements(idx_from, idx_to)
    size = get_size(movements, fixed_var_idx, diff)
    part_n, part_m = size
    weights = np.full((part_n, part_m), PLAIN_TYPE)
    to_coords = hills_map[idx_to][:-1]

    for i in range(part_n):
        row_coords = to_coords + (possible_movements[0] * i)
        for j in range(part_m):
            cell_coords = row_coords + (possible_movements[1] * j)
            cell_idx = cell_coords[0] * N ** 2 + cell_coords[1] * N + cell_coords[2]
            cell = hills_map[cell_idx]
            cell_d = cell[-1]

            elems_to_compare = []

            if i > 0:
                elems_to_compare.append(weights[i - 1][j])

            if j > 0:
                elems_to_compare.append(weights[i][j - 1])

            weights[i][j] = cell_d

            if len(elems_to_compare):
                weights[i][j] += min(elems_to_compare)

    path = []
    curr_i, curr_j = np.array(weights.shape) - 1

    while curr_i > 0 or curr_j > 0:
        cell_coords = to_coords + (possible_movements[0] * curr_i) + (possible_movements[1] * curr_j)
        cell_idx = cell_coords[0] * N ** 2 + cell_coords[1] * N + cell_coords[2]

        path.append(cell_idx)

        left_value = weights[curr_i][curr_j - 1] if curr_j > 0 else 1000
        top_value = weights[curr_i - 1][curr_j] if curr_i > 0 else 1000

        if left_value <= top_value:
            curr_j -= 1
        else:
            curr_i -= 1

    path.append(idx_to)

    return path


new_path = solve_turtle_task(new_from_idx, new_to_idx)


def draw_map_with_turtle_route(from_idx, to_idx):
    solve_path = solve_turtle_task(from_idx, to_idx)
    visualize_solve(solve_path)


draw_map_with_turtle_route(37_620, 51_500)
