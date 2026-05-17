import warnings
from collections import defaultdict

import bpy
import networkx
import numpy as np
import taichi as ti
from scipy.spatial import cKDTree

# =========================
# 0. Global Constants & Taichi Init
# =========================

if not ti.lang.impl.get_runtime().prog:
    try:
        ti.init(arch=ti.gpu)
    except Exception:
        ti.init(arch=ti.cpu)

# =========================
# 1. Taichi Kernels
# =========================


# =========================
# 2. Geometry Utilities
# =========================


def right_handed(coords: np.ndarray):
    if len(coords) < 3:
        return True
    x = coords[:, 0]
    y = coords[:, 1]
    return np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0


def clear_paths(paths: list[np.ndarray], paths_closed: list[bool], atol=1e-6):
    cleared = []
    for path, closed in zip(paths, paths_closed):
        if closed:
            path = np.append(path, [path[0]], axis=0)
        coords = path[:, :2]
        if len(coords) < 2:
            continue
        dup_mask = np.ones(len(coords), dtype=bool)
        dup_mask[1:] = np.linalg.norm(coords[1:] - coords[:-1], axis=1) > atol
        coords = coords[dup_mask]
        if len(coords) > 2:
            colinr_mask = np.ones(len(coords), dtype=bool)
            v1 = coords[1:-1] - coords[:-2]
            v2 = coords[2:] - coords[1:-1]
            cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            colinr_mask[1:-1] = np.abs(cross) >= atol
            coords = coords[colinr_mask]
        if len(coords) > 1:
            cleared.append(coords)
    return cleared


def merge_paths(paths: list[np.ndarray], atol=1e-6):
    decimals = int(np.log10(1 / atol))
    coords_all = np.concatenate(paths, axis=0)
    coords_all = np.around(coords_all, decimals)

    coord_tuples = [tuple(coord) for coord in coords_all]
    d_coord_node = {coord_key: i for i, coord_key in enumerate(coord_tuples)}

    edges = []
    for path in paths:
        path_tuples = [tuple(np.around(coord, decimals)) for coord in path]
        path_indices = [d_coord_node[t] for t in path_tuples]
        for i in range(len(path_indices) - 1):
            edges.append((path_indices[i], path_indices[i + 1]))

    networkx_graph = networkx.Graph()
    networkx_graph.add_edges_from(edges)

    merged = []
    merged_closed = []
    path_starts = []
    for path in paths:
        start_tuple = tuple(np.around(path[0], decimals))
        path_starts.append(d_coord_node[start_tuple])

    cycles = networkx.cycle_basis(networkx_graph)
    for cycle in cycles:
        if len(cycle) < 3:
            continue
        target_start = next((ps for ps in path_starts if ps in cycle), None)
        if target_start is not None:
            start_idx = cycle.index(target_start)
            cycle = cycle[start_idx:] + cycle[:start_idx]

        coord_list = [coords_all[idx] for idx in cycle]
        coords = np.array(coord_list)
        if not right_handed(coords):
            coords = coords[::-1]
        merged.append(coords)
        merged_closed.append(True)

    for island in networkx.connected_components(networkx_graph):
        island_list = list(island)
        if is_cycle(networkx_graph, island_list):
            continue
        if is_open_walk(networkx_graph, island_list):
            edges_island = [(u, v) for u, v in networkx_graph.edges if u in island_list and v in island_list]
            nodes = longest_chain(edges_island)
            if nodes:
                coord_list = [coords_all[idx] for idx in nodes]
                coords = np.array(coord_list)
                if not right_handed(coords):
                    coord_list.reverse()
                merged.append(np.array(coord_list))
                merged_closed.append(False)

    return merged, merged_closed


def longest_chain(edges):
    if not edges:
        return []
    adj: dict[int, list[int]] = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    start = None
    end = None
    for node, neighbors in adj.items():
        if len(neighbors) == 1:
            if start is None:
                start = node
            else:
                end = node
                break
    if start is None:
        start = edges[0][0]

    chain = [start]
    visited_edges = set()
    current = start
    while True:
        neighbors = adj.get(current, [])
        next_node = None
        for n in neighbors:
            edge = tuple(sorted((current, n)))
            if edge not in visited_edges:
                next_node = n
                visited_edges.add(edge)
                break
        if next_node is None:
            break
        chain.append(next_node)
        current = next_node
        if end and current == end:
            break
    if len(chain) > 1 and chain[0] == chain[-1]:
        chain = chain[:-1]
    return chain


def is_cycle(graph: networkx.Graph, island):
    degrees = [graph.degree(node) for node in island]
    return set(degrees) == {2}


def is_open_walk(graph: networkx.Graph, island):
    if len(island) == 2:
        return True
    degrees = [graph.degree(node) for node in island]
    return set(degrees) == {1, 2} and degrees.count(1) == 2


def vectorized_intersection(P1, P2, P3, P4):
    d1, d2, d13 = P2 - P1, P4 - P3, P1 - P3
    denom = d1[0] * d2[:, 1] - d1[1] * d2[:, 0]
    is_p = np.abs(denom) < 1e-9
    denom_s = np.where(is_p, 1.0, denom)
    ua = (d2[:, 0] * d13[:, 1] - d2[:, 1] * d13[:, 0]) / denom_s
    ub = (d1[0] * d13[:, 1] - d1[1] * d13[:, 0]) / denom_s
    mask = (~is_p) & (ua >= -1e-9) & (ua <= 1 + 1e-9) & (ub >= -1e-9) & (ub <= 1 + 1e-9)
    return P1 + ua[:, None] * d1, mask


def generate_offset_lines(coords: np.ndarray, closed: bool, offset=0.1, limit=5.0):
    coords = coords[:, :2]

    if closed:
        tangents = np.roll(coords, -1, axis=0) - coords
    else:
        tangents = np.diff(coords, axis=0)

    norms = np.linalg.norm(tangents, axis=1)
    norms[norms < 1e-6] = 1.0
    tangents /= norms[:, None]

    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    vn = np.zeros_like(coords)

    if closed:
        n_in = np.roll(normals, 1, axis=0)
        n_out = normals
        denom = 1 + np.sum(n_in * n_out, axis=1)
        denom[denom < 1e-6] = 1e-6
        vn = (n_in + n_out) / denom[:, None]
    else:
        n_in = normals[:-1]
        n_out = normals[1:]
        denom = 1 + np.sum(n_in * n_out, axis=1)
        denom[denom < 1e-6] = 1e-6
        vn[1:-1] = (n_in + n_out) / denom[:, None]
        vn[0] = normals[0]
        vn[-1] = normals[-1]

    lens = np.linalg.norm(vn, axis=1)
    mask = lens > limit
    if np.any(mask):
        vn[mask] *= limit / lens[mask][:, None]

    return coords + vn * offset, coords - vn * offset


class Interlacing:
    def __init__(self, paths: list[np.ndarray], paths_closed: list[bool], offset=0.1, use_preprocess=True):
        paths = [p[:, :2] for p in paths]

        if use_preprocess:
            self.paths, self.paths_closed = merge_paths(clear_paths(paths, paths_closed))
        else:
            self.paths = paths
            self.paths_closed = paths_closed

        self.paths_flag = []
        # self.paths.sort(key=lambda path: (path[0][0], path[0][1]))
        flag = True
        for path, closed in zip(self.paths, self.paths_closed):
            if closed:
                self.paths_flag.append(True)
            else:
                self.paths_flag.append(flag)
                flag = not flag
        self.paths_l = []
        self.paths_r = []
        for path, closed in zip(self.paths, self.paths_closed):
            path_l, path_r = generate_offset_lines(path, closed, offset)
            self.paths_l.append(path_l)
            self.paths_r.append(path_r)
        self.num_segs = [len(path) if closed else len(path) - 1 for path, closed in zip(self.paths, self.paths_closed)]
        self.paths_all = (self.paths, self.paths_l, self.paths_r)
        self.d_segments = {}
        self.d_intersections = {}
        self.d_seg_intersects = defaultdict(list)
        for i in range(len(self.paths)):
            for j in range(3):
                path = self.paths_all[j][i]
                for k in range(len(path) - 1):
                    self.d_segments[(i, j, k)] = (path[k], path[k + 1])
                if self.paths_closed[i]:
                    self.d_segments[(i, j, self.num_segs[i] - 1)] = (path[-1], path[0])

        self._cal_intersections()
        self._set_holes()
        self._set_ribbons()

    def _cal_intersections(self):
        seg_coords_list = []
        seg_ids_list = []
        for (p_idx, t_idx, s_idx), (p_start, p_end) in self.d_segments.items():
            if t_idx == 0:
                continue
            coord_row = np.hstack((p_start, p_end))
            seg_coords_list.append(coord_row)
            seg_ids_list.append((p_idx, t_idx, s_idx))

        if not seg_coords_list:
            return

        seg_arr = np.vstack(seg_coords_list)
        seg_ids_arr = np.array(seg_ids_list, dtype=object)
        num_segs = len(seg_arr)

        x1, y1, x2, y2 = seg_arr[:, 0], seg_arr[:, 1], seg_arr[:, 2], seg_arr[:, 3]
        xmin, xmax = np.minimum(x1, x2), np.maximum(x1, x2)
        ymin, ymax = np.minimum(y1, y2), np.maximum(y1, y2)
        aug_segs = np.column_stack((seg_arr, xmin, ymin, xmax, ymax))

        sort_idx = np.argsort(xmin)
        aug_segs = aug_segs[sort_idx]
        sorted_ids = seg_ids_arr[sort_idx]
        s_xmin = aug_segs[:, 4]

        intersection_clusters = defaultdict(set)

        for i in range(num_segs - 1):
            curr_seg = aug_segs[i]
            curr_id = tuple(sorted_ids[i])
            search_limit = np.searchsorted(s_xmin, curr_seg[6], side="right")
            if search_limit <= i + 1:
                continue
            candidates_idx = np.arange(i + 1, search_limit)
            y_mask = (curr_seg[7] >= aug_segs[candidates_idx, 5]) & (curr_seg[5] <= aug_segs[candidates_idx, 7])
            valid_candidates_indices = candidates_idx[y_mask]
            if len(valid_candidates_indices) == 0:
                continue

            P1, P2 = curr_seg[:2], curr_seg[2:4]
            cand_segs = aug_segs[valid_candidates_indices]
            P3, P4 = cand_segs[:, 0:2], cand_segs[:, 2:4]
            points, valid_mask = vectorized_intersection(P1, P2, P3, P4)

            if not np.any(valid_mask):
                continue

            for point, m_idx in zip(points[valid_mask], valid_candidates_indices[valid_mask]):
                target_id = tuple(sorted_ids[m_idx])
                self.d_intersections[frozenset((curr_id, target_id))] = point
                self.d_seg_intersects[curr_id].append(target_id)
                self.d_seg_intersects[target_id].append(curr_id)
                coord_key = (round(point[0], 6), round(point[1], 6))
                intersection_clusters[coord_key].add(curr_id)
                intersection_clusters[coord_key].add(target_id)

        for coord, seg_set in intersection_clusters.items():
            if len(seg_set) >= 3:
                pass  # warnings.warn(f"Multi-segment intersection at {coord} involves {len(seg_set)} segments.", UserWarning)

        for seg_id in self.d_seg_intersects:
            seg_start = self.d_segments[seg_id][0]
            self.d_seg_intersects[seg_id].sort(
                key=lambda tgt_id: np.linalg.norm(self.d_intersections[frozenset((seg_id, tgt_id))] - seg_start),
            )

    def _set_holes(self):
        # 添加开放路径的端点, 这些端点也会被 _set_ribbons 用到
        for i, (paths_l, paths_r) in enumerate(zip(self.paths_l, self.paths_r)):
            if self.paths_closed[i]:
                continue
            num_seg = len(paths_l) - 1
            seg_id_l, tgt_id_l = (i, 1, 0), 0
            self.d_intersections[frozenset((seg_id_l, tgt_id_l))] = paths_l[0]
            self.d_seg_intersects[seg_id_l].insert(0, tgt_id_l)

            seg_id_l, tgt_id_l = (i, 1, num_seg - 1), -1
            self.d_intersections[frozenset((seg_id_l, tgt_id_l))] = paths_l[-1]
            self.d_seg_intersects[seg_id_l].append(tgt_id_l)

            seg_id_r, tgt_id_r = (i, 2, 0), 0
            self.d_intersections[frozenset((seg_id_r, tgt_id_r))] = paths_r[0]
            self.d_seg_intersects[seg_id_r].insert(0, tgt_id_r)

            seg_id_r, tgt_id_r = (i, 2, num_seg - 1), -1
            self.d_intersections[frozenset((seg_id_r, tgt_id_r))] = paths_r[-1]
            self.d_seg_intersects[seg_id_r].append(tgt_id_r)

        self.holes = []
        nx_graph = networkx.Graph()

        # === 1. 收集所有"断头"节点 (Valid Caps) ===
        cap_nodes = []
        cap_coords = []
        cap_info = []

        path_idxs = [i for i in range(len(self.paths)) if not self.paths_closed[i]]

        for i in path_idxs:
            num_seg = self.num_segs[i]

            # 格式: (seg_idx, end_point_flag, side_idx)
            # end_point_flag: 0 是起点, -1 是终点
            # side_idx: 1 是左, 2 是右
            endpoints = [
                (0, 0, 1),  # 左起点
                (0, 0, 2),  # 右起点
                (num_seg - 1, -1, 1),  # 左终点
                (num_seg - 1, -1, 2),  # 右终点
            ]

            for s_idx, e_flag, side in endpoints:
                node_id = frozenset(((i, side, s_idx), e_flag))

                if node_id in self.d_intersections:
                    coord = self.d_intersections[node_id]
                    cap_nodes.append(node_id)
                    cap_coords.append(coord)
                    cap_info.append((i, side, e_flag))

        # === 2. 使用 KDTree 查找邻居并缝合 ===
        if len(cap_coords) > 1:
            tree = cKDTree(cap_coords)
            dists_all, idxs_all = tree.query(cap_coords, k=min(3, len(cap_coords)))

            added_edges = set()

            for i, (dists, idxs) in enumerate(zip(dists_all, idxs_all)):
                current_node = cap_nodes[i]
                current_path_idx = cap_info[i][0]
                current_eflag = cap_info[i][2]

                valid_neighbor_found = False
                for d, neighbor_idx in zip(dists, idxs):
                    if i == neighbor_idx:
                        continue

                    neighbor_path_idx = cap_info[neighbor_idx][0]
                    neighbor_eflag = cap_info[neighbor_idx][2]

                    # 排除孪生: 同一条路径同一端点的左右端点不连
                    if current_path_idx == neighbor_path_idx and current_eflag == neighbor_eflag:
                        valid_neighbor_found = True
                        continue

                    if d > 10.0:
                        continue

                    target_node = cap_nodes[neighbor_idx]

                    edge_sig = frozenset((current_node, target_node))
                    if edge_sig not in added_edges:
                        nx_graph.add_edge(current_node, target_node)
                        added_edges.add(edge_sig)
                        valid_neighbor_found = True

                    if valid_neighbor_found:
                        break
                if not valid_neighbor_found:
                    warnings.warn(f"Node {current_node} has no valid neighbors.", UserWarning)

        # === 3. 连接丝带内部的 edges ===
        for seg_id, tgt_ids in self.d_seg_intersects.items():
            for k in range(0, len(tgt_ids), 2):
                if k + 1 < len(tgt_ids):
                    nx_graph.add_edge(
                        frozenset((seg_id, tgt_ids[k])),
                        frozenset((seg_id, tgt_ids[k + 1])),
                    )

        # === 4. 计算 Loop 面积并提取 Holes ===
        cycles = networkx.cycle_basis(nx_graph)

        valid_cycle_data = []
        areas = []
        for cycle in cycles:
            if len(cycle) < 3:
                continue

            try:
                coords = np.array([self.d_intersections[node] for node in cycle])
            except KeyError:
                continue

            # Shoelace formula for area
            x = coords[:, 0]
            y = coords[:, 1]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

            valid_cycle_data.append((coords, area))
            areas.append(area)

        if areas:
            areas.sort()
            total_area = sum(areas[:-1])

            for coords, area in valid_cycle_data:
                if area > total_area:
                    continue

                if not right_handed(coords):
                    coords = coords[::-1]
                self.holes.append(coords)

    def _set_ribbons(self):
        self.ribbons: list[np.ndarray] = []
        for i, path in enumerate(self.paths):
            flag = self.paths_flag[i]
            nx_graph = networkx.Graph()
            num_seg = self.num_segs[i]

            if not self.paths_closed[i]:
                seg_id_l = (i, 1, 0)
                seg_id_r = (i, 2, 0)
                nx_graph.add_edge(
                    frozenset((seg_id_l, 0)),
                    frozenset((seg_id_r, 0)),
                )
                seg_id_l = (i, 1, num_seg - 1)
                seg_id_r = (i, 2, num_seg - 1)
                nx_graph.add_edge(
                    frozenset((seg_id_l, -1)),
                    frozenset((seg_id_r, -1)),
                )

            for j in range(num_seg):
                seg_id_l = (i, 1, j)
                seg_id_r = (i, 2, j)
                tgt_ids_l = self.d_seg_intersects[seg_id_l]
                tgt_ids_r = self.d_seg_intersects[seg_id_r]

                if len(tgt_ids_l) < 2:
                    continue
                # Left and right must have same count for paired traversal
                if len(tgt_ids_l) != len(tgt_ids_r):
                    continue

                prev_node_l = frozenset((seg_id_l, tgt_ids_l[0]))
                prev_node_r = frozenset((seg_id_r, tgt_ids_r[0]))

                for k in range(2, len(tgt_ids_l) - 1, 2):
                    if flag:
                        curr_start_l = frozenset((seg_id_l, tgt_ids_l[k - 2]))
                        curr_start_r = frozenset((seg_id_r, tgt_ids_r[k - 2]))

                        if prev_node_l != curr_start_l:
                            nx_graph.add_edge(prev_node_l, curr_start_l)
                        if prev_node_r != curr_start_r:
                            nx_graph.add_edge(prev_node_r, curr_start_r)

                        nx_graph.add_edge(
                            curr_start_l,
                            frozenset((seg_id_l, tgt_ids_l[k - 1])),
                        )
                        nx_graph.add_edge(
                            curr_start_r,
                            frozenset((seg_id_r, tgt_ids_r[k - 1])),
                        )
                        nx_graph.add_edge(
                            frozenset((seg_id_r, tgt_ids_r[k - 1])),
                            frozenset((seg_id_l, tgt_ids_l[k - 1])),
                        )

                        next_start_l = frozenset((seg_id_l, tgt_ids_l[k]))
                        next_start_r = frozenset((seg_id_r, tgt_ids_r[k]))
                        nx_graph.add_edge(next_start_l, next_start_r)

                        prev_node_l = next_start_l
                        prev_node_r = next_start_r

                    flag = not flag

                end_node_l = frozenset((seg_id_l, tgt_ids_l[-1]))
                end_node_r = frozenset((seg_id_r, tgt_ids_r[-1]))

                if prev_node_l != end_node_l:
                    nx_graph.add_edge(prev_node_l, end_node_l)
                if prev_node_r != end_node_r:
                    nx_graph.add_edge(prev_node_r, end_node_r)

            cycles = networkx.cycle_basis(nx_graph)
            for cycle in cycles:
                if len(cycle) < 3:
                    continue
                try:
                    coords = np.array([self.d_intersections[node] for node in cycle])
                except KeyError:
                    continue
                if not right_handed(coords):
                    coords = coords[::-1]
                self.ribbons.append(coords)


# =========================
# 3. Curve Data Sync
# =========================


def extract_curve_data(obj_ref: bpy.types.Object, depsgraph=None):
    """Extract world-space spline data from a reference curve object."""
    if depsgraph is None:
        depsgraph = bpy.context.evaluated_depsgraph_get()

    eval_obj = obj_ref.evaluated_get(depsgraph)
    src_curve = eval_obj.data

    if not isinstance(src_curve, bpy.types.Curve) or not src_curve.splines:
        return [], []

    src_matrix = np.array(eval_obj.matrix_world, dtype=np.float64).T

    all_paths = []
    paths_closed = []

    for src_spline in src_curve.splines:
        is_poly = src_spline.type == "POLY"
        points = src_spline.points if is_poly else src_spline.bezier_points
        p_len = len(points)
        if p_len < 2:
            continue

        coords = np.zeros((p_len, 3), dtype=np.float64)
        if is_poly:
            raw = np.empty(p_len * 4, dtype=np.float64)
            points.foreach_get("co", raw)
            coords = raw.reshape((p_len, 4))[:, :3]
        else:
            points.foreach_get("co", coords.ravel())

        ones = np.ones((p_len, 1), dtype=np.float64)
        coords_homo = np.hstack((coords, ones))
        coords_world = (coords_homo @ src_matrix)[:, :3]

        all_paths.append(coords_world)
        paths_closed.append(src_spline.use_cyclic_u)

    return all_paths, paths_closed


def sync_curve_object(curve_data: bpy.types.Curve, ribbons, offset_lines=None, paths_closed=None):
    """Write ribbon polygons as closed POLY splines into the target curve data."""
    curve_data.splines.clear()

    for ribbon in ribbons:
        if ribbon is None or len(ribbon) < 3:
            continue
        s = curve_data.splines.new("POLY")
        s.use_cyclic_u = True
        count = len(ribbon)
        s.points.add(count - 1)

        d = ribbon.astype(np.float64)
        if d.shape[1] == 2:
            z = np.zeros((count, 1), dtype=np.float64)
            d = np.hstack((d, z))
        elif d.shape[1] > 3:
            d = d[:, :3]

        w = np.ones((count, 1), dtype=np.float64)
        d4 = np.hstack((d, w))
        s.points.foreach_set("co", np.ascontiguousarray(d4).ravel())

    if offset_lines:
        for (out_left, out_right), closed in zip(offset_lines, paths_closed):
            for data in (out_left, out_right):
                if data is None or len(data) < 2:
                    continue
                s = curve_data.splines.new("POLY")
                s.use_cyclic_u = closed
                count = len(data)
                s.points.add(count - 1)

                d = data.astype(np.float64)
                if d.shape[1] == 2:
                    z = np.zeros((count, 1), dtype=np.float64)
                    d = np.hstack((d, z))
                elif d.shape[1] > 3:
                    d = d[:, :3]

                w = np.ones((count, 1), dtype=np.float64)
                d4 = np.hstack((d, w))
                s.points.foreach_set("co", np.ascontiguousarray(d4).ravel())


# =========================
# 4. State & Properties
# =========================


def on_dirty_update(self, context):
    self.is_dirty = True


class TAI_InterlacingProperties(bpy.types.PropertyGroup):
    obj_ref: bpy.props.PointerProperty(
        name="Reference", type=bpy.types.Object, poll=lambda s, o: o.type == "CURVE", update=on_dirty_update
    )
    obj_gen: bpy.props.PointerProperty(name="Generated", type=bpy.types.Object, poll=lambda s, o: o.type == "CURVE")

    output_type: bpy.props.EnumProperty(
        name="Output",
        items=[
            ("OFFSET", "Offset", "Simple offset curves"),
            ("LACE", "Lace", "Interlaced ribbon pattern"),
        ],
        default="LACE",
        update=on_dirty_update,
    )
    use_preprocess: bpy.props.BoolProperty(
        name="Preprocess",
        default=True,
        description="Merge and clear paths before interlacing",
        update=on_dirty_update,
    )
    offset: bpy.props.FloatProperty(
        name="Offset",
        default=0.04,
        min=0.001,
        soft_max=1.0,
        subtype="DISTANCE",
        unit="LENGTH",
        description="Offset distance, half width of the lace",
        update=on_dirty_update,
    )
    miter_limit: bpy.props.FloatProperty(
        name="Miter Limit",
        default=10.0,
        min=1.0,
        soft_max=20.0,
        description="Limit for the miter length at sharp corners",
        update=on_dirty_update,
    )
    ribbon_count: bpy.props.IntProperty(name="Ribbon Count")
    is_dirty: bpy.props.BoolProperty(default=True)


# =========================
# 5. UI Panels & Operators
# =========================


class TAI_OT_init_interlacing(bpy.types.Operator):
    bl_idname = "taichi.init_interlacing"
    bl_label = "Initialize Interlacing"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.taichi_interlacing
        if not props.obj_ref:
            # Create a default reference curve (circle)
            curve_data = bpy.data.curves.new("Interlacing_Ref", type="CURVE")
            curve_data.dimensions = "2D"
            s = curve_data.splines.new("POLY")
            n = 4
            s.points.add(n - 1)
            import math

            for i in range(n):
                angle = 2.0 * math.pi * i / n
                x, y = math.cos(angle), math.sin(angle)
                s.points[i].co = (x, y, 0.0, 1.0)
            s.use_cyclic_u = True

            o = bpy.data.objects.new("Interlacing_Ref", curve_data)
            context.collection.objects.link(o)
            props.obj_ref = o

        gen_name = f"{props.obj_ref.name}_lace"
        if not props.obj_gen or props.obj_gen.name != gen_name:
            gen_data = bpy.data.curves.new(gen_name, type="CURVE")
            gen_data.dimensions = "2D"
            o_gen = bpy.data.objects.get(gen_name) or bpy.data.objects.new(gen_name, gen_data)
            if o_gen.name not in context.collection.objects:
                context.collection.objects.link(o_gen)
            props.obj_gen = o_gen

        update_interlacing(context.scene)
        return {"FINISHED"}


class TAI_OT_select_interlacing(bpy.types.Operator):
    bl_idname = "taichi.select_interlacing"
    bl_label = "Select"
    target: bpy.props.EnumProperty(items=[("REF", "Reference", ""), ("GEN", "Generated", "")])

    def execute(self, context):
        props = context.scene.taichi_interlacing
        obj = props.obj_ref if self.target == "REF" else props.obj_gen
        if obj:
            if context.object and context.object.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            context.view_layer.objects.active = obj
        return {"FINISHED"}


class TAI_OT_toggle_ref_wire(bpy.types.Operator):
    bl_idname = "taichi.toggle_interlacing_wire"
    bl_label = "Toggle Reference Wire"
    bl_options = {"UNDO"}

    def execute(self, context):
        props = context.scene.taichi_interlacing
        if props.obj_ref:
            props.obj_ref.display_type = "WIRE" if props.obj_ref.display_type != "WIRE" else "TEXTURED"
        return {"FINISHED"}


class TAI_PT_interlacing_panel(bpy.types.Panel):
    bl_label = "Taichi Interlacing"
    bl_idname = "TAI_PT_interlacing_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Taichi"

    def draw(self, context):
        layout, props = self.layout, context.scene.taichi_interlacing

        # Reference row
        row = layout.row(align=True)
        row.prop(props, "obj_ref", text="", placeholder="Select Source Curve")
        if props.obj_ref:
            is_wire = props.obj_ref.display_type == "WIRE"
            row.operator(
                TAI_OT_toggle_ref_wire.bl_idname, text="", icon="SHADING_WIRE" if is_wire else "SHADING_TEXTURE"
            )

        layout.operator(TAI_OT_init_interlacing.bl_idname, icon="CURVE_DATA")

        if props.obj_gen:
            box = layout.box()
            box.label(text="Parameters", icon="SETTINGS")
            row = box.row(align=True)
            row.prop(props, "output_type", expand=True)

            col = box.column(align=True)
            col.prop(props, "offset")
            col.prop(props, "miter_limit")
            col.prop(props, "use_preprocess")

            box = layout.box()
            box.label(text="Components", icon="OBJECT_DATA")
            row = box.row(align=True)
            row.prop(props, "obj_gen", text="")
            row = box.row(align=True)
            for t, l in [("REF", "Reference"), ("GEN", "Generated")]:
                row.operator(TAI_OT_select_interlacing.bl_idname, text=l).target = t

            if props.ribbon_count > 0:
                box = layout.box()
                row = box.row()
                row.label(text=f"Ribbons: {props.ribbon_count}", icon="INFO")

            # Data Interface
            sec = layout.box()
            sec.label(text="Data Interface", icon="SPREADSHEET")
            col = sec.column(align=True)
            for names in [
                ("Attribute", "Domain"),
                ("spline", "Spline"),
            ]:
                row = col.row(align=True)
                b = row.box()
                b.scale_y = 0.5
                r = b.row(align=True)
                split = r.split(factor=0.6)
                split.label(text=names[0])
                split.label(text=names[1])


# =========================
# 6. Core Update Logic
# =========================


def update_interlacing(scene: bpy.types.Scene, depsgraph=None):
    """Coordinate the full interlacing update cycle."""
    props = scene.taichi_interlacing
    obj_ref = props.obj_ref
    obj_gen = props.obj_gen

    if not obj_ref or not obj_gen or not props.is_dirty:
        return

    try:
        all_paths, paths_closed = extract_curve_data(obj_ref, depsgraph)
        if not all_paths:
            props.is_dirty = False
            return

        if props.output_type == "OFFSET":
            offset_lines = []
            for path, closed in zip(all_paths, paths_closed):
                out_left, out_right = generate_offset_lines(path, closed, offset=props.offset, limit=props.miter_limit)
                offset_lines.append((out_left, out_right))

            sync_curve_object(obj_gen.data, ribbons=[], offset_lines=offset_lines, paths_closed=paths_closed)
            props.ribbon_count = len(offset_lines) * 2
        else:
            interlacing = Interlacing(all_paths, paths_closed, offset=props.offset, use_preprocess=props.use_preprocess)
            sync_curve_object(obj_gen.data, ribbons=interlacing.ribbons)
            props.ribbon_count = len(interlacing.ribbons)

    except Exception as e:
        print(f"Interlacing update error: {e}")
        import traceback

        traceback.print_exc()

    props.is_dirty = False


# =========================
# 7. Lifecycle & Registration
# =========================


@bpy.app.handlers.persistent
def interlacing_handler(scene, depsgraph):
    props = scene.taichi_interlacing
    obj_ref = props.obj_ref
    if not obj_ref:
        return

    should_update = False
    for update in depsgraph.updates:
        if update.id.name in [obj_ref.name, getattr(obj_ref.data, "name", "")]:
            should_update = True
            break

    if should_update:
        props.is_dirty = True

    update_interlacing(scene, depsgraph=depsgraph)


classes = (
    TAI_InterlacingProperties,
    TAI_OT_init_interlacing,
    TAI_OT_select_interlacing,
    TAI_OT_toggle_ref_wire,
    TAI_PT_interlacing_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.taichi_interlacing = bpy.props.PointerProperty(type=TAI_InterlacingProperties)

    bpy.app.handlers.depsgraph_update_post[:] = [
        h for h in bpy.app.handlers.depsgraph_update_post if h.__name__ != "interlacing_handler"
    ]
    bpy.app.handlers.depsgraph_update_post.append(interlacing_handler)


def unregister():
    bpy.app.handlers.depsgraph_update_post[:] = [
        h for h in bpy.app.handlers.depsgraph_update_post if h.__name__ != "interlacing_handler"
    ]
    del bpy.types.Scene.taichi_interlacing
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__" or __name__ == "<run_path>":
    register()
