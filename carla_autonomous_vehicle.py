import carla
import cv2
import math
import heapq
import numpy as np
import time as _time

# =========================
# CONFIG
# =========================
TARGET_SPEED    = 25.0   # km/h
LOOKAHEAD_MIN   = 4.0    # m
LOOKAHEAD_GAIN  = 0.3
WHEELBASE       = 2.8    # m
MAX_STEER_DEG   = 70.0
ROUTE_LEN       = 500
WP_SPACING      = 2.0    # m
ROUTE_EXTEND_THRESH = 60  # còn bao nhiêu wp thì extend

IMG_W, IMG_H    = 1280, 720

# ── Obstacle sensor config ───────────────────────────────────────────────────
# Logic distance: tốc độ 25km/h ≈ 7m/s, cần ~1.5s để settle+đổi lane ≈ 10m
#
#   |←——— OBS_FRONT_DIST=20m ————→|
#   |←—OBS_HIT=12m—→|             |   phát hiện → bắt đầu CHECKING + giảm tốc
#   |←OBS_STOP=5m→|               |   phanh khẩn cấp (chỉ khi WAITING_CLEAR)
#   [xe]
#   khoảng HIT→STOP = 7m ≈ 1s @ 25km/h → đủ để quyết định + đánh lái

OBS_FRONT_DIST      = 20.0  # m — range sensor front
OBS_HIT_THRESHOLD   = 12.0  # m — bắt đầu CHECKING_LANES
OBS_STOP_THRESHOLD  = 5.0   # m — phanh khẩn cấp nếu không có lane nào

# ── Sensor lane kề ───────────────────────────────────────────────────────────
# Lane width CARLA ≈ 3.5-4m
# LANE_CLEAR_THRESH phải < LANE_CHECK_DIST để có vùng phân biệt rõ
#
#   sensor đặt offset 3.5m → nằm giữa lane kề
#   nếu có xe trong 6m → không rẽ (quá gần, nguy hiểm khi merge)
#   nếu không có xe hoặc xe cách >6m → rẽ được

LANE_SENSOR_OFFSET_Y = 3.5  # m — offset ngang = 1 lane width
LANE_CHECK_DIST      = 12.0 # m — range sensor lane kề (phát hiện xe đang đến)
LANE_CLEAR_THRESH    = 4.0  # m — lane kề "trống" khi không có xe trong 6m

# Cooldown giữa 2 lần đổi lane
LC_COOLDOWN         = 2.5    # s

# Khoảng cách đến đích để dừng lại
GOAL_REACH_DIST     = 4.0    # m

# Obstacle sensor warmup — bỏ qua N tick đầu để sensor ổn định
SENSOR_WARMUP_TICKS  = 30

# Số frame chờ sau khi vào CHECKING_LANES để sensor lane kề có thời gian update
SENSOR_SETTLE_FRAMES  = 3

# Thời gian (giây) giữ giá trị front sensor trước khi coi là "trống"
# Tránh trường hợp 1 frame không có callback → front về inf → thoát CHECKING
OBS_FRONT_DECAY      = 0.5   # s — giữ giá trị front tối thiểu 0.5s

# Số giây chờ sau khi đổi lane xong trước khi cho phép CHECKING lại
# (sensor cần thời gian ổn định ở vị trí mới)
POST_LC_SETTLE_TIME  = 2.0   # s

# ── A* config ────────────────────────────────────────────────────────────────
ASTAR_STEP      = 3.0

# ── Mini-map config ──────────────────────────────────────────────────────────
MAP_WIN_W, MAP_WIN_H = 700, 700
MAP_WP_RADIUS   = 2
MAP_CACHE_STEP  = 4

# Màu overlay
COLOR_ROUTE     = (0,   200, 80)
COLOR_TARGET    = (0,   80,  255)
COLOR_LC_TGT    = (255, 0,   200)
COLOR_NEAR      = (0,   220, 255)

# ── Sensor fan visualization ─────────────────────────────────────────────────
# Mỗi sensor được vẽ như 1 hình quạt (arc) project 3D → camera 2D
FAN_SEGMENTS    = 24        # số đoạn polygon chia cung quạt
FAN_HALF_ANGLE  = 20.0      # độ — góc mở mỗi bên của quạt (tổng 40°)
FAN_HEIGHT      = 0.15      # m — chiều cao quạt so với mặt đường (để project lên camera)

# Màu quạt: (clear_color, blocked_color) theo BGR
FAN_COLOR_FRONT_CLEAR   = (0,   210,  80)   # xanh lá  — trước trống
FAN_COLOR_FRONT_BLOCKED = (0,    40, 255)   # đỏ       — trước bị chặn
FAN_COLOR_SIDE_CLEAR    = (0,   200, 220)   # vàng xanh — lane kề trống
FAN_COLOR_SIDE_BLOCKED  = (30,   80, 200)   # cam đậm   — lane kề bị chặn
FAN_ALPHA               = 0.30              # độ trong suốt overlay

# =========================
# SENSOR STATE  (shared globals — updated by callbacks)
# =========================
# Obstacle distances — dùng dict để tránh vấn đề global scope trong try/while
obs = {
    "front":      float("inf"),
    "left":       float("inf"),
    "right":      float("inf"),
    "front_time": 0.0,   # timestamp lần cuối front callback fire
}

camera_img          = None
cam_transform_cache = None
lc_target_wp        = None

_f    = IMG_W / (2.0 * math.tan(math.radians(90.0 / 2.0)))
CAM_K = np.array([
    [_f,  0,   IMG_W / 2.0],
    [0,   _f,  IMG_H / 2.0],
    [0,   0,   1.0         ],
], dtype=float)


# =========================
# SENSOR CALLBACKS
# =========================
def camera_cb(img):
    global camera_img, cam_transform_cache
    camera_img = (
        np.frombuffer(img.raw_data, dtype=np.uint8)
        .reshape(img.height, img.width, 4)[:, :, :3]
        .copy()
    )
    cam_transform_cache = img.transform


def obs_front_cb(event):
    if event.distance > 0.1:
        obs["front"]      = event.distance
        obs["front_time"] = _time.time()

def obs_left_cb(event):
    if event.distance > 0.1:
        obs["left"] = event.distance

def obs_right_cb(event):
    if event.distance > 0.1:
        obs["right"] = event.distance

def reset_obs_each_frame():
    """
    - left/right: reset mỗi frame (cần data fresh để check lane có trống không)
    - front: KHÔNG reset mỗi frame — dùng decay timeout thay vì reset ngay.
      Tránh 1 frame miss callback làm front về inf → CHECKING thoát sai.
    """
    now_t = _time.time()
    # Front: giữ giá trị nếu callback fire trong OBS_FRONT_DECAY giây gần nhất
    if now_t - obs["front_time"] > OBS_FRONT_DECAY:
        obs["front"] = float("inf")   # đã quá lâu không có callback → thực sự trống
    # Left/right: luôn reset, callback frame này sẽ ghi đè nếu có vật
    obs["left"]  = float("inf")
    obs["right"] = float("inf")


# =========================
# UTIL
# =========================
def get_speed(vehicle):
    v = vehicle.get_velocity()
    return 3.6 * math.hypot(v.x, v.y)


def carla_dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


# =========================
# ROUTE BUILDER
# =========================
def build_route(start_wp, max_len=ROUTE_LEN, spacing=WP_SPACING):
    route = [start_wp]
    wp    = start_wp
    for _ in range(max_len):
        nxt = wp.next(spacing)
        if not nxt:
            break
        wp = nxt[0]
        route.append(wp)
    return route


# =========================
# A* SHORTEST PATH
# =========================
def astar_route(start_wp, goal_loc, m, step=ASTAR_STEP, max_nodes=2000):
    def h(wp):
        l = wp.transform.location
        return math.hypot(l.x - goal_loc.x, l.y - goal_loc.y)

    def key(wp):
        # Key chỉ dùng road_id + lane_id + vị trí làm tròn
        # KHÔNG dùng section_id vì đôi khi cùng điểm nhưng section khác nhau
        return (wp.road_id, int(wp.lane_id),
                round(wp.transform.location.x, 1),
                round(wp.transform.location.y, 1))

    open_heap = []
    start_key = key(start_wp)
    heapq.heappush(open_heap, (h(start_wp), 0.0, start_key, start_wp, None))

    came_from  = {}
    g_score    = {start_key: 0.0}
    closed     = set()
    best_wp    = start_wp
    best_dist  = h(start_wp)
    nodes_exp  = 0

    while open_heap and nodes_exp < max_nodes:
        f, g, k, wp, par_k = heapq.heappop(open_heap)
        if k in closed:
            continue
        closed.add(k)
        came_from[k] = (par_k, wp)
        nodes_exp += 1

        d = h(wp)
        if d < best_dist:
            best_dist = d
            best_wp   = wp
        if d < step * 1.5:
            best_wp = wp
            break

        cur_lane = int(wp.lane_id)
        # ── Chỉ đi thẳng trong lane hiện tại, không expand sang lane kề ────────
        # Đổi lane là quyết định riêng (manual A/D hoặc obstacle avoidance),
        # không để A* tự chọn lane — tránh route nhảy lung tung giữa các lane.
        for nwp in wp.next(step):
            if nwp.lane_type != carla.LaneType.Driving:
                continue
            if int(nwp.lane_id) != cur_lane:
                continue   # bỏ qua nếu next() trả về lane khác (junction)
            nk = key(nwp)
            if nk in closed:
                continue
            ng = g + step
            if ng < g_score.get(nk, 1e18):
                g_score[nk] = ng
                heapq.heappush(open_heap, (ng + h(nwp), ng, nk, nwp, k))

    path_keys = []
    cur = key(best_wp)
    while cur is not None:
        path_keys.append(cur)
        par_k, _ = came_from.get(cur, (None, None))
        cur = par_k
    path_keys.reverse()

    path_wps = [came_from[k][1] if k in came_from else start_wp
                for k in path_keys]
    if not path_wps:
        path_wps = [start_wp]

    # Chỉ nối tail nếu best_wp còn xa goal — tránh vượt qua đích
    dist_best_to_goal = h(best_wp)
    if dist_best_to_goal > step * 3:
        tail = build_route(best_wp, max_len=60, spacing=WP_SPACING)
        return path_wps + tail[1:]
    else:
        # Đã đến gần goal → dừng đúng tại best_wp, không đi thêm
        return path_wps


# =========================
# PURE PURSUIT
# =========================
class PurePursuit:
    def __init__(self):
        self.route = []
        self.idx   = 0

    def set_route(self, route, loc):
        self.route = route
        self.idx   = 0
        if not route:
            return
        best_d = 1e9
        for i, wp in enumerate(route):
            d = carla_dist(loc, wp.transform.location)
            if d < best_d:
                best_d   = d
                self.idx = i

    def reset(self, route, loc):
        self.set_route(route, loc)

    def set_route_lc(self, route):
        """
        Dùng riêng khi đổi lane — force idx=0 để đi từ đầu route mới,
        không snap về điểm gần nhất (vì xe vẫn đang ở lane cũ).
        """
        self.route = route
        self.idx   = 0

    def target(self, loc, lookahead):
        route = self.route
        n     = len(route)
        if n == 0:
            return None
        best_d = carla_dist(loc, route[self.idx].transform.location)
        for i in range(self.idx + 1, n):
            d = carla_dist(loc, route[i].transform.location)
            if d < best_d:
                best_d   = d
                self.idx = i
            elif d > best_d + lookahead * 3:
                break
        for i in range(self.idx, n):
            if carla_dist(loc, route[i].transform.location) >= lookahead:
                return route[i]
        return route[-1] if route else None

    def steer(self, vehicle, wp):
        if wp is None:
            return 0.0
        tf  = vehicle.get_transform()
        yaw = math.radians(tf.rotation.yaw)
        dx  = wp.transform.location.x - tf.location.x
        dy  = wp.transform.location.y - tf.location.y
        local_x =  math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        ld = math.hypot(local_x, local_y)
        if ld < 1e-3:
            return 0.0
        alpha = math.atan2(local_y, local_x)
        delta = math.atan2(2.0 * WHEELBASE * math.sin(alpha), ld)
        return float(np.clip(delta / math.radians(MAX_STEER_DEG), -1.0, 1.0))

    @property
    def has_route(self):
        return len(self.route) > 0


# =========================
# LANE CHANGE
# =========================
def do_lane_change(current_wp, direction, m, goal_loc=None):
    """
    CARLA lane_id convention:
      - Âm  (-1, -2, ...): lane chạy thuận chiều road (forward direction)
      - Dương (+1, +2, ...): lane chạy ngược chiều road
    get_left_lane() / get_right_lane() trả về lane kề theo góc nhìn TÀI XẾ.

    Validate đúng: adj phải có cùng dấu với current (cùng chiều đường),
    và |adj_lane_id| phải khác |cur_lane_id| (thực sự sang lane khác).
    """
    if direction == "left":
        adj = current_wp.get_left_lane()
    else:
        adj = current_wp.get_right_lane()

    if adj is None:
        print(f"[LC] Không có lane {direction}")
        return None, None
    if adj.lane_type != carla.LaneType.Driving:
        print(f"[LC] Lane {direction} không phải Driving ({adj.lane_type})")
        return None, None

    cur_lane_id = int(current_wp.lane_id)
    adj_lane_id = int(adj.lane_id)

    # ── Guard 1: adj trả về chính lane hiện tại ──────────────────────────────
    if adj_lane_id == cur_lane_id:
        print(f"[LC] adj == current ({cur_lane_id}), bỏ qua")
        return None, None

    # ── Guard 2: adj sang chiều ngược (lane_id đổi dấu) ─────────────────────
    # Ví dụ: đang ở lane -1, get_left_lane() trả về lane +1 → đường ngược chiều
    if (cur_lane_id < 0) != (adj_lane_id < 0):
        print(f"[LC] adj lane {adj_lane_id} ngược chiều với current {cur_lane_id}, bỏ qua")
        return None, None

    # Guard 1 + 2 đã đủ — CARLA đảm bảo get_left/right_lane() trả về
    # đúng bên trái/phải theo góc nhìn tài xế, không cần check delta nữa.
    # Chỉ log để debug:
    delta = adj_lane_id - cur_lane_id
    print(f"[LC] {direction}: lane {cur_lane_id} → {adj_lane_id} (delta={delta})")

    if goal_loc is not None:
        new_route = astar_route(adj, goal_loc, m)
        print(f"[LC] Sang lane {adj_lane_id} (từ {cur_lane_id}), A* → goal, {len(new_route)} wp")
    else:
        new_route = build_route(adj)
        print(f"[LC] Sang lane {adj_lane_id} (từ {cur_lane_id}), build_route {len(new_route)} wp")

    return adj, new_route


# =========================
# OBSTACLE-BASED LANE DECISION
# =========================
def decide_lane_change(current_wp):
    """
    Dùng dữ liệu từ 3 obstacle sensor để quyết định:
      - Phía trước trống     → None (không làm gì)
      - Phía trước bị chặn:
          * Trái trống        → "left"
          * Phải trống        → "right"
          * Cả hai đều bị chặn → "stop"

    Trả về: "left" | "right" | "stop" | None
    """
    front_blocked = obs["front"] < OBS_HIT_THRESHOLD
    if not front_blocked:
        return None

    left_has_lane  = current_wp.get_left_lane()  is not None and \
                     current_wp.get_left_lane().lane_type == carla.LaneType.Driving
    right_has_lane = current_wp.get_right_lane() is not None and \
                     current_wp.get_right_lane().lane_type == carla.LaneType.Driving

    # inf = không có vật trong range → lane TRỐNG (đây là trường hợp tốt)
    # Sensor chỉ fire khi CÓ vật — không fire = không có vật = trống
    # (warmup đã được xử lý riêng ở đầu loop, không cần check lại ở đây)
    left_clear  = left_has_lane  and obs["left"]  >= LANE_CLEAR_THRESH
    right_clear = right_has_lane and obs["right"] >= LANE_CLEAR_THRESH

    print(f"[OBS] front={obs['front']:.1f}m "
          f"left={obs['left']:.1f}m(has_lane={left_has_lane},clear={left_clear}) "
          f"right={obs['right']:.1f}m(has_lane={right_has_lane},clear={right_clear})")

    if left_clear and right_clear:
        # Cả hai trống — ưu tiên trái
        return "left"
    elif left_clear:
        return "left"
    elif right_clear:
        return "right"
    else:
        return "stop"


# =========================
# 3D → 2D PROJECTION
# =========================
def world_to_pixel(world_xyz, cam_transform):
    inv       = np.array(cam_transform.get_inverse_matrix())
    p_world   = np.array([world_xyz[0], world_xyz[1], world_xyz[2], 1.0])
    p_cam_ue4 = inv @ p_world
    x_opt =  p_cam_ue4[1]
    y_opt = -p_cam_ue4[2]
    z_opt =  p_cam_ue4[0]
    if z_opt <= 0.1:
        return None
    u = int(CAM_K[0, 0] * x_opt / z_opt + CAM_K[0, 2])
    v = int(CAM_K[1, 1] * y_opt / z_opt + CAM_K[1, 2])
    if 0 <= u < IMG_W and 0 <= v < IMG_H:
        return (u, v)
    return None


# =========================
# SENSOR FAN VISUALIZATION
# =========================
def _fan_polygon_world(origin_x, origin_y, origin_z,
                       yaw_rad, half_angle_rad,
                       radius, segments):
    """
    Tạo list các điểm world (x,y,z) tạo thành hình quạt.
    origin_x/y/z : tâm quạt (vị trí sensor trên thế giới)
    yaw_rad       : hướng chính của quạt (hướng mũi xe + offset)
    half_angle_rad: góc mở mỗi bên
    radius        : bán kính quạt (= range sensor)
    """
    pts = [(origin_x, origin_y, origin_z)]   # điểm tâm
    for i in range(segments + 1):
        a = yaw_rad - half_angle_rad + (2 * half_angle_rad * i / segments)
        px = origin_x + radius * math.cos(a)
        py = origin_y + radius * math.sin(a)
        pts.append((px, py, origin_z))
    pts.append((origin_x, origin_y, origin_z))  # đóng polygon
    return pts


def draw_sensor_fans(img, vehicle_tf, cam_tf,
                     front_dist, left_dist, right_dist):
    """
    Vẽ 3 hình quạt (front / left-lane / right-lane) lên ảnh camera.
    Mỗi quạt:
      - Tâm = vị trí sensor trong world (tính từ xe + offset)
      - Bán kính = range sensor
      - Màu đổi theo trạng thái clear/blocked
      - Được fill bán trong suốt + viền đặc
    """
    vx  = vehicle_tf.location.x
    vy  = vehicle_tf.location.y
    vz  = vehicle_tf.location.z
    yaw = math.radians(vehicle_tf.rotation.yaw)

    cos_y, sin_y = math.cos(yaw), math.sin(yaw)

    # Hàm tính world position của sensor (offset trong local frame xe)
    def sensor_world(local_x, local_y):
        wx = vx + cos_y * local_x - sin_y * local_y
        wy = vy + sin_y * local_x + cos_y * local_y
        wz = vz + FAN_HEIGHT
        return wx, wy, wz

    half_rad = math.radians(FAN_HALF_ANGLE)

    # ── Định nghĩa 3 sensor ──────────────────────────────────────────────────
    sensors = [
        {
            "name":     "FRONT",
            "origin":   sensor_world(2.5, 0.0),
            "yaw":      yaw,                        # hướng thẳng trước
            "radius":   OBS_FRONT_DIST,
            "dist":     front_dist,
            "thresh":   OBS_HIT_THRESHOLD,
            "col_clear":   FAN_COLOR_FRONT_CLEAR,
            "col_blocked": FAN_COLOR_FRONT_BLOCKED,
        },
        {
            "name":     "LEFT",
            "origin":   sensor_world(2.5, -LANE_SENSOR_OFFSET_Y),
            "yaw":      yaw,
            "radius":   LANE_CHECK_DIST,
            "dist":     left_dist,
            "thresh":   LANE_CLEAR_THRESH,
            "col_clear":   FAN_COLOR_SIDE_CLEAR,
            "col_blocked": FAN_COLOR_SIDE_BLOCKED,
        },
        {
            "name":     "RIGHT",
            "origin":   sensor_world(2.5,  LANE_SENSOR_OFFSET_Y),
            "yaw":      yaw,
            "radius":   LANE_CHECK_DIST,
            "dist":     right_dist,
            "thresh":   LANE_CLEAR_THRESH,
            "col_clear":   FAN_COLOR_SIDE_CLEAR,
            "col_blocked": FAN_COLOR_SIDE_BLOCKED,
        },
    ]

    overlay = img.copy()

    for s in sensors:
        ox, oy, oz = s["origin"]
        blocked    = s["dist"] < s["thresh"]
        color      = s["col_blocked"] if blocked else s["col_clear"]

        # ── Polygon toàn bộ vùng quét (bán kính đầy đủ) ─────────────────────
        full_pts_world = _fan_polygon_world(
            ox, oy, oz, s["yaw"], half_rad, s["radius"], FAN_SEGMENTS)

        full_px = []
        for (wx, wy, wz) in full_pts_world:
            px = world_to_pixel(np.array([wx, wy, wz]), cam_tf)
            if px is not None:
                full_px.append(px)

        if len(full_px) >= 3:
            poly = np.array(full_px, dtype=np.int32)
            cv2.fillPoly(overlay, [poly], color)
            cv2.polylines(overlay, [poly], True, color, 2)

        # ── Polygon phần đã "dùng" (đến khoảng cách detect thực tế) ─────────
        actual_r = min(s["dist"], s["radius"])
        if actual_r > 0.5:
            hit_pts_world = _fan_polygon_world(
                ox, oy, oz, s["yaw"], half_rad, actual_r, FAN_SEGMENTS)
            hit_px = []
            for (wx, wy, wz) in hit_pts_world:
                px = world_to_pixel(np.array([wx, wy, wz]), cam_tf)
                if px is not None:
                    hit_px.append(px)

            if len(hit_px) >= 3 and blocked:
                # Khi bị chặn: tô đậm phần từ tâm đến vật
                poly_hit = np.array(hit_px, dtype=np.int32)
                bright   = tuple(min(255, c + 80) for c in color)
                cv2.fillPoly(overlay, [poly_hit], bright)

        # ── Label tên + distance ─────────────────────────────────────────────
        # Tính điểm giữa cung (đỉnh xa nhất của quạt) để đặt text
        mid_wx = ox + s["radius"] * 0.6 * math.cos(s["yaw"])
        mid_wy = oy + s["radius"] * 0.6 * math.sin(s["yaw"])
        mid_px = world_to_pixel(np.array([mid_wx, mid_wy, oz]), cam_tf)
        if mid_px:
            status = "BLOCKED" if blocked else "CLEAR"
            label  = f"{s['name']} {s['dist']:.1f}m {status}"
            txt_col = (80, 80, 255) if blocked else (80, 255, 120)
            cv2.putText(overlay, label, (mid_px[0] - 40, mid_px[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, txt_col, 2)

    # Blend overlay lên ảnh gốc
    cv2.addWeighted(overlay, FAN_ALPHA, img, 1.0 - FAN_ALPHA, 0, img)

    # ── Vẽ viền đặc (không blend) để quạt nổi rõ ────────────────────────────
    for s in sensors:
        ox, oy, oz = s["origin"]
        blocked    = s["dist"] < s["thresh"]
        color      = s["col_blocked"] if blocked else s["col_clear"]

        full_pts_world = _fan_polygon_world(
            ox, oy, oz, s["yaw"], half_rad, s["radius"], FAN_SEGMENTS)
        full_px = []
        for (wx, wy, wz) in full_pts_world:
            px = world_to_pixel(np.array([wx, wy, wz]), cam_tf)
            if px is not None:
                full_px.append(px)
        if len(full_px) >= 3:
            cv2.polylines(img, [np.array(full_px, dtype=np.int32)],
                          True, color, 2)

        # Vẽ tâm sensor
        origin_px = world_to_pixel(np.array([ox, oy, oz]), cam_tf)
        if origin_px:
            cv2.circle(img, origin_px, 5, color, -1)
            cv2.circle(img, origin_px, 6, (255, 255, 255), 1)


# =========================
# DRAW WAYPOINTS (camera view)
# =========================
def draw_waypoints(img, cam_tf, route, pp_idx, target_wp, lc_wp, vehicle_tf, draw_n=40):
    if not route:
        return
    yaw  = math.radians(vehicle_tf.rotation.yaw)
    fwd  = np.array([math.cos(yaw), math.sin(yaw)])
    vpos = np.array([vehicle_tf.location.x, vehicle_tf.location.y])

    drawn = 0
    for i in range(pp_idx, len(route)):
        if drawn >= draw_n:
            break
        loc  = route[i].transform.location
        npos = np.array([loc.x, loc.y])
        d2   = np.linalg.norm(npos - vpos)
        if d2 > 1e-3 and np.dot((npos - vpos) / d2, fwd) <= 0:
            continue
        xyz = np.array([loc.x, loc.y, loc.z + 0.3])
        px  = world_to_pixel(xyz, cam_tf)
        if px is None:
            continue
        if i == pp_idx:
            cv2.circle(img, px, 8, COLOR_NEAR, -1)
            cv2.circle(img, px, 9, (0, 0, 0), 1)
            cv2.putText(img, "NEAR", (px[0]+10, px[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_NEAR, 1)
        else:
            r = max(3, 6 - drawn // 7)
            cv2.circle(img, px, r, COLOR_ROUTE, -1)
        drawn += 1

    if target_wp is not None:
        tl = target_wp.transform.location
        px = world_to_pixel(np.array([tl.x, tl.y, tl.z + 0.5]), cam_tf)
        if px:
            cv2.circle(img, px, 12, COLOR_TARGET, -1)
            cv2.circle(img, px, 13, (255, 255, 255), 2)
            cv2.putText(img, "TARGET", (px[0]+14, px[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TARGET, 2)

    if lc_wp is not None:
        ll = lc_wp.transform.location
        px = world_to_pixel(np.array([ll.x, ll.y, ll.z + 0.5]), cam_tf)
        if px:
            cv2.circle(img, px, 14, COLOR_LC_TGT, -1)
            cv2.circle(img, px, 15, (255, 255, 255), 2)
            cv2.putText(img, "LC TARGET", (px[0]+14, px[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LC_TGT, 2)


# =========================
# MINI-MAP
# =========================
class MiniMap:
    WIN_NAME = "MiniMap | Click to set destination"

    def __init__(self, m, all_wps, vehicle):
        self.m        = m
        self.vehicle  = vehicle
        self.goal_loc = None
        self.route    = []
        self._need_path = False

        xs = [w.transform.location.x for w in all_wps]
        ys = [w.transform.location.y for w in all_wps]
        self.map_xmin, self.map_xmax = min(xs), max(xs)
        self.map_ymin, self.map_ymax = min(ys), max(ys)

        margin  = 20
        range_x = self.map_xmax - self.map_xmin or 1
        range_y = self.map_ymax - self.map_ymin or 1
        # Dùng scale đồng nhất (giữ tỉ lệ) và căn giữa map trong window
        sx = (MAP_WIN_W - 2 * margin) / range_x
        sy = (MAP_WIN_H - 2 * margin) / range_y
        self.scale  = min(sx, sy)   # giữ aspect ratio
        # Tính offset để căn giữa trên cả 2 chiều
        self.off_x  = (MAP_WIN_W - int(range_x * self.scale)) // 2
        self.off_y  = (MAP_WIN_H - int(range_y * self.scale)) // 2
        self.margin = margin

        self._base_img = np.zeros((MAP_WIN_H, MAP_WIN_W, 3), dtype=np.uint8)
        self._base_img[:] = (18, 20, 28)

        # Grid mờ
        for gx in range(0, MAP_WIN_W, 60):
            cv2.line(self._base_img, (gx, 0), (gx, MAP_WIN_H), (28, 30, 38), 1)
        for gy in range(0, MAP_WIN_H, 60):
            cv2.line(self._base_img, (0, gy), (MAP_WIN_W, gy), (28, 30, 38), 1)

        # Vẽ road topology: nối từng waypoint với các waypoint kế tiếp
        # → thấy đầy đủ layout đường, không bị thưa như scatter dots
        print("[MAP] Drawing road topology …")
        drawn_segments = set()
        for wp in all_wps:
            p1 = self._world_to_map(wp.transform.location)
            for nwp in wp.next(4.0):   # spacing 4m để không quá dày
                p2 = self._world_to_map(nwp.transform.location)
                seg = (min(p1, p2), max(p1, p2))
                if seg in drawn_segments:
                    continue
                drawn_segments.add(seg)
                cv2.line(self._base_img, p1, p2, (60, 70, 90), 1)
        print(f"[MAP] Road topology drawn ({len(drawn_segments)} segments)")

        cv2.namedWindow(self.WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WIN_NAME, MAP_WIN_W, MAP_WIN_H)
        cv2.setMouseCallback(self.WIN_NAME, self._mouse_cb)

    def _world_to_map(self, loc):
        u = int((loc.x - self.map_xmin) * self.scale) + self.off_x
        v = int((loc.y - self.map_ymin) * self.scale) + self.off_y
        return (max(0, min(MAP_WIN_W-1, u)), max(0, min(MAP_WIN_H-1, v)))

    def _map_to_world(self, px, py):
        wx = (px - self.off_x) / self.scale + self.map_xmin
        wy = (py - self.off_y) / self.scale + self.map_ymin
        return carla.Location(wx, wy, 0.0)

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            world_loc = self._map_to_world(x, y)
            snap_wp = self.m.get_waypoint(
                world_loc, project_to_road=True,
                lane_type=carla.LaneType.Driving)
            if snap_wp:
                # Snap goal về cùng chiều đường với ego
                # (tránh click nhầm làn đối diện trên minimap)
                ego_wp = self.m.get_waypoint(
                    self.vehicle.get_location(), project_to_road=True,
                    lane_type=carla.LaneType.Driving)
                if ego_wp:
                    ego_sign = int(ego_wp.lane_id) < 0
                    snap_sign = int(snap_wp.lane_id) < 0
                    if ego_sign != snap_sign:
                        # Thử lấy lane cùng chiều gần nhất
                        alt = snap_wp.get_left_lane() or snap_wp.get_right_lane()
                        if alt and alt.lane_type == carla.LaneType.Driving:
                            snap_wp = alt
                            print(f"[MAP] Goal snapped to same-direction lane "
                                  f"{int(snap_wp.lane_id)}")
                self.goal_loc = snap_wp.transform.location
            else:
                self.goal_loc = world_loc
            self._need_path = True
            print(f"[MAP] Goal → ({self.goal_loc.x:.1f}, {self.goal_loc.y:.1f}) "
                  f"lane={int(snap_wp.lane_id) if snap_wp else '?'}")

    def maybe_plan(self, current_wp, ego_loc):
        if not self._need_path or self.goal_loc is None:
            return None
        self._need_path = False
        print("[MAP] Running A* …")
        new_route = astar_route(current_wp, self.goal_loc, self.m)
        self.route = new_route
        print(f"[MAP] A* done — {len(new_route)} waypoints")
        return new_route

    def draw(self, vehicle_tf, route, pp_idx):
        img = self._base_img.copy()

        if not route:
            cv2.putText(img, "Click on map to set destination",
                        (MAP_WIN_W//2 - 170, MAP_WIN_H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
        else:
            for i in range(1, len(route)):
                p1 = self._world_to_map(route[i-1].transform.location)
                p2 = self._world_to_map(route[i].transform.location)
                cv2.line(img, p1, p2, (0, 200, 80), 2)
            if pp_idx < len(route):
                pn = self._world_to_map(route[pp_idx].transform.location)
                cv2.circle(img, pn, 5, COLOR_NEAR, -1)

        yaw_rad = math.radians(vehicle_tf.rotation.yaw)
        ep   = self._world_to_map(vehicle_tf.location)
        size = 10
        tip  = (int(ep[0] + size * math.cos(yaw_rad)),
                int(ep[1] + size * math.sin(yaw_rad)))
        bl   = (int(ep[0] - size*0.5*math.cos(yaw_rad) + size*0.6*math.sin(yaw_rad)),
                int(ep[1] - size*0.5*math.sin(yaw_rad) - size*0.6*math.cos(yaw_rad)))
        br   = (int(ep[0] - size*0.5*math.cos(yaw_rad) - size*0.6*math.sin(yaw_rad)),
                int(ep[1] - size*0.5*math.sin(yaw_rad) + size*0.6*math.cos(yaw_rad)))
        cv2.fillPoly(img, [np.array([tip, bl, br], dtype=np.int32)], (255, 220, 0))

        if self.goal_loc:
            gp = self._world_to_map(self.goal_loc)
            cv2.drawMarker(img, gp, (0, 80, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.circle(img, gp, 8, (0, 80, 255), 2)
            cv2.putText(img, "GOAL", (gp[0]+10, gp[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1)

        cv2.putText(img, "Click to set destination",
                    (MAP_WIN_W//2 - 120, MAP_WIN_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 120, 120), 1)
        cv2.imshow(self.WIN_NAME, img)


# =========================
# HUD (camera view)
# =========================
def draw_hud(img, steer, speed_kmh, stop_reason, lc_decision,
             has_route, front_dist, left_dist, right_dist,
             avoid_state="CRUISING"):
    # ── Obstacle sensor bar ──────────────────────────────────────────────────
    bar_y   = IMG_H - 80
    bar_h   = 18
    max_d   = OBS_FRONT_DIST

    # Front
    front_ratio = min(front_dist / max_d, 1.0)
    front_col   = (0, 0, 255) if front_dist < OBS_HIT_THRESHOLD else (0, 200, 80)
    cv2.rectangle(img, (20, bar_y), (220, bar_y + bar_h), (50, 50, 50), -1)
    cv2.rectangle(img, (20, bar_y), (20 + int(200 * front_ratio), bar_y + bar_h), front_col, -1)
    cv2.putText(img, f"FRONT {front_dist:.1f}m", (20, bar_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, front_col, 2)

    # Left
    left_ratio = min(left_dist / max_d, 1.0)
    left_col   = (0, 200, 80) if left_dist >= LANE_CLEAR_THRESH else (0, 120, 255)
    cv2.rectangle(img, (20, bar_y + 26), (220, bar_y + 26 + bar_h), (50, 50, 50), -1)
    cv2.rectangle(img, (20, bar_y + 26),
                  (20 + int(200 * left_ratio), bar_y + 26 + bar_h), left_col, -1)
    cv2.putText(img, f"LEFT  {left_dist:.1f}m", (20, bar_y + 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_col, 2)

    # Right
    right_ratio = min(right_dist / max_d, 1.0)
    right_col   = (0, 200, 80) if right_dist >= LANE_CLEAR_THRESH else (0, 120, 255)
    cv2.rectangle(img, (240, bar_y + 26), (440, bar_y + 26 + bar_h), (50, 50, 50), -1)
    cv2.rectangle(img, (240, bar_y + 26),
                  (240 + int(200 * right_ratio), bar_y + 26 + bar_h), right_col, -1)
    cv2.putText(img, f"RIGHT {right_dist:.1f}m", (240, bar_y + 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_col, 2)

    # ── Speed / Steer ────────────────────────────────────────────────────────
    cv2.putText(img, f"Speed : {speed_kmh:.1f} km/h", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(img, f"Steer : {steer:.3f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if not has_route:
        cv2.putText(img, "WAITING FOR DESTINATION (click on map)",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
    elif stop_reason:
        cv2.putText(img, f"STOP: {stop_reason}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    # ── Avoidance state badge ────────────────────────────────────────────────
    state_colors = {
        "CRUISING":       (0,   200,  80),
        "CHECKING_LANES": (0,   200, 255),
        "CHANGING":       (0,   140, 255),
        "WAITING_CLEAR":  (0,    40, 220),
    }
    s_col = state_colors.get(avoid_state, (180, 180, 180))
    cv2.putText(img, f"STATE: {avoid_state}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, s_col, 2)

    if lc_decision:
        cv2.putText(img, f"AUTO-LC → {lc_decision.upper()}", (20, 198),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 100, 0), 2)

    cv2.putText(img, "A/D: lane change | ESC: quit | Map: click to route",
                (20, IMG_H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


# =========================
# CARLA INIT
# =========================
client = carla.Client("localhost", 2000)
client.set_timeout(30)
world  = client.get_world()

for actor in world.get_actors().filter('vehicle.*'):
    actor.destroy()
for actor in world.get_actors().filter('walker.pedestrian.*'):
    actor.destroy()

m      = world.get_map()
bp_lib = world.get_blueprint_library()

# ── Chọn spawn point cho ego vehicle ─────────────────────────────────────────
all_spawns = m.get_spawn_points()
print(f"\n[SPAWN] {len(all_spawns)} spawn points available.")
print("[SPAWN] Nhập số thứ tự spawn point (0 to {}) hoặc Enter để random: ".format(
    len(all_spawns) - 1), end="")
try:
    _inp = input().strip()
    _idx = int(_inp) if _inp else np.random.randint(len(all_spawns))
    _idx = max(0, min(_idx, len(all_spawns) - 1))
except Exception:
    _idx = 0
spawn = all_spawns[_idx]
print(f"[SPAWN] Using spawn point {_idx}: "
      f"({spawn.location.x:.1f}, {spawn.location.y:.1f})")

vehicle = world.spawn_actor(bp_lib.filter("vehicle.tesla.model3")[0], spawn)

print("[SEED] Nhập seed cho NPC spawn (Enter = 42): ", end="")
try:
    _seed_inp  = input().strip()
    RANDOM_SEED = int(_seed_inp) if _seed_inp else 42
except Exception:
    RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
print(f"[SEED] Using seed = {RANDOM_SEED}")

# =========================
# TRAFFIC MANAGER
# =========================
tm = client.get_trafficmanager(8000)
tm.set_global_distance_to_leading_vehicle(2.0)
tm.set_synchronous_mode(False)
tm.ignore_lights_percentage(vehicle, 100)  # ego bỏ qua đèn đỏ
tm.auto_lane_change(vehicle, False)         # ego không tự đổi làn

# =========================
# SPAWN NPCs
# =========================
NPC_VEHICLES    = 20
NPC_PEDESTRIANS = 10

spawn_points = m.get_spawn_points()
npc_vehicles = []
npc_walkers  = []
npc_controllers = []

vehicle_bps = [b for b in bp_lib.filter("vehicle.*")
               if int(b.get_attribute("number_of_wheels")) >= 4]

# Tạo danh sách spawn transform được snap về waypoint
# → xe spawn đúng hướng đường, không bị góc chéo
valid_spawns = []
for sp in spawn_points:
    if math.hypot(sp.location.x - spawn.location.x,
                  sp.location.y - spawn.location.y) < 8.0:
        continue  # quá gần ego
    wp = m.get_waypoint(sp.location, project_to_road=True,
                        lane_type=carla.LaneType.Driving)
    if wp is None:
        continue
    # Dùng rotation của waypoint (đúng hướng lane), nâng z 0.3m tránh chui đất
    valid_spawns.append(carla.Transform(
        carla.Location(wp.transform.location.x,
                       wp.transform.location.y,
                       wp.transform.location.z + 0.3),
        wp.transform.rotation,
    ))

np.random.shuffle(valid_spawns)

# # Spawn theo batch nhỏ, tick giữa mỗi batch để physics settle
# BATCH = 10
# for i in range(0, min(NPC_VEHICLES, len(valid_spawns)), BATCH):
#     for sp in valid_spawns[i:i + BATCH]:
#         bp = np.random.choice(vehicle_bps)
#         if bp.has_attribute("color"):
#             bp.set_attribute("color",
#                 np.random.choice(bp.get_attribute("color").recommended_values))
#         npc = world.try_spawn_actor(bp, sp)
#         if npc:
#             npc.set_autopilot(True, 8000)
#             tm.ignore_lights_percentage(npc, 0)
#             tm.vehicle_percentage_speed_difference(npc, 0)
#             tm.auto_lane_change(npc, True)
#             npc_vehicles.append(npc)
#     if hasattr(world, "tick"):
#         world.tick()

# print(f"[INIT] Spawned {len(npc_vehicles)}/{NPC_VEHICLES} NPC vehicles")

# walker_bps           = bp_lib.filter("walker.pedestrian.*")
# walker_controller_bp = bp_lib.find("controller.ai.walker")
# for _ in range(NPC_PEDESTRIANS):
#     bp = np.random.choice(walker_bps)
#     if bp.has_attribute("is_invincible"):
#         bp.set_attribute("is_invincible", "false")
#     loc = world.get_random_location_from_navigation()
#     if loc is None:
#         continue
#     walker = world.try_spawn_actor(bp, carla.Transform(loc))
#     if walker is None:
#         continue
#     ctrl = world.spawn_actor(walker_controller_bp, carla.Transform(),
#                              attach_to=walker)
#     npc_walkers.append(walker)
#     npc_controllers.append(ctrl)

# world.tick() if hasattr(world, "tick") else None
# for ctrl in npc_controllers:
#     ctrl.start()
#     dest = world.get_random_location_from_navigation()
#     if dest:
#         ctrl.go_to_location(dest)
#     ctrl.set_max_speed(1.0 + np.random.random() * 1.5)
# print(f"[INIT] Spawned {len(npc_walkers)} pedestrians")

# Spawn theo batch nhỏ, tick giữa mỗi batch để physics settle
BATCH = 10
for i in range(0, min(NPC_VEHICLES, len(valid_spawns)), BATCH):
    for sp in valid_spawns[i:i + BATCH]:
        bp = np.random.choice(vehicle_bps)
        if bp.has_attribute("color"):
            bp.set_attribute("color",
                np.random.choice(bp.get_attribute("color").recommended_values))
        npc = world.try_spawn_actor(bp, sp)
        if npc:
            # ← Không autopilot, đứng yên tại chỗ
            npc.apply_control(carla.VehicleControl(
                throttle=0.0, steer=0.0, brake=1.0, hand_brake=True))
            npc_vehicles.append(npc)
    if hasattr(world, "tick"):
        world.tick()

print(f"[INIT] Spawned {len(npc_vehicles)}/{NPC_VEHICLES} NPC vehicles (STATIONARY)")

walker_bps           = bp_lib.filter("walker.pedestrian.*")
walker_controller_bp = bp_lib.find("controller.ai.walker")
for _ in range(NPC_PEDESTRIANS):
    bp = np.random.choice(walker_bps)
    if bp.has_attribute("is_invincible"):
        bp.set_attribute("is_invincible", "false")
    loc = world.get_random_location_from_navigation()
    if loc is None:
        continue
    walker = world.try_spawn_actor(bp, carla.Transform(loc))
    if walker is None:
        continue
    ctrl = world.spawn_actor(walker_controller_bp, carla.Transform(),
                             attach_to=walker)
    npc_walkers.append(walker)
    npc_controllers.append(ctrl)

world.tick() if hasattr(world, "tick") else None
for ctrl in npc_controllers:
    ctrl.start()
    ctrl.stop()   # ← đứng yên, không đi đâu cả
print(f"[INIT] Spawned {len(npc_walkers)} pedestrians (STATIONARY)")

# =========================
# CAMERA SENSOR
# =========================
cam_bp = bp_lib.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", str(IMG_W))
cam_bp.set_attribute("image_size_y", str(IMG_H))
camera = world.spawn_actor(
    cam_bp,
    carla.Transform(carla.Location(x=0.5, z=1.3), carla.Rotation(pitch=0)),
    attach_to=vehicle,
)
camera.listen(camera_cb)

# =========================
# OBSTACLE SENSORS
# ─ 3 sensor: front / left-lane / right-lane
# ─ Gắn cố định lên xe, CARLA tự tính transform theo xe
# =========================
obs_bp = bp_lib.find("sensor.other.obstacle")
obs_bp.set_attribute("distance",        str(OBS_FRONT_DIST))
obs_bp.set_attribute("hit_radius",      "0.5")   # m — bán kính quét
obs_bp.set_attribute("only_dynamics",   "false")  # detect cả static + dynamic
obs_bp.set_attribute("debug_linetrace", "false")

# ── Obstacle sensor helper ──────────────────────────────────────────────────
def make_obs_sensor(bp_lib, distance, hit_radius=0.5):
    """
    only_dynamics=true  → chỉ detect actor động (xe, người), KHÔNG detect
                          static mesh hay body của chính xe mình.
    hit_radius nhỏ (0.5m) → tránh quét quá rộng sang lane khác.
    """
    bp = bp_lib.find("sensor.other.obstacle")
    bp.set_attribute("distance",        str(distance))
    bp.set_attribute("hit_radius",      str(hit_radius))
    bp.set_attribute("only_dynamics",   "true")   # ← chỉ dynamic actor
    bp.set_attribute("debug_linetrace", "false")
    return bp

# Front sensor — thẳng mũi xe, z=0.8m (tránh hit mặt đường)
sensor_front = world.spawn_actor(
    make_obs_sensor(bp_lib, OBS_FRONT_DIST, hit_radius=0.5),
    carla.Transform(carla.Location(x=2.5, y=0.0, z=0.8)),
    attach_to=vehicle,
)
sensor_front.listen(obs_front_cb)

# Left-lane sensor — offset sang trái ~1 lane, z=0.8m
sensor_left = world.spawn_actor(
    make_obs_sensor(bp_lib, LANE_CHECK_DIST, hit_radius=0.5),
    carla.Transform(carla.Location(x=2.5, y=-LANE_SENSOR_OFFSET_Y, z=0.8)),
    attach_to=vehicle,
)
sensor_left.listen(obs_left_cb)

# Right-lane sensor — offset sang phải ~1 lane, z=0.8m
sensor_right = world.spawn_actor(
    make_obs_sensor(bp_lib, LANE_CHECK_DIST, hit_radius=0.5),
    carla.Transform(carla.Location(x=2.5, y=LANE_SENSOR_OFFSET_Y, z=0.8)),
    attach_to=vehicle,
)
sensor_right.listen(obs_right_cb)

print("[INIT] Obstacle sensors attached (front / left-lane / right-lane)")

# =========================
# ROUTE / PP INIT
# =========================
start_wp = m.get_waypoint(spawn.location, project_to_road=True,
                          lane_type=carla.LaneType.Driving)
pp    = PurePursuit()
route = []

# =========================
# MINI-MAP INIT
# =========================
print("[INIT] Loading waypoints for mini-map …")
all_wps = [w for w in m.generate_waypoints(2.0)
           if w.lane_type == carla.LaneType.Driving]
mini_map = MiniMap(m, all_wps, vehicle)
print(f"[INIT] Mini-map ready ({len(all_wps)} wps)")
print("[INIT] Vehicle ready. Click on mini-map to set destination.")

# =========================
# AUTO-LC cooldown
# =========================
_last_lc_time      = -LC_COOLDOWN
lane_change_request = None
_tick_count         = 0   # đếm frame để bỏ qua warmup

# =========================
# AVOIDANCE STATE MACHINE
# =========================
# Trạng thái:
#   "CRUISING"       — đi bình thường, front trống
#   "CHECKING_LANES" — phát hiện vật cản, kiểm tra lane kề (1 lần duy nhất)
#   "CHANGING"       — đã ra lệnh đổi lane, chờ xe sang lane xong
#   "WAITING_CLEAR"  — không lane nào trống, đứng yên chờ đường thông
# State machine: dùng dict để tránh vấn đề global trong try/while scope
avd = {
    "state":          "CRUISING",
    "clear_since":    None,
    "settle_frames":  0,
    "settling_since": None,
    "blocked_since":  None,   # timestamp khi bắt đầu bị chặn
}
CLEAR_HYSTERESIS = 2.0   # s — hysteresis trước khi về CRUISING

# =========================
# MAIN LOOP
# =========================
try:
    while True:
        loc       = vehicle.get_location()
        tf        = vehicle.get_transform()
        speed_kmh = get_speed(vehicle)
        lookahead = LOOKAHEAD_MIN + LOOKAHEAD_GAIN * speed_kmh / 3.6
        now       = _time.time()

        current_wp = m.get_waypoint(loc, project_to_road=True,
                                    lane_type=carla.LaneType.Driving)

        # ── Reset obs mỗi frame (sensor chỉ fire khi có vật → tự về inf khi trống)
        reset_obs_each_frame()

        # ── Warmup: bỏ qua N tick đầu để sensor ổn định ─────────────────────
        _tick_count += 1
        if _tick_count % 60 == 0:
            left_adj  = current_wp.get_left_lane()
            right_adj = current_wp.get_right_lane()
            print(f"[DBG] lane={int(current_wp.lane_id)} road={current_wp.road_id} "
                  f"section={current_wp.section_id} | "
                  f"left={'None' if left_adj is None else f'{int(left_adj.lane_id)}({left_adj.lane_type})'} "
                  f"right={'None' if right_adj is None else f'{int(right_adj.lane_id)}({right_adj.lane_type})'}")
        if _tick_count <= SENSOR_WARMUP_TICKS:
            obs["front"] = float("inf")
            obs["left"]  = float("inf")
            obs["right"] = float("inf")

        # ── Mini-map: plan khi user click ────────────────────────────────────
        new_route = mini_map.maybe_plan(current_wp, loc)
        if new_route:
            route = new_route
            mini_map.route = route
            pp.set_route(route, loc)
            lc_target_wp = None
            print(f"[ROUTE] Activated ({len(route)} wp)")

        # ── Keyboard ─────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            lane_change_request = "left"
        elif key == ord('d'):
            lane_change_request = "right"
        elif key == 27:
            break

        # ── Manual lane change ────────────────────────────────────────────────
        if lane_change_request and pp.has_route:
            tgt_wp, new_r = do_lane_change(current_wp, lane_change_request, m,
                                           goal_loc=mini_map.goal_loc)
            if tgt_wp and new_r:
                route          = new_r
                mini_map.route = route
                lc_target_wp   = tgt_wp
                pp.set_route_lc(route)
                _last_lc_time  = now
                # Override bất kể state hiện tại — kể cả đang CHANGING
                avd["state"]         = "CHANGING"
                avd["settle_frames"] = 0
                print(f"[MANUAL-LC] {lane_change_request} → lane {int(new_r[0].lane_id)}, "
                      f"{len(new_r)} wp (override state → CHANGING)")
            else:
                adj_dbg = current_wp.get_left_lane() if lane_change_request == "left"                           else current_wp.get_right_lane()
                print(f"[MANUAL-LC] {lane_change_request} failed | "
                      f"cur lane={int(current_wp.lane_id)} | "
                      f"adj={'None' if adj_dbg is None else f'lane={int(adj_dbg.lane_id)} type={adj_dbg.lane_type}'}")
            lane_change_request = None
        elif lane_change_request:
            lane_change_request = None

        # ── Pure pursuit ──────────────────────────────────────────────────────
        steer  = 0.0
        target = None
        if pp.has_route:
            target = pp.target(loc, lookahead)
            steer  = pp.steer(vehicle, target)

        if lc_target_wp is not None:
            if carla_dist(loc, lc_target_wp.transform.location) < lookahead:
                lc_target_wp = None

        # ── Stop / throttle ───────────────────────────────────────────────────
        stop_reason = ""
        lc_decision = ""
        throttle    = 0.0
        brake       = 1.0

        if not pp.has_route:
            # Chưa có đích — đứng yên
            throttle = 0.0
            brake    = 1.0
        else:
            # ── Đến đích ─────────────────────────────────────────────────────
            if mini_map.goal_loc is not None:
                dist_to_goal = carla_dist(loc, mini_map.goal_loc)
                if dist_to_goal < GOAL_REACH_DIST:
                    # Apply brake ngay lập tức, không đợi xuống cuối loop
                    throttle = 0.0
                    brake    = 1.0
                    stop_reason = f"ARRIVED AT GOAL ({dist_to_goal:.1f}m)"
                    # Clear route và goal — xe sẽ đứng yên frame tiếp theo
                    pp.route          = []
                    route             = []
                    mini_map.route    = []
                    mini_map.goal_loc = None
                    avd["state"]      = "CRUISING"
                    print("[GOAL] Arrived — waiting for new destination")
                    # Apply control ngay và skip phần còn lại của loop
                    vehicle.apply_control(carla.VehicleControl(
                        throttle=0.0, steer=0.0, brake=1.0))
                    # Vẫn render camera và minimap trước khi continue
                    if camera_img is not None and cam_transform_cache is not None:
                        img = camera_img.copy()
                        draw_waypoints(img, cam_transform_cache,
                                       [], 0, None, None, tf)
                        draw_hud(img, 0.0, speed_kmh, stop_reason, "",
                                 has_route=False,
                                 front_dist=obs["front"],
                                 left_dist=obs["left"],
                                 right_dist=obs["right"],
                                 avoid_state="CRUISING")
                        cv2.imshow("CARLA | Obstacle Sensor + Pure Pursuit", img)
                    mini_map.draw(tf, [], 0)
                    cv2.waitKey(1)
                    continue   # ← skip toàn bộ state machine và throttle logic

            # ── Đèn đỏ — TẠM TẮT (ego đã set ignore_lights 100%) ──────────────
            # if not stop_reason:
            #     tl = vehicle.get_traffic_light()
            #     if tl and tl.get_state() == carla.TrafficLightState.Red:
            #         stop_reason = "RED LIGHT"

            # ── Avoidance state machine ───────────────────────────────────────
            front_blocked = obs["front"] < OBS_HIT_THRESHOLD

            if avd["state"] == "CRUISING":
                if front_blocked and not stop_reason:
                    avd["state"]          = "CHECKING_LANES"
                    avd["settle_frames"]  = 0
                    avd["clear_since"]    = None
                    avd["blocked_since"]  = now   # bắt đầu đếm thời gian bị chặn
                    print(f"[SM] CRUISING → CHECKING_LANES (front={obs['front']:.1f}m)")

            elif avd["state"] == "CHECKING_LANES":
                if not front_blocked:
                    avd["state"] = "CRUISING"
                    print("[SM] CHECKING_LANES → CRUISING (cleared)")
                else:
                    avd["settle_frames"] += 1
                    if avd["settle_frames"] >= SENSOR_SETTLE_FRAMES:
                        decision = decide_lane_change(current_wp)
                        print(f"[SM] decision={decision} "
                              f"L={obs['left']:.1f}m R={obs['right']:.1f}m")

                        if decision in ("left", "right"):
                            tgt_wp, new_r = do_lane_change(
                                current_wp, decision, m,
                                goal_loc=mini_map.goal_loc)
                            if tgt_wp and new_r:
                                route = new_r; mini_map.route = route
                                lc_target_wp  = tgt_wp
                                pp.set_route_lc(route)
                                _last_lc_time = now
                                lc_decision   = decision
                                steer = pp.steer(vehicle, pp.target(loc, lookahead))
                                avd["state"]  = "CHANGING"
                                print(f"[SM] CHECKING_LANES → CHANGING ({decision})")
                            else:
                                avd["state"] = "WAITING_CLEAR"
                                print("[SM] CHECKING_LANES → WAITING_CLEAR (no valid lane)")
                        else:
                            avd["state"] = "WAITING_CLEAR"
                            print("[SM] CHECKING_LANES → WAITING_CLEAR (all blocked)")

            elif avd["state"] == "CHANGING":
                # ── Manual override: user bấm A/D trong khi đang CHANGING ────
                # lane_change_request đã được xử lý ở trên (set route mới)
                # chỉ cần reset state về CHANGING để tiếp tục với route mới
                if lc_target_wp is None:
                    avd["state"]          = "SETTLING"
                    avd["settling_since"] = now
                    print("[SM] CHANGING → SETTLING")

            elif avd["state"] == "SETTLING":
                if front_blocked:
                    # Lane mới cũng có xe → về CHECKING ngay
                    avd["state"]         = "CHECKING_LANES"
                    avd["settle_frames"] = 0
                    avd["blocked_since"] = now
                    print("[SM] SETTLING → CHECKING_LANES (new lane blocked)")
                elif now - avd["settling_since"] >= POST_LC_SETTLE_TIME:
                    avd["state"] = "CRUISING"
                    print("[SM] SETTLING → CRUISING")

            elif avd["state"] == "WAITING_CLEAR":
                blocked_duration = now - (avd["blocked_since"] or now)
                if not front_blocked:
                    if avd["clear_since"] is None:
                        avd["clear_since"] = now
                    elif now - avd["clear_since"] >= CLEAR_HYSTERESIS:
                        avd["state"]         = "CHECKING_LANES"
                        avd["clear_since"]   = None
                        avd["settle_frames"] = 0
                        avd["blocked_since"] = now
                        print("[SM] WAITING_CLEAR → CHECKING_LANES")
                else:
                    avd["clear_since"] = None
                    # Đã dừng > 3s → retry check lane (xe kề có thể đã đi qua)
                    if blocked_duration > 3.0:
                        avd["state"]         = "CHECKING_LANES"
                        avd["settle_frames"] = 0
                        avd["blocked_since"] = now   # reset để timer chạy lại
                        print("[SM] WAITING_CLEAR → CHECKING_LANES (retry after 3s)")

            # ── Throttle / brake ──────────────────────────────────────────────
            blocked_time = now - (avd["blocked_since"] or now)

            if avd["state"] == "CHANGING":
                # Đang đánh lái — tiếp tục di chuyển, steer mạnh hơn bình thường
                throttle = float(np.clip(
                    0.35 + 0.02 * (TARGET_SPEED - speed_kmh), 0.0, 0.7))
                brake    = 0.0
                # Steer đã được set ở trên khi CHANGING bắt đầu

            elif avd["state"] == "WAITING_CLEAR":
                throttle    = 0.0
                brake       = 1.0
                stop_reason = stop_reason or "BLOCKED — NO CLEAR LANE"

            elif avd["state"] == "CHECKING_LANES":
                # Di chuyển chậm trong ~1s đầu, chỉ dừng hẳn nếu quá gần
                time_to_stop = max(0.0, 1.0 - blocked_time)   # còn bao lâu trước khi dừng
                ratio = max(0.0, (obs["front"] - OBS_STOP_THRESHOLD) /
                                 max(0.1, OBS_HIT_THRESHOLD - OBS_STOP_THRESHOLD))
                # Giữ throttle thấp (không dừng ngay) trong 1s đầu
                min_throttle = 0.15 * max(0.0, time_to_stop)
                throttle = max(min_throttle, float(np.clip(
                    0.4 + 0.03 * (TARGET_SPEED - speed_kmh), 0.0, 1.0)) * ratio)
                brake    = 0.0 if throttle > 0.05 else 0.1

            elif stop_reason:
                throttle = 0.0
                brake    = 1.0

            else:
                # CRUISING / SETTLING
                throttle = float(np.clip(
                    0.4 + 0.03 * (TARGET_SPEED - speed_kmh), 0.0, 1.0))
                brake    = 0.2 if speed_kmh > TARGET_SPEED + 3 else 0.0

            # ── Phanh khẩn cấp — chỉ khi rất gần VÀ không đang đổi lane ─────
            if (obs["front"] < OBS_STOP_THRESHOLD
                    and avd["state"] not in ("CHANGING", "SETTLING")
                    and not stop_reason):
                throttle    = 0.0
                brake       = 1.0
                stop_reason = f"TOO CLOSE ({obs['front']:.1f}m)"

            # Route extension đã được bỏ — A* plan thẳng đến goal,
            # goal-reached logic ở trên sẽ dừng xe đúng chỗ.

        # ── Camera view ───────────────────────────────────────────────────────
        if camera_img is not None and cam_transform_cache is not None:
            img = camera_img.copy()

            # ── Sensor fan overlay (vẽ trước waypoints để không che)
            draw_sensor_fans(img, tf, cam_transform_cache,
                             obs["front"],
                             obs["left"],
                             obs["right"])

            draw_waypoints(img, cam_transform_cache,
                           route, pp.idx, target, lc_target_wp, tf)

            draw_hud(img, steer, speed_kmh, stop_reason, lc_decision,
                     has_route=pp.has_route,
                     front_dist=obs["front"],
                     left_dist=obs["left"],
                     right_dist=obs["right"],
                     avoid_state=avd["state"])

            cv2.imshow("CARLA | Obstacle Sensor + Pure Pursuit", img)

        # ── Mini-map ──────────────────────────────────────────────────────────
        mini_map.draw(tf, route, pp.idx)

        # ── Apply control ─────────────────────────────────────────────────────
        vehicle.apply_control(carla.VehicleControl(
            throttle=throttle, steer=steer, brake=brake))

finally:
    sensor_front.stop(); sensor_front.destroy()
    sensor_left.stop();  sensor_left.destroy()
    sensor_right.stop(); sensor_right.destroy()
    camera.stop();       camera.destroy()
    vehicle.destroy()
    for ctrl in npc_controllers:
        ctrl.stop()
    for w in npc_walkers:
        w.destroy()
    for v in npc_vehicles:
        v.destroy()
    cv2.destroyAllWindows()
    print("[EXIT] Done.")