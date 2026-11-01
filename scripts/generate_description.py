#!/usr/bin/env python3
"""Generate the derived files of aibot_description from the source URDF.

Inputs:
    src/aibot_description/urdf/robot_original.urdf     source URDF (BSD-3)
    src/aibot_description/meshes/visual/*.STL          source meshes
Outputs:
    src/aibot_description/urdf/aibot_body.xacro        links + joints, package:// paths
    src/aibot_description/meshes/collision/*.STL       convex hulls of the visual meshes
    src/aibot_description/config/joint_limits.yaml     limits of the 43 controlled joints

Every number in the outputs comes from the source file or its meshes; this script only
re-expresses them. Transformations:

  1. Mesh paths ../meshes/X.STL -> package://aibot_description/meshes/{visual,collision}/X.STL
  2. Re-root at the pelvis. The source model's root is base_link (the TORSO); the waist chain
     base_link -> waist_pitch -> waist_roll -> waist_yaw_link hangs *below* it and the legs hang
     from waist_yaw_link. Fixing base_link to the world would make the waist joints swing the
     legs. The chain is inverted so waist_yaw_link (pelvis) is the root and the torso sits on
     the waist. Joint axes are negated so a joint value q means the *same* relative
     configuration as the source model (limits and sign conventions stay valid for the real robot).
  3. Legs: wrapped in xacro:if lock_legs -> fixed joints at the README upright pose
     (hip_pitch 30 deg, knee -60 deg, ankle_pitch 30 deg, others 0). The source revolute
     definitions are kept in the xacro:unless branch.
  4. Collisions: xacro:if simple_collision -> convex hull of the visual mesh (a few hundred
     triangles instead of up to 20k); sub-6 mm dummy links (tips, flanges, mic) get none.
     Otherwise the original visual mesh, as the source model does.
  5. Hand joints: source dynamics (damping=1, friction=1) exceed the finger effort limits
     (0.4-1.1 Nm), so the fingers could never move in simulation; damping/friction become
     macro parameters.
  6. Inertia tensors violating the triangle inequality are balanced the way MuJoCo's
     balanceinertia="true" does (the source file relies on it): principal moments -> mean.

Usage (from the workspace root):  python3 scripts/generate_description.py
"""
from __future__ import annotations

import argparse
import copy
import math
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

XACRO_NS = "http://www.ros.org/wiki/xacro"
PKG = "aibot_description"
NEW_ROOT = "waist_yaw_link"
LEG_JOINT_SUFFIXES = ("hip_roll_joint", "hip_yaw_joint", "hip_pitch_joint",
                      "knee_joint", "ankle_pitch_joint", "ankle_roll_joint")
# README "fully upright" leg pose (degrees). Leg joints not listed are 0.
UPRIGHT_DEG = {"hip_pitch_joint": 30.0, "knee_joint": -60.0, "ankle_pitch_joint": 30.0}
TINY_LINK_EXTENT = 0.006  # m; links whose mesh fits in this cube get no simple collision
HULL_GRID = 0.0005        # m; vertices snapped to this grid before hulling (fewer facets)

# Order of the controlled joint groups in joint_limits.yaml (prefix match on real joint names).
GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("waist", ("waist_",)),
    ("head", ("neck_",)),
    ("left_arm", ("left_shoulder_", "left_arm_", "left_elbow_", "left_wrist_")),
    ("right_arm", ("right_shoulder_", "right_arm_", "right_elbow_", "right_wrist_")),
    ("left_hand", ("left_hand_",)),
    ("right_hand", ("right_hand_",)),
]


# --------------------------------------------------------------------------- math helpers
def rot_rpy(r: float, p: float, y: float) -> np.ndarray:
    """URDF convention: fixed-axis XYZ, i.e. R = Rz(y) @ Ry(p) @ Rx(r)."""
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def rpy_from_rot(R: np.ndarray) -> tuple[float, float, float]:
    sp = -R[2, 0]
    if abs(sp) < 1.0 - 1e-12:
        return (math.atan2(R[2, 1], R[2, 2]), math.asin(sp), math.atan2(R[1, 0], R[0, 0]))
    return (math.atan2(-R[1, 2], R[1, 1]), math.copysign(math.pi / 2, sp), 0.0)  # gimbal lock


def rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation about a (normalised) axis."""
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def tf(xyz: np.ndarray, R: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, xyz
    return T


def tf_inv(T: np.ndarray) -> np.ndarray:
    R, p = T[:3, :3], T[:3, 3]
    return tf(-R.T @ p, R.T)


def is_identity(T: np.ndarray) -> bool:
    return bool(np.allclose(T, np.eye(4), atol=1e-12))


def fmt(v: float) -> str:
    return "0" if abs(v) < 1e-12 else f"{v:.9g}"


def vec(s: str | None, default: str = "0 0 0") -> np.ndarray:
    return np.array([float(x) for x in (s or default).split()])


# --------------------------------------------------------------------------- xml helpers
def origin_tf(el: ET.Element) -> np.ndarray:
    o = el.find("origin")
    if o is None:
        return np.eye(4)
    return tf(vec(o.get("xyz")), rot_rpy(*vec(o.get("rpy"))))


def set_origin(el: ET.Element, T: np.ndarray) -> None:
    o = el.find("origin")
    if o is None:
        o = ET.Element("origin")
        el.insert(0, o)
    o.set("xyz", " ".join(fmt(v) for v in T[:3, 3]))
    o.set("rpy", " ".join(fmt(v) for v in rpy_from_rot(T[:3, :3])))


def premultiply_origin(el: ET.Element, T: np.ndarray) -> None:
    """el.origin := T @ el.origin (re-express a frame after its parent frame moved)."""
    set_origin(el, T @ origin_tf(el))


@dataclass
class Robot:
    root: ET.Element
    links: dict[str, ET.Element]
    joints: list[ET.Element]

    @staticmethod
    def parent(j: ET.Element) -> str:
        return j.find("parent").get("link")

    @staticmethod
    def child(j: ET.Element) -> str:
        return j.find("child").get("link")

    def joint_by_child(self, link: str) -> ET.Element | None:
        return next((j for j in self.joints if self.child(j) == link), None)

    def child_joints(self, link: str) -> list[ET.Element]:
        return [j for j in self.joints if self.parent(j) == link]

    def root_link(self) -> str:
        kids = {self.child(j) for j in self.joints}
        return next(n for n in self.links if n not in kids)


def is_leg(j: ET.Element) -> bool:
    return j.get("type") == "revolute" and j.get("name").endswith(LEG_JOINT_SUFFIXES)


# --------------------------------------------------------------------------- transformations
def reroot(robot: Robot, new_root: str) -> list[str]:
    """Invert the joint chain old_root -> ... -> new_root.

    Chain L0..Ln with joints J1..Jn (Ji: L(i-1) -> Li, origin Ti, axis ai). Source motion:
        F_i = F_(i-1) @ T_i @ R(a_i, q_i)
    After inversion L(i-1) is the child of Ji. URDF needs a child frame to sit on its joint, so
    L(k) is relocated to F'_k = F_k @ T_(k+1) (a no-op when T is identity). Then
        new Ji: parent Li, child L(i-1), origin T_(i+1)^-1 (identity for i = n), axis -a_i
    and everything attached to a relocated link is pre-multiplied by T_(k+1)^-1.
    """
    old_root = robot.root_link()
    path: list[ET.Element] = []
    cur = new_root
    while cur != old_root:
        j = robot.joint_by_child(cur)
        assert j is not None, f"{cur} has no parent joint"
        path.append(j)
        cur = robot.parent(j)
    path.reverse()
    chain = [old_root] + [robot.child(j) for j in path]
    T = [origin_tf(j) for j in path]  # T[i-1] is T_i
    n = len(path)
    log = [f"re-rooted at {new_root}; inverted chain {' > '.join(chain)}"]

    for k in range(n):  # relocate L0..L(n-1) and their non-chain children
        Tinv = tf_inv(T[k])
        if is_identity(Tinv):
            continue
        link = robot.links[chain[k]]
        for tag in ("inertial", "visual", "collision"):
            for el in link.findall(tag):
                premultiply_origin(el, Tinv)
        for j in robot.child_joints(chain[k]):
            if j not in path:
                premultiply_origin(j, Tinv)
        log.append(f"  frame of {chain[k]} moved to its joint pivot (offset {' '.join(fmt(v) for v in T[k][:3, 3])})")

    for i in range(1, n + 1):  # invert the joints
        j = path[i - 1]
        j.find("parent").set("link", chain[i])
        j.find("child").set("link", chain[i - 1])
        set_origin(j, tf_inv(T[i]) if i < n else np.eye(4))
        if j.get("type") == "fixed":
            for tag in ("axis", "limit", "dynamics"):
                for el in j.findall(tag):
                    j.remove(el)
            note = "fixed"
        else:
            ax = j.find("axis")
            ax.set("xyz", " ".join(fmt(-v) for v in vec(ax.get("xyz"))))
            note = f"axis negated to {ax.get('xyz')}"
        log.append(f"  {j.get('name')}: parent {chain[i]}, child {chain[i - 1]}, {note}")
    return log


def locked_leg_joint(j: ET.Element) -> ET.Element:
    """Fixed copy of a leg joint frozen at the README upright angle."""
    name = j.get("name")
    deg = next((v for suf, v in UPRIGHT_DEG.items() if name.endswith(suf)), 0.0)
    R = rot_axis(vec(j.find("axis").get("xyz")), math.radians(deg))
    f = ET.Element("joint", {"name": name, "type": "fixed"})
    f.append(copy.deepcopy(j.find("parent")))
    f.append(copy.deepcopy(j.find("child")))
    set_origin(f, origin_tf(j) @ tf(np.zeros(3), R))
    return f


def balance_inertia(link: ET.Element) -> str | None:
    inert = link.find("inertial")
    if inert is None:
        return None
    i = inert.find("inertia")
    I = np.array([[float(i.get("ixx")), float(i.get("ixy")), float(i.get("ixz"))],
                  [float(i.get("ixy")), float(i.get("iyy")), float(i.get("iyz"))],
                  [float(i.get("ixz")), float(i.get("iyz")), float(i.get("izz"))]])
    ev = np.sort(np.linalg.eigvalsh(I))
    if ev[0] > 0 and ev[0] + ev[1] >= ev[2] * (1 - 1e-9):
        return None
    mean = float(ev.mean())
    for k in ("ixy", "ixz", "iyz"):
        i.set(k, "0")
    for k in ("ixx", "iyy", "izz"):
        i.set(k, fmt(mean))
    return (f"  inertia of {link.get('name')} violated the triangle inequality "
            f"(principal {', '.join(f'{e:.3g}' for e in ev)}); set isotropic {mean:.3g}")


# --------------------------------------------------------------------------- meshes
def read_stl(path: Path) -> np.ndarray:
    """Return (n, 3) vertices of a binary or ASCII STL."""
    data = path.read_bytes()
    if data[:5] == b"solid" and b"facet" in data[:512]:
        pts = [[float(x) for x in ln.split()[1:4]] for ln in data.decode(errors="ignore").splitlines()
               if ln.strip().startswith("vertex")]
        return np.array(pts)
    n = struct.unpack("<I", data[80:84])[0]
    rec = np.frombuffer(data[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return rec["v"].reshape(-1, 3).astype(float)


def write_convex_hull(verts: np.ndarray, out: Path) -> int:
    """Write the convex hull of `verts` as a binary STL with outward normals. Returns #facets."""
    pts = np.unique(np.round(verts / HULL_GRID) * HULL_GRID, axis=0)
    hull = ConvexHull(pts)
    tris = pts[hull.simplices]
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    flip = np.einsum("ij,ij->i", nrm, hull.equations[:, :3]) < 0  # qhull gives no winding guarantee
    tris[flip] = tris[flip][:, [0, 2, 1]]
    nrm[flip] *= -1
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    rec = np.zeros(len(tris), dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    rec["n"], rec["v"] = nrm, tris.reshape(-1, 9)
    out.write_bytes(b"aibot convex hull".ljust(80, b"\0") + struct.pack("<I", len(tris)) + rec.tobytes())
    return len(tris)


def mesh_name(link: ET.Element) -> str | None:
    m = link.find("visual/geometry/mesh")
    return Path(m.get("filename")).name if m is not None else None


# --------------------------------------------------------------------------- forward kinematics
def link_poses(robot: Robot, root: str) -> dict[str, np.ndarray]:
    """Link poses in the root frame at q = 0 with legs locked upright."""
    poses = {root: np.eye(4)}
    stack = [root]
    while stack:
        p = stack.pop()
        for j in robot.child_joints(p):
            T = origin_tf(locked_leg_joint(j) if is_leg(j) else j)
            poses[robot.child(j)] = poses[p] @ T
            stack.append(robot.child(j))
    return poses


# --------------------------------------------------------------------------- outputs
def write_joint_limits(robot: Robot, out: Path) -> int:
    movable = [j for j in robot.joints if j.get("type") == "revolute" and not is_leg(j)]
    lines = ["# AUTO-GENERATED by scripts/generate_description.py from robot_original.urdf - DO NOT EDIT.",
             "# ros2_control joint_limits format. Positions in rad, velocities in rad/s, efforts in Nm.",
             "joint_limits:"]
    count = 0
    for group, prefixes in GROUPS:
        members = [j for j in movable if j.get("name").startswith(prefixes)]
        lines.append(f"  # ---- {group} ({len(members)})")
        for j in members:
            lim = j.find("limit")
            lines += [f"  {j.get('name')}:",
                      "    has_position_limits: true",
                      f"    min_position: {fmt(float(lim.get('lower')))}",
                      f"    max_position: {fmt(float(lim.get('upper')))}",
                      "    has_velocity_limits: true",
                      f"    max_velocity: {fmt(float(lim.get('velocity')))}",
                      "    has_acceleration_limits: false",
                      "    has_effort_limits: true",
                      f"    max_effort: {fmt(float(lim.get('effort')))}"]
            count += 1
    assert count == len(movable), "a movable joint matched no group"
    out.write_text("\n".join(lines) + "\n")
    return count


def xif(cond: str, *children: ET.Element, negate: bool = False) -> ET.Element:
    el = ET.Element(f"{{{XACRO_NS}}}{'unless' if negate else 'if'}", {"value": cond})
    el.extend(children)
    return el


def main() -> None:
    ws = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package-dir", type=Path, default=ws / "src" / PKG)
    args = ap.parse_args()
    pkg: Path = args.package_dir
    vis_dir, col_dir = pkg / "meshes" / "visual", pkg / "meshes" / "collision"
    col_dir.mkdir(parents=True, exist_ok=True)
    (pkg / "config").mkdir(exist_ok=True)

    root = ET.parse(pkg / "urdf" / "robot_original.urdf").getroot()
    for el in root.findall("mujoco"):
        root.remove(el)
    robot = Robot(root, {l.get("name"): l for l in root.findall("link")}, root.findall("joint"))
    log: list[str] = []

    # 1) mesh paths (visual); unnamed inline materials get unique names
    for link in robot.links.values():
        for m in link.iter("mesh"):
            m.set("filename", f"package://{PKG}/meshes/visual/{Path(m.get('filename')).name}")
        for k, mat in enumerate(link.iter("material")):
            if not mat.get("name"):
                mat.set("name", f"{link.get('name')}_color{k or ''}")

    log += reroot(robot, NEW_ROOT)                                   # 2)
    log += [m for m in (balance_inertia(l) for l in robot.links.values()) if m]  # 6)

    # 4) convex-hull collision meshes
    hull_facets, src_facets, hulls = 0, 0, 0
    collision_for: dict[str, ET.Element | None] = {}
    for name, link in robot.links.items():
        mname = mesh_name(link)
        collision_for[name] = None
        if mname is None:
            continue
        v = read_stl(vis_dir / mname)
        if np.ptp(v, axis=0).max() < TINY_LINK_EXTENT:
            continue
        hull_facets += write_convex_hull(v, col_dir / mname)
        src_facets += len(v) // 3
        hulls += 1
        c = ET.Element("collision")
        c.append(copy.deepcopy(link.find("visual/origin")))
        ET.SubElement(ET.SubElement(c, "geometry"), "mesh",
                      {"filename": f"package://{PKG}/meshes/collision/{mname}"})
        collision_for[name] = c
    log.append(f"collision: {hulls} convex hulls, {src_facets} -> {hull_facets} triangles")

    # FK at the locked upright pose: how far below the pelvis frame is the lowest sole point?
    poses = link_poses(robot, NEW_ROOT)
    sole = 0.0
    for name, link in robot.links.items():
        mname = mesh_name(link)
        if mname is None:
            continue
        T = poses[name] @ origin_tf(link.find("visual"))
        z = (T[:3, :3] @ read_stl(vis_dir / mname).T)[2] + T[2, 3]
        sole = min(sole, float(z.min()))
    log.append(f"lowest sole point is {-sole:.4f} m below the pelvis frame (legs upright)")
    for name in ("base_link", "left_arm_base_link", "right_arm_base_link", "left_hand_link", "right_hand_link"):
        p = poses[name][:3, 3]
        log.append(f"  {name} at q=0: ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) m from pelvis")

    # ---- emit xacro
    ET.register_namespace("xacro", XACRO_NS)
    out_root = ET.Element("robot", {"name": "aibot_body"})
    ET.SubElement(out_root, f"{{{XACRO_NS}}}property",
                  {"name": "sole_depth", "value": f"{-sole:.4f}"})
    macro = ET.SubElement(out_root, f"{{{XACRO_NS}}}macro", {
        "name": "aibot_body",
        "params": "lock_legs:=true simple_collision:=true hand_damping:=0.05 hand_friction:=0.0"})

    for name, link in robot.links.items():
        new = ET.SubElement(macro, "link", {"name": name})
        new.extend(link.findall("inertial") + link.findall("visual"))
        simple = collision_for[name]
        new.append(xif("${simple_collision}", simple if simple is not None
                       else ET.Comment(" tiny dummy link: no collision ")))
        orig = link.findall("collision")
        if orig:
            new.append(xif("${simple_collision}", *orig, negate=True))

    for j in robot.joints:
        if j.get("type") == "revolute" and "_hand_" in j.get("name"):
            dyn = j.find("dynamics")
            dyn.attrib.clear()
            dyn.set("damping", "${hand_damping}")
            dyn.set("friction", "${hand_friction}")
        if is_leg(j):
            macro.append(xif("${lock_legs}", locked_leg_joint(j)))
            macro.append(xif("${lock_legs}", j, negate=True))
        else:
            macro.append(j)

    ET.indent(out_root, space="  ")
    header = ("<?xml version=\"1.0\"?>\n"
              "<!-- AUTO-GENERATED by scripts/generate_description.py - DO NOT EDIT.\n"
              "     Source: urdf/robot_original.urdf.\n"
              "     BSD 3-Clause License, see LICENSE.\n"
              "     Hand joints: source dynamics damping=1 friction=1 replaced by macro params.\n"
              "     Generator log:\n"
              + "".join(f"       {line.replace('--', '-')}\n" for line in log)
              + "-->\n")
    (pkg / "urdf" / "aibot_body.xacro").write_text(header + ET.tostring(out_root, encoding="unicode") + "\n")
    n = write_joint_limits(robot, pkg / "config" / "joint_limits.yaml")
    print("\n".join(log))
    print(f"wrote urdf/aibot_body.xacro, config/joint_limits.yaml ({n} joints), meshes/collision/")


if __name__ == "__main__":
    main()
