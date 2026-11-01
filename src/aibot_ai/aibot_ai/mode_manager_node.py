"""mode_manager_node: the single owner of the operating mode (mimic | ai), with safe hand-over.

Service   /aibot/set_mode     aibot_msgs/SetMode   {mode: mimic|ai} -> {success, message}
          /aibot/stop         std_srvs/Trigger     cancel everything, hold the pose (mode -> hold)
Publishes /aibot/mode         std_msgs/String, transient-local: boot | mimic | ai | switching | hold
          /aibot/mode_status  std_msgs/String      human-readable outcome of the last request

The transition logic lives in mode_machine.py (pure, unit-tested); this node only maps each step
to a service call or a wait. While a transition runs the published mode is "switching", and the
commanding nodes (retarget_node, skill_server_node) refuse to publish unless the mode is their
own, so a hand-over never has two active sources.

"settle" waits until no joint has moved more than settle.position_tolerance over the last
settle.hold_time. Then the robot's actual pose equals the last command, so the next owner, which
always starts from the actual pose, continues without a jump. Positions, not velocities: fingers
closed in contact chatter at up to ~1.5 rad/s while moving only ~5 mrad, which a velocity
threshold mistakes for motion.
"""
from __future__ import annotations

import contextlib
import threading
import time

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

from aibot_msgs.srv import SetMode
from aibot_teleop.utils.mode_gate import MODE_TOPIC, mode_qos

from .mode_machine import Mode, ModeMachine, Step, Transition

SETBOOL_STEPS = {   # step name -> (service, value)
    "pause_retarget": ("/teleop/enable", False), "resume_retarget": ("/teleop/enable", True),
    "pause_webcam": ("/teleop/webcam_enable", False), "resume_webcam": ("/teleop/webcam_enable", True),
    "enable_ai": ("/aibot/skills/enable", True), "disable_ai": ("/aibot/skills/enable", False),
}
TRIGGER_STEPS = {
    "cancel_plan": "/aibot/planner/cancel", "cancel_skills": "/aibot/skills/stop",
    "release_objects": "/aibot/grasp/release_all",
}


class ModeManagerNode(Node):
    def __init__(self) -> None:
        super().__init__("mode_manager_node")
        p = self.declare_parameter
        self.initial_mode: str = p("initial_mode", "mimic").value
        self.startup_delay: float = p("startup_delay", 4.0).value
        self.startup_wait: float = p("startup_wait", 120.0).value
        self._t_start = time.monotonic()
        self.service_timeout: float = p("service_timeout", 3.0).value
        self.settle_tol: float = p("settle.position_tolerance", 0.02).value
        self.settle_hold: float = p("settle.hold_time", 0.3).value
        self.settle_timeout: float = p("settle.timeout", 4.0).value

        self.machine = ModeMachine()
        self.cb = ReentrantCallbackGroup()
        self._lock = threading.Lock()          # one transition at a time (STOP may pre-empt)
        self._positions: list[tuple[float, np.ndarray]] = []    # (monotonic time, joint positions)
        self.mode_pub = self.create_publisher(String, MODE_TOPIC, mode_qos())
        self.status_pub = self.create_publisher(String, "/aibot/mode_status", 5)
        self.create_subscription(JointState, "/joint_states", self._on_joint_states, 10, callback_group=self.cb)
        self.setbool = {srv: self.create_client(SetBool, srv, callback_group=self.cb)
                        for srv, _ in SETBOOL_STEPS.values()}
        self.trigger = {srv: self.create_client(Trigger, srv, callback_group=self.cb) for srv in TRIGGER_STEPS.values()}
        self.create_service(SetMode, "/aibot/set_mode", self._srv_set_mode, callback_group=self.cb)
        self.create_service(Trigger, "/aibot/stop", self._srv_stop, callback_group=self.cb)
        self._publish_mode()
        self._startup_timer = self.create_timer(self.startup_delay, self._startup, callback_group=self.cb)
        self.get_logger().info(f"mode manager up; switching to '{self.initial_mode}' in {self.startup_delay:.0f} s")

    # ------------------------------------------------------------------ plumbing
    def _publish_mode(self) -> None:
        self.mode_pub.publish(String(data=self.machine.published_mode.value))

    def _status(self, text: str) -> None:
        self.status_pub.publish(String(data=text))
        self.get_logger().info(text)

    def _on_joint_states(self, msg: JointState) -> None:
        now = time.monotonic()
        self._positions.append((now, np.array(msg.position)))
        while self._positions and now - self._positions[0][0] > self.settle_hold + 0.2:
            self._positions.pop(0)

    def _motion_in_window(self) -> float | None:
        """Largest position range of any joint over the last settle.hold_time (None: too little data)."""
        now = time.monotonic()
        window = [q for t, q in list(self._positions) if now - t <= self.settle_hold]
        if len(window) < 3 or now - self._positions[0][0] < self.settle_hold:
            return None
        return float(np.max(np.ptp(np.array(window), axis=0)))

    def _startup(self) -> None:
        """First transition, once the robot exists: on a loaded machine Gazebo and the
        controllers can take longer than startup_delay, and a switch without /joint_states
        cannot settle (it would end in HOLD)."""
        if not self._positions and time.monotonic() - self._t_start < self.startup_wait:
            self.get_logger().info("waiting for /joint_states before the startup transition",
                                   throttle_duration_sec=10.0)
            return
        self._startup_timer.cancel()
        ok, message = self.switch(self.initial_mode)
        self._status(f"startup -> {self.initial_mode}: {message}")

    # ------------------------------------------------------------------ steps
    def _call(self, client, request, required: bool = True) -> tuple[bool, str]:
        # an optional node that is not running (e.g. planner in a mimic-only bringup) is skipped fast
        wait = self.service_timeout if required else min(0.3, self.service_timeout)
        if not client.wait_for_service(timeout_sec=wait):
            return False, f"{client.srv_name} not available"
        done = threading.Event()
        future = client.call_async(request)
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(self.service_timeout):
            return False, f"{client.srv_name} timed out"
        res = future.result()
        return bool(res.success), res.message

    def _settle(self) -> tuple[bool, str]:
        t_end = time.monotonic() + self.settle_timeout
        motion = None
        while time.monotonic() < t_end:
            motion = self._motion_in_window()
            if motion is not None and motion < self.settle_tol:
                return True, f"settled (max joint motion {motion * 1000:.1f} mrad over {self.settle_hold:.1f} s)"
            time.sleep(0.02)
        shown = "no /joint_states" if motion is None else f"still moving {motion:.3f} rad"
        return False, f"robot did not settle within {self.settle_timeout:.0f} s ({shown})"

    def _run_step(self, step: Step) -> tuple[bool, str]:
        if step.name == "settle":
            return self._settle()
        if step.name in SETBOOL_STEPS:
            srv, value = SETBOOL_STEPS[step.name]
            return self._call(self.setbool[srv], SetBool.Request(data=value), step.required)
        return self._call(self.trigger[TRIGGER_STEPS[step.name]], Trigger.Request(), step.required)

    def _run(self, tr: Transition) -> Mode:
        self._publish_mode()                     # "switching" from here on
        aborted = False
        for step in tr.steps:
            if self.machine.transition is not tr:   # pre-empted by STOP
                return self.machine.mode
            ok, detail = self._run_step(step)
            if self.machine.transition is not tr:   # STOP arrived while this step ran
                return self.machine.mode
            self.get_logger().info(f"  {step.name}: {'ok' if ok else 'FAILED'} {detail}")
            if not self.machine.step_result(step, ok, detail):
                aborted = True
                break
        mode = self.machine.finish(aborted)
        self._publish_mode()
        return mode

    # ------------------------------------------------------------------ public
    def switch(self, target: str) -> tuple[bool, str]:
        with self._lock:
            tr, message = self.machine.request(target)
            if tr is None:
                return message.startswith("already"), message
            t0 = time.monotonic()
            mode = self._run(tr)
        ok = mode.value == target
        warn = f" (warnings: {self.machine.last_error})" if self.machine.last_error else ""
        text = (f"now in {mode.value} after {time.monotonic() - t0:.2f} s{warn}" if ok
                else f"switch to {target} FAILED, holding: {self.machine.last_error}")
        self._status(text)
        return ok, text

    def _srv_set_mode(self, req: SetMode.Request, res: SetMode.Response) -> SetMode.Response:
        res.success, res.message = self.switch(req.mode.strip().lower())
        return res

    def _srv_stop(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        tr = self.machine.request_hold()          # pre-empts a running transition
        self._publish_mode()                      # nobody commands from this instant
        mode = self._run(tr)
        res.success, res.message = True, f"STOPPED: holding pose (mode {mode.value}); choose a mode to continue"
        self._status(res.message)
        return res


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ModeManagerNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        with contextlib.suppress(KeyboardInterrupt):
            node.destroy_node()
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
