# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Camera modality interaction handlers."""

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from threading import Lock
from typing import ClassVar, Literal, cast, override

import torch

from vllm_omni.diffusion.interaction.modality_handlers.base import InteractionHandler
from vllm_omni.diffusion.interaction.types import (
    InteractionChunkMetadata,
    InteractionEventArrival,
    InteractionPayload,
    resolve_event_frame_offset,
)
from vllm_omni.diffusion.worker.utils import StepRequestState

CameraMode = Literal["target", "velocity"]
Vec3 = tuple[float, float, float]
# Unit quaternion in ``(x, y, z, w)`` order. Identity is ``(0, 0, 0, 1)``.
Quat = tuple[float, float, float, float]

_IDENTITY_QUAT: Quat = (0.0, 0.0, 0.0, 1.0)


@dataclass
class CameraPose:
    """Canonical absolute camera pose used by the shared timeline."""

    translation: Vec3 = (0.0, 0.0, 0.0)
    rotation: Quat = _IDENTITY_QUAT

    @classmethod
    def identity(cls) -> CameraPose:
        return cls()

    def clone(self) -> CameraPose:
        return CameraPose(translation=self.translation, rotation=self.rotation)

    def as_matrix(self) -> torch.Tensor:
        """Return a 4x4 rigid transform from translation + unit quaternion."""
        mat = torch.eye(4, dtype=torch.float32)
        mat[:3, :3] = _quat_to_rotmat(self.rotation)
        mat[0, 3] = float(self.translation[0])
        mat[1, 3] = float(self.translation[1])
        mat[2, 3] = float(self.translation[2])
        return mat


@dataclass
class CameraVelocity:
    """Per-frame velocity applied once every ``1/fps`` second.

    ``rotation`` is a per-frame delta quaternion composed onto the current pose.
    """

    translation: Vec3 = (0.0, 0.0, 0.0)
    rotation: Quat = _IDENTITY_QUAT
    event_id: str = ""


@dataclass
class CameraTarget:
    pose: CameraPose
    event_id: str
    transition_chunks: int


@dataclass
class QueuedCameraEvent:
    """Timestamped camera command waiting for the next chunk boundary."""

    event_id: str
    received_at: float
    mode: CameraMode
    pose: CameraPose
    transition_chunks: int


@dataclass
class CameraSession:
    """Request-local camera timeline stored under ``state.extra['camera_session']``."""

    lock: Lock = field(default_factory=Lock)
    current_pose: CameraPose = field(default_factory=CameraPose.identity)
    pending_events: list[QueuedCameraEvent] = field(default_factory=list)
    last_boundary_at: float | None = None
    active_velocity: CameraVelocity | None = None
    active_target: CameraTarget | None = None
    target_source: CameraPose | None = None
    # Fractional chunk progress while a target transition is active.
    elapsed_transition_chunks: float = 0.0


class CameraModalityHandler(InteractionHandler):
    """Shared camera timeline; subclasses only pack model-specific tensors.

    Events are queued with arrival timestamps and resolved altogether to a list
    of frame actions at the next chunk boundary.
    This class should be subclassed to implement the model-specific projection
    (i.e., converting to the model's expected representation format).

    Some concrete subclasses are provided below for common cases.
    """

    modality: ClassVar[str] = "camera"
    default_transition_chunks: ClassVar[int] = 1

    @override
    def enqueue(
        self,
        state: StepRequestState,
        *,
        event_arrival: InteractionEventArrival,
        payload: InteractionPayload,
        transition_chunks: int | None,
    ) -> None:
        if not event_arrival.event_id:
            raise ValueError("event_id must be non-empty")

        mode = payload.get("mode", "target")
        if mode not in ("target", "velocity"):
            raise ValueError("camera mode must be 'target' or 'velocity'")
        camera_mode = cast(CameraMode, mode)
        data = payload.get("data")
        if not isinstance(data, Mapping):
            raise ValueError("camera data must be an object")

        pose = _parse_pose(data)
        if camera_mode == "target":
            duration = self.default_transition_chunks if transition_chunks is None else int(transition_chunks)
            if duration < 0:
                raise ValueError("transition_chunks must be >= 0")
        else:
            duration = 0

        session = _get_camera_session(state)
        with session.lock:
            session.pending_events.append(
                QueuedCameraEvent(
                    event_id=event_arrival.event_id,
                    received_at=event_arrival.received_at,
                    mode=camera_mode,
                    pose=pose,
                    transition_chunks=duration,
                )
            )

    @override
    def apply_at_chunk_boundary(
        self,
        state: StepRequestState,
        *,
        chunk_index: int,
        num_frames: int,
        fps: float,
        boundary_at: float,
    ) -> InteractionChunkMetadata | None:
        del chunk_index
        session = _get_camera_session(state)
        with session.lock:
            samples, started, active, completed = self._step_one_chunk(
                session,
                num_frames=num_frames,
                fps=fps,
                boundary_at=boundary_at,
            )
            tensor = self._project(samples)
            conditioning_raw = state.extra.setdefault("conditioning", {})
            if not isinstance(conditioning_raw, dict):
                conditioning: dict[str, torch.Tensor] = {}
                state.extra["conditioning"] = conditioning
            else:
                conditioning = cast(dict[str, torch.Tensor], conditioning_raw)
            conditioning[self.modality] = tensor

        return InteractionChunkMetadata(
            started_event_ids=started,
            active_event_ids=active,
            completed_event_ids=completed,
        )

    def _step_one_chunk(
        self,
        session: CameraSession,
        *,
        num_frames: int,
        fps: float,
        boundary_at: float,
    ) -> tuple[list[CameraPose], list[str], list[str], list[str]]:
        """
        Sample absolute poses for this chunk under target/velocity semantics.

        Returns:
            tuple:
                - poses (list[CameraPose]): Sampled absolute camera poses for each frame in the chunk.
                - started_event_ids (list[str])
                - active_event_ids (list[str])
                - completed_event_ids (list[str])
        """

        num_frames = max(int(num_frames), 1)
        pending = list(session.pending_events)
        session.pending_events.clear()

        resolved: list[tuple[int, QueuedCameraEvent]] = []  # (frame number in chunk, event)
        for event in pending:
            frame = resolve_event_frame_offset(
                received_at=event.received_at,
                previous_boundary_at=session.last_boundary_at,
                num_frames=num_frames,
                fps=fps,
            )
            resolved.append((frame, event))
        resolved.sort(key=lambda item: item[0])

        by_frame: dict[int, list[QueuedCameraEvent]] = {}  # (frame number in chunk, event)
        for frame, event in resolved:
            by_frame.setdefault(frame, []).append(event)

        started: list[str] = []
        completed: list[str] = []
        poses: list[CameraPose] = []

        for frame_idx in range(num_frames):
            for event in by_frame.get(frame_idx, []):
                started.append(event.event_id)
                self._activate_event(session, event)
            just_completed = self._step_one_frame(session, num_frames)
            poses.append(session.current_pose.clone())
            if just_completed is not None:
                completed.append(just_completed)

        active: list[str] = []
        if session.active_target is not None:
            active.append(session.active_target.event_id)
            duration = session.active_target.transition_chunks
            if duration <= 0 or session.elapsed_transition_chunks >= duration:
                if session.active_target.event_id not in completed:
                    completed.append(session.active_target.event_id)
        elif session.active_velocity is not None:
            active.append(session.active_velocity.event_id)

        session.last_boundary_at = boundary_at
        return poses, started, active, completed

    def _activate_event(self, session: CameraSession, event: QueuedCameraEvent) -> None:
        if event.mode == "velocity":
            session.active_velocity = CameraVelocity(
                translation=event.pose.translation,
                rotation=event.pose.rotation,
                event_id=event.event_id,
            )
            session.active_target = None
            session.target_source = None
            session.elapsed_transition_chunks = 0.0
            return

        # else, mode == "target"
        session.active_velocity = None
        session.active_target = CameraTarget(
            pose=event.pose,
            event_id=event.event_id,
            transition_chunks=event.transition_chunks,
        )
        session.target_source = session.current_pose.clone()
        session.elapsed_transition_chunks = 0.0

    def _step_one_frame(self, session: CameraSession, total_num_frames_this_chunk: int) -> str | None:
        """Advance the active camera command by one output frame."""
        if session.active_target is not None:
            duration = session.active_target.transition_chunks
            source = session.target_source or session.current_pose
            target = session.active_target.pose
            if duration <= 0:
                session.current_pose = target.clone()
                return session.active_target.event_id

            session.elapsed_transition_chunks += 1.0 / float(total_num_frames_this_chunk)
            alpha = min(1.0, session.elapsed_transition_chunks / float(duration))
            session.current_pose = _lerp_pose(source, target, alpha)
            if session.elapsed_transition_chunks >= duration:
                session.current_pose = target.clone()
                return session.active_target.event_id
            return None

        if session.active_velocity is not None:
            session.current_pose = _add_velocity(session.current_pose, session.active_velocity)
        return None

    @abstractmethod
    def _project(self, poses: list[CameraPose]) -> torch.Tensor:
        """Pack absolute poses into the tensor layout expected by a specific model."""


class SE3DeltaCameraHandler(CameraModalityHandler):
    """Project absolute poses to dense frame-to-frame 4x4 deltas: ``[T, 4, 4]``.

    Dense layout preserves per-frame absolute positions from the shared timeline.
    """

    @override
    def _project(self, poses: list[CameraPose]) -> torch.Tensor:
        if not poses:
            return torch.zeros((0, 4, 4), dtype=torch.float32)
        mats = [p.as_matrix() for p in poses]
        deltas: list[torch.Tensor] = [torch.eye(4, dtype=torch.float32)]
        prev = mats[0]
        for mat in mats[1:]:
            # Relative rigid transform: T_delta = inv(T_prev) @ T_curr.
            deltas.append(torch.linalg.inv(prev) @ mat)
            prev = mat
        return torch.stack(deltas, dim=0)


class WASDEventCameraHandler(CameraModalityHandler):
    """Project pose motion into sparse WASDIJKL-style events: ``[E, 8]``.

    Channels: ``[W, A, S, D, I, J, K, L]`` — translation + pan axes.
    Pan axes use small-angle approximations from the relative quaternion.

    Sparse packing collapses non-zero frame-to-frame motion and may discard
    absolute frame indices that dense projectors retain.
    """

    @override
    def _project(self, poses: list[CameraPose]) -> torch.Tensor:
        if len(poses) < 2:
            return torch.zeros((0, 8), dtype=torch.float32)
        events: list[torch.Tensor] = []
        prev = poses[0]
        for pose in poses[1:]:
            dx = pose.translation[0] - prev.translation[0]
            dy = pose.translation[1] - prev.translation[1]
            dz = pose.translation[2] - prev.translation[2]
            del dy  # vertical unused in the 8-channel WASD packing below
            rel = _quat_mul(_quat_conjugate(prev.rotation), pose.rotation)
            # Small-angle: relative quat (x, y, z, w) ≈ (rx/2, ry/2, rz/2, 1).
            rx = 2.0 * float(rel[0])
            ry = 2.0 * float(rel[1])
            event = torch.tensor(
                [
                    max(dz, 0.0),  # W forward
                    max(-dx, 0.0),  # A left
                    max(-dz, 0.0),  # S back
                    max(dx, 0.0),  # D right
                    max(rx, 0.0),  # I
                    max(-ry, 0.0),  # J
                    max(-rx, 0.0),  # K
                    max(ry, 0.0),  # L
                ],
                dtype=torch.float32,
            )
            if float(event.abs().sum()) > 0:
                events.append(event)
            prev = pose
        if not events:
            return torch.zeros((0, 8), dtype=torch.float32)
        return torch.stack(events, dim=0)


def _get_camera_session(state: StepRequestState) -> CameraSession:
    session = state.extra.get("camera_session")
    if isinstance(session, CameraSession):
        return session
    session = CameraSession()
    state.extra["camera_session"] = session
    return session


def _as_xyz(value: object, *, name: str) -> Vec3:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"camera {name} must be a length-3 list/tuple")
    return (float(value[0]), float(value[1]), float(value[2]))


def _as_quat(value: object, *, name: str) -> Quat:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"camera {name} must be a length-4 list/tuple (x, y, z, w)")
    return _quat_normalize((float(value[0]), float(value[1]), float(value[2]), float(value[3])))


def _parse_pose(data: Mapping[str, object]) -> CameraPose:
    translation = _as_xyz(data.get("translation", (0.0, 0.0, 0.0)), name="translation")
    rotation = _as_quat(data.get("rotation", _IDENTITY_QUAT), name="rotation")
    return CameraPose(translation=translation, rotation=rotation)


def _lerp(a: float, b: float, alpha: float) -> float:
    return a + (b - a) * alpha


def _lerp_pose(source: CameraPose, target: CameraPose, alpha: float) -> CameraPose:
    alpha = min(1.0, max(0.0, alpha))
    return CameraPose(
        translation=cast(
            Vec3,
            tuple(_lerp(s, t, alpha) for s, t in zip(source.translation, target.translation)),
        ),
        rotation=_quat_nlerp(source.rotation, target.rotation, alpha),
    )


def _add_velocity(pose: CameraPose, velocity: CameraVelocity) -> CameraPose:
    return CameraPose(
        translation=cast(
            Vec3,
            tuple(p + v for p, v in zip(pose.translation, velocity.translation)),
        ),
        rotation=_quat_normalize(_quat_mul(pose.rotation, velocity.rotation)),
    )


def _quat_normalize(q: Quat) -> Quat:
    x, y, z, w = q
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        raise ValueError("camera rotation quaternion must be non-zero")
    return (x / norm, y / norm, z / norm, w / norm)


def _quat_conjugate(q: Quat) -> Quat:
    return (-q[0], -q[1], -q[2], q[3])


def _quat_mul(a: Quat, b: Quat) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _quat_nlerp(a: Quat, b: Quat, alpha: float) -> Quat:
    """Normalized linear interpolation; flips ``b`` when needed for shortest path."""
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    if dot < 0.0:
        b = (-b[0], -b[1], -b[2], -b[3])
    return _quat_normalize(
        (
            _lerp(a[0], b[0], alpha),
            _lerp(a[1], b[1], alpha),
            _lerp(a[2], b[2], alpha),
            _lerp(a[3], b[3], alpha),
        )
    )


def _quat_to_rotmat(q: Quat) -> torch.Tensor:
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.tensor(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=torch.float32,
    )
