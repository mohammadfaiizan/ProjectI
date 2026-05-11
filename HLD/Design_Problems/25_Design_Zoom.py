"""
Zoom System Design - Python Implementation
Demonstrates: SignalingServer (SDP/ICE exchange), SFURouter (stream selection),
              ParticipantManager, BandwidthAdapter (simulcast), WaitingRoom,
              BreakoutRoomManager, RecordingJob.
No external dependencies - standard library only.
"""

import uuid
import time
import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Enums & Constants
# ---------------------------------------------------------------------------

class ParticipantStatus(Enum):
    WAITING  = "waiting"
    ACTIVE   = "active"
    LEFT     = "left"

class StreamQuality(Enum):
    LOW    = "low"     # 180p @ 150 kbps
    MEDIUM = "medium"  # 360p @ 500 kbps
    HIGH   = "high"    # 720p @ 1500 kbps

class RecordingStatus(Enum):
    NOT_STARTED = "not_started"
    RECORDING   = "recording"
    PROCESSING  = "processing"
    READY       = "ready"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class Stream:
    """Represents one participant's outgoing video/audio stream."""
    participant_id: str
    stream_type: str                    # video, audio, screen
    quality: StreamQuality = StreamQuality.HIGH
    bitrate_kbps: int = 1500
    is_active: bool = True

@dataclass
class SDPMessage:
    """Simplified SDP offer/answer container."""
    msg_type: str                       # offer | answer | ice_candidate
    from_id: str
    to_id: str
    payload: str                        # SDP string or ICE candidate JSON

@dataclass
class Participant:
    id: str
    display_name: str
    meeting_id: str
    is_host: bool = False
    is_muted: bool = False
    is_video_on: bool = True
    status: ParticipantStatus = ParticipantStatus.WAITING
    available_bandwidth_kbps: int = 5000    # estimated downlink
    streams: list = field(default_factory=list)
    join_time: Optional[datetime] = None
    leave_time: Optional[datetime] = None

@dataclass
class Meeting:
    id: str
    host_id: str
    title: str
    password: str = ""
    waiting_room_enabled: bool = True
    recording_enabled: bool = False
    max_participants: int = 100
    status: str = "waiting"             # waiting | active | ended
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# 1. WaitingRoom — hold participants before host admits
# ---------------------------------------------------------------------------

class WaitingRoom:
    """
    Holds participants who have connected but not yet been admitted.
    In production: Redis LIST per meeting_id; host polls via WebSocket.
    """

    def __init__(self):
        # meeting_id -> list of participant IDs in waiting order
        self._rooms: dict[str, list] = defaultdict(list)
        # participant_id -> join timestamp
        self._join_times: dict[str, float] = {}

    def add(self, meeting_id: str, participant_id: str) -> None:
        if participant_id not in self._rooms[meeting_id]:
            self._rooms[meeting_id].append(participant_id)
            self._join_times[participant_id] = time.time()

    def remove(self, meeting_id: str, participant_id: str) -> None:
        lst = self._rooms[meeting_id]
        if participant_id in lst:
            lst.remove(participant_id)
            self._join_times.pop(participant_id, None)

    def get_waiting(self, meeting_id: str) -> list[str]:
        return self._rooms[meeting_id].copy()

    def admit_all(self, meeting_id: str) -> list[str]:
        admitted = self._rooms[meeting_id].copy()
        self._rooms[meeting_id].clear()
        return admitted

    def is_waiting(self, meeting_id: str, participant_id: str) -> bool:
        return participant_id in self._rooms[meeting_id]

    def wait_time_secs(self, participant_id: str) -> float:
        if participant_id not in self._join_times:
            return 0.0
        return round(time.time() - self._join_times[participant_id], 1)


# ---------------------------------------------------------------------------
# 2. BandwidthAdapter — simulcast stream selection
# ---------------------------------------------------------------------------

class BandwidthAdapter:
    """
    Selects appropriate simulcast layer for each receiver based on
    their available downlink bandwidth.

    Simulcast layers:
      LOW:    180p,  150 kbps
      MEDIUM: 360p,  500 kbps
      HIGH:   720p, 1500 kbps
    """

    LAYERS = {
        StreamQuality.LOW:    {"resolution": "180p",  "bitrate_kbps": 150},
        StreamQuality.MEDIUM: {"resolution": "360p",  "bitrate_kbps": 500},
        StreamQuality.HIGH:   {"resolution": "720p",  "bitrate_kbps": 1500},
    }

    def select_quality(self, available_bandwidth_kbps: int, stream_count: int) -> StreamQuality:
        """
        Selects highest quality that fits within available bandwidth.
        Divides available bandwidth across all streams.
        """
        if stream_count == 0:
            return StreamQuality.HIGH

        # Reserve bandwidth: each of N streams shares available bandwidth
        per_stream_budget = available_bandwidth_kbps / stream_count

        # Also reserve 30% for audio (Opus ~32kbps per participant) + overhead
        video_budget = per_stream_budget * 0.7

        if video_budget >= self.LAYERS[StreamQuality.HIGH]["bitrate_kbps"]:
            return StreamQuality.HIGH
        elif video_budget >= self.LAYERS[StreamQuality.MEDIUM]["bitrate_kbps"]:
            return StreamQuality.MEDIUM
        else:
            return StreamQuality.LOW

    def get_layer_info(self, quality: StreamQuality) -> dict:
        return self.LAYERS[quality]

    def estimate_total_download_kbps(
        self, participant: Participant, active_stream_count: int
    ) -> int:
        """Estimate download bandwidth required for this participant."""
        quality = self.select_quality(participant.available_bandwidth_kbps, active_stream_count)
        layer   = self.LAYERS[quality]
        # N-1 video streams + N-1 audio streams (32 kbps each)
        video_total = layer["bitrate_kbps"] * active_stream_count
        audio_total = 32 * active_stream_count
        return video_total + audio_total


# ---------------------------------------------------------------------------
# 3. SFURouter — selective stream forwarding
# ---------------------------------------------------------------------------

class SFURouter:
    """
    Selective Forwarding Unit: decides which streams each participant receives.
    Key decisions:
    - Active speaker (by audio energy) gets promoted to prominent view
    - Each receiver gets streams at appropriate quality level
    - Screen share gets full bandwidth priority
    In production: implemented as a media server (Janus, mediasoup, etc.)
    """

    def __init__(self):
        self._adapter = BandwidthAdapter()
        # meeting_id -> {participant_id -> Stream}
        self._streams: dict[str, dict] = defaultdict(dict)
        # meeting_id -> active_speaker_id
        self._active_speakers: dict[str, str] = {}
        # Simulated audio energy levels: participant_id -> float (0-1)
        self._audio_energy: dict[str, float] = {}

    def publish_stream(self, meeting_id: str, stream: Stream) -> None:
        """Participant starts sending their stream to SFU."""
        self._streams[meeting_id][stream.participant_id] = stream

    def unpublish_stream(self, meeting_id: str, participant_id: str) -> None:
        self._streams[meeting_id].pop(participant_id, None)

    def update_audio_energy(self, participant_id: str, energy: float) -> None:
        """Called by SFU when audio energy level changes."""
        self._audio_energy[participant_id] = max(0.0, min(1.0, energy))

    def get_active_speaker(self, meeting_id: str, participants: list[str]) -> Optional[str]:
        """Returns participant with highest recent audio energy."""
        if not participants:
            return None
        return max(
            (p for p in participants if p in self._audio_energy),
            key=lambda p: self._audio_energy.get(p, 0.0),
            default=None,
        )

    def get_streams_for_participant(
        self,
        meeting_id: str,
        receiver_id: str,
        receiver_bandwidth_kbps: int,
        all_participant_ids: list[str],
    ) -> list[dict]:
        """
        Returns list of stream descriptions the receiver should receive.
        Applies simulcast layer selection per their bandwidth.
        """
        meeting_streams = self._streams.get(meeting_id, {})
        # All streams except the receiver's own
        sender_ids = [pid for pid in meeting_streams if pid != receiver_id]
        stream_count = len(sender_ids)

        result = []
        for sender_id in sender_ids:
            stream = meeting_streams[sender_id]
            if not stream.is_active:
                continue

            # Screen share always gets max available bandwidth slice
            if stream.stream_type == "screen":
                quality = StreamQuality.HIGH
            else:
                quality = self._adapter.select_quality(receiver_bandwidth_kbps, stream_count)

            layer = self._adapter.get_layer_info(quality)
            result.append({
                "sender_id":   sender_id,
                "stream_type": stream.stream_type,
                "quality":     quality.value,
                "resolution":  layer["resolution"],
                "bitrate_kbps": layer["bitrate_kbps"],
            })

        # Sort: active speaker first, then by audio energy
        active_speaker = self.get_active_speaker(meeting_id, sender_ids)
        result.sort(key=lambda s: (s["sender_id"] != active_speaker, s["stream_type"] != "screen"))
        return result


# ---------------------------------------------------------------------------
# 4. SignalingServer — SDP/ICE exchange simulation
# ---------------------------------------------------------------------------

class SignalingServer:
    """
    Manages WebRTC signaling: SDP offer/answer and ICE candidate exchange.
    In production: WebSocket server with Redis pub/sub for cross-server routing.
    """

    def __init__(self):
        # participant_id -> [list of pending messages]
        self._message_queues: dict[str, list] = defaultdict(list)
        # meeting_id -> {participant_id -> connection_state}
        self._connections: dict[str, dict] = defaultdict(dict)

    def send_offer(self, from_id: str, to_id: str, sdp: str) -> None:
        """Caller sends SDP offer to callee."""
        msg = SDPMessage("offer", from_id, to_id, sdp)
        self._message_queues[to_id].append(msg)
        self._connections[from_id][to_id] = "offer_sent"

    def send_answer(self, from_id: str, to_id: str, sdp: str) -> None:
        """Callee responds with SDP answer."""
        msg = SDPMessage("answer", from_id, to_id, sdp)
        self._message_queues[to_id].append(msg)
        self._connections[from_id][to_id] = "answered"
        self._connections[to_id][from_id] = "answered"

    def send_ice_candidate(self, from_id: str, to_id: str, candidate: str) -> None:
        """Send ICE candidate to peer."""
        msg = SDPMessage("ice_candidate", from_id, to_id, candidate)
        self._message_queues[to_id].append(msg)

    def get_pending_messages(self, participant_id: str) -> list[SDPMessage]:
        """Drain the message queue for a participant."""
        msgs = self._message_queues[participant_id].copy()
        self._message_queues[participant_id].clear()
        return msgs

    def simulate_peer_connection(self, participant_a: str, participant_b: str) -> str:
        """
        Simulate the full offer/answer/ICE exchange.
        Returns 'connected' or 'failed'.
        In production: this happens asynchronously via WebSocket.
        """
        # Offer
        self.send_offer(participant_a, participant_b, f"sdp_offer_from_{participant_a}")
        # Answer
        self.send_answer(participant_b, participant_a, f"sdp_answer_from_{participant_b}")
        # ICE candidates (simulate STUN success)
        self.send_ice_candidate(participant_a, participant_b, f"candidate:stun:{participant_a}")
        self.send_ice_candidate(participant_b, participant_a, f"candidate:stun:{participant_b}")
        return "connected"

    def get_connection_state(self, from_id: str, to_id: str) -> str:
        return self._connections.get(from_id, {}).get(to_id, "none")


# ---------------------------------------------------------------------------
# 5. ParticipantManager — track state per meeting
# ---------------------------------------------------------------------------

class ParticipantManager:
    """Tracks all participants in a meeting and their A/V states."""

    def __init__(self):
        # meeting_id -> {participant_id -> Participant}
        self._participants: dict[str, dict] = defaultdict(dict)

    def add_participant(self, meeting_id: str, participant: Participant) -> None:
        self._participants[meeting_id][participant.id] = participant

    def admit_participant(self, meeting_id: str, participant_id: str) -> None:
        p = self._participants[meeting_id].get(participant_id)
        if p:
            p.status = ParticipantStatus.ACTIVE
            p.join_time = datetime.now(timezone.utc)

    def remove_participant(self, meeting_id: str, participant_id: str) -> None:
        p = self._participants[meeting_id].get(participant_id)
        if p:
            p.status    = ParticipantStatus.LEFT
            p.leave_time = datetime.now(timezone.utc)

    def mute(self, meeting_id: str, participant_id: str, muted: bool) -> None:
        p = self._participants[meeting_id].get(participant_id)
        if p:
            p.is_muted = muted

    def set_video(self, meeting_id: str, participant_id: str, on: bool) -> None:
        p = self._participants[meeting_id].get(participant_id)
        if p:
            p.is_video_on = on

    def get_active(self, meeting_id: str) -> list[Participant]:
        return [
            p for p in self._participants[meeting_id].values()
            if p.status == ParticipantStatus.ACTIVE
        ]

    def get_all(self, meeting_id: str) -> list[Participant]:
        return list(self._participants[meeting_id].values())

    def get_participant(self, meeting_id: str, participant_id: str) -> Optional[Participant]:
        return self._participants[meeting_id].get(participant_id)


# ---------------------------------------------------------------------------
# 6. BreakoutRoomManager
# ---------------------------------------------------------------------------

class BreakoutRoomManager:
    """
    Manages breakout rooms (sub-meetings within a meeting).
    Each breakout room is a mini-meeting with its own routing context.
    """

    def __init__(self, sfu: SFURouter, signaling: SignalingServer):
        self._sfu        = sfu
        self._signaling  = signaling
        # main_meeting_id -> {room_id -> [participant_ids]}
        self._rooms: dict[str, dict] = defaultdict(dict)
        # participant_id -> current room_id (None = main room)
        self._participant_room: dict[str, Optional[str]] = {}

    def create_rooms(
        self,
        main_meeting_id: str,
        num_rooms: int,
        participant_ids: list[str],
        auto_assign: bool = True,
    ) -> dict[str, list]:
        """
        Creates N breakout rooms and optionally assigns participants.
        Returns {room_id -> [participant_ids]}.
        """
        room_ids = [f"{main_meeting_id}_room_{i}" for i in range(1, num_rooms + 1)]
        for rid in room_ids:
            self._rooms[main_meeting_id][rid] = []

        if auto_assign:
            for i, pid in enumerate(participant_ids):
                room_id = room_ids[i % num_rooms]
                self._rooms[main_meeting_id][room_id].append(pid)
                self._participant_room[pid] = room_id

        return self._rooms[main_meeting_id].copy()

    def move_participant(
        self, main_meeting_id: str, participant_id: str, target_room_id: str
    ) -> None:
        """Move participant from one room to another."""
        # Remove from current room
        current_room = self._participant_room.get(participant_id)
        if current_room and current_room in self._rooms[main_meeting_id]:
            room_list = self._rooms[main_meeting_id][current_room]
            if participant_id in room_list:
                room_list.remove(participant_id)

        # Add to target room
        self._rooms[main_meeting_id][target_room_id].append(participant_id)
        self._participant_room[participant_id] = target_room_id

    def end_breakout_rooms(
        self, main_meeting_id: str
    ) -> list[str]:
        """Closes all breakout rooms, returns list of participant IDs to move back."""
        affected = []
        for room_id, participants in self._rooms[main_meeting_id].items():
            affected.extend(participants)
        self._rooms[main_meeting_id].clear()
        for pid in affected:
            self._participant_room.pop(pid, None)
        return affected

    def get_room_participants(self, main_meeting_id: str, room_id: str) -> list[str]:
        return self._rooms[main_meeting_id].get(room_id, [])

    def broadcast_to_all_rooms(
        self, main_meeting_id: str, message: str
    ) -> int:
        """Broadcast a message to all breakout rooms. Returns count of rooms reached."""
        room_count = len(self._rooms[main_meeting_id])
        # In production: send WebSocket message to all participants in all rooms
        return room_count


# ---------------------------------------------------------------------------
# 7. RecordingJob — capture + transcode simulation
# ---------------------------------------------------------------------------

class RecordingJob:
    """
    Manages recording lifecycle: start → capture → stop → transcode → ready.
    In production: SFU writes raw RTP to S3; FFmpeg workers transcode async.
    """

    def __init__(self):
        # meeting_id -> recording state
        self._recordings: dict[str, dict] = {}
        self._job_id_counter = 0

    def start_recording(self, meeting_id: str, host_id: str) -> str:
        if meeting_id in self._recordings and \
                self._recordings[meeting_id]["status"] == RecordingStatus.RECORDING:
            raise ValueError("Recording already in progress")

        self._job_id_counter += 1
        job_id = f"rec_{meeting_id}_{self._job_id_counter}"
        self._recordings[meeting_id] = {
            "job_id":      job_id,
            "meeting_id":  meeting_id,
            "host_id":     host_id,
            "status":      RecordingStatus.RECORDING,
            "start_time":  time.time(),
            "end_time":    None,
            "duration_s":  None,
            "s3_key":      None,
            "file_size_mb": None,
        }
        return job_id

    def stop_recording(self, meeting_id: str) -> dict:
        rec = self._recordings.get(meeting_id)
        if not rec or rec["status"] != RecordingStatus.RECORDING:
            raise ValueError("No active recording for this meeting")

        end_time = time.time()
        duration = end_time - rec["start_time"]
        rec.update({
            "status":     RecordingStatus.PROCESSING,
            "end_time":   end_time,
            "duration_s": round(duration, 1),
        })
        # Simulate async transcoding
        self._simulate_transcode(meeting_id)
        return rec

    def _simulate_transcode(self, meeting_id: str) -> None:
        """
        In production: FFmpeg worker picks up raw streams from S3,
        composites audio and video, outputs MP4.
        This simulation marks it ready immediately.
        """
        rec = self._recordings[meeting_id]
        duration = rec["duration_s"] or 60
        # Estimate: ~500 MB/hour of recording
        file_size_mb = round((duration / 3600) * 500, 1)
        s3_key = f"recordings/{meeting_id}/{rec['job_id']}.mp4"
        rec.update({
            "status":      RecordingStatus.READY,
            "s3_key":      s3_key,
            "file_size_mb": file_size_mb,
        })

    def get_status(self, meeting_id: str) -> Optional[dict]:
        return self._recordings.get(meeting_id)


# ---------------------------------------------------------------------------
# 8. ZoomSystem — Facade
# ---------------------------------------------------------------------------

class ZoomSystem:
    def __init__(self):
        self._meetings: dict[str, Meeting]    = {}
        self._signaling   = SignalingServer()
        self._sfu         = SFURouter()
        self._participants = ParticipantManager()
        self._waiting_room = WaitingRoom()
        self._bandwidth    = BandwidthAdapter()
        self._breakout     = BreakoutRoomManager(self._sfu, self._signaling)
        self._recording    = RecordingJob()

    def create_meeting(self, host_id: str, title: str, password: str = "") -> Meeting:
        meeting = Meeting(
            id       = str(uuid.uuid4())[:8],
            host_id  = host_id,
            title    = title,
            password = password,
        )
        self._meetings[meeting.id] = meeting
        return meeting

    def join_meeting(
        self, meeting_id: str, participant_id: str, display_name: str,
        bandwidth_kbps: int = 5000
    ) -> Participant:
        meeting = self._meetings.get(meeting_id)
        if not meeting:
            raise KeyError(f"Meeting {meeting_id} not found")

        is_host = (participant_id == meeting.host_id)
        p = Participant(
            id=participant_id, display_name=display_name,
            meeting_id=meeting_id, is_host=is_host,
            available_bandwidth_kbps=bandwidth_kbps,
        )
        self._participants.add_participant(meeting_id, p)

        if meeting.waiting_room_enabled and not is_host:
            # Host goes directly to active; others wait
            self._waiting_room.add(meeting_id, participant_id)
        else:
            self._participants.admit_participant(meeting_id, participant_id)
            meeting.status = "active"
            # Publish their stream to SFU
            stream = Stream(participant_id, "video")
            self._sfu.publish_stream(meeting_id, stream)
        return p

    def admit_participant(self, meeting_id: str, host_id: str, participant_id: str) -> None:
        meeting = self._meetings.get(meeting_id)
        if not meeting or meeting.host_id != host_id:
            raise PermissionError("Only host can admit participants")
        self._waiting_room.remove(meeting_id, participant_id)
        self._participants.admit_participant(meeting_id, participant_id)
        stream = Stream(participant_id, "video")
        self._sfu.publish_stream(meeting_id, stream)

    def leave_meeting(self, meeting_id: str, participant_id: str) -> None:
        self._participants.remove_participant(meeting_id, participant_id)
        self._sfu.unpublish_stream(meeting_id, participant_id)

    def start_recording(self, meeting_id: str, host_id: str) -> str:
        meeting = self._meetings.get(meeting_id)
        if not meeting or meeting.host_id != host_id:
            raise PermissionError("Only host can start recording")
        return self._recording.start_recording(meeting_id, host_id)

    def end_meeting(self, meeting_id: str, host_id: str) -> None:
        meeting = self._meetings.get(meeting_id)
        if not meeting or meeting.host_id != host_id:
            raise PermissionError("Only host can end meeting")
        # Stop recording if active
        rec = self._recording.get_status(meeting_id)
        if rec and rec["status"] == RecordingStatus.RECORDING:
            self._recording.stop_recording(meeting_id)
        # Remove all active participants
        for p in self._participants.get_active(meeting_id):
            self.leave_meeting(meeting_id, p.id)
        meeting.status = "ended"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    system = ZoomSystem()

    # Create and start a meeting
    print("=== Create Meeting ===")
    meeting = system.create_meeting("host_1", "Team Standup", password="1234")
    print(f"  Meeting ID: {meeting.id} | Title: {meeting.title}")

    # Host joins (bypasses waiting room)
    host_p = system.join_meeting(meeting.id, "host_1", "Alice (Host)", bandwidth_kbps=10000)
    print(f"  Host joined: {host_p.display_name} | status={host_p.status.value}")

    # Guests join (go to waiting room)
    g1 = system.join_meeting(meeting.id, "guest_1", "Bob",   bandwidth_kbps=3000)
    g2 = system.join_meeting(meeting.id, "guest_2", "Carol", bandwidth_kbps=800)
    g3 = system.join_meeting(meeting.id, "guest_3", "Dave",  bandwidth_kbps=300)

    print(f"\n=== Waiting Room ===")
    waiting = system._waiting_room.get_waiting(meeting.id)
    print(f"  Waiting: {waiting}")

    # Admit guests
    for gid in ["guest_1", "guest_2", "guest_3"]:
        system.admit_participant(meeting.id, "host_1", gid)
    print(f"  All admitted. Active: {len(system._participants.get_active(meeting.id))}")

    # Simulcast / Bandwidth adaptation
    print("\n=== Bandwidth Adaptation (Simulcast) ===")
    adapter = system._bandwidth
    for participant_id, name, bw in [("guest_1", "Bob", 3000), ("guest_2", "Carol", 800), ("guest_3", "Dave", 300)]:
        quality = adapter.select_quality(bw, stream_count=3)  # 3 other streams
        layer   = adapter.get_layer_info(quality)
        total_dl = adapter.estimate_total_download_kbps(
            system._participants.get_participant(meeting.id, participant_id), 3
        )
        print(f"  {name} ({bw}kbps): receives {quality.value} ({layer['resolution']}) | "
              f"est. download={total_dl}kbps")

    # SFU stream selection
    print("\n=== SFU Stream Selection for Bob ===")
    bob_streams = system._sfu.get_streams_for_participant(
        meeting.id, "guest_1", 3000,
        ["host_1", "guest_1", "guest_2", "guest_3"]
    )
    for s in bob_streams:
        print(f"  From {s['sender_id']}: {s['quality']} ({s['resolution']})")

    # Signaling simulation
    print("\n=== Signaling (SDP/ICE) ===")
    state = system._signaling.simulate_peer_connection("host_1", "guest_1")
    print(f"  host_1 <-> guest_1: {state}")
    msgs = system._signaling.get_pending_messages("host_1")
    for m in msgs:
        print(f"  Pending for host_1: {m.msg_type} from {m.from_id}")

    # Recording
    print("\n=== Recording ===")
    job_id = system.start_recording(meeting.id, "host_1")
    print(f"  Recording started: {job_id}")
    rec_status = system._recording.get_status(meeting.id)
    print(f"  Status: {rec_status['status'].value}")

    time.sleep(0.01)  # Simulate brief meeting duration
    system._recording.stop_recording(meeting.id)
    rec_final = system._recording.get_status(meeting.id)
    print(f"  After stop: status={rec_final['status'].value} | s3_key={rec_final['s3_key']}")

    # Breakout rooms
    print("\n=== Breakout Rooms ===")
    active_ids = [p.id for p in system._participants.get_active(meeting.id) if not p.is_host]
    rooms = system._breakout.create_rooms(meeting.id, num_rooms=2,
                                           participant_ids=active_ids, auto_assign=True)
    for room_id, members in rooms.items():
        print(f"  {room_id}: {members}")

    broadcast_count = system._breakout.broadcast_to_all_rooms(meeting.id, "Wrapping up in 5 minutes!")
    print(f"  Broadcast sent to {broadcast_count} rooms")

    returned = system._breakout.end_breakout_rooms(meeting.id)
    print(f"  Returned to main: {returned}")

    # Mute / video
    print("\n=== Participant Controls ===")
    system._participants.mute(meeting.id, "guest_2", True)
    system._participants.set_video(meeting.id, "guest_3", False)
    for p in system._participants.get_active(meeting.id):
        print(f"  {p.display_name}: muted={p.is_muted} | video={p.is_video_on}")

    # End meeting
    print("\n=== End Meeting ===")
    system.end_meeting(meeting.id, "host_1")
    print(f"  Meeting status: {system._meetings[meeting.id].status}")
    print(f"  Active participants after end: {len(system._participants.get_active(meeting.id))}")
