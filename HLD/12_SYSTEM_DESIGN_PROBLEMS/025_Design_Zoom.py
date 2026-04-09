"""
ZOOM — Video Conferencing Platform
====================================

FUNCTIONAL REQUIREMENTS:
- Create and join meetings (by meeting ID + passcode)
- Audio/video streaming with adaptive quality
- Screen sharing
- Chat (in-meeting), reactions, hand raise
- Waiting room, host controls (mute all, remove participant)
- Recording (local and cloud)
- Breakout rooms

NON-FUNCTIONAL REQUIREMENTS:
- 300 M daily meeting participants at peak
- Video latency < 150 ms end-to-end (p95)
- Support 1000 participants in a webinar
- 99.9% uptime for scheduled meetings

ARCHITECTURE:
  Client ──WebSocket──▶ Signal Server (meeting coordination)
  Client ──WebRTC/UDP──▶ Media Server (ZMS) ──▶ MCU/SFU ──▶ Other clients

  Signal Server: handles join/leave, participant list, host controls
  ZMS (Zoom Media Server): receives all participant streams, forwards selectively

KEY DESIGN DECISIONS:
1. MEDIA ARCHITECTURE — Selective Forwarding Unit (SFU) model:
   Each participant sends one stream to the media server.
   Server selects which streams to forward to each viewer.
   Scales better than MCU (which mixes all streams on server).
   Zoom uses a hybrid: SFU for small meetings, MCU encoding for recordings.

2. ADAPTIVE VIDEO — SimulCast: each client sends 3 quality layers
   (e.g. 1080p/720p/360p).  Server selects per-subscriber.
   NACK (Negative ACK) for packet loss recovery.
   FEC (Forward Error Correction) for low-latency recovery.

3. NETWORK PATH SELECTION — Zoom's Global Network Routing:
   ZSR (Zoom Server Router) selects lowest-latency media server.
   Clients connect to nearest ZMS; ZMSes are peered in Zoom's backbone.
   Avoids public internet for inter-datacenter audio/video.

4. SIGNAL PROTOCOL — custom protocol over WebSocket:
   - JOIN: authenticate, assign to meeting
   - LEAVE: cleanup
   - PARTICIPANT_UPDATE: mute/unmute/video-on-off
   - HOST_COMMAND: kick, mute all, start recording

5. WAITING ROOM — participants held in "lobby" state.
   Host can admit individually or all at once.
   Meeting participants don't see waiting room users.

6. RECORDING — cloud recording: media server writes frames to S3 in realtime.
   After meeting: transcoding job merges audio/video tracks.
"""

from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Callable
from enum import Enum
from collections import defaultdict
import threading
import math


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ParticipantRole(Enum):
    HOST = "host"
    CO_HOST = "co_host"
    PANELIST = "panelist"   # webinar
    ATTENDEE = "attendee"


class ParticipantStatus(Enum):
    IN_WAITING_ROOM = "waiting_room"
    IN_MEETING = "in_meeting"
    LEFT = "left"
    REMOVED = "removed"


class VideoQuality(Enum):
    QUALITY_360P = ("360p", 360_000)   # bps
    QUALITY_720P = ("720p", 1_500_000)
    QUALITY_1080P = ("1080p", 4_000_000)

    def __init__(self, label: str, bitrate_bps: int):
        self.label = label
        self.bitrate_bps = bitrate_bps


@dataclass
class NetworkStats:
    packet_loss_pct: float = 0.0
    rtt_ms: float = 50.0
    jitter_ms: float = 5.0
    bandwidth_bps: int = 10_000_000


@dataclass
class Participant:
    participant_id: str
    user_id: str
    display_name: str
    role: ParticipantRole
    status: ParticipantStatus = ParticipantStatus.IN_WAITING_ROOM
    is_muted: bool = False
    is_video_on: bool = True
    is_screen_sharing: bool = False
    hand_raised: bool = False
    network: NetworkStats = field(default_factory=NetworkStats)
    joined_at: Optional[float] = None
    left_at: Optional[float] = None


@dataclass
class Meeting:
    meeting_id: str
    topic: str
    host_id: str
    passcode: str
    scheduled_start: float
    duration_minutes: int
    waiting_room_enabled: bool = True
    recording_enabled: bool = False
    max_participants: int = 100
    is_webinar: bool = False
    started_at: Optional[float] = None
    ended_at: Optional[float] = None

    @property
    def is_active(self) -> bool:
        return self.started_at is not None and self.ended_at is None

    @property
    def meeting_url(self) -> str:
        return f"https://zoom.us/j/{self.meeting_id}"


# ---------------------------------------------------------------------------
# Signal Server — meeting coordination
# ---------------------------------------------------------------------------

class SignalServer:
    """Handles participant join/leave, host controls, waiting room."""

    def __init__(self):
        self._meetings: Dict[str, Meeting] = {}
        self._participants: Dict[str, Dict[str, Participant]] = defaultdict(dict)
        self._event_log: List[Dict] = []

    def create_meeting(self, host_id: str, topic: str,
                       passcode: str = "", duration: int = 60,
                       waiting_room: bool = True) -> Meeting:
        meeting = Meeting(
            meeting_id=str(uuid.uuid4())[:9].replace("-", ""),
            topic=topic,
            host_id=host_id,
            passcode=passcode or str(uuid.uuid4())[:6],
            scheduled_start=time.time(),
            duration_minutes=duration,
            waiting_room_enabled=waiting_room,
        )
        self._meetings[meeting.meeting_id] = meeting
        return meeting

    def start_meeting(self, meeting_id: str, host_id: str) -> bool:
        meeting = self._meetings.get(meeting_id)
        if not meeting or meeting.host_id != host_id:
            return False
        meeting.started_at = time.time()
        self._emit(meeting_id, "meeting.started", {"host_id": host_id})
        return True

    def join(self, meeting_id: str, user_id: str, display_name: str,
             passcode: str = "") -> Optional[Participant]:
        meeting = self._meetings.get(meeting_id)
        if not meeting:
            return None
        if meeting.passcode and meeting.passcode != passcode:
            return None

        role = ParticipantRole.HOST if user_id == meeting.host_id else ParticipantRole.ATTENDEE
        status = (ParticipantStatus.IN_WAITING_ROOM if meeting.waiting_room_enabled
                  and role != ParticipantRole.HOST
                  else ParticipantStatus.IN_MEETING)

        participant = Participant(
            participant_id=str(uuid.uuid4())[:8],
            user_id=user_id,
            display_name=display_name,
            role=role,
            status=status,
            joined_at=time.time() if status == ParticipantStatus.IN_MEETING else None,
        )
        self._participants[meeting_id][participant.participant_id] = participant
        self._emit(meeting_id, "participant.joined", {
            "pid": participant.participant_id,
            "name": display_name,
            "status": status.value,
        })
        return participant

    def admit_from_waiting_room(self, meeting_id: str, host_pid: str,
                                 participant_pid: str) -> bool:
        parts = self._participants.get(meeting_id, {})
        host = parts.get(host_pid)
        target = parts.get(participant_pid)
        if not host or host.role not in (ParticipantRole.HOST, ParticipantRole.CO_HOST):
            return False
        if not target or target.status != ParticipantStatus.IN_WAITING_ROOM:
            return False
        target.status = ParticipantStatus.IN_MEETING
        target.joined_at = time.time()
        self._emit(meeting_id, "participant.admitted", {"pid": participant_pid})
        return True

    def admit_all(self, meeting_id: str, host_pid: str) -> int:
        parts = self._participants.get(meeting_id, {})
        host = parts.get(host_pid)
        if not host or host.role not in (ParticipantRole.HOST, ParticipantRole.CO_HOST):
            return 0
        admitted = 0
        for p in parts.values():
            if p.status == ParticipantStatus.IN_WAITING_ROOM:
                p.status = ParticipantStatus.IN_MEETING
                p.joined_at = time.time()
                admitted += 1
        if admitted:
            self._emit(meeting_id, "waiting_room.admit_all", {"count": admitted})
        return admitted

    def mute(self, meeting_id: str, actor_pid: str, target_pid: str) -> bool:
        parts = self._participants.get(meeting_id, {})
        actor = parts.get(actor_pid)
        target = parts.get(target_pid)
        if not actor or not target:
            return False
        # Host can mute anyone; participants can only self-mute
        if actor.role not in (ParticipantRole.HOST, ParticipantRole.CO_HOST):
            if actor_pid != target_pid:
                return False
        target.is_muted = True
        self._emit(meeting_id, "participant.muted", {"pid": target_pid, "by": actor_pid})
        return True

    def mute_all(self, meeting_id: str, host_pid: str) -> int:
        parts = self._participants.get(meeting_id, {})
        host = parts.get(host_pid)
        if not host or host.role not in (ParticipantRole.HOST, ParticipantRole.CO_HOST):
            return 0
        count = 0
        for p in parts.values():
            if p.participant_id != host_pid and not p.is_muted:
                p.is_muted = True
                count += 1
        return count

    def remove_participant(self, meeting_id: str, host_pid: str, target_pid: str) -> bool:
        parts = self._participants.get(meeting_id, {})
        host = parts.get(host_pid)
        target = parts.get(target_pid)
        if not host or host.role not in (ParticipantRole.HOST, ParticipantRole.CO_HOST):
            return False
        if not target:
            return False
        target.status = ParticipantStatus.REMOVED
        target.left_at = time.time()
        self._emit(meeting_id, "participant.removed",
                   {"pid": target_pid, "by": host_pid})
        return True

    def leave(self, meeting_id: str, participant_pid: str) -> bool:
        parts = self._participants.get(meeting_id, {})
        participant = parts.get(participant_pid)
        if not participant:
            return False
        participant.status = ParticipantStatus.LEFT
        participant.left_at = time.time()
        self._emit(meeting_id, "participant.left", {"pid": participant_pid})
        return True

    def end_meeting(self, meeting_id: str, host_pid: str) -> bool:
        meeting = self._meetings.get(meeting_id)
        parts = self._participants.get(meeting_id, {})
        host = parts.get(host_pid)
        if not meeting or not host or host.role != ParticipantRole.HOST:
            return False
        meeting.ended_at = time.time()
        self._emit(meeting_id, "meeting.ended", {"host_pid": host_pid})
        return True

    def in_meeting_participants(self, meeting_id: str) -> List[Participant]:
        return [p for p in self._participants.get(meeting_id, {}).values()
                if p.status == ParticipantStatus.IN_MEETING]

    def waiting_room(self, meeting_id: str) -> List[Participant]:
        return [p for p in self._participants.get(meeting_id, {}).values()
                if p.status == ParticipantStatus.IN_WAITING_ROOM]

    def get_meeting(self, meeting_id: str) -> Optional[Meeting]:
        return self._meetings.get(meeting_id)

    def _emit(self, meeting_id: str, event: str, data: Dict):
        self._event_log.append({"meeting_id": meeting_id, "event": event,
                                 "data": data, "ts": time.time()})


# ---------------------------------------------------------------------------
# Media Server (SFU simulation)
# ---------------------------------------------------------------------------

@dataclass
class MediaStream:
    stream_id: str
    participant_id: str
    stream_type: str      # "video" | "audio" | "screen"
    quality: VideoQuality = VideoQuality.QUALITY_720P
    is_active: bool = True
    bytes_sent: int = 0


class SFUMediaServer:
    """
    Selective Forwarding Unit: receives streams, forwards selectively.
    Real Zoom: ZMS with WebRTC DataChannels.
    """

    def __init__(self):
        self._streams: Dict[str, MediaStream] = {}
        self._subscriptions: Dict[str, Set[str]] = defaultdict(set)  # subscriber → stream_ids
        self._meeting_streams: Dict[str, Set[str]] = defaultdict(set)  # meeting → stream_ids

    def publish(self, meeting_id: str, participant_id: str,
                stream_type: str = "video") -> MediaStream:
        stream = MediaStream(
            stream_id=str(uuid.uuid4())[:8],
            participant_id=participant_id,
            stream_type=stream_type,
        )
        self._streams[stream.stream_id] = stream
        self._meeting_streams[meeting_id].add(stream.stream_id)
        return stream

    def subscribe(self, subscriber_id: str, stream_id: str) -> bool:
        if stream_id not in self._streams:
            return False
        self._subscriptions[subscriber_id].add(stream_id)
        return True

    def unsubscribe(self, subscriber_id: str, stream_id: str) -> None:
        self._subscriptions[subscriber_id].discard(stream_id)

    def adapt_quality(self, stream_id: str, network: NetworkStats) -> VideoQuality:
        """Choose quality based on network conditions."""
        bw = network.bandwidth_bps
        loss = network.packet_loss_pct
        if loss > 5 or bw < 500_000:
            return VideoQuality.QUALITY_360P
        elif loss > 2 or bw < 2_000_000:
            return VideoQuality.QUALITY_720P
        else:
            return VideoQuality.QUALITY_1080P

    def active_streams(self, meeting_id: str) -> List[MediaStream]:
        return [self._streams[sid] for sid in self._meeting_streams.get(meeting_id, set())
                if sid in self._streams and self._streams[sid].is_active]

    def meeting_bandwidth_kbps(self, meeting_id: str) -> float:
        """Total inbound bandwidth for the meeting."""
        total_bps = sum(s.quality.bitrate_bps for s in self.active_streams(meeting_id))
        return total_bps / 1000


# ---------------------------------------------------------------------------
# Chat & Reactions
# ---------------------------------------------------------------------------

@dataclass
class ChatMessage:
    message_id: str
    meeting_id: str
    sender_id: str
    sender_name: str
    text: str
    is_private: bool = False
    recipient_id: Optional[str] = None
    ts: float = field(default_factory=time.time)


@dataclass
class Reaction:
    participant_id: str
    emoji: str     # "thumbs_up" | "clap" | "heart" | "surprised"
    ts: float = field(default_factory=time.time)


class MeetingChat:
    def __init__(self):
        self._messages: Dict[str, List[ChatMessage]] = defaultdict(list)
        self._reactions: Dict[str, List[Reaction]] = defaultdict(list)

    def send(self, meeting_id: str, sender_id: str, sender_name: str,
             text: str, recipient_id: Optional[str] = None) -> ChatMessage:
        msg = ChatMessage(
            message_id=str(uuid.uuid4())[:8],
            meeting_id=meeting_id,
            sender_id=sender_id,
            sender_name=sender_name,
            text=text,
            is_private=recipient_id is not None,
            recipient_id=recipient_id,
        )
        self._messages[meeting_id].append(msg)
        return msg

    def react(self, meeting_id: str, participant_id: str, emoji: str) -> Reaction:
        r = Reaction(participant_id, emoji)
        self._reactions[meeting_id].append(r)
        return r

    def get_messages(self, meeting_id: str, viewer_id: str) -> List[ChatMessage]:
        msgs = []
        for m in self._messages.get(meeting_id, []):
            if not m.is_private or m.sender_id == viewer_id or m.recipient_id == viewer_id:
                msgs.append(m)
        return msgs


# ---------------------------------------------------------------------------
# Breakout Rooms
# ---------------------------------------------------------------------------

@dataclass
class BreakoutRoom:
    room_id: str
    name: str
    meeting_id: str
    assigned_participants: Set[str] = field(default_factory=set)


class BreakoutRoomService:
    def __init__(self):
        self._rooms: Dict[str, List[BreakoutRoom]] = defaultdict(list)

    def create_rooms(self, meeting_id: str, count: int,
                     participant_pids: List[str]) -> List[BreakoutRoom]:
        rooms = []
        for i in range(count):
            room = BreakoutRoom(
                room_id=str(uuid.uuid4())[:6],
                name=f"Breakout Room {i + 1}",
                meeting_id=meeting_id,
            )
            rooms.append(room)
            self._rooms[meeting_id].append(room)

        # Assign participants evenly
        for i, pid in enumerate(participant_pids):
            rooms[i % count].assigned_participants.add(pid)

        return rooms

    def get_rooms(self, meeting_id: str) -> List[BreakoutRoom]:
        return self._rooms.get(meeting_id, [])


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demonstrate_1_meeting_lifecycle():
    print("\n=== 1. Meeting Creation & Lifecycle ===")
    signal = SignalServer()

    # Host creates meeting
    meeting = signal.create_meeting("host_alice", "Weekly Team Sync",
                                     passcode="1234", duration=60)
    print(f"Meeting created: {meeting.meeting_id}")
    print(f"URL: {meeting.meeting_url}")
    print(f"Passcode: {meeting.passcode}")

    # Host joins and starts
    host_p = signal.join(meeting.meeting_id, "host_alice", "Alice (Host)",
                          meeting.passcode)
    signal.start_meeting(meeting.meeting_id, "host_alice")
    print(f"\nHost joined: {host_p.display_name}, role={host_p.role.value}, "
          f"status={host_p.status.value}")

    # Participants join (go to waiting room)
    p_bob = signal.join(meeting.meeting_id, "user_bob", "Bob", meeting.passcode)
    p_carol = signal.join(meeting.meeting_id, "user_carol", "Carol", meeting.passcode)
    p_dave = signal.join(meeting.meeting_id, "user_dave", "Dave", meeting.passcode)

    waiting = signal.waiting_room(meeting.meeting_id)
    print(f"\nWaiting room: {[p.display_name for p in waiting]}")

    # Host admits all
    admitted = signal.admit_all(meeting.meeting_id, host_p.participant_id)
    print(f"Host admitted all: {admitted} participants")

    in_meeting = signal.in_meeting_participants(meeting.meeting_id)
    print(f"In meeting: {[p.display_name for p in in_meeting]}")

    return signal, meeting, host_p, p_bob, p_carol, p_dave


def demonstrate_2_host_controls(signal, meeting, host_p, p_bob, p_carol, p_dave):
    print("\n=== 2. Host Controls ===")

    # Mute all
    muted = signal.mute_all(meeting.meeting_id, host_p.participant_id)
    print(f"Mute all: {muted} participants muted")

    # Bob raises hand (simulated)
    bob = signal._participants[meeting.meeting_id][p_bob.participant_id]
    bob.hand_raised = True
    print(f"Bob raised hand: {bob.hand_raised}")

    # Host unmutes Bob
    bob.is_muted = False
    print(f"Host unmuted Bob: muted={bob.is_muted}")

    # Remove Dave (disruptive participant)
    removed = signal.remove_participant(meeting.meeting_id,
                                         host_p.participant_id,
                                         p_dave.participant_id)
    dave = signal._participants[meeting.meeting_id][p_dave.participant_id]
    print(f"Dave removed: {removed}, status={dave.status.value}")

    # End meeting
    signal.end_meeting(meeting.meeting_id, host_p.participant_id)
    mtg = signal.get_meeting(meeting.meeting_id)
    duration_mins = (mtg.ended_at - mtg.started_at) / 60
    print(f"Meeting ended. Duration: {duration_mins:.2f} min")


def demonstrate_3_media_sfu():
    print("\n=== 3. SFU Media Server & Adaptive Quality ===")
    sfu = SFUMediaServer()
    meeting_id = "mtg_001"

    # Participants publish streams
    participants = [
        ("p001", NetworkStats(packet_loss_pct=0.1, rtt_ms=30, bandwidth_bps=20_000_000)),
        ("p002", NetworkStats(packet_loss_pct=1.5, rtt_ms=80, bandwidth_bps=3_000_000)),
        ("p003", NetworkStats(packet_loss_pct=8.0, rtt_ms=200, bandwidth_bps=400_000)),
    ]

    streams = []
    for pid, network in participants:
        stream = sfu.publish(meeting_id, pid, "video")
        quality = sfu.adapt_quality(stream.stream_id, network)
        stream.quality = quality
        streams.append(stream)
        print(f"  Participant {pid}: loss={network.packet_loss_pct:.1f}%, "
              f"bw={network.bandwidth_bps//1000}Kbps → quality={quality.label}")

    total_bw = sfu.meeting_bandwidth_kbps(meeting_id)
    print(f"\nMeeting inbound bandwidth: {total_bw:.0f} Kbps")

    # p001 subscribes to p002's stream
    sfu.subscribe("p001", streams[1].stream_id)
    subs = len(sfu._subscriptions["p001"])
    print(f"p001 subscribing to {subs} stream(s)")


def demonstrate_4_chat_and_reactions():
    print("\n=== 4. In-meeting Chat & Reactions ===")
    chat = MeetingChat()
    meeting_id = "mtg_chat_001"

    # Public messages
    m1 = chat.send(meeting_id, "alice", "Alice", "Can everyone see my screen?")
    m2 = chat.send(meeting_id, "bob", "Bob", "Yes, looks good!")
    m3 = chat.send(meeting_id, "carol", "Carol", "Same here 👍")

    # Private message
    m4 = chat.send(meeting_id, "alice", "Alice", "Bob, are you presenting next?",
                    recipient_id="bob")

    # Reactions
    chat.react(meeting_id, "bob", "thumbs_up")
    chat.react(meeting_id, "carol", "clap")

    # Alice sees all messages (including her private to Bob)
    alice_msgs = chat.get_messages(meeting_id, "alice")
    print(f"Alice sees {len(alice_msgs)} messages:")
    for m in alice_msgs:
        priv = " [PRIVATE]" if m.is_private else ""
        print(f"  [{m.sender_name}]{priv}: {m.text}")

    # Dave (non-participant) sees only public messages
    dave_msgs = chat.get_messages(meeting_id, "dave")
    print(f"\nDave sees {len(dave_msgs)} messages (no private ones)")

    reactions = chat._reactions.get(meeting_id, [])
    print(f"Reactions: {[(r.participant_id, r.emoji) for r in reactions]}")


def demonstrate_5_breakout_rooms():
    print("\n=== 5. Breakout Rooms ===")
    breakouts = BreakoutRoomService()
    meeting_id = "mtg_breakout"

    participants = ["p_alice", "p_bob", "p_carol", "p_dave", "p_eve", "p_frank"]
    rooms = breakouts.create_rooms(meeting_id, count=3, participant_pids=participants)

    print(f"Created {len(rooms)} breakout rooms for {len(participants)} participants:")
    for room in rooms:
        print(f"  {room.name}: {list(room.assigned_participants)}")


if __name__ == "__main__":
    signal, meeting, host_p, p_bob, p_carol, p_dave = demonstrate_1_meeting_lifecycle()
    demonstrate_2_host_controls(signal, meeting, host_p, p_bob, p_carol, p_dave)
    demonstrate_3_media_sfu()
    demonstrate_4_chat_and_reactions()
    demonstrate_5_breakout_rooms()
