"""
WEBRTC AND PEER-TO-PEER DESIGN
================================

Problem Statement:
Real-time video/audio (Zoom, Google Meet) and file transfer between browsers
needs sub-100ms latency, which is impossible via server relay. WebRTC enables
direct peer-to-peer connections through NAT traversal, but requires signaling
servers and STUN/TURN infrastructure.

WebRTC Architecture:
  1. Signaling Server (your server): exchange SDP offers/answers, ICE candidates
  2. STUN Server: helps peers discover their public IP/port
  3. TURN Server: relay when direct P2P fails (symmetric NAT)
  4. P2P Channel: direct UDP connection (DTLS encrypted)

  Peer A                  Signaling             Peer B
    │── SDP Offer ──────→ │ ──── SDP Offer ───→ │
    │← SDP Answer ──────  │ ←─── SDP Answer ─── │
    │── ICE Cand. ───────→ │ ──── ICE Cand. ───→ │
    │                      │                      │
    │←═══════════ P2P UDP (DTLS/SRTP) ═══════════→│

Mesh vs SFU vs MCU (multi-party video):
  Mesh: every peer connects to every other (2^N connections) — bad at scale
  SFU (Selective Forwarding Unit): peers upload once → SFU forwards — good
  MCU (Multipoint Control Unit): server mixes streams — CPU heavy
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import time
import uuid
import random


class ConnectionState(Enum):
    NEW            = "new"
    CHECKING       = "checking"
    CONNECTED      = "connected"
    COMPLETED      = "completed"
    FAILED         = "failed"
    DISCONNECTED   = "disconnected"


class ICECandidateType(Enum):
    HOST       = "host"       # local interface IP (LAN)
    SRFLX      = "srflx"      # server-reflexive (STUN — public IP behind NAT)
    RELAY      = "relay"      # TURN relay (symmetric NAT fallback)
    PRFLX      = "prflx"      # peer-reflexive (discovered during connectivity)


class NATType(Enum):
    FULL_CONE          = "full_cone"          # easiest to traverse
    RESTRICTED_CONE    = "restricted_cone"    # medium
    PORT_RESTRICTED    = "port_restricted"    # harder
    SYMMETRIC          = "symmetric"          # requires TURN relay


@dataclass
class SDPOffer:
    """Session Description Protocol — describes media capabilities."""
    peer_id     : str
    sdp_type    : str   # "offer" | "answer"
    audio_codecs: List[str] = field(default_factory=lambda: ["opus"])
    video_codecs: List[str] = field(default_factory=lambda: ["VP8", "H264"])
    ice_ufrag   : str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    ice_pwd     : str = field(default_factory=lambda: uuid.uuid4().hex[:16])


@dataclass
class ICECandidate:
    """Network path candidate for connectivity checking."""
    candidate_id  : str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    candidate_type: ICECandidateType = ICECandidateType.HOST
    ip_address    : str = "127.0.0.1"
    port          : int = 0
    priority      : int = 0
    protocol      : str = "udp"

    @property
    def description(self) -> str:
        return (f"candidate:{self.candidate_id} 1 {self.protocol} {self.priority} "
                f"{self.ip_address} {self.port} typ {self.candidate_type.value}")


# ─────────────────────────────────────────────
# STUN SERVER
# ─────────────────────────────────────────────

class STUNServer:
    """
    Session Traversal Utilities for NAT.
    Tells a peer what its public IP:port looks like from outside its NAT.
    """

    def __init__(self, server_ip: str = "stun.example.com"):
        self.server_ip = server_ip
        self.queries   = 0

    def get_public_endpoint(self, private_ip: str, nat_type: NATType) -> Tuple[str, int]:
        """Returns (public_ip, public_port) as seen from outside NAT."""
        self.queries += 1
        # Simulate NAT address translation
        public_ip   = f"203.0.{hash(private_ip) % 255}.{hash(private_ip + 'x') % 255}"
        public_port = 10000 + (hash(private_ip) % 55000)
        print(f"  STUN: {private_ip} → public={public_ip}:{public_port} (NAT={nat_type.value})")
        return public_ip, public_port


# ─────────────────────────────────────────────
# TURN SERVER
# ─────────────────────────────────────────────

class TURNServer:
    """
    Traversal Using Relays around NAT.
    When P2P fails (symmetric NAT), traffic is relayed through TURN.
    TURN is expensive: all media flows through the server.
    """

    def __init__(self, server_ip: str = "turn.example.com"):
        self.server_ip    = server_ip
        self.active_relays: Dict[str, Tuple[str, str]] = {}   # relay_id → (peer_a, peer_b)
        self.bytes_relayed = 0

    def allocate_relay(self, peer_a: str, peer_b: str) -> str:
        relay_id = f"relay-{uuid.uuid4().hex[:8]}"
        self.active_relays[relay_id] = (peer_a, peer_b)
        print(f"  TURN: allocated relay {relay_id} for {peer_a} ↔ {peer_b}")
        return relay_id

    def relay(self, relay_id: str, data_bytes: int):
        self.bytes_relayed += data_bytes

    @property
    def relay_count(self) -> int:
        return len(self.active_relays)


# ─────────────────────────────────────────────
# SIGNALING SERVER
# ─────────────────────────────────────────────

class SignalingServer:
    """
    Coordinates session setup: exchanges SDP and ICE candidates.
    Once P2P is established, signaling is no longer needed for media.
    """

    def __init__(self):
        self._rooms    : Dict[str, Set[str]] = {}
        self._sdp_store: Dict[str, SDPOffer] = {}   # peer_id → SDP
        self._ice_store: Dict[str, List[ICECandidate]] = {}
        self.messages  = 0

    def join_room(self, room_id: str, peer_id: str):
        self._rooms.setdefault(room_id, set()).add(peer_id)
        print(f"  Signal: {peer_id} joined room {room_id}")

    def send_offer(self, from_peer: str, to_peer: str, offer: SDPOffer):
        self._sdp_store[f"{from_peer}→{to_peer}"] = offer
        self.messages += 1
        print(f"  Signal: SDP {offer.sdp_type} {from_peer} → {to_peer}  "
              f"(audio={offer.audio_codecs}, video={offer.video_codecs})")

    def send_ice_candidate(self, from_peer: str, to_peer: str, candidate: ICECandidate):
        self._ice_store.setdefault(f"{from_peer}→{to_peer}", []).append(candidate)
        self.messages += 1
        print(f"  Signal: ICE candidate {from_peer} → {to_peer}  [{candidate.candidate_type.value}] "
              f"{candidate.ip_address}:{candidate.port}")

    def get_peers_in_room(self, room_id: str, exclude: str) -> List[str]:
        return [p for p in self._rooms.get(room_id, set()) if p != exclude]


# ─────────────────────────────────────────────
# PEER CONNECTION
# ─────────────────────────────────────────────

class PeerConnection:
    """
    Simulates a WebRTC RTCPeerConnection.
    Handles ICE gathering, offer/answer exchange, and state tracking.
    """

    def __init__(self, peer_id: str, private_ip: str, nat_type: NATType,
                 stun: STUNServer, turn: TURNServer):
        self.peer_id    = peer_id
        self.private_ip = private_ip
        self.nat_type   = nat_type
        self.stun       = stun
        self.turn       = turn
        self.state      = ConnectionState.NEW
        self.candidates : List[ICECandidate] = []
        self.relay_id   : Optional[str] = None

    def gather_ice_candidates(self) -> List[ICECandidate]:
        """Gather HOST, SRFLX, and optionally RELAY candidates."""
        # HOST candidate (local interface)
        host = ICECandidate(
            candidate_type=ICECandidateType.HOST,
            ip_address=self.private_ip,
            port=random.randint(40000, 60000),
            priority=2130706431,   # highest
            protocol="udp"
        )
        self.candidates.append(host)

        # SRFLX candidate via STUN
        pub_ip, pub_port = self.stun.get_public_endpoint(self.private_ip, self.nat_type)
        srflx = ICECandidate(
            candidate_type=ICECandidateType.SRFLX,
            ip_address=pub_ip,
            port=pub_port,
            priority=1677721855,
            protocol="udp"
        )
        self.candidates.append(srflx)

        # RELAY candidate via TURN (always gather as fallback)
        relay_ip   = "198.51.100.10"
        relay_port = random.randint(49152, 65535)
        relay_cand = ICECandidate(
            candidate_type=ICECandidateType.RELAY,
            ip_address=relay_ip,
            port=relay_port,
            priority=16777215,   # lowest — last resort
            protocol="udp"
        )
        self.candidates.append(relay_cand)

        return self.candidates

    def connect(self, remote_nat: NATType) -> ConnectionState:
        """Determine if P2P succeeds or TURN relay is needed."""
        # Symmetric NAT on either side → must use TURN
        needs_turn = (
            self.nat_type == NATType.SYMMETRIC or
            remote_nat   == NATType.SYMMETRIC
        )
        if needs_turn:
            self.state    = ConnectionState.CONNECTED
            self.relay_id = self.turn.allocate_relay(self.peer_id, "remote")
            print(f"  ICE: {self.peer_id} → TURN relay required (symmetric NAT)")
        else:
            self.state = ConnectionState.COMPLETED
            print(f"  ICE: {self.peer_id} → P2P direct connection established")
        return self.state


# ─────────────────────────────────────────────
# SFU (Selective Forwarding Unit)
# ─────────────────────────────────────────────

class SFU:
    """
    Each participant uploads one stream to the SFU.
    SFU selectively forwards each participant's stream to others.
    No transcoding — just forwarding. CPU efficient.
    """

    def __init__(self, server_id: str = "sfu-1"):
        self.server_id   = server_id
        self._rooms      : Dict[str, Set[str]] = {}
        self._streams    : Dict[str, int] = {}   # peer_id → upload_kbps
        self.total_upload_kbps   = 0
        self.total_download_kbps = 0

    def join(self, room_id: str, peer_id: str, upload_kbps: int):
        self._rooms.setdefault(room_id, set()).add(peer_id)
        self._streams[peer_id] = upload_kbps
        self.total_upload_kbps += upload_kbps

    def calculate_bandwidth(self, room_id: str) -> Dict:
        peers = list(self._rooms.get(room_id, []))
        n = len(peers)
        if n < 2:
            return {"peers": n, "upload_each": 0, "download_each": 0, "server_total": 0}

        # Each peer uploads once, downloads (n-1) streams
        upload_each   = 1500   # kbps per stream (720p video)
        download_each = (n - 1) * upload_each
        server_total  = n * upload_each  # SFU receives all, forwards all
        return {
            "peers"         : n,
            "upload_each_kbps"  : upload_each,
            "download_each_kbps": download_each,
            "server_bw_kbps"    : server_total,
        }


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_webrtc():
    print("=" * 65)
    print("WEBRTC AND PEER-TO-PEER DESIGN")
    print("=" * 65)

    stun = STUNServer("stun.example.com")
    turn = TURNServer("turn.example.com")
    signaling = SignalingServer()

    # ── P2P Call Setup ────────────────────────
    print("\n[1] WEBRTC CALL SETUP — ALICE & BOB")
    print("─" * 55)
    print("  Step 1: Both peers join signaling server")
    signaling.join_room("room-abc", "alice")
    signaling.join_room("room-abc", "bob")

    print("\n  Step 2: ICE candidate gathering (STUN)")
    alice_conn = PeerConnection("alice", "192.168.1.10", NATType.RESTRICTED_CONE, stun, turn)
    bob_conn   = PeerConnection("bob",   "10.0.2.5",    NATType.FULL_CONE,       stun, turn)
    alice_candidates = alice_conn.gather_ice_candidates()
    bob_candidates   = bob_conn.gather_ice_candidates()

    print("\n  Step 3: SDP offer/answer exchange via signaling")
    offer  = SDPOffer("alice", "offer")
    answer = SDPOffer("bob",   "answer")
    signaling.send_offer("alice", "bob", offer)
    signaling.send_offer("bob", "alice", answer)

    print("\n  Step 4: ICE candidates exchanged via signaling")
    for cand in alice_candidates:
        signaling.send_ice_candidate("alice", "bob", cand)

    print("\n  Step 5: ICE connectivity check + connection")
    state = alice_conn.connect(bob_conn.nat_type)
    print(f"  Connection state: {state.value}")

    # ── Symmetric NAT (needs TURN) ────────────
    print("\n\n[2] SYMMETRIC NAT — TURN RELAY REQUIRED")
    print("─" * 55)
    alice2 = PeerConnection("alice2", "192.168.2.10", NATType.SYMMETRIC, stun, turn)
    charlie = PeerConnection("charlie", "10.0.3.5",  NATType.PORT_RESTRICTED, stun, turn)
    alice2.gather_ice_candidates()
    charlie.gather_ice_candidates()
    state2 = alice2.connect(charlie.nat_type)
    print(f"  Connection state: {state2.value}  relay_id={alice2.relay_id}")

    # ── Multi-party: Mesh vs SFU ──────────────
    print("\n\n[3] MULTI-PARTY VIDEO — MESH vs SFU BANDWIDTH")
    print("─" * 55)
    print(f"  {'N peers':<10} {'Mesh upload':<18} {'Mesh download':<18} "
          f"{'SFU upload':<15} {'SFU download':<15} {'SFU server BW'}")
    print(f"  {'─'*85}")

    for n in [2, 4, 6, 10]:
        mesh_upload   = (n - 1) * 1500
        mesh_download = (n - 1) * 1500
        sfu = SFU()
        for i in range(n):
            sfu.join("room", f"peer-{i}", 1500)
        sfu_bw = sfu.calculate_bandwidth("room")
        print(f"  {n:<10} {mesh_upload:<18} {mesh_download:<18} "
              f"{sfu_bw['upload_each_kbps']:<15} "
              f"{sfu_bw['download_each_kbps']:<15} "
              f"{sfu_bw['server_bw_kbps']} kbps")

    # ── ICE Candidate Priority ────────────────
    print("\n\n[4] ICE CANDIDATE TYPES & PRIORITY")
    print("─" * 55)
    cand_guide = [
        (ICECandidateType.HOST,  "192.168.1.1:50000", "Highest", "Direct LAN — fastest, no traversal"),
        (ICECandidateType.SRFLX, "203.1.2.3:51234",   "Medium",  "Public IP via STUN — works for most NATs"),
        (ICECandidateType.PRFLX, "203.1.2.4:52000",   "Medium",  "Discovered during connectivity check"),
        (ICECandidateType.RELAY, "198.51.100.10:49152","Lowest",  "TURN relay — symmetric NAT fallback"),
    ]
    for ctype, example, priority, notes in cand_guide:
        print(f"  {ctype.value:<16} {example:<22} {priority:<10} {notes}")

    # ── Architecture Summary ──────────────────
    print("\n\n[5] WEBRTC INFRASTRUCTURE COMPONENTS")
    print("─" * 55)
    components = [
        ("Signaling Server", "Your server",           "Exchange SDP/ICE before P2P (WebSocket)"),
        ("STUN Server",      "Public (Google/Twilio)","Discover public IP:port behind NAT"),
        ("TURN Server",      "Your server (costly)",  "Media relay for symmetric NAT (~10-15% calls)"),
        ("SFU",              "mediasoup, Janus, Jitsi","Multi-party video forwarding server"),
        ("DTLS",             "Built into WebRTC",     "Encryption for data channels"),
        ("SRTP",             "Built into WebRTC",     "Encryption for audio/video streams"),
    ]
    print(f"  {'Component':<20} {'Provider':<25} {'Purpose'}")
    print(f"  {'─'*75}")
    for comp, prov, purpose in components:
        print(f"  {comp:<20} {prov:<25} {purpose}")

    print(f"\n  Signaling messages exchanged: {signaling.messages}")
    print(f"  STUN queries made: {stun.queries}")
    print(f"  TURN relays active: {turn.relay_count}")


if __name__ == "__main__":
    demonstrate_webrtc()
