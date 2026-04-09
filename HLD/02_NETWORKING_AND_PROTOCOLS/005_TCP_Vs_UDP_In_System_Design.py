"""
TCP VS UDP IN SYSTEM DESIGN
=============================

Problem Statement:
The choice of transport protocol fundamentally affects reliability, latency,
and complexity. TCP guarantees delivery and ordering; UDP sacrifices these
guarantees for lower overhead and latency. Engineers must choose wisely.

TCP (Transmission Control Protocol):
  - 3-way handshake (SYN → SYN-ACK → ACK)
  - Guaranteed delivery (retransmission on loss)
  - Ordered delivery
  - Flow control + congestion control
  - Higher overhead: ~20-byte header, handshake delay

UDP (User Datagram Protocol):
  - No connection setup
  - No guarantee of delivery
  - No ordering guarantee
  - Low overhead: 8-byte header
  - Lower latency; suitable for time-sensitive data

Decision Guide:
  TCP: HTTP, databases, file transfer, payments, authentication
  UDP: DNS, video streaming, VoIP, online gaming, IoT telemetry
  UDP + App-layer reliability (QUIC): HTTP/3, WebRTC data channels
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import random
import time


class TransportProtocol(Enum):
    TCP = "TCP"
    UDP = "UDP"


class ConnectionState(Enum):
    CLOSED      = "CLOSED"
    SYN_SENT    = "SYN_SENT"
    SYN_RECEIVED= "SYN_RECEIVED"
    ESTABLISHED = "ESTABLISHED"
    FIN_WAIT    = "FIN_WAIT"
    TIME_WAIT   = "TIME_WAIT"


@dataclass
class Packet:
    seq_num    : int
    payload    : str
    protocol   : TransportProtocol
    delivered  : bool = False
    dropped    : bool = False
    latency_ms : float = 0.0

    @property
    def size_bytes(self) -> int:
        header = 20 if self.protocol == TransportProtocol.TCP else 8
        return header + len(self.payload.encode())


# ─────────────────────────────────────────────
# TCP CONNECTION
# ─────────────────────────────────────────────

class TCPConnection:
    """
    Simulates TCP connection lifecycle with 3-way handshake,
    ordered delivery, and retransmission.
    """

    def __init__(self, name: str, packet_loss_rate: float = 0.05):
        self.name             = name
        self.packet_loss_rate = packet_loss_rate
        self.state            = ConnectionState.CLOSED
        self.send_seq         = 0
        self.recv_seq         = 0
        self._acked           : Dict[int, Packet] = {}
        self.retransmissions  = 0
        self.packets_sent     = 0
        self.connect_overhead_ms = 0.0

    def connect(self) -> float:
        """3-way handshake: SYN → SYN-ACK → ACK"""
        print(f"  TCP [{self.name}] Handshake:")
        rtt = 30.0  # simulated RTT
        print(f"    → SYN       (seq=0)")
        time.sleep(0.01)
        self.state = ConnectionState.SYN_SENT
        print(f"    ← SYN-ACK   (seq=0, ack=1)")
        time.sleep(0.01)
        print(f"    → ACK       (ack=1)")
        self.state = ConnectionState.ESTABLISHED
        self.connect_overhead_ms = rtt
        print(f"    ✅ Established (overhead: {rtt}ms RTT)")
        return rtt

    def send(self, data: str) -> Optional[Packet]:
        if self.state != ConnectionState.ESTABLISHED:
            raise RuntimeError("Cannot send: not connected")

        self.send_seq    += 1
        self.packets_sent += 1
        pkt = Packet(self.send_seq, data, TransportProtocol.TCP)

        # Simulate packet loss + retransmission
        attempt = 0
        while attempt < 5:
            if random.random() > self.packet_loss_rate:
                pkt.delivered  = True
                pkt.latency_ms = 10.0 + attempt * 30.0   # retransmit adds 30ms each
                self._acked[pkt.seq_num] = pkt
                if attempt > 0:
                    self.retransmissions += 1
                break
            attempt += 1
            print(f"    ⚠  packet seq={pkt.seq_num} dropped, retransmitting (attempt {attempt+1})")

        return pkt if pkt.delivered else None

    def close(self):
        print(f"  TCP [{self.name}] FIN handshake (4-way close)")
        self.state = ConnectionState.TIME_WAIT
        print(f"    → FIN  ← FIN-ACK  → ACK  ← FIN  → ACK  → TIME_WAIT(30s)")
        self.state = ConnectionState.CLOSED

    def report(self):
        delivered = sum(1 for p in self._acked.values() if p.delivered)
        avg_lat   = sum(p.latency_ms for p in self._acked.values()) / max(1, len(self._acked))
        print(f"\n  TCP [{self.name}] Stats:")
        print(f"    Packets sent      : {self.packets_sent}")
        print(f"    Delivered         : {delivered}")
        print(f"    Retransmissions   : {self.retransmissions}")
        print(f"    Avg latency       : {avg_lat:.1f} ms")
        print(f"    Connect overhead  : {self.connect_overhead_ms:.0f} ms")


# ─────────────────────────────────────────────
# UDP SOCKET
# ─────────────────────────────────────────────

class UDPSocket:
    """
    UDP: no connection, no retransmission.
    Fire-and-forget — application handles reliability if needed.
    """

    def __init__(self, name: str, packet_loss_rate: float = 0.05):
        self.name             = name
        self.packet_loss_rate = packet_loss_rate
        self.packets_sent     = 0
        self.packets_lost     = 0
        self.packets_delivered= 0
        self.out_of_order     = 0

    def send(self, data: str, seq: int) -> Optional[Packet]:
        self.packets_sent += 1
        pkt = Packet(seq, data, TransportProtocol.UDP)

        if random.random() < self.packet_loss_rate:
            pkt.dropped = True
            self.packets_lost += 1
            return None   # dropped — no retransmit

        pkt.delivered  = True
        pkt.latency_ms = 5.0 + random.uniform(0, 10)  # jitter
        self.packets_delivered += 1
        return pkt

    def report(self):
        loss_pct = self.packets_lost / self.packets_sent * 100 if self.packets_sent else 0
        print(f"\n  UDP [{self.name}] Stats:")
        print(f"    Packets sent      : {self.packets_sent}")
        print(f"    Delivered         : {self.packets_delivered}")
        print(f"    Lost              : {self.packets_lost}  ({loss_pct:.1f}% loss rate)")
        print(f"    No retransmissions (fire-and-forget)")
        print(f"    No connect overhead (connectionless)")


# ─────────────────────────────────────────────
# APPLICATION-LAYER RELIABILITY (QUIC-like)
# ─────────────────────────────────────────────

class ReliabilityLayer:
    """
    Application-layer reliability on top of UDP:
    - Sequence numbers for ordering
    - Selective ACKs for retransmission
    - Used in QUIC (HTTP/3), WebRTC data channels
    """

    def __init__(self, name: str, packet_loss_rate: float = 0.05):
        self.name    = name
        self.udp     = UDPSocket(name, packet_loss_rate)
        self._recv   : Dict[int, Packet] = {}
        self._next_expected = 1
        self.retransmissions = 0

    def send_reliable(self, data: str, seq: int) -> bool:
        for attempt in range(4):
            pkt = self.udp.send(data, seq)
            if pkt:
                self._recv[seq] = pkt
                return True
            self.retransmissions += 1
            print(f"    QUIC-like: seq={seq} lost, selective retransmit (attempt {attempt+1})")
        return False

    def get_ordered(self) -> List[Packet]:
        ordered = []
        seq = 1
        while seq in self._recv:
            ordered.append(self._recv[seq])
            seq += 1
        return ordered


@dataclass
class UseCase:
    name        : str
    protocol    : TransportProtocol
    reason      : str
    examples    : List[str]


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_tcp_vs_udp():
    print("=" * 65)
    print("TCP VS UDP IN SYSTEM DESIGN")
    print("=" * 65)
    random.seed(42)

    # ── TCP ───────────────────────────────────
    print("\n[1] TCP — RELIABLE, ORDERED DELIVERY")
    print("─" * 50)
    tcp = TCPConnection("web-client→API", packet_loss_rate=0.15)
    overhead = tcp.connect()

    payloads = ["GET /api/users", "POST /api/orders", "GET /api/products",
                "DELETE /api/cart/1", "PATCH /api/profile"]
    for i, payload in enumerate(payloads, 1):
        pkt = tcp.send(payload)
        status = f"✅ delivered ({pkt.latency_ms:.0f}ms)" if pkt else "❌ undeliverable"
        print(f"  seq={i:02d}: {payload:<25} {status}")

    tcp.close()
    tcp.report()

    # ── UDP ───────────────────────────────────
    print("\n\n[2] UDP — FAST, UNRELIABLE")
    print("─" * 50)
    udp = UDPSocket("game-client→server", packet_loss_rate=0.10)

    for i in range(1, 11):
        pkt = udp.send(f"PlayerPos(x={i*10}, y={i*5}, t={i})", seq=i)
        status = f"✅ {pkt.latency_ms:.0f}ms" if pkt else "❌ dropped (no retry)"
        print(f"  frame {i:02d}: {status}")

    udp.report()

    # ── UDP + Reliability Layer (QUIC) ────────
    print("\n\n[3] UDP + APP-LAYER RELIABILITY (QUIC-like)")
    print("─" * 50)
    quic = ReliabilityLayer("quic-client", packet_loss_rate=0.15)

    for i in range(1, 8):
        ok = quic.send_reliable(f"HTTP/3 stream frame {i}", seq=i)
        print(f"  seq={i}: {'✅ delivered' if ok else '❌ failed'}")

    ordered = quic.get_ordered()
    print(f"\n  Received {len(ordered)} frames in order")
    print(f"  Total retransmissions: {quic.retransmissions}")

    # ── Protocol Decision Guide ───────────────
    print("\n\n[4] PROTOCOL DECISION GUIDE")
    print("─" * 50)
    use_cases = [
        UseCase("Web APIs (HTTP)",      TransportProtocol.TCP, "Must not lose requests or corrupt data",
                ["REST APIs", "GraphQL", "WebSocket"]),
        UseCase("Database queries",     TransportProtocol.TCP, "ACID transactions require ordered delivery",
                ["PostgreSQL", "MySQL", "MongoDB"]),
        UseCase("File transfer (FTP/SFTP)",TransportProtocol.TCP,"Files must arrive intact and complete",
                ["S3 upload", "SFTP", "rsync"]),
        UseCase("Online gaming",        TransportProtocol.UDP, "Old position data is useless; latest frame matters",
                ["Counter-Strike", "Fortnite", "WebRTC games"]),
        UseCase("Live video/audio",     TransportProtocol.UDP, "Dropped frame < frozen video (jitter > loss)",
                ["Zoom", "Twitch ingest", "VoIP"]),
        UseCase("DNS queries",          TransportProtocol.UDP, "Short query/response; retry if no reply",
                ["All DNS lookups"]),
        UseCase("IoT telemetry",        TransportProtocol.UDP, "Sensors flood data; old readings are stale",
                ["MQTT over UDP", "CoAP"]),
        UseCase("HTTP/3 (QUIC)",        TransportProtocol.UDP, "UDP + app-layer reliability + 0-RTT connect",
                ["Chrome", "Cloudflare", "Google"]),
    ]
    print(f"  {'Use Case':<28} {'Protocol':<8} {'Reason'}")
    print(f"  {'─'*80}")
    for uc in use_cases:
        print(f"  {uc.name:<28} {uc.protocol.value:<8} {uc.reason}")

    # ── Header Overhead Comparison ────────────
    print("\n\n[5] HEADER OVERHEAD COMPARISON")
    print("─" * 50)
    payload_size = 100  # bytes
    tcp_total    = 20 + payload_size
    udp_total    = 8  + payload_size
    print(f"  Payload  : {payload_size} bytes")
    print(f"  TCP total: {tcp_total} bytes  (20-byte header = {20/tcp_total*100:.0f}% overhead)")
    print(f"  UDP total: {udp_total} bytes  ( 8-byte header = {8/udp_total*100:.0f}% overhead)")
    print(f"\n  At 1M packets/sec:")
    print(f"    TCP header waste: {20 * 1_000_000 / 1e6:.0f} MB/s")
    print(f"    UDP header waste: { 8 * 1_000_000 / 1e6:.0f} MB/s")


if __name__ == "__main__":
    demonstrate_tcp_vs_udp()
