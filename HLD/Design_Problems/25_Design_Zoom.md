# System Design: Zoom

## 1. Problem Statement & Clarifying Questions

### Problem Statement
Design a video conferencing platform like Zoom that supports real-time audio/video communication between participants, screen sharing, recording, breakout rooms, and scheduling — at the scale of 300M daily meeting participants.

### Clarifying Questions
1. **Scale**: 300M daily participants — how many concurrent meetings? (~10M concurrent peak)
2. **Meeting size**: Max participants per meeting? (1000+ for webinar, 100 for normal meeting)
3. **Protocols**: WebRTC for browser? (yes) Native apps? (yes, may use proprietary protocols)
4. **Recording**: Cloud recording required? (yes, stored in S3)
5. **Screen sharing**: Separate video track? (yes)
6. **Breakout rooms**: Auto-assign or manual? (both)
7. **Waiting room**: Required before host admits? (yes)
8. **Chat**: In-meeting chat messages? (yes, stored for 30 days)
9. **Bandwidth adaptation**: Should quality adapt to network conditions? (yes, simulcast)
10. **Geographic distribution**: Global deployment? (yes, regional media servers)

---

## 2. Functional & Non-Functional Requirements

### Functional Requirements
- Host creates a meeting with a meeting ID and password
- Participants join via meeting ID; enter waiting room if enabled
- Host admits participants from waiting room
- Real-time audio and video streaming between all participants
- Selective audio/video mute per participant
- Screen sharing as a separate video track
- In-meeting text chat with message persistence (30 days)
- Host can split participants into breakout rooms (sub-meetings)
- Cloud recording: audio + video → S3 → transcoded → downloadable
- End-to-end scheduling: create meeting, calendar invite, join link
- Bandwidth adaptation: auto-lower video quality on poor connections
- Virtual backgrounds and noise suppression (client-side processing)

### Non-Functional Requirements
- **Scale**: 300M daily participants, ~10M concurrent meetings peak
- **Latency**: Audio/video delay < 150ms end-to-end
- **Availability**: 99.99% (< 53 min downtime/year)
- **Bandwidth**: Each participant: 1.5–3 Mbps upload (1080p); 1–2 Mbps download per stream
- **Recording storage**: ~100PB total stored recordings
- **Resilience**: Meeting continues if one media server fails (failover < 2s)

---

## 3. Capacity Estimation

### Traffic
- **Concurrent meetings**: 10M peak
- **Participants/meeting average**: 10 → 100M concurrent participants peak
- **Bandwidth per participant**: 2 Mbps upload, 10×1 Mbps download = 12 Mbps total
- **Total bandwidth**: 100M participants × 12 Mbps = 1.2 Exabits/second (EBS)
- **Signaling messages**: 100M participants × 10 signals/session = 1B signals at session start
- **Chat messages**: 100M participants × 5 msgs/hour × 1 hour avg = 500M msgs/day

### Storage
- **Recording**: 100M meetings/day × 1% recorded × 1 hour × 500MB/hour = ~500TB/day
- **Chat logs**: 500M × 200 bytes = 100GB/day
- **Meeting metadata**: 100M meetings/day × 500 bytes = ~50GB/day

### Servers
- **SFU servers**: 1 SFU handles ~100 meeting rooms of 10 participants each
  - 10M meetings / 100 = 100K SFU server instances at peak
- **Signaling servers**: 1 WebSocket server handles ~10K connections; 100M / 10K = 10K signaling servers

---

## 4. High-Level Architecture

```
                        ┌──────────────────────────────────────────────────────┐
                        │               Client Applications                     │
                        │     Web (WebRTC) / Desktop / iOS / Android           │
                        └──────────────┬─────────────────┬────────────────────┘
                                       │ Signaling        │ Media (SRTP/DTLS)
                        ┌──────────────▼──────┐          │
                        │   Signaling Server  │          │
                        │   (WebSocket)       │          │
                        │   SDP/ICE Exchange  │          │
                        └──────────────┬──────┘          │
                                       │                  │
                        ┌──────────────▼──────────────────▼──────────────────┐
                        │              Media Layer                             │
                        │                                                      │
                        │  ┌──────────────────────────────────────────────┐   │
                        │  │           SFU (Selective Forwarding Unit)     │   │
                        │  │  - Each participant sends 1 stream to SFU    │   │
                        │  │  - SFU selects which streams to forward       │   │
                        │  │  - Bandwidth adaptation per receiver          │   │
                        │  │  - Simulcast: 3 resolution layers per stream  │   │
                        │  └──────────────────────────────────────────────┘   │
                        │                                                      │
                        │  ┌─────────────────┐   ┌────────────────────────┐  │
                        │  │  TURN Servers   │   │  Recording Service     │  │
                        │  │  (NAT Traversal)│   │  (Mix → S3 → Transcode)│  │
                        │  └─────────────────┘   └────────────────────────┘  │
                        └──────────────────────────────────────────────────────┘
                                       │
                        ┌──────────────▼──────────────────────────────────────┐
                        │              Control Plane                           │
                        │                                                      │
                        │  ┌──────────────┐  ┌─────────────┐  ┌───────────┐  │
                        │  │ Meeting Svc  │  │  User Svc   │  │ Schedule  │  │
                        │  │ (CRUD + WR)  │  │             │  │ Svc       │  │
                        │  └──────┬───────┘  └─────────────┘  └───────────┘  │
                        │         │                                            │
                        │  ┌──────▼────────────────────────────────────────┐  │
                        │  │  PostgreSQL (meetings, participants, schedule) │  │
                        │  │  Redis (active meeting state, waiting room)   │  │
                        │  │  Cassandra (chat messages)                     │  │
                        │  └───────────────────────────────────────────────┘  │
                        └──────────────────────────────────────────────────────┘

  STUN: client discovers its public IP/port (no server relay needed)
  TURN: server relays media when direct P2P fails (symmetric NAT)
  ICE: framework that tries P2P first, falls back to TURN
```

---

## 5. Component Deep-Dive

### 5.1 WebRTC Stack

**Key Components:**
- **STUN (Session Traversal Utilities for NAT)**: Client sends STUN request to discover its public IP:port. No relay — just discovery. Lightweight, cheap.
- **TURN (Traversal Using Relays around NAT)**: Full relay server. Required when symmetric NAT blocks direct P2P. ~10–15% of connections need TURN.
- **ICE (Interactive Connectivity Establishment)**: Framework that gathers all possible connection candidates (host, server-reflexive, relay), tries them in priority order, selects best path.
- **SDP (Session Description Protocol)**: Offer/answer exchange describing codecs, bandwidth, encryption params.
- **DTLS (Datagram TLS)**: Encryption layer for media streams.
- **SRTP (Secure RTP)**: Encrypted audio/video transport.

**Signaling Flow:**
```
1. Caller: create offer SDP → send to Signaling Server
2. Signaling Server: forward offer to callee
3. Callee: create answer SDP → send back
4. Both: gather ICE candidates → send via signaling
5. Both: try all ICE candidate pairs → select best → media flows
6. If all direct paths fail → fallback to TURN relay
```

### 5.2 SFU vs MCU vs P2P

| Architecture | Description | Pros | Cons | Use Case |
|---|---|---|---|---|
| **P2P mesh** | Every participant sends to every other participant | Low latency, no server media processing | Upload = (N-1) × bitrate; breaks at ~4 participants | Small group calls |
| **MCU (Multipoint Control Unit)** | Server receives all streams, mixes into 1, sends 1 stream back | Each client sends/receives 1 stream; low client bandwidth | Server compute-intensive, adds mixing latency (~200ms), no individual stream control | Legacy conferencing |
| **SFU (Selective Forwarding Unit)** | Server receives N streams, forwards selected subset to each participant | Low latency, individual stream control, simulcast support | Client still downloads multiple streams; N-1 download streams | Modern video conferencing |

**SFU Deep-Dive:**
- Each participant uploads ONE stream (or 3 for simulcast) to the SFU
- SFU decides which streams each participant receives based on:
  - Active speaker detection (audio energy level)
  - Participant's available bandwidth
  - UI layout (grid, speaker view)
- Simulcast: participant uploads 3 layers (360p, 720p, 1080p). SFU delivers appropriate layer per receiver.

### 5.3 Bandwidth Adaptation (Simulcast)

```
Participant uploads 3 streams:
  Layer 0: 180p @ 150 kbps  (low)
  Layer 1: 360p @ 500 kbps  (medium)
  Layer 2: 720p @ 1500 kbps (high)

SFU determines per-receiver which layer to forward:
  - Receiver available downlink < 300 kbps → forward Layer 0
  - Receiver available downlink < 1 Mbps  → forward Layer 1
  - Receiver available downlink >= 1 Mbps → forward Layer 2

REMB (Receiver Estimated Maximum Bitrate): receiver sends RTCP feedback to SFU
SFU adjusts forwarded layer in real-time
```

### 5.4 Signaling Server

The signaling server facilitates WebRTC negotiation. It does NOT handle media.

**Protocol**: WebSocket (persistent connection per participant)

**Messages:**
```json
{ "type": "join",   "meeting_id": "...", "participant_id": "..." }
{ "type": "offer",  "sdp": "...",        "target_id": "..." }
{ "type": "answer", "sdp": "...",        "target_id": "..." }
{ "type": "ice",    "candidate": "...",  "target_id": "..." }
{ "type": "leave",  "participant_id": "..." }
```

**Scaling signaling servers**: Stateful (WebSocket connections must stay on same server). Solution: use Redis pub/sub to route messages between signaling server instances. When A (on server 1) sends ICE candidate to B (on server 2): server 1 publishes to `participant:{B_id}` Redis channel → server 2 subscribes and delivers to B.

### 5.5 Recording Pipeline

1. **Capture**: SFU can record raw RTP streams per participant
2. **Storage**: Raw streams → S3 (temporary)
3. **Transcoding**: FFmpeg workers pull raw streams, mix audio (overlay), composite video (grid layout)
4. **Output**: MP4 file → S3 (final) → CDN for download
5. **Thumbnail**: Generate preview image at meeting end

**Recording latency**: Live recording means ~30 second delay to first processable chunk. Full recording available ~5-10 minutes after meeting ends (transcoding time).

### 5.6 Waiting Room

The waiting room is a holding state before the host admits participants.

**Implementation:**
- Participant connects to signaling server, enters waiting room state
- `waiting_room:{meeting_id}` Redis list stores pending participant IDs
- Host sees list of participants in waiting room (via WebSocket push)
- Host admits individual participants or "Admit All"
- On admit: signaling server sends join-approved message → participant proceeds to ICE negotiation → joins meeting

### 5.7 Breakout Rooms

Breakout rooms are sub-meetings within the parent meeting.

**Implementation:**
- Host triggers "create breakout rooms" with N rooms and assignment (auto/manual)
- System creates N new meeting rooms (logically, using same SFU infrastructure)
- Participants get a "move to room X" signal → WebSocket message
- Client disconnects from main SFU, reconnects to breakout room SFU
- Breakout SFU is lighter (smaller group) — may share physical SFU with multiple breakouts
- Host can broadcast to all breakout rooms simultaneously (one-way message)
- "Return to main session" signal moves all participants back

---

## 6. Database Design

### Meetings Table
```sql
CREATE TABLE meetings (
    id              VARCHAR(11) PRIMARY KEY,  -- "123-456-7890" format
    host_id         BIGINT REFERENCES users(id),
    title           VARCHAR(200),
    password        VARCHAR(10),
    meeting_type    VARCHAR(10),    -- instant, scheduled, recurring
    scheduled_at    TIMESTAMPTZ,
    duration_mins   INT,
    status          VARCHAR(10) DEFAULT 'waiting',  -- waiting, active, ended
    recording_enabled BOOLEAN DEFAULT false,
    waiting_room    BOOLEAN DEFAULT true,
    max_participants INT DEFAULT 100,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    ended_at        TIMESTAMPTZ
);
```

### Participants Table
```sql
CREATE TABLE participants (
    id              BIGSERIAL PRIMARY KEY,
    meeting_id      VARCHAR(11) REFERENCES meetings(id),
    user_id         BIGINT,        -- NULL for non-authenticated guests
    display_name    VARCHAR(100),
    status          VARCHAR(10) DEFAULT 'waiting',  -- waiting, active, left
    is_muted        BOOLEAN DEFAULT false,
    is_video_on     BOOLEAN DEFAULT true,
    is_host         BOOLEAN DEFAULT false,
    join_time       TIMESTAMPTZ,
    leave_time      TIMESTAMPTZ,
    sfu_server_id   VARCHAR(50)    -- which SFU server handles this participant
);
CREATE INDEX idx_participants_meeting ON participants(meeting_id, status);
```

### Recordings Table
```sql
CREATE TABLE recordings (
    id              BIGSERIAL PRIMARY KEY,
    meeting_id      VARCHAR(11) REFERENCES meetings(id),
    status          VARCHAR(10) DEFAULT 'processing',   -- processing, ready, failed
    duration_secs   INT,
    file_size_bytes BIGINT,
    s3_key          VARCHAR(500),
    thumbnail_url   VARCHAR(500),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    available_at    TIMESTAMPTZ
);
```

### Chat Messages (Cassandra Schema)
```sql
-- Wide-column for high write throughput
CREATE TABLE chat_messages (
    meeting_id      TEXT,
    sent_at         TIMESTAMP,
    message_id      UUID,
    sender_id       BIGINT,
    sender_name     TEXT,
    body            TEXT,
    recipient_type  TEXT,   -- everyone, host, specific user
    PRIMARY KEY ((meeting_id), sent_at, message_id)
) WITH CLUSTERING ORDER BY (sent_at ASC);
```

---

## 7. API Design

### Meeting Management API
```
POST /api/v1/meetings
Body: { title, scheduled_at, duration_mins, password, settings: {waiting_room, recording} }
Response: { meeting_id, join_url, host_key }

GET /api/v1/meetings/{meeting_id}
Response: { meeting details, participant_count, status }

DELETE /api/v1/meetings/{meeting_id}
(End meeting — signals all participants to disconnect)

GET /api/v1/meetings/{meeting_id}/participants
Response: { participants: [{id, name, is_muted, is_video_on}] }
```

### Meeting Control API (WebSocket events)
```
// Client → Server
{ "action": "join",        "meeting_id": "123", "display_name": "Alice" }
{ "action": "mute",        "target_id": "self"|participant_id }
{ "action": "admit",       "participant_id": "..." }          // host only
{ "action": "kick",        "participant_id": "..." }          // host only
{ "action": "start_record" }                                  // host only
{ "action": "breakout_create", "rooms": 3, "auto_assign": true } // host only

// Server → Client
{ "event": "participant_joined", "participant": {...} }
{ "event": "participant_left",   "participant_id": "..." }
{ "event": "admitted",           "meeting_id": "..." }       // to waiting participant
{ "event": "recording_started" }
{ "event": "move_to_breakout",   "room_id": "...", "sfu_url": "..." }
```

### Recording API
```
GET /api/v1/recordings/{recording_id}
Response: { status, duration, download_url, thumbnail_url }

GET /api/v1/meetings/{meeting_id}/recordings
DELETE /api/v1/recordings/{recording_id}
```

---

## 8. Scalability & Bottlenecks

### Bottleneck 1: SFU Server Scalability
- 10M concurrent meetings → 100K SFU instances
- SFU must handle up to 100 participants × 2 Mbps each = 200 Mbps per SFU
- Solution: Auto-scaling SFU pool (Kubernetes); assign meetings to SFU with available capacity
- Meeting placement: Consistent hashing by meeting_id → SFU assignment

### Bottleneck 2: TURN Server Bandwidth
- ~15% of connections require TURN relay: 15M participants × 2 Mbps = 30 Tbps TURN bandwidth
- Most expensive part of Zoom's infrastructure
- Solution: Global TURN server network with GeoDNS → nearest TURN server
- Optimize: aggressive ICE candidate prioritization to avoid TURN when possible

### Bottleneck 3: Signaling at Scale
- 100M participants × 10 WebSocket messages at session start = 1B messages/minute peak
- Solution: Stateless signaling via Redis pub/sub for cross-server routing
- Signaling servers are CPU-light (just routing JSON); can handle 10K connections each
- 100M / 10K = 10K signaling server instances

### Bottleneck 4: Recording Storage
- 500TB/day of recording data
- Solution: S3 with lifecycle policies (hot → warm → cold → Glacier)
- Auto-delete after 30 days by default (unless host pays for extended storage)
- Transcoding fleet: FFmpeg workers on EC2 Spot Instances (cost-effective for batch workloads)

### Bottleneck 5: SFU Failure Handling
- If SFU crashes mid-meeting, all participants disconnect
- Solution: Meeting state stored in Redis; on SFU failure, meeting redirected to backup SFU
- Participants' WebSocket connection triggers reconnect → rejoin on new SFU
- Target: < 2s failover with client-side reconnect logic

---

## 9. Trade-offs & Design Decisions

### Decision 1: P2P vs SFU vs MCU
- **P2P**: Works for 2-person calls, but N=10 means each uploads 9 streams
- **MCU**: Lowest client bandwidth but server mixing latency + complexity
- **SFU**: Best balance: 1 upload per participant, low latency, simulcast support
- **Choice**: SFU for meetings ≥ 3 participants; P2P for 1:1 calls (no server relay needed)
- **Trade-off**: SFU still requires N-1 downloads per participant (mitigated by simulcast)

### Decision 2: Cloud Recording Architecture
- **Option A**: Record at SFU (raw streams) → transcode offline
  - Pros: No processing delay during meeting, full quality
  - Cons: Huge raw storage, transcoding backlog
- **Option B**: Record at client (screen capture) → upload at end
  - Pros: No server-side media processing
  - Cons: Client-side resource usage, upload unreliable on poor connections
- **Choice**: Option A (server-side recording) with async transcoding
- **Trade-off**: Recording available with delay after meeting ends

### Decision 3: Codec Choice
- **VP8**: Older, wide support, software decode only
- **VP9**: Better compression (30% better than VP8), hardware decode on modern devices
- **H.264/AVC**: Widely hardware-accelerated, standard for video
- **AV1**: Best compression (~50% better than VP9), but heavy encoding — not real-time viable for most devices yet
- **Choice**: H.264 primary (hardware acceleration everywhere), VP9 for browsers, AV1 for recordings (offline encoding OK)

### Decision 4: Opus Codec for Audio
- Opus is the clear winner for VoIP: variable bitrate (6–510 kbps), forward error correction (FEC), packet loss concealment (PLC)
- At 32 kbps Opus, audio quality exceeds G.711 at 64 kbps
- Built-in: echo cancellation algorithms, DTX (discontinuous transmission) for silence

### Decision 5: Waiting Room Implementation
- **Option A**: Block at signaling server (participant connects but SDP not exchanged until admitted)
- **Option B**: Separate lobby server/SFU that only does 1-way preview
- **Choice**: Option A (block at signaling layer — no media flows until admitted)
- **Trade-off**: Waiting room participants hold a WebSocket connection; at scale, need to limit waiting room size

---

## 10. Key Interview Talking Points

### 1. WebRTC Core Flow
Must explain: SDP offer/answer, ICE candidate gathering, STUN vs TURN vs ICE relationship. Key insight: STUN is just address discovery (cheap); TURN is full relay (expensive, ~15% of connections). ICE tries all paths and picks best. Signaling channel (WebSocket) is only for SDP/ICE exchange — not for media.

### 2. SFU Architecture
The critical insight: with N=100 participants and MCU, server mixes all 100 streams into 1 composite — simple for client but huge server compute. With SFU, server is just a router: forward the right stream to the right receiver. Much more scalable. Simulcast makes SFU even better: SFU picks which quality layer to forward per receiver, no re-encoding.

### 3. Simulcast Deep-Dive
Three layers: 360p/150kbps, 720p/500kbps, 1080p/1500kbps. The SFU makes per-receiver forwarding decisions using RTCP REMB (receiver bitrate feedback). This is what makes Zoom quality degrade gracefully on poor connections — the SFU drops to a lower layer automatically.

### 4. Scale Numbers
- 300M daily participants → ~10M concurrent meetings peak
- 100K SFU servers needed at peak
- 10K signaling servers
- TURN bandwidth: ~30 Tbps — the most expensive infrastructure component
- Recording: 500TB/day → storage tiering is essential

### 5. Geographic Distribution
Regional deployment is essential for < 150ms latency. Architecture:
- Client connects to nearest signaling/SFU cluster via GeoDNS
- Cross-region meetings: SFU-cascade topology (two regional SFUs connected via high-bandwidth link, forwarding streams between regions)
- Recording: written to S3 in same region, replicated to home region for access

### 6. End-to-End Encryption (E2EE)
Standard Zoom uses transport encryption (DTLS-SRTP) — server can decrypt at SFU. True E2EE means SFU can't decrypt. This creates problems:
- Active speaker detection requires server to see audio energy → impossible with E2EE
- Recording impossible with E2EE (server can't mix)
- Solution: Zoom offers E2EE mode with these limitations explicitly stated

### 7. Breakout Rooms
Implementation detail: each breakout room is a mini-meeting with its own SFU routing context. When participants move to breakout rooms, the client reconnects to a new SFU room (same or different physical SFU). The original meeting SFU might serve both main and breakout rooms, just with different routing tables. Host "broadcast to all breakouts" = WebSocket message sent to all room signaling channels simultaneously.
