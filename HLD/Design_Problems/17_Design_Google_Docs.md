# Design Google Docs — Real-Time Collaborative Document Editor

---

## 1. Problem Statement & Clarifying Questions

Design a real-time collaborative document editing system where multiple users can simultaneously edit the same document and see each other's changes in real time.

### Clarifying Questions

| Question | Assumption |
|---|---|
| How many concurrent collaborators per document? | Up to 100 concurrent editors per doc |
| What is the total document count? | 1 billion documents |
| How many concurrent collaborators globally? | 1 million simultaneous collaborators |
| Do we need offline editing support? | Yes — queue ops locally, sync on reconnect |
| Do we need comment threads? | Yes — inline comments with replies |
| What types of content? | Text documents (not spreadsheets/presentations) |
| Do we need revision history? | Yes — named versions, 30-day auto-save history |
| What latency target for change propagation? | < 100ms for collaborators in the same region |
| Do we need access control? | Yes — owner, editor, commenter, viewer roles |

---

## 2. Functional & Non-Functional Requirements

### Functional Requirements
1. **Real-Time Editing** — Multiple users edit simultaneously; changes appear for all within milliseconds
2. **Operational Transformation (OT)** — Resolve concurrent conflicting operations correctly
3. **Cursor Broadcasting** — Show other users' cursor positions and selections with name labels
4. **Revision History** — View and restore any past version of the document
5. **Access Control** — Owner, editor, commenter, and viewer permission levels
6. **Comments** — Inline comment threads anchored to text ranges
7. **Offline Editing** — Queue operations while offline; sync on reconnect
8. **Auto-Save** — Continuous persistence; no manual save required
9. **Presence Awareness** — Show who is currently viewing/editing the document

### Non-Functional Requirements
1. **Latency** — < 100ms for collaborative changes in same region
2. **Availability** — 99.99% uptime
3. **Consistency** — Eventual consistency for document state; all users converge to identical final state
4. **Scalability** — 1B docs, 1M concurrent collaborators
5. **Durability** — No data loss; operations logged before applying

---

## 3. Capacity Estimation

### Document Storage
- Total documents: 1 billion
- Average document size: 100 KB (text + metadata)
- Total storage: 1B × 100 KB = **100 TB** for current state
- With revision history (10x overhead): ~1 PB

### Traffic Estimation
- 1M concurrent collaborators
- Each user generates ~1 keystroke/second = 1 operation/second
- Total operations/second: **1M ops/sec**
- Each operation payload: ~100 bytes (JSON)
- Total bandwidth: 1M × 100 bytes = **100 MB/s inbound**
- Fan-out to collaborators (avg 5 per doc): **500 MB/s outbound**

### WebSocket Connections
- 1M concurrent connections
- Each connection server handles 10,000 WebSocket connections
- Number of connection servers needed: 1M / 10,000 = **100 connection servers**

### Operation Log Storage
- 1M ops/sec × 100 bytes = 100 MB/sec
- Per day: 100 MB/sec × 86,400 = ~8.6 TB/day
- Stored in time-series log, pruned after compaction

---

## 4. High-Level Architecture

```
              ┌──────────────────────────────────────┐
              │           Browser / Mobile App        │
              │  ┌────────────────────────────────┐  │
              │  │  Local OT Engine + Op Buffer   │  │
              │  └────────────────────────────────┘  │
              └───────────────┬──────────────────────┘
                              │ WebSocket (persistent)
              ┌───────────────▼──────────────────────┐
              │         WebSocket Gateway             │
              │   (Connection Servers — 100 nodes)    │
              └──────┬─────────────────┬─────────────┘
                     │                 │
         ┌───────────▼───┐     ┌───────▼───────────┐
         │  OT Service   │     │  Presence Service │
         │  (Transform + │     │  (cursor, online  │
         │   order ops)  │     │   status)         │
         └───────┬───────┘     └───────────────────┘
                 │
      ┌──────────▼───────────────────────────────┐
      │              Kafka (Operations Log)       │
      │  Topic: doc_ops_{doc_id} — ordered log   │
      └──────┬──────────────────┬────────────────┘
             │                  │
  ┌──────────▼────┐    ┌────────▼──────────┐
  │  PostgreSQL   │    │    Cassandra       │
  │  (documents,  │    │  (operations log, │
  │   snapshots,  │    │   per-doc ops     │
  │   comments,   │    │   time series)    │
  │   collabs)    │    └───────────────────┘
  └───────────────┘
              │
      ┌───────▼────────┐
      │   Redis        │
      │  (op sequence  │
      │   counter,     │
      │   cursor state)│
      └────────────────┘
```

---

## 5. Component Deep-Dive

### 5.1 Operational Transformation (OT) — Core Algorithm

OT is the heart of Google Docs-style collaboration. It ensures that concurrent operations from multiple users converge to the same final document state.

**Three Operation Types:**
- `Insert(position, text)` — Insert text at position
- `Delete(position, length)` — Delete `length` characters starting at position
- `Retain(length)` — Keep `length` characters unchanged (used in composed ops)

**The Problem Without OT:**
```
Document: "Hello World"
           0123456789...

User A: Insert(5, ",")  → "Hello, World"
User B: Insert(11, "!")  → "Hello World!"  (concurrent, based on original)

Without OT, applying B after A:
  Apply A: "Hello, World"   (positions shifted right by 1 after position 5)
  Apply B naively: Insert at position 11 → "Hello, World!"  ← WRONG position
                   Should insert at position 12 to account for A's insertion

With OT: transform(B, A) → Insert(12, "!") → "Hello, World!"  ← CORRECT
```

**Transformation Rules:**
```
transform_insert_insert(A=Insert(pa, ta), B=Insert(pb, tb)):
  if pa <= pb:
    return Insert(pb + len(ta), tb)   # A shifted B's position right
  else:
    return B                           # B is before A, unaffected

transform_delete_insert(A=Delete(pa, la), B=Insert(pb, tb)):
  if pb <= pa:
    return Insert(pb, tb)              # B before A's delete, unaffected
  elif pb >= pa + la:
    return Insert(pb - la, tb)         # B after A's delete, shift left
  else:
    return Insert(pa, tb)              # B inside A's delete range, move to start of delete
```

### 5.2 Server Authority Model

Google Docs uses a **server-centric OT** model:

```
1. Client A sends op_a with seq_num=5 (last seen server op)
2. Client B sends op_b with seq_num=5 (concurrent!)
3. Server receives op_a first → assigns seq_num=6, broadcasts to all
4. Server receives op_b → transforms op_b against op_a → assigns seq_num=7
5. Both clients apply in seq_num order → identical final state
```

This is the key insight: **the server defines total ordering**. All clients trust the server sequence and transform their pending local ops against any server ops they haven't yet seen.

### 5.3 Client-Side State Machine

Each client maintains:
- **Committed State:** Last known server-confirmed document state
- **In-Flight Op:** Operation sent to server, awaiting acknowledgment
- **Pending Buffer:** Operations typed after in-flight, not yet sent

```
States:     Synchronized → Awaiting → Buffering
Transitions:
  Type → Awaiting (send op to server)
  Receive own op ack → apply pending as next in-flight → Awaiting
  Receive other's op while Awaiting → transform in-flight and pending against it
```

### 5.4 Snapshot + Delta Loading

Loading a large document efficiently:

1. **Snapshot**: Periodic materialized view of document state (every 1000 ops or 1 hour)
2. **Delta**: Apply only operations since the last snapshot

Loading a doc: `fetch_snapshot(doc_id) + fetch_ops_since(snapshot_seq_num)`

This bounds load time regardless of document age.

### 5.5 CRDTs as Alternative to OT

**Conflict-free Replicated Data Types (CRDTs)** offer an alternative:
- Each character assigned a globally unique, stable identifier (fractional index)
- No transformation needed — IDs are immutable
- Example: LSEQ, Logoot, RGA algorithms
- **Pros:** No server authority required, pure peer-to-peer
- **Cons:** Metadata overhead per character, tombstone accumulation for deleted chars
- **Google's choice:** OT with server authority (simpler convergence guarantees)

### 5.6 Cursor Broadcasting

Cursor positions are transmitted as metadata alongside operations:
- Each operation includes the sender's cursor position post-operation
- Server fans out cursor updates via WebSocket to all document collaborators
- Cursor positions are ephemeral — stored in Redis, not persisted to DB
- Cursor shows: user name, color-coded caret, and selection highlight

### 5.7 WebSocket Connection Management

- Each connection server handles ~10,000 WebSocket connections
- Problem: Two users on the same document may be on different servers
- Solution: Redis pub/sub — each document has a channel; all servers subscribe
  - Server A has User A → publishes op to channel `doc:{doc_id}`
  - Server B has User B → subscribes to same channel → pushes op to User B

---

## 6. Database Design

### 6.1 Documents Table
```sql
CREATE TABLE documents (
    doc_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title        VARCHAR(512),
    owner_id     UUID NOT NULL,
    snapshot     TEXT,                        -- Latest materialized content
    snapshot_seq BIGINT DEFAULT 0,           -- Seq num of last snapshot
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    updated_at   TIMESTAMPTZ DEFAULT NOW(),
    is_deleted   BOOLEAN DEFAULT FALSE
);
```

### 6.2 Operations Table (Cassandra)
```cql
CREATE TABLE operations (
    doc_id       UUID,
    seq_num      BIGINT,
    op_id        UUID,
    user_id      UUID,
    op_type      TEXT,         -- 'insert' | 'delete' | 'retain'
    position     INT,
    content      TEXT,         -- For insert
    length       INT,          -- For delete
    created_at   TIMESTAMP,
    PRIMARY KEY (doc_id, seq_num)
) WITH CLUSTERING ORDER BY (seq_num ASC);
```

### 6.3 Snapshots Table
```sql
CREATE TABLE snapshots (
    snapshot_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id       UUID REFERENCES documents(doc_id),
    seq_num      BIGINT NOT NULL,
    content      TEXT NOT NULL,       -- Full document state
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    INDEX idx_snapshots_doc_seq (doc_id, seq_num DESC)
);
```

### 6.4 Collaborators Table
```sql
CREATE TABLE collaborators (
    doc_id       UUID REFERENCES documents(doc_id),
    user_id      UUID,
    role         ENUM('owner','editor','commenter','viewer'),
    invited_by   UUID,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (doc_id, user_id)
);
```

### 6.5 Comments Table
```sql
CREATE TABLE comments (
    comment_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id        UUID REFERENCES documents(doc_id),
    parent_id     UUID REFERENCES comments(comment_id),  -- For replies
    author_id     UUID NOT NULL,
    content       TEXT NOT NULL,
    anchor_start  INT,           -- Character position in document
    anchor_end    INT,
    is_resolved   BOOLEAN DEFAULT FALSE,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
```

---

## 7. API Design

### WebSocket Messages

**Client → Server:**
```json
{
  "type": "operation",
  "doc_id": "uuid",
  "op": { "type": "insert", "position": 42, "content": "Hello" },
  "seq_num": 15,
  "client_id": "uuid"
}
```

**Server → Client (operation broadcast):**
```json
{
  "type": "operation",
  "op": { "type": "insert", "position": 42, "content": "Hello" },
  "seq_num": 16,
  "user_id": "uuid",
  "cursor": { "position": 47, "name": "Alice", "color": "#4285F4" }
}
```

**Server → Client (acknowledgment):**
```json
{
  "type": "ack",
  "seq_num": 16,
  "client_seq": 15
}
```

### REST APIs

**Get Document (initial load):**
```
GET /api/v1/documents/{doc_id}
Response: { doc_id, title, content, seq_num, collaborators }
```

**Get Revision History:**
```
GET /api/v1/documents/{doc_id}/revisions?limit=50&before={seq_num}
Response: { revisions: [{seq_num, user_id, summary, timestamp}] }
```

**Share Document:**
```
POST /api/v1/documents/{doc_id}/collaborators
Body: { email, role }
Response: { collaborator_id, invite_sent }
```

**Add Comment:**
```
POST /api/v1/documents/{doc_id}/comments
Body: { content, anchor_start, anchor_end }
Response: { comment_id, created_at }
```

---

## 8. Scalability & Bottlenecks

### Bottleneck 1: OT Service is Single-Threaded Per Document
- **Problem:** OT requires total ordering — all ops for a document must be serialized
- **Solution:** Assign each document to exactly one OT service node (consistent hashing by doc_id). If node fails, reassign (stateless — operation log in Kafka/Cassandra). Scale: 1M ops/sec ÷ 10K ops/doc = 100 OT nodes needed.

### Bottleneck 2: WebSocket Fan-Out
- **Problem:** 100 collaborators on one doc → 1 operation → 100 WebSocket pushes
- **Solution:** Redis pub/sub per document. Connection servers subscribe to the doc channel. Fan-out happens at the connection layer, not OT layer.

### Bottleneck 3: Cassandra Operations Log Growth
- **Problem:** Every keystroke is an operation — billions/day
- **Solution:** Periodic snapshot creation. After snapshot at seq_num N, ops 1..N can be archived to cold storage (S3). Hot path only needs ops since last snapshot.

### Bottleneck 4: Document Load for Large Docs
- **Problem:** A document with 1M operations takes seconds to replay
- **Solution:** Snapshot every 1000 ops. Load = snapshot + < 1000 ops to replay. Max load time bounded regardless of document age.

### Bottleneck 5: Presence State at Scale
- **Problem:** 1M cursors updating every few seconds = 100K cursor updates/sec
- **Solution:** Redis stores ephemeral cursor state per document. TTL-based expiry (cursor disappears if no update in 5 seconds). Pub/sub for distribution. Cursor updates batched (send every 100ms max).

---

## 9. Trade-offs & Design Decisions

### Decision 1: OT vs. CRDT
- **OT:** Requires server authority for ordering. Well-understood, efficient for linear text. Google Docs uses OT.
- **CRDT (e.g., Yjs):** Decentralized, no server needed for correctness. Better for P2P. Used by Figma's live collaboration.
- **Choice:** OT with server authority — simpler conflict resolution guarantees, lower per-character metadata overhead.

### Decision 2: Server-Centric vs. Peer-to-Peer OT
- **Server-centric:** All ops routed through server for total ordering. Simpler client logic.
- **P2P OT:** Clients can transform against each other directly. Complex diamond-dependency problem.
- **Choice:** Server-centric. Server is source of truth for sequence numbers.

### Decision 3: Cassandra vs. PostgreSQL for Operations Log
- **Cassandra:** Write-optimized, excellent for append-only time-series ops. Partition by doc_id.
- **PostgreSQL:** Flexible queries but write throughput limited.
- **Choice:** Cassandra for operations log (append-only, high write rate). PostgreSQL for document metadata and collaborator records.

### Decision 4: Snapshot Frequency
- **Every op:** Perfect load time, massive storage overhead
- **Every 1000 ops:** Bounded load time (< 1000 ops to replay), acceptable overhead
- **Every 24 hours:** Simple but terrible UX for large documents
- **Choice:** Every 1000 ops OR every 1 hour, whichever comes first.

### Decision 5: Comment Anchoring
- **Problem:** If someone inserts text before a comment's anchor position, the anchor breaks
- **Solution:** Store anchor as character-level OT-aware position. When ops are applied, transform comment anchors the same way as any other position reference.

---

## 10. Key Interview Talking Points

1. **OT Core Insight:** The problem isn't conflict *prevention* — it's conflict *resolution*. OT transforms concurrent operations so that applying them in different orders still produces the same final document. The transform function is the critical piece.

2. **Server as Total Order Oracle:** The server assigns sequence numbers. Every client transforms their unacknowledged local ops against server ops they see with higher sequence numbers. This guarantees convergence.

3. **Client State Machine:** Clients have three states: synchronized, awaiting-ack, and buffering. The "in-flight + pending buffer" pattern ensures ops are never lost and the client remains responsive.

4. **OT vs. CRDT Trade-off:** OT needs a server authority but has low metadata overhead. CRDTs are decentralized but require stable IDs per character (heavier). For Google Docs-scale centralized service, OT is the right call.

5. **Snapshot + Delta for Efficient Loading:** Without snapshots, loading a popular 5-year-old document means replaying millions of operations. Periodic snapshots bound load time to O(1000) ops regardless of document age.

6. **WebSocket + Redis Pub/Sub for Multi-Server Fan-Out:** The trick is that the document's "room" lives in Redis, not in a single server. Any connection server can join a document's pub/sub channel and forward messages to its local clients.

7. **Cursor Broadcast is Separate from Document Ops:** Cursor positions are ephemeral and lossy — it's fine to drop a cursor update. Document operations are durable and must not be dropped. Separate channels/protocols for each.

8. **Idempotency for Offline Replay:** When the client reconnects after offline editing, it replays its buffered ops. The server must handle duplicate ops gracefully (detect by client op ID, ignore duplicates).

9. **Scale Bottleneck: OT is Per-Document Serial:** All operations on a document must be totally ordered. This limits horizontal scaling — you can't parallelize OT for a single document. Scale by distributing documents across nodes, not by parallelizing within a document.

10. **Numbers to Know:** 1M concurrent collaborators × 1 op/sec = 1M ops/sec. At 100 bytes/op = 100 MB/sec inbound. With 5x fan-out = 500 MB/sec outbound. 100 WebSocket servers at 10K connections each.
