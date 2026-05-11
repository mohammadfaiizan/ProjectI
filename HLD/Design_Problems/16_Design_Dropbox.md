# Design Dropbox — Distributed File Storage & Sync System

---

## 1. Problem Statement & Clarifying Questions

Design a cloud file storage and synchronization service like Dropbox that allows users to store files, sync them across multiple devices, and share them with others.

### Clarifying Questions

| Question | Assumption |
|---|---|
| What is the target user scale? | 500M registered users, 100M DAU |
| What file size limits apply? | Up to 5GB per file, 2GB average storage per user |
| Do we need real-time sync or eventual consistency? | Eventual consistency acceptable, target < 30 seconds |
| How many file versions to retain? | 30 versions per file |
| Do we support collaborative editing? | No — file sync only, not real-time co-editing |
| Do we need sharing/permissions? | Yes — public links, shared folders, per-user permissions |
| What upload/download patterns are expected? | 1 upload/day on average per user |
| Mobile support required? | Yes — iOS, Android, Windows, macOS, Linux clients |

---

## 2. Functional & Non-Functional Requirements

### Functional Requirements
1. **File Upload** — Users can upload files of any type up to 5GB
2. **File Download** — Users can download any stored file
3. **File Sync** — Changes sync automatically across all devices
4. **Delta Sync** — Only changed portions of a file are transferred, not the full file
5. **File Versioning** — Retain up to 30 previous versions per file
6. **Sharing** — Share files via public links or share folders with specific users
7. **Conflict Resolution** — Handle concurrent edits gracefully (last-write-wins + conflict copy)
8. **Offline Support** — Queue changes made offline and sync when connection restores
9. **Deduplication** — Avoid storing the same data twice across users

### Non-Functional Requirements
1. **Durability** — 99.999999999% (11 nines) data durability
2. **Availability** — 99.99% uptime (< 52 minutes downtime/year)
3. **Consistency** — Eventual consistency for sync; strong consistency for metadata
4. **Latency** — Upload/download should utilize full available bandwidth
5. **Scalability** — Handle 500M users, petabytes of data
6. **Security** — Encryption at rest (AES-256) and in transit (TLS 1.3)

---

## 3. Capacity Estimation

### Storage Estimation
- Users: 500M registered, 100M DAU
- Average storage per user: 2 GB
- Total storage: 500M × 2 GB = **1 Exabyte (1,000 PB)**
- With replication (3x): 3 EB total raw storage

### Traffic Estimation
- Uploads per day: 100M DAU × 1 upload = 100M uploads/day
- Average file size: 500 KB
- Upload bandwidth: 100M × 500 KB / 86,400 = **~580 MB/s upload throughput**
- Downloads (3x read-to-write ratio): **~1.7 GB/s download throughput**

### Chunk Estimation
- Chunk size: 4 MB (fixed)
- Average file size: 500 KB → most files are < 1 chunk
- Large file (1 GB) = 256 chunks
- 100M uploads/day with 10% being > 1 chunk → ~110M chunk operations/day

### Metadata Estimation
- Files per user: ~1,000
- Total files: 500M × 1,000 = 500 Billion file records
- Metadata per file: ~1 KB
- Total metadata: ~500 TB

---

## 4. High-Level Architecture

```
                        ┌─────────────────────────────┐
                        │         Client Apps          │
                        │  (Desktop / Mobile / Web)    │
                        └─────────────┬───────────────┘
                                      │ HTTPS / WebSocket
                        ┌─────────────▼───────────────┐
                        │         API Gateway          │
                        │    (Load Balancer + Auth)    │
                        └──┬──────────┬──────────┬────┘
                           │          │          │
              ┌────────────▼──┐  ┌────▼────┐  ┌─▼──────────────┐
              │  Metadata     │  │  Block  │  │   Notification  │
              │  Service      │  │  Server │  │   Service       │
              │  (File tree,  │  │  (Dedup,│  │   (WebSocket /  │
              │   versions,   │  │  chunk  │  │   Long Poll)    │
              │   shares)     │  │  mgmt)  │  └─────────────────┘
              └───────┬───────┘  └────┬────┘
                      │               │
          ┌───────────▼──┐    ┌───────▼─────────┐
          │  PostgreSQL  │    │   Object Store   │
          │  (metadata,  │    │   (AWS S3 /      │
          │   versions,  │    │   GCS)           │
          │   shares)    │    │   Chunk Storage  │
          └──────────────┘    └─────────┬────────┘
                                        │
                              ┌─────────▼────────┐
                              │   CDN (CloudFront│
                              │   / Fastly)      │
                              │   Download Edge  │
                              └──────────────────┘

         ┌────────────────────────────────────────────┐
         │              Supporting Services            │
         │  ┌────────────┐  ┌──────────┐  ┌────────┐ │
         │  │   Redis    │  │  Kafka   │  │ Elastic│ │
         │  │  (sessions,│  │ (sync    │  │ Search │ │
         │  │   cache)   │  │  events) │  │(search)│ │
         │  └────────────┘  └──────────┘  └────────┘ │
         └────────────────────────────────────────────┘
```

---

## 5. Component Deep-Dive

### 5.1 Client Sync Agent
The desktop/mobile client is the most complex component:

**Responsibilities:**
- Watch local filesystem for changes (inotify on Linux, FSEvents on macOS, ReadDirectoryChangesW on Windows)
- Chunk files and compute SHA-256 hashes
- Compare local chunk list with server chunk list (delta computation)
- Upload only new/changed chunks via presigned S3 URLs
- Maintain local SQLite database of file states
- Handle offline queue — buffer changes locally, replay on reconnect

**Chunking Strategy:**
- Fixed-size 4 MB chunks (simplicity over variable-size CDC chunking)
- Each chunk identified by SHA-256 hash of its content
- Content-addressable: same data across any user/file → same chunk hash

### 5.2 Block Server (Deduplication Layer)
Before writing a chunk to S3, the block server checks if that chunk hash already exists in the deduplication store.

**Flow:**
1. Client requests upload token for chunk hash `H`
2. Block server checks `dedup_store[H]` → exists? Return existing S3 URL (cross-user dedup)
3. If not exists: generate presigned S3 upload URL, record mapping after upload confirms
4. Client uploads directly to S3 using presigned URL (block server not in data path)

**Deduplication store:** Redis or dedicated KV store mapping `SHA-256 hash → S3 object key`

### 5.3 Metadata Service
Manages the logical file system tree for each user.

**Responsibilities:**
- Maintain file/folder hierarchy
- Map files to their ordered list of chunks
- Track file versions (30-version history)
- Manage share permissions
- Generate change feeds for notification service

### 5.4 Notification Service
Keeps clients aware of changes made on other devices.

**Mechanism:** WebSocket preferred (persistent connection); Long Poll as fallback
- When file changes, metadata service publishes event to Kafka
- Notification service consumes events, pushes to connected client WebSockets
- On reconnect, client provides last-known `sync_cursor`; server replays missed events

### 5.5 Upload Flow (Step-by-Step)
```
1. Client chunks file → computes SHA-256 per chunk
2. Client sends chunk hash list to Metadata Service
3. Metadata Service returns: which chunks already exist (server has them)
4. For NEW chunks only:
   a. Client requests presigned S3 URL from Block Server
   b. Client uploads chunk bytes directly to S3
   c. Block Server records chunk hash → S3 key in dedup store
5. Client commits file metadata (name, chunk list, version) to Metadata Service
6. Metadata Service publishes "file_changed" event to Kafka
7. Notification Service pushes change to other devices
```

### 5.6 Download Flow
```
1. Client requests file metadata (chunk list + S3 keys) from Metadata Service
2. Metadata Service returns ordered list of chunk S3/CDN URLs
3. Client fetches chunks in parallel from CDN (CloudFront)
4. Client reassembles chunks in order → reconstructs file
```

### 5.7 Delta Sync Algorithm
When a file is modified (e.g., edit middle of a 1GB video):

```
Old chunks: [C1, C2, C3, C4, C5, C6, C7, C8]
New chunks: [C1, C2, C3_new, C4, C5, C6_new, C7, C8]

Delta = only C3_new and C6_new need uploading
Savings: 75% bandwidth reduction
```

---

## 6. Database Design

### 6.1 Files Table
```sql
CREATE TABLE files (
    file_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID NOT NULL,
    parent_folder UUID REFERENCES folders(folder_id),
    name          VARCHAR(255) NOT NULL,
    size_bytes    BIGINT,
    mime_type     VARCHAR(128),
    is_deleted    BOOLEAN DEFAULT FALSE,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW(),
    INDEX idx_files_user_folder (user_id, parent_folder)
);
```

### 6.2 Chunks Table
```sql
CREATE TABLE chunks (
    chunk_hash    CHAR(64) PRIMARY KEY,  -- SHA-256 hex
    s3_key        VARCHAR(512) NOT NULL,
    size_bytes    INT NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
```

### 6.3 File_Chunks Table (Maps files to their chunks)
```sql
CREATE TABLE file_chunks (
    file_version_id UUID NOT NULL,
    chunk_index     INT NOT NULL,
    chunk_hash      CHAR(64) REFERENCES chunks(chunk_hash),
    PRIMARY KEY (file_version_id, chunk_index)
);
```

### 6.4 File_Versions Table
```sql
CREATE TABLE file_versions (
    version_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_id       UUID REFERENCES files(file_id),
    version_num   INT NOT NULL,
    size_bytes    BIGINT,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    created_by    UUID,
    is_current    BOOLEAN DEFAULT FALSE,
    INDEX idx_versions_file (file_id, version_num DESC)
);
```

### 6.5 Shares Table
```sql
CREATE TABLE shares (
    share_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resource_id   UUID NOT NULL,           -- file_id or folder_id
    resource_type ENUM('file','folder'),
    owner_id      UUID NOT NULL,
    shared_with   UUID,                    -- NULL = public link
    permission    ENUM('view','edit'),
    public_token  VARCHAR(32) UNIQUE,      -- for public link sharing
    expires_at    TIMESTAMPTZ,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
```

### 6.6 Folders Table
```sql
CREATE TABLE folders (
    folder_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID NOT NULL,
    parent_folder UUID REFERENCES folders(folder_id),
    name          VARCHAR(255) NOT NULL,
    is_deleted    BOOLEAN DEFAULT FALSE,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
```

---

## 7. API Design

### Upload File
```
POST /api/v1/files/upload/init
Body: { filename, size_bytes, chunk_hashes: [hash1, hash2, ...] }
Response: { file_id, missing_chunks: [hash1, hash3], upload_urls: {hash1: presigned_url} }

PUT /api/v1/files/upload/complete
Body: { file_id, chunk_hashes: [ordered list] }
Response: { file_id, version_id, created_at }
```

### Download File
```
GET /api/v1/files/{file_id}
Response: { file_id, name, size_bytes, chunk_urls: [{index, url}] }
```

### List Directory
```
GET /api/v1/folders/{folder_id}/contents
Response: { folders: [...], files: [...], cursor: "pagination_cursor" }
```

### Get File Versions
```
GET /api/v1/files/{file_id}/versions
Response: { versions: [{version_id, version_num, size_bytes, created_at}] }
```

### Share File
```
POST /api/v1/shares
Body: { resource_id, resource_type, shared_with (optional), permission, expires_at }
Response: { share_id, public_url (if public), permission }
```

### Sync Delta
```
GET /api/v1/sync/changes?since_cursor={cursor}
Response: { changes: [{event_type, file_id, version_id, timestamp}], next_cursor }
```

---

## 8. Scalability & Bottlenecks

### Bottleneck 1: Metadata Service at Scale
- **Problem:** 500B file records — single PostgreSQL can't handle
- **Solution:** Shard by `user_id` — each shard owns a range of users. File operations are almost always scoped to one user.

### Bottleneck 2: Deduplication Store Lookup
- **Problem:** Every chunk upload requires a hash lookup — billions of chunks
- **Solution:** Redis cluster for hot chunks; write-through to PostgreSQL for durability. Bloom filter as pre-check (avoid KV lookup if definitely not present).

### Bottleneck 3: S3 Write Throughput
- **Problem:** 100M uploads/day peak creates hot partitions in S3
- **Solution:** Use content-addressable key prefix (first 4 chars of hash) → ensures random distribution across S3 partitions

### Bottleneck 4: Notification Fan-Out for Shared Folders
- **Problem:** Shared folder with 1000 members → 1 change creates 1000 notifications
- **Solution:** Kafka topic per shared folder; notification service consumes and fans out. Cap fan-out at connection time (lazy delivery).

### Bottleneck 5: CDN Cache Miss
- **Problem:** First download of a new chunk is a cache miss → high S3 origin load
- **Solution:** CloudFront caches chunks by S3 key. Chunks are immutable (content-addressed) → perfect cache hit rate after first access

### Horizontal Scaling
- **API Gateway:** Stateless, auto-scale behind ALB
- **Metadata Service:** Sharded PostgreSQL + read replicas
- **Block Server:** Stateless, auto-scale (dedup state in Redis)
- **Notification Service:** Stateless WebSocket servers + Redis pub/sub for cross-server delivery

---

## 9. Trade-offs & Design Decisions

### Decision 1: Fixed-Size vs. Variable-Size (CDC) Chunking
- **Fixed-Size 4MB:** Simple to implement, predictable behavior. Weakness: inserting bytes at file start invalidates all subsequent chunk hashes.
- **Content-Defined Chunking (CDC):** Rolling hash detects natural boundaries; better dedup for text edits. Complexity: variable chunk sizes, more complex reassembly.
- **Choice:** Fixed-size 4MB for simplicity. Most Dropbox files are small (< 4MB) so chunking rarely kicks in.

### Decision 2: Deduplication Scope
- **Per-user:** Simple but misses savings from two users storing the same file
- **Cross-user (global):** Massive storage savings (academic papers, software installers)
- **Choice:** Cross-user dedup via global SHA-256 chunk store. Note: security implication — two users' data shares underlying storage. Mitigated by access control at metadata layer.

### Decision 3: Conflict Resolution
- **Last-Write-Wins (LWW):** Simple. User who syncs last wins. Other version saved as `filename (John's conflicted copy 2024-01-15).txt`
- **3-way merge:** Complex, format-specific. Works for text files.
- **Choice:** LWW with conflict copy creation. Simpler to implement, acceptable UX for most use cases.

### Decision 4: Metadata Storage — SQL vs NoSQL
- **SQL (PostgreSQL):** Strong consistency, ACID transactions, familiar tooling. Shardable.
- **NoSQL (DynamoDB/Cassandra):** Better horizontal scale, but complex queries (folder listing with pagination) are harder.
- **Choice:** PostgreSQL sharded by user_id. File tree queries are complex (recursive folder listing) — SQL wins.

### Decision 5: Upload Path — Through Server vs. Direct to S3
- **Through Server:** Simpler auth, but server becomes bandwidth bottleneck
- **Direct to S3 via Presigned URL:** Server not in data path, massive bandwidth scaling
- **Choice:** Direct to S3 with presigned URLs. Block server only coordinates metadata.

---

## 10. Key Interview Talking Points

1. **Content-Addressable Storage:** SHA-256 hash of chunk content is the identity. Same data = same hash = stored once. This enables cross-user deduplication transparently.

2. **Delta Sync is a Client Concern:** The client computes chunk hashes locally before upload. Server only stores a list of chunk hashes per file version. Delta = set difference of new vs old chunk hash lists.

3. **Presigned URLs Eliminate Bandwidth Bottleneck:** Block server issues a time-limited AWS presigned URL; client uploads 4GB directly to S3. Your API server never touches the bytes.

4. **Version History = Immutable Chunk Snapshots:** Versions are cheap — they only store the ordered list of chunk hashes, not duplicate data. Old and new versions sharing unchanged chunks don't duplicate storage.

5. **Conflict Copy Strategy:** When two devices edit the same file offline and both sync, the later upload wins (LWW by server timestamp). The losing version is renamed and saved alongside — user sees both, can choose.

6. **Notification via WebSocket + Kafka:** WebSocket for low-latency push to clients. Kafka as the event bus ensures no notifications are dropped even if notification service restarts.

7. **Bloom Filter for Dedup:** Before querying Redis for chunk existence, use a Bloom filter. If bloom filter says "definitely not present" → skip Redis lookup. Reduces dedup overhead by 90%.

8. **Offline Queue:** Client maintains a local SQLite event log of file changes. When network restores, replays events in order. Idempotent by design (same chunk hash = same result regardless of how many times you attempt upload).

9. **30-Version Limit:** After 30 versions, oldest version's chunks are GC'd — but only if no other file version references those chunks (reference counting). This prevents accidental deletion of shared chunks.

10. **Scale Numbers to Remember:** 500M users × 2GB = 1EB storage. 100M uploads/day = ~1,160 uploads/sec average. Peak (10x) = ~11,600 uploads/sec. Chunk size 4MB → 290 MB/sec average upload bandwidth just for metadata-free chunk bytes.
