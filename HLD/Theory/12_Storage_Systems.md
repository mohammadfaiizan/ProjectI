# 12 — Storage Systems

---

## Table of Contents
1. [Storage Types Overview](#1-storage-types-overview)
2. [When to Use Each Storage Type](#2-when-to-use-each-storage-type)
3. [Object Storage Design](#3-object-storage-design)
4. [Distributed File Systems](#4-distributed-file-systems)
5. [Blob Storage for Media](#5-blob-storage-for-media)
6. [Content-Addressable Storage](#6-content-addressable-storage)
7. [Data Deduplication](#7-data-deduplication)
8. [Erasure Coding vs Replication](#8-erasure-coding-vs-replication)
9. [Data Tiering](#9-data-tiering)
10. [Data Lake Architecture](#10-data-lake-architecture)
11. [Data Warehouse Design](#11-data-warehouse-design)
12. [LSM Tree vs B-Tree](#12-lsm-tree-vs-b-tree)
13. [Write-Ahead Logging (WAL)](#13-write-ahead-logging-wal)
14. [Compaction Strategies in LSM Trees](#14-compaction-strategies-in-lsm-trees)
15. [Backup and Disaster Recovery](#15-backup-and-disaster-recovery)
16. [RAID Levels](#16-raid-levels)
17. [Storage Scalability](#17-storage-scalability)
18. [NVMe over Fabrics](#18-nvme-over-fabrics)
19. [Quick Reference](#19-quick-reference)

---

## 1. Storage Types Overview

### Block Storage

Raw storage presented as a block device. OS/filesystem manages the blocks. No metadata at storage layer — just read/write at byte offsets.

```
Block Device:
  [Block 0][Block 1][Block 2]...[Block N]
  Each block: fixed size (512B or 4KB)
  
OS layer formats with filesystem:
  ext4, XFS, NTFS → manages files, directories, inodes
  
Application sees: /dev/sdb → formatted as ext4 → /mnt/data/
```

**Examples:** AWS EBS, Azure Managed Disks, SAN (iSCSI/Fibre Channel), local NVMe

**Properties:**
- Low latency (direct I/O to blocks)
- No inherent metadata or versioning
- Must be formatted with a filesystem before use
- Attached to one instance at a time (typically)
- Ideal for databases, OS volumes

### Object Storage

Flat namespace of objects. Each object has: data (arbitrary bytes) + metadata + unique key.

```
Object:
  key:      "user-photos/alice/profile-pic.jpg"
  data:     [binary photo data]
  metadata: {content-type: "image/jpeg", size: 2MB, created: "2024-01-01"}
  
No directory hierarchy (key can contain "/" as naming convention)
Accessed via HTTP REST API (GET/PUT/DELETE)
```

**Examples:** AWS S3, Google Cloud Storage, Azure Blob Storage, MinIO

**Properties:**
- Unlimited scalability
- Globally accessible via URL
- Built-in redundancy (11 nines durability in S3)
- No latency for sequential I/O compared to block
- Cannot be mounted as filesystem (though NFS mounts exist)
- Ideal for unstructured data: images, videos, backups, logs

### File Storage (Network File System)

Shared filesystem accessible over network. Familiar filesystem semantics (directories, files, permissions).

```
NFS mount:
  mount -t nfs nas-server:/exports/shared /mnt/shared
  
All clients see same filesystem namespace:
  /mnt/shared/
    reports/
      daily-report.xlsx
    uploads/
      user123/
        avatar.png
```

**Examples:** AWS EFS (NFS), Azure Files (SMB/NFS), NetApp, GlusterFS

**Properties:**
- Shared access by multiple instances simultaneously
- POSIX filesystem semantics
- Higher latency than block storage
- Good for shared application state, CMS uploads, home directories

### Comparison Table

| Aspect | Block Storage | Object Storage | File Storage |
|---|---|---|---|
| Access method | Block I/O (byte offset) | HTTP REST API | POSIX filesystem (NFS/SMB) |
| Metadata | None (filesystem layer) | Rich (key-value) | Filesystem metadata |
| Scalability | Vertical (disk size) | Nearly unlimited | Limited (NFS bottleneck) |
| Latency | Lowest (~0.1ms NVMe) | Medium (ms) | Medium (ms) |
| Concurrency | Single instance (usually) | Any client | Multiple clients |
| Versioning | No | Yes (S3 versioning) | No |
| Cost | Moderate-high | Low | Moderate |
| Use case | Databases, OS volumes | Media, backups, artifacts | Shared filesystems |

---

## 2. When to Use Each Storage Type

### Decision Framework

```
What are you storing?
│
├── Structured data that needs ACID transactions?
│   └── Relational DB on Block Storage (EBS + PostgreSQL)
│
├── Semi-structured data at massive scale?
│   └── NoSQL DB (Cassandra, DynamoDB) — manages own block storage
│
├── Large binary objects (images, videos, files)?
│   └── Object Storage (S3, GCS)
│
├── Need shared filesystem across multiple servers?
│   └── File Storage (EFS, NFS)
│
├── Sequential analytics / log processing?
│   └── Object Storage + columnar format (Parquet on S3)
│
└── Temporary scratch space for compute?
    └── Local NVMe (ephemeral instance store)
```

### Practical Examples

| Workload | Storage Choice | Reasoning |
|---|---|---|
| PostgreSQL database | Block (EBS gp3) | ACID transactions, low latency |
| User avatar uploads | Object (S3) | Unstructured, CDN distribution |
| Kubernetes persistent volumes | Block (EBS) | Single-attach, low latency |
| ML training datasets | Object (S3) | Large files, cost-effective |
| Shared config files | File (EFS) | Multiple containers need access |
| Video streaming files | Object (S3) + CloudFront | Range requests, CDN |
| Application logs archival | Object (S3 Glacier) | Cold storage, low cost |
| In-memory cache | Redis (memory-backed) | Ultra-low latency |
| Data warehouse | Columnar on Object (S3 + Parquet) | Analytics optimization |

---

## 3. Object Storage Design

### Core Concepts

**Bucket:** Top-level container (namespace isolation)
**Key:** Full path within bucket (e.g., `photos/2024/jan/img001.jpg`)
**Object:** Data + metadata + key
**Metadata:** System metadata (ETag, content-type, size) + user-defined metadata

### S3 Key Design

Keys are flat strings, not real directories. `/` is a naming convention.

```
Bad design:
  user-123.jpg
  user-456.jpg
  (hot prefix: all objects in same partition)

Good design:
  abcdef/user-123.jpg    (hash prefix for distribution)
  xyz789/user-456.jpg
  
S3 internally partitions by key prefix.
High-request patterns benefit from distributed prefixes.
```

### Multipart Upload

For large files (> 100 MB). Upload in parallel chunks.

```
1. Initiate multipart upload → get upload_id
2. Upload Part 1 (5MB)  → ETag1
   Upload Part 2 (5MB)  → ETag2  (parallel)
   Upload Part 3 (5MB)  → ETag3
3. Complete multipart upload (send all ETags)
   S3 assembles final object

Benefits:
  - Retry individual failed parts (not whole file)
  - Parallel upload for speed
  - Resume interrupted uploads
  - Maximum part size: 5 GB; minimum (except last): 5 MB
```

### Presigned URLs

Allow temporary access without exposing credentials:

```python
import boto3
s3 = boto3.client('s3')

# Generate presigned URL for upload
presigned_url = s3.generate_presigned_url(
    'put_object',
    Params={'Bucket': 'my-bucket', 'Key': 'uploads/photo.jpg'},
    ExpiresIn=3600  # 1 hour
)

# Client uploads directly to S3 — no server proxying needed
# curl -X PUT "{presigned_url}" -T photo.jpg
```

**Flow:**
```
Client → App Server → generate presigned URL → return to client
Client → PUT directly to S3 (with presigned URL)
App Server → notified via S3 event (SNS/SQS/Lambda)
```

### S3 Consistency Model

As of December 2020: **strong read-after-write consistency** for all S3 operations.

### Versioning and Lifecycle

```
Versioning: Every PUT creates a new version; DELETE adds delete marker
  GET without version ID → returns latest version
  GET with version ID → returns specific version

Lifecycle rules:
  30 days → move to S3-IA (Infrequent Access)
  90 days → move to S3-Glacier Instant Retrieval
  365 days → move to S3-Glacier Deep Archive
  5 years → expire (delete)
```

---

## 4. Distributed File Systems

### Google File System (GFS)

Designed for large files, sequential access, and fault tolerance at commodity hardware scale.

```
Architecture:
  1 Master node: metadata (file→chunk mapping), chunk location
  Thousands of ChunkServers: store 64MB chunks
  Multiple clients: access data directly from ChunkServers

Read flow:
  Client → Master: "Where are chunks for file X, offset Y?"
  Master → Client: "Chunk 42 is on servers S1, S2, S3"
  Client → S1: "Read chunk 42" (bypasses master)
  
Write flow:
  Client → Master: "Lease for writing to chunk 42"
  Master → Client: "Primary=S1, replicas=S2,S3"
  Client → S1 (primary): data pipeline starts
  S1 → S2 → S3 (pipeline replication)
  S1 (after all acks) → Client: success
```

**GFS Design Decisions:**

| Decision | Reasoning |
|---|---|
| 64MB chunks (large) | Reduce master metadata; fewer master requests |
| Single master | Simplicity; master is bottleneck (partially mitigated by caching) |
| No data through master | Scalability; master only handles metadata |
| Relaxed consistency | Performance; designed for append-heavy workloads |
| Pipelined replication | Maximize network bandwidth |

### HDFS (Hadoop Distributed File System)

GFS-inspired, open-source. Powers Hadoop ecosystem.

```
Architecture:
  NameNode (1 active, 1 standby): stores filesystem namespace
    - File → block mapping
    - Block → DataNode mapping
    - Stored in memory for fast access
    
  DataNodes (N): store actual data blocks (128MB default)
    - Report to NameNode via heartbeat
    - Send block reports
    
  Secondary NameNode: periodically checkpoints NameNode (not a hot standby)
```

**HDFS Block Replication:**
```
Block A (128MB):
  DataNode 1: replica 1 (same rack)
  DataNode 2: replica 2 (same rack) 
  DataNode 3: replica 3 (different rack)
  
Rack-aware placement:
  2 replicas in rack 1 (fast local copy)
  1 replica in rack 2 (cross-rack for rack failure tolerance)
```

**HDFS Read:**
```
Client → NameNode: "Block locations for /data/file.csv?"
NameNode → Client: [{block1: [DN1, DN2, DN3]}, ...]
Client → nearest DataNode: read block 1 (topology-aware)
```

**HDFS vs GFS:**

| Feature | GFS | HDFS |
|---|---|---|
| Language | C++ | Java |
| Chunk/Block size | 64MB | 128MB |
| Consistency | Relaxed | Strong |
| Snapshots | Yes | Yes |
| HA NameNode | Complex | HANameNode (Quorum Journal) |
| Ecosystem | Google internal | Open-source Hadoop |

---

## 5. Blob Storage for Media

### Chunking Strategy

Large media files split into chunks for parallel upload, download, and partial failures.

```
Video file: 2GB
Chunk size: 5MB
Chunks: 2048/5 = ~410 chunks

Upload:
  Client splits → uploads chunks in parallel (5 concurrent)
  Server reassembles after all chunks received
  
Storage:
  chunks/video-uuid/chunk-000001
  chunks/video-uuid/chunk-000002
  ...
  metadata: {total_chunks: 410, chunk_size: 5MB, status: "complete"}
```

### CDN Integration

```
Upload flow:
  User → App Server → upload to S3 (origin)
                   → trigger CDN cache invalidation if replacing existing

Download flow (without CDN):
  User → App Server → S3 → back to user  (single datacenter bandwidth)

Download flow (with CDN):
  User → CloudFront PoP (nearest edge) → served from edge cache
  Cache miss: CloudFront → S3 (origin) → cache at edge → serve user
  
Benefits:
  - Geographically distributed
  - Reduced origin load
  - Lower latency for end users
```

### Range Requests

HTTP Range header enables partial content retrieval (video seeking, resumable downloads).

```
Request:
  GET /videos/movie.mp4
  Range: bytes=10485760-20971519   (10MB-20MB)

Response:
  HTTP/1.1 206 Partial Content
  Content-Range: bytes 10485760-20971519/2147483648
  Content-Length: 10485760
  [10MB chunk of data]

Video player uses range requests to:
  - Load only the part of video being watched
  - Enable seeking without downloading full file
  - Buffer ahead while watching
```

### Media Processing Pipeline

```
User uploads video
     ↓
S3 (raw upload)
     ↓
Event trigger (S3 → SNS → SQS)
     ↓
Video transcoding job (AWS MediaConvert / FFmpeg on EC2)
     ↓ Generates multiple resolutions:
  video-uuid/360p.mp4
  video-uuid/720p.mp4
  video-uuid/1080p.mp4
  video-uuid/manifest.m3u8  (HLS adaptive streaming)
     ↓
S3 (processed)
     ↓
CloudFront (CDN distribution)
```

---

## 6. Content-Addressable Storage

### Concept

Instead of naming objects by location (path), name them by **content** (hash of content).

```
Traditional: /files/user123/profile-pic.jpg
             Location-based; two identical files stored twice

Content-addressable: sha256(content) = "d4f3e2a1..."
                     /objects/d4f3e2a1...
                     Same content → same key → deduplicated automatically!
```

### Git Object Storage

Git is a content-addressable store.

```
Blob (file content): sha1(content)
Tree (directory): sha1(list of blobs + trees)
Commit: sha1(tree + parent commit + metadata)

When you commit:
  1. Each file content → blob object: .git/objects/ab/cd1234...
  2. Directory structure → tree object
  3. Commit → commit object

Two repos with same file content → same blob hash → identical objects
```

### IPFS (InterPlanetary File System)

Content-addressed distributed filesystem.

```
Add file to IPFS:
  hash(content) → CID (Content Identifier)
  
Retrieve:
  ipfs get QmXxx...  (CID)
  
IPFS finds node that has content with this hash
No single server owns it — anyone who has it can serve it
Immutable: changing content changes CID
```

### Deduplication via Content Addressing

```
Enterprise backup with content-addressing:
  backup1/file_a.log  → hash: abc123 → stored as objects/abc123
  backup1/file_b.log  → hash: def456 → stored as objects/def456
  backup2/file_a.log  → hash: abc123 → ALREADY EXISTS → no storage!
  
100TB of backups that are 90% duplicate → only 10TB stored
```

---

## 7. Data Deduplication

### File-Level Deduplication

Compute hash of entire file. If same hash exists → don't store duplicate.

```
Files:
  report-jan.pdf  (1MB, hash: aaa)
  report-feb.pdf  (1MB, hash: bbb)  — different
  report-jan-copy.pdf (1MB, hash: aaa)  — duplicate!

Storage: only store 2 files, maintain reference count
```

**Problem:** Even 1 byte difference → different hash → no dedup.

### Block/Chunk-Level Deduplication

Split files into chunks; dedup at chunk level.

```
File 1: [chunk_A][chunk_B][chunk_C]
File 2: [chunk_A][chunk_D][chunk_C]  ← shares chunk_A and chunk_C

Storage:
  chunk_A → stored once (2 references)
  chunk_B → stored (1 reference)
  chunk_C → stored once (2 references)
  chunk_D → stored (1 reference)
  
File 1 metadata: [chunk_A, chunk_B, chunk_C]
File 2 metadata: [chunk_A, chunk_D, chunk_C]
```

### Variable-Length Chunking (Rabin Fingerprinting)

Fixed-size chunks are inefficient when files shift by even 1 byte (all subsequent chunks change).

**Rabin fingerprinting** uses a rolling hash to find content-defined chunk boundaries:

```
Content: [AABBCCDDEE]FFGG[HHIIJJKK]

Rolling hash over sliding window:
  When hash % prime == target_value → chunk boundary
  
Result: chunk boundaries based on content, not position
  Chunk 1: AABBCCDDEE  (natural boundary at position 10)
  Chunk 2: FFGG
  Chunk 3: HHIIJJKK

If we insert 2 bytes at position 3:
  Fixed chunking: every chunk after position 3 changes
  Rabin chunking: only chunk 1 affected; chunks 2,3 unchanged
```

Used by: Dropbox, ZFS, NetApp ONTAP, Perforce.

### Deduplication Ratios

| Workload | Typical Dedup Ratio |
|---|---|
| Virtual machine backups | 10:1 — 30:1 |
| Email servers | 20:1 — 40:1 |
| File servers | 5:1 — 10:1 |
| Database backups | 2:1 — 5:1 |
| Video files | 1:1 (already compressed) |

---

## 8. Erasure Coding vs Replication

### Replication

Store N identical copies.

```
3-way replication:
  Data: [D] → [D][D][D] on 3 different nodes
  
Storage overhead: 3x
Can tolerate: 2 node failures (with 3 copies)
```

### Erasure Coding

Encode data into k+m chunks where:
- k = data chunks
- m = parity chunks
- Can recover from any m failures using any k chunks

**Reed-Solomon (6+3) example:**

```
Original data: 6 data blocks [D1][D2][D3][D4][D5][D6]
Erasure coded: 6 data + 3 parity = [D1][D2][D3][D4][D5][D6][P1][P2][P3]
Distributed across 9 nodes

Storage overhead: 9/6 = 1.5x (vs 3x for 3-way replication)
Can tolerate: any 3 node failures → reconstruct from any 6 of 9

Math: P1 = XOR(D1, D2)  (simplified; real RS is GF(2^8) arithmetic)
```

### Comparison

| Aspect | Replication (3x) | Erasure Coding (6+3) |
|---|---|---|
| Storage overhead | 3x | 1.5x |
| Fault tolerance | 2 failures | 3 failures |
| Read latency | Low (read nearest replica) | Higher (may need to read multiple) |
| Write latency | Medium (3 writes) | Higher (encoding CPU + 9 writes) |
| CPU cost | Low | High (encoding/decoding) |
| Use case | Hot data, low latency | Cold/warm data, cost efficiency |
| Examples | Cassandra, MongoDB | HDFS (EC), S3, Ceph |

### S3 Erasure Coding

Amazon S3 uses erasure coding (not disclosed, but approximately equivalent to 8+4 or similar scheme):
- 11 nines (99.999999999%) durability
- Objects stored across multiple Availability Zones
- Storage overhead much lower than 3x replication

### When to Use What

```
Use replication when:
  - Read-heavy workloads (serve from nearest replica)
  - Latency-sensitive (hot data, real-time)
  - Simple failure recovery (no decoding overhead)
  
Use erasure coding when:
  - Write-once, read-rarely (cold storage, archival)
  - Storage cost is priority
  - Large objects (encoding overhead amortized)
  - High durability with less overhead
```

---

## 9. Data Tiering

### Tiers by Access Frequency and Cost

```
HOT (Tier 1):    Frequently accessed (daily)
                 Storage: NVMe SSD, in-memory
                 Latency: < 1ms
                 Cost: $$$$
                 Examples: Redis, NVMe EBS, DynamoDB

WARM (Tier 2):   Occasionally accessed (weekly/monthly)
                 Storage: SSD or HDD
                 Latency: 1-10ms
                 Cost: $$
                 Examples: S3 Standard-IA, HDD-backed S3, EBS st1

COLD (Tier 3):   Rarely accessed (quarterly/annual)
                 Storage: HDD array or tape
                 Latency: seconds to minutes
                 Cost: $
                 Examples: S3 Glacier Instant, S3 Glacier Flexible

FROZEN (Tier 4): Compliance archival (years)
                 Storage: Tape, deep archive
                 Latency: hours
                 Cost: ¢
                 Examples: S3 Glacier Deep Archive, AWS Tape Gateway
```

### S3 Storage Classes

| Class | Latency | Minimum duration | Use case | Cost/GB/mo |
|---|---|---|---|---|
| S3 Standard | ms | None | Hot data | $0.023 |
| S3 Standard-IA | ms | 30 days | Infrequent access | $0.0125 |
| S3 One Zone-IA | ms | 30 days | Single AZ, lower cost | $0.01 |
| S3 Glacier Instant | ms | 90 days | Archives with quick access | $0.004 |
| S3 Glacier Flexible | minutes-hours | 90 days | Archives, rare access | $0.0036 |
| S3 Glacier Deep Archive | hours | 180 days | Long-term compliance | $0.00099 |

### Automated Tiering (S3 Intelligent-Tiering)

```
S3 Intelligent-Tiering:
  Monitors access patterns
  Moves objects automatically between tiers:
    30 days without access → Infrequent Access tier
    90 days without access → Archive Instant Access tier
    180 days → Archive Access tier
    
  No retrieval fees; objects moved to hot tier on access
  Management fee: $0.0025 per 1000 objects
```

### Database Tiering Example

```
E-commerce platform:
  
  Hot tier (Redis):     Last 24 hours of orders (< 5% of data, 60% of queries)
  Warm tier (PostgreSQL): Last 2 years of orders (95% of data, 39% of queries)
  Cold tier (S3 + Parquet): All historical orders (100% of data, 1% of queries)
```

---

## 10. Data Lake Architecture

### What is a Data Lake?

Central repository storing structured, semi-structured, and unstructured data at scale, in native format.

**Key principle:** Schema-on-read (impose schema when querying, not when writing).

```
Data Lake (S3/GCS):
  /raw/       → data as-is from sources (CSV, JSON, Avro, binary)
  /refined/   → cleaned, validated, standardized (Parquet)
  /curated/   → aggregated, business-ready datasets (Parquet)
  
Same data stored in all zones; each zone adds transformation value
```

### Zones

**Raw Zone (Bronze):**
- Exact copy of source data
- Immutable — never modified
- Full historical record
- Formats: CSV, JSON, XML, binary, images

**Refined Zone (Silver):**
- Cleaned: nulls handled, duplicates removed
- Validated: schema enforced
- Standardized: consistent types, formats
- Partitioned for efficient query
- Format: Parquet or ORC (columnar)

**Curated Zone (Gold):**
- Aggregated business metrics
- Joined datasets
- Ready for BI tools and ML
- Format: Parquet, Delta Lake, or Iceberg

### Schema-on-Read vs Schema-on-Write

```
Schema-on-Write (data warehouse):
  Define schema first → ETL enforces schema → store structured data
  Pros: Fast queries  Cons: Schema changes are painful; lose raw data

Schema-on-Read (data lake):
  Store any format → define schema when querying
  Pros: Flexible; preserve raw data  Cons: Query slower; schema chaos
```

### Modern Data Lake (Lakehouse)

Combines data lake (scale, cost) with data warehouse (ACID, performance):
- **Delta Lake (Databricks):** ACID transactions on S3; time travel
- **Apache Iceberg:** Open table format; schema evolution; partition evolution
- **Apache Hudi:** Upserts/deletes on data lake; incremental processing

```
Lakehouse Architecture:
  S3 (storage) + Delta Lake (table format) + Spark/Trino (query engine)
  
Features:
  ACID transactions on S3 objects
  Time travel: query data as of yesterday
  Schema evolution: add columns without rewriting
  Z-ordering: cluster related data together for fast queries
```

---

## 11. Data Warehouse Design

### OLTP vs OLAP

| Aspect | OLTP (Transactional) | OLAP (Analytical) |
|---|---|---|
| Query pattern | Many small read/write | Few large read queries |
| Optimization | Write performance, indexes | Full table scans, aggregations |
| Schema | Normalized (3NF) | Denormalized (star/snowflake) |
| Data volume | GB | TB-PB |
| Latency | ms | seconds-minutes |
| Users | Application backend | Analysts, BI tools |
| Examples | PostgreSQL, MySQL | Redshift, BigQuery, Snowflake |

### Star Schema

Fact table (metrics) surrounded by dimension tables.

```
Fact Table: fact_orders
  order_id, date_key, customer_key, product_key, store_key
  amount, quantity, discount

Dimension Tables:
  dim_date:     date_key, year, month, day, day_of_week, quarter
  dim_customer: customer_key, name, segment, region
  dim_product:  product_key, name, category, brand, price
  dim_store:    store_key, name, city, state, country
```

**Query:**
```sql
SELECT d.month, c.region, SUM(f.amount) as revenue
FROM fact_orders f
JOIN dim_date d ON f.date_key = d.date_key
JOIN dim_customer c ON f.customer_key = c.customer_key
WHERE d.year = 2024
GROUP BY d.month, c.region;
```

**Advantages:** Simple queries; denormalized (fewer joins).

### Snowflake Schema

Dimension tables further normalized.

```
dim_product → dim_category → dim_department
           → dim_brand
           
More tables, more joins, more storage efficient
```

### Columnar Storage

Row-based vs columnar:

```
Row storage (PostgreSQL heap):
  [row1: id=1, name="Alice", age=30, city="NYC"]
  [row2: id=2, name="Bob",   age=25, city="LA" ]
  
  Full row read for any query → fast for OLTP (read whole row)

Columnar storage (Parquet, Redshift, BigQuery):
  id column:   [1, 2, 3, 4, ...]
  name column: ["Alice", "Bob", "Charlie", ...]
  age column:  [30, 25, 35, ...]
  
  Analytical query: SELECT avg(age) → reads only age column
  Reads 100x less data for selective column queries
```

**Compression in columnar:**
- Same-type values compress well (repeated patterns)
- Dictionary encoding for low-cardinality strings
- Run-length encoding for sorted columns
- Bit packing for integers

### OLAP Cubes

Pre-aggregated multidimensional data structures for fast analytics.

```
Cube dimensions: Product × Region × Time
Pre-compute: SUM(sales) for every (product, region, time) combination

Query: "Sales for Electronics in US Q4 2024"
  Without cube: full table scan + aggregation
  With cube: direct lookup in pre-computed cell
```

Modern columnar databases (Redshift, BigQuery) are fast enough that cubes are less common today.

---

## 12. LSM Tree vs B-Tree

### B-Tree (PostgreSQL, MySQL, SQLite)

Balanced tree structure. All data on disk. Updates in-place.

```
B-Tree structure (B+ Tree):
           [50]
          /    \
      [20,30]   [70,80]
     /  |  \   /  |  \
  [10][25][35][60][75][90]  ← leaf nodes contain actual data + pointers

Read: O(log N) tree traversal
Write: Find leaf → update in place → may cause page split → write to random disk location
```

**Write behavior:**
- Random I/O for writes (find page in tree, update)
- Good for read-heavy workloads
- In-place updates: old value overwritten

### LSM Tree (Cassandra, RocksDB, LevelDB, HBase)

Log-Structured Merge Tree. Optimized for write-heavy workloads.

```
Write path:
  1. Write to WAL (sequential, crash recovery)
  2. Write to MemTable (in-memory sorted structure)
  3. When MemTable full → flush to SSTable (Sorted String Table) on disk
  
SSTable files:
  L0: [SSTable1] [SSTable2] [SSTable3]  ← freshly flushed
  L1: [SSTable4] [SSTable5]             ← after L0 compaction
  L2: [SSTable6]                        ← after L1 compaction
  
Read path:
  Check MemTable → check L0 SSTables (bloom filter first) → L1 → L2 → ...
  May read multiple levels → read amplification
```

### LSM Write Optimization

```
Sequential writes only:
  Every write: WAL append (sequential) + MemTable (in-memory)
  SSTable flush: sequential write to disk
  
No in-place updates:
  "Delete" = write a tombstone marker
  "Update" = write new version; old version merged later during compaction
  
Result: 10-1000x write throughput vs B-Tree for write-heavy workloads
```

### Comparison

| Aspect | B-Tree | LSM Tree |
|---|---|---|
| Write I/O | Random (in-place update) | Sequential (append-only) |
| Write throughput | Moderate | Very high |
| Read I/O | Low (single lookup) | Higher (multiple SSTables) |
| Read throughput | High | Moderate (bloom filters help) |
| Space amplification | Low | Higher (multiple versions until compaction) |
| Write amplification | Low | Higher (data written multiple times during compaction) |
| Use case | Read-heavy (RDBMS, OLTP) | Write-heavy (Cassandra, HBase, analytics) |
| Examples | PostgreSQL, MySQL, SQLite | Cassandra, RocksDB, HBase, LevelDB |

---

## 13. Write-Ahead Logging (WAL)

### Purpose

Guarantee durability without expensive synchronous disk writes for every operation.

**Rule:** Every change written to WAL before modifying actual data pages.

```
Without WAL:
  1. Modify data page in memory
  2. Flush data page to disk
  (crash between 1 and 2 → data lost)

With WAL:
  1. Write change to WAL (sequential, fast)
  2. Flush WAL to disk (fsync)
  3. Modify data page in memory
  4. Flush data page to disk (async, batched)
  (crash between 2 and 4 → replay WAL to recover)
```

### PostgreSQL WAL

```
WAL record structure:
  {LSN (log sequence number), transaction ID, resource manager, data}

WAL files: pg_wal/000000010000000000000001 (16MB segments)

Crash recovery:
  PostgreSQL reads WAL from last checkpoint
  Replays every WAL record
  Returns to consistent state
```

### WAL Benefits Beyond Durability

**1. Replication:**
```
PostgreSQL streaming replication:
  Primary writes WAL → sends WAL stream to standby
  Standby replays WAL → stays in sync
  
WAL is the replication stream!
```

**2. Change Data Capture (CDC):**
```
Debezium connects to PostgreSQL WAL:
  Reads every INSERT/UPDATE/DELETE as WAL record
  Publishes to Kafka as CDC event
  
No polling needed — WAL provides ordered stream of all changes
```

**3. Point-in-Time Recovery (PITR):**
```
Base backup + WAL archive → recover to any point in time

"Restore to 2024-01-15 14:30:00"
  → restore base backup from 2024-01-14
  → replay WAL from 2024-01-14 to 2024-01-15 14:30:00
  → stop
```

### WAL in LSM Trees

LSM tree's WAL is simpler: append-only log, replayed to rebuild MemTable after crash.

```
LSM WAL:
  Write(key=A, val=1)  → WAL entry 1
  Write(key=B, val=2)  → WAL entry 2
  Write(key=A, val=3)  → WAL entry 3

Crash → on restart:
  Replay WAL → MemTable: {A:3, B:2}
  Continue as normal
  
Once MemTable flushed to SSTable → WAL entries truncated
```

---

## 14. Compaction Strategies in LSM Trees

### Why Compaction?

Over time, LSM trees accumulate many SSTable files:
- Multiple versions of same key
- Tombstone markers for deleted keys
- Wasted storage

Compaction merges SSTables to reclaim space and improve read performance.

### Size-Tiered Compaction (STCS)

Group SSTables by size. When enough SSTables of similar size, merge them into one larger SSTable.

```
Initial:  [1MB][1MB][1MB][1MB] → compact → [4MB]
Next:     [4MB][4MB][4MB][4MB] → compact → [16MB]
Next:     [16MB][16MB]...

Characteristics:
  - Write amplification: low (merge only when many similar size)
  - Space amplification: high (during compaction, need 2x space)
  - Read amplification: moderate
  - Best for: write-heavy workloads (Cassandra default for time-series)
```

### Leveled Compaction (LCS)

SSTables organized in levels. Each level has bounded total size. L0 SSTables compacted into L1; L1 into L2, etc.

```
L0: Up to 4 SSTables (any size)
L1: Up to 10MB total (non-overlapping key ranges)
L2: Up to 100MB total
L3: Up to 1000MB total

Key range coverage:
  L1: [A-C] [D-F] [G-I] [J-M] ... (no overlap)
  
Compaction: when L0 has 4+ SSTables → merge into L1
            when Ln exceeds size limit → merge into Ln+1

Characteristics:
  - Write amplification: higher (rewrite data per level)
  - Space amplification: low (non-overlapping; ~10% extra)
  - Read amplification: low (one SSTable per level for a key)
  - Best for: read-heavy workloads
```

### FIFO Compaction

First-in, first-out. Oldest SSTables deleted when total size exceeds limit.

```
Characteristics:
  - No merging, just deletion
  - Fastest compaction strategy
  - Only works for time-series data with TTL (oldest data expired anyway)
```

### Compaction Strategy Comparison

| Strategy | Read Amp | Write Amp | Space Amp | Use Case |
|---|---|---|---|---|
| Size-Tiered | Medium | Low | High | Write-heavy, time-series |
| Leveled | Low | High | Low | Read-heavy, general |
| FIFO | High | None | None | TTL time-series data |
| TWCS (Time Window) | Medium | Low | Medium | Time-series with TTL |

---

## 15. Backup and Disaster Recovery

### Key Metrics

**RPO (Recovery Point Objective):** Maximum acceptable data loss, measured in time.
- RPO = 0: zero data loss (synchronous replication)
- RPO = 1 hour: can lose up to 1 hour of data
- RPO = 24 hours: daily backups sufficient

**RTO (Recovery Time Objective):** Maximum acceptable downtime.
- RTO = 0: zero downtime (active-active multi-region)
- RTO = 1 hour: system must be up within 1 hour of failure
- RTO = 24 hours: DR allowed to take up to a day

```
Cost vs RTO/RPO:
                High cost
                   │
  Active-Active ───┤ RTO=0, RPO=0
  (Multi-region)   │
                   │
  Active-Passive ──┤ RTO=minutes, RPO=seconds
  (Hot standby)    │
                   │
  Warm standby ────┤ RTO=1 hour, RPO=minutes
                   │
  Cold backup ─────┤ RTO=hours, RPO=hours/days
                   │
                Low cost
```

### 3-2-1 Backup Rule

- **3** copies of data
- **2** different media types
- **1** offsite copy

```
Production database:
  Copy 1: Primary DB (EBS, us-east-1)
  Copy 2: Read replica (EBS, us-east-1b — different AZ)
  Copy 3: S3 backup (us-west-2 — different region) ← offsite
  
Media types: EBS (block) + S3 (object)
```

### Point-in-Time Recovery (PITR)

```
PostgreSQL PITR:
  1. Take base backup: pg_basebackup → stored in S3
  2. Continuously archive WAL segments to S3
  
  Recovery to any point:
    → restore base backup
    → replay WAL until desired point
    → start database
    
Recovery time depends on:
  - How old the base backup is
  - Volume of WAL to replay
  - Database size

RDS automated backups: daily base backup + 5-minute transaction log uploads
Can restore to any 5-minute window in retention period (default 7 days, max 35)
```

### Disaster Recovery Strategies

```
1. Backup and Restore (cheapest, slowest):
   RPO: hours  RTO: hours-days
   
2. Pilot Light (minimal running infrastructure):
   Core DB replication running; app servers off
   RPO: minutes  RTO: 30-60 minutes
   
3. Warm Standby (scaled-down full stack):
   DB replication + scaled-down app tier in DR region
   RPO: seconds  RTO: minutes
   
4. Multi-Site Active-Active (most expensive, fastest):
   Full stack in multiple regions, all serving traffic
   RPO: 0  RTO: 0 (automatic failover)
```

---

## 16. RAID Levels

RAID (Redundant Array of Independent Disks) — combine multiple physical disks for performance and/or redundancy.

### RAID 0 (Striping, No Redundancy)

```
Disk1: [A1][A3][A5][A7]
Disk2: [A2][A4][A6][A8]

Data striped across disks
Performance: 2x read and write throughput
Redundancy: NONE — any disk failure = total data loss
Use case: Scratch space, video editing (where speed matters, data is reproducible)
```

### RAID 1 (Mirroring)

```
Disk1: [A1][A2][A3][A4]
Disk2: [A1][A2][A3][A4]  (exact mirror)

Performance: 2x read throughput; write = slowest disk
Redundancy: 1 disk failure tolerated
Storage efficiency: 50%
Use case: OS drives, critical small databases
```

### RAID 5 (Striping with Distributed Parity)

```
Disk1: [A1][B1][C1][P4]
Disk2: [A2][B2][P3][C2]
Disk3: [A3][P2][B3][C3]
Disk4: [P1][A4][B4][C4]

Parity (P) distributed across all disks
Can reconstruct from any 1 disk failure
Storage efficiency: (N-1)/N  → 4 disks = 75%
Minimum: 3 disks
Read performance: excellent
Write performance: good (parity calculation needed)
Use case: File servers, NAS
```

### RAID 6 (Striping with Double Parity)

```
Like RAID 5 but with 2 parity blocks per stripe
Can tolerate 2 simultaneous disk failures
Storage efficiency: (N-2)/N → 6 disks = 67%
Minimum: 4 disks
Slower writes than RAID 5 (two parity calculations)
Use case: Large disk arrays where rebuild time creates risk
```

### RAID 10 (RAID 1+0: Mirror + Stripe)

```
Disk1: [A1][A3]    Mirror pair 1
Disk2: [A1][A3]    Mirror pair 1

Disk3: [A2][A4]    Mirror pair 2
Disk4: [A2][A4]    Mirror pair 2

Data striped across mirror pairs
Can tolerate 1 disk per mirror pair (2+ disks if in different pairs)
Performance: excellent (reads from both, writes to mirror pair)
Storage efficiency: 50%
Use case: High-performance databases
```

### RAID Comparison Table

| Level | Min Disks | Fault Tolerance | Read Perf | Write Perf | Storage Efficiency |
|---|---|---|---|---|---|
| RAID 0 | 2 | None | N× | N× | 100% |
| RAID 1 | 2 | 1 disk | 2× | 1× | 50% |
| RAID 5 | 3 | 1 disk | Good | Good | (N-1)/N |
| RAID 6 | 4 | 2 disks | Good | Moderate | (N-2)/N |
| RAID 10 | 4 | 1 per pair | N/2× | N/2× | 50% |

---

## 17. Storage Scalability

### Vertical Scaling (Scale Up)

Add more capacity to existing storage node:
- Larger disks
- More disks in the same server
- Faster drives (HDD → SSD → NVMe)

```
Limits:
  - Physical server rack space
  - Controller bandwidth
  - Cost (NVMe > SSD > HDD per GB)
  - Single point of failure
  
Use case: Database servers where you want more fast storage
```

### Horizontal Scaling (Scale Out)

Add more storage nodes to the cluster.

```
Distributed storage options:
  
1. Shared-Nothing Architecture (Cassandra, HDFS):
   Each node owns its data
   No shared storage layer
   Add nodes → add capacity + throughput
   
2. Shared Storage with Compute Separation (S3 + Athena, Snowflake):
   Storage tier: unlimited object storage
   Compute tier: query engines that scale independently
   
3. Distributed Block Storage (Ceph, GlusterFS):
   Block devices distributed across many nodes
   Client mounts as single block device
```

### Compute-Storage Separation

Modern trend in cloud data warehouses:

```
Traditional (Coupled):
  [Node 1: CPU + Storage]
  [Node 2: CPU + Storage]
  Scale storage → must scale compute too (waste)

Separated (Snowflake, BigQuery):
  Storage: S3 / GCS (unlimited, cheap)
  Compute: Virtual warehouse clusters (spin up/down in seconds)
  
  Scale storage → just store more in S3
  Scale compute → add warehouse nodes for more query parallelism
  Cost: pay for compute only while querying
```

---

## 18. NVMe over Fabrics

### Traditional Storage Network Bottleneck

```
Server CPU ──PCIe──► Local NVMe (0.1ms latency, 7GB/s bandwidth)
Server CPU ──NIC──► iSCSI / Fibre Channel → SAN (1-10ms latency, limited bandwidth)
```

NVMe drives are so fast that network becomes the bottleneck with traditional protocols.

### NVMe-oF (NVMe over Fabrics)

Extends NVMe protocol over high-speed networks to access remote NVMe drives at near-local performance.

```
Fabric Types:
  RDMA over Converged Ethernet (RoCE): 25-400Gbps, < 5μs latency
  InfiniBand: 200Gbps, ~1μs latency
  Fibre Channel: 64Gbps

Performance:
  Local NVMe: 100μs latency
  NVMe-oF/RDMA: 200-300μs latency (2-3× overhead)
  vs iSCSI: 1-10ms latency (10-100× NVMe-oF)
```

### Use Cases

- **Disaggregated storage:** Separate compute nodes from storage nodes; storage pool shared
- **GPU clusters (AI/ML):** NVMe-oF for high-speed dataset access from shared storage
- **Financial trading:** Ultra-low latency storage for market data
- **High-performance computing:** Parallel filesystem access

### AWS Equivalent

AWS EBS uses a similar concept — network-attached block storage optimized for low latency:
- EBS gp3: 1ms latency, 16,000 IOPS, 1000 MB/s throughput
- EBS io2 Block Express: Sub-millisecond latency, 256,000 IOPS

---

## 19. Quick Reference

### Storage Type Decision Matrix

| Requirement | Storage Type | Example |
|---|---|---|
| Database (RDBMS) | Block | EBS + PostgreSQL |
| Images / Videos / Files | Object | S3 |
| Shared filesystem | File | EFS, NFS |
| ML datasets | Object | S3 + Parquet |
| Logs archival | Object (cold tier) | S3 Glacier |
| In-memory cache | In-memory | Redis, ElastiCache |
| Data warehouse | Columnar + Object | Snowflake, Redshift |
| Container scratch | Local block (ephemeral) | Instance store |
| Shared config | File or Object | EFS or S3 |

### Erasure Coding vs Replication Trade-offs

| Factor | Replication (3×) | Erasure Coding (6+3) |
|---|---|---|
| Storage overhead | 3× | 1.5× |
| Read performance | Fast (no decode needed) | Slower (decode on degraded read) |
| Write performance | Moderate | Slower (encoding computation) |
| Fault tolerance | 2 failures | 3 failures (any 3 of 9) |
| Recovery speed | Fast (copy from replica) | Slower (reconstruct from chunks) |
| CPU cost | Low | High (encoding/decoding) |
| Best for | Hot data, latency-sensitive | Cold data, cost-sensitive |

### Storage Systems Interview Cheat Sheet

1. **Block vs Object vs File:** Block=DB volumes; Object=media/backups; File=shared filesystem
2. **S3 multipart upload:** Parallel chunks > 5MB; retry failed parts; max 5TB per object
3. **Presigned URLs:** Temporary client direct upload to S3; no proxying through server
4. **HDFS NameNode:** Metadata only (in memory); DataNodes store blocks; rack-aware replication
5. **Content-addressable storage:** Hash = address; deduplication implicit; used in Git, IPFS
6. **Rabin fingerprinting:** Variable-length chunking; boundary based on content hash, not position
7. **LSM vs B-Tree:** LSM=write-heavy (Cassandra); B-Tree=read-heavy (PostgreSQL)
8. **WAL:** Sequential write before data page update; enables crash recovery, replication, CDC
9. **Leveled compaction:** Non-overlapping key ranges per level; better reads; higher write amp
10. **RAID 10:** Mirror + stripe; best performance and redundancy for databases
11. **RPO/RTO:** RPO=max data loss (hours→zero); RTO=max downtime (days→zero)
12. **3-2-1 rule:** 3 copies, 2 media, 1 offsite
13. **Columnar storage:** Read only needed columns; compress well; best for OLAP aggregations
14. **Data tiering:** Hot=SSD/Redis; Warm=HDD/S3-IA; Cold=Glacier; Frozen=Deep Archive
15. **Erasure coding:** Lower storage overhead than replication; CPU cost; S3/HDFS use it
```
