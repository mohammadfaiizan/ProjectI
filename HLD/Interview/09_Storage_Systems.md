# Storage Systems — HLD Interview Q&A

---

## Easy (Q1–Q7)

---

### Q1. What is the difference between block storage, object storage, and file storage? When do you use each?

**Answer:**

Three fundamentally different paradigms for storing data, each optimized for specific access patterns.

**Block Storage**
Data divided into fixed-size blocks. The OS manages the filesystem on top. Like a raw hard drive.

```
Physical disk divided into:
  Block 0: [4KB raw data]
  Block 1: [4KB raw data]
  Block 2: [4KB raw data]
  ...
OS filesystem maps files to blocks; direct random I/O access.
```

- Examples: AWS EBS, Azure Disk, GCP Persistent Disk, SAN (Storage Area Network).
- Access: Attached to a single machine; accessed via OS file system calls.
- Use cases: Database storage (MySQL, PostgreSQL), OS volumes, applications needing low-latency random read/write (< 1ms).
- Pros: Low latency, high IOPS, random access.
- Cons: Expensive, attached to one instance, doesn't scale globally.

**File Storage (NAS — Network Attached Storage)**
Files organized in a hierarchical directory structure. Shared across multiple machines via NFS or SMB/CIFS protocols.

```
/mnt/shared-drive/
  /reports/
    report_2026.pdf
    report_2025.pdf
  /configs/
    app.config
```

- Examples: AWS EFS, Azure Files, NFS, GlusterFS.
- Access: Multiple machines mount the same network filesystem.
- Use cases: Shared configuration files, home directories, legacy apps expecting POSIX filesystem, CMS media.
- Pros: Shared access, POSIX semantics, familiar filesystem interface.
- Cons: Higher latency than block, not infinitely scalable.

**Object Storage**
Data stored as discrete objects (files) with a flat namespace (no directory hierarchy), accessed via HTTP API (PUT/GET/DELETE). Each object has an ID, data, and metadata.

```
Bucket: my-photos
  Key: user-123/photo-001.jpg  -> [blob data] + {size: 4MB, content-type: image/jpeg, ...}
  Key: user-456/avatar.png     -> [blob data] + {size: 200KB, ...}
  (No real directory hierarchy — keys just look like paths)
```

- Examples: AWS S3, GCP Cloud Storage, Azure Blob, MinIO.
- Access: HTTP REST API. No mounting required. Globally accessible.
- Use cases: Images, videos, backups, logs, static web assets, data lakes, ML training data.
- Pros: Infinitely scalable, cheap, durable (11 nines on S3), HTTP access anywhere.
- Cons: High latency (~10-100ms per request), no random write/update (must replace entire object), no POSIX semantics.

| Dimension | Block | File | Object |
|-----------|-------|------|--------|
| Latency | < 1ms | 1-10ms | 10-100ms |
| Scalability | Limited | Limited | Infinite |
| Access method | Block I/O | POSIX/NFS | HTTP API |
| Sharing | Single machine | Multiple machines | Global |
| Use case | DB, OS volumes | Shared files | Media, backups |
| Cost | High | Medium | Low |

---

### Q2. How does Amazon S3 (object storage) work internally?

**Answer:**

S3 is Amazon's massively scalable object storage service. While AWS doesn't publish full implementation details, the architecture is well-understood from academic papers, patents, and Jeff Barr's blog posts.

**High-level architecture:**
```
Client
  |
  v
[S3 Front End (Request Router)]
  - Authenticates request (SigV4)
  - Routes to correct index layer based on bucket/key
  |
  v
[Index Layer (Metadata service)]
  - Key -> physical location mapping
  - Stores: object metadata, ACLs, versioning info
  - Distributed key-value store (hash ring based)
  |
  v
[Storage Layer]
  - Actual object bytes
  - Distributed across many storage nodes
  - Each object replicated across multiple AZs
  - Chunk-based for large objects (multipart)
```

**Key design principles:**

**1. Flat namespace with hash-based routing:**
All objects in a bucket share a flat key namespace. Internally, keys are hashed to determine storage nodes. This is why sequential key prefixes (e.g., `2026-01-01-log`, `2026-01-02-log`) used to cause hotspots — they all hash to the same prefix and hit the same storage partitions. AWS now handles this with automatic key partitioning, but randomizing prefixes is still best practice for very high throughput.

**2. Eventual consistency (historically) → Strong consistency (2020):**
As of December 2020, S3 provides strong read-after-write consistency for all operations (including overwrite PUTs and DELETEs) without any additional configuration.

**3. Durability via erasure coding:**
S3 Standard achieves 11 nines (99.999999999%) durability by:
- Replicating data across at least 3 Availability Zones.
- Using erasure coding for storage efficiency (discussed in Q6).

**4. Multipart upload:**
For objects > 100MB, use multipart upload:
```
1. Initiate multipart upload -> get upload_id
2. Upload part 1 (5MB min per part): PUT /key?partNumber=1&uploadId=xxx
3. Upload part 2, 3, ..., N in parallel
4. Complete multipart upload with ETags -> S3 assembles the object
```
Benefits: Parallel upload, resume failed parts, no re-upload of entire object on failure.

**5. Storage classes (tiering):**
S3 Standard -> S3-IA (Infrequent Access) -> S3 Glacier -> S3 Glacier Deep Archive
Lifecycle policies automatically transition objects based on age.

---

### Q3. What is the HDFS architecture, and how does data replication work?

**Answer:**

**HDFS (Hadoop Distributed File System)** is designed for large-scale batch processing of very large files (GBs to TBs). It optimizes for sequential reads and high throughput over random access and low latency.

**Architecture:**
```
            [NameNode]
            (Metadata)
           /     |      \
          /      |       \
    [DataNode1][DataNode2][DataNode3]
    [Block A1] [Block A2] [Block A3]  <- Replicas of same block
    [Block B1] [Block B3]
    [Block C2] [Block C1] [Block C3]
```

**NameNode (Master):**
- Stores all filesystem metadata: file-to-block mapping, block-to-DataNode mapping, directory structure, permissions.
- Runs entirely in memory for fast lookups.
- **Single point of failure** (mitigated by High Availability mode with standby NameNode).
- Does NOT store actual data.

**DataNode (Worker):**
- Stores actual data blocks (default 128MB per block).
- Sends heartbeats to NameNode every 3 seconds.
- Reports block inventory to NameNode on startup.
- Handles read/write requests from clients directly (after getting block locations from NameNode).

**Data replication process (write):**
```
Client writes file.txt (500MB) to HDFS:

1. Client -> NameNode: "I want to write file.txt"
2. NameNode: "Divide into 4 blocks. Replicate to these DataNodes:"
   Block 1 -> DN1, DN2, DN3
   Block 2 -> DN2, DN3, DN1
   Block 3 -> DN3, DN1, DN2
   Block 4 -> DN1, DN3, DN2

3. Client writes Block 1 directly to DN1 (pipeline):
   Client -> DN1 -> DN2 -> DN3 (daisy-chain replication)
   Each node receives block and forwards to next in pipeline.
   ACK flows back: DN3 -> DN2 -> DN1 -> Client

4. Repeat for Blocks 2, 3, 4.
5. Client -> NameNode: "All blocks written successfully"
```

**Read process:**
```
Client reads file.txt:
1. Client -> NameNode: "Where are the blocks for file.txt?"
2. NameNode returns: block locations with rack topology (prefer same rack/AZ)
3. Client reads directly from nearest DataNode for each block in parallel
```

**Failure handling:**
- DataNode failure: NameNode detects missing heartbeat, initiates re-replication of that node's blocks to other DataNodes.
- NameNode failure: Standby NameNode (shared edit log on NFS/QJM) takes over. Automated failover using ZooKeeper.

---

### Q4. What is an LSM tree, and how does it differ from a B-tree?

**Answer:**

These are the two dominant data structures for database storage engines, each optimized for different workloads.

**B-Tree (Balanced Tree):**
The standard for traditional RDBMS (MySQL InnoDB, PostgreSQL). Data is stored in a balanced tree of fixed-size pages.

```
         [50]
        /     \
    [25]       [75]
   /    \      /   \
[10,20] [30,40] [60,70] [80,90]

Read: Traverse tree to find value -> O(log N) page reads
Write: Find page, insert in-place -> may trigger page splits and rewrites
```

- Reads: O(log N) — very efficient.
- Writes: Random I/O (update in-place) — each write may read a page, modify, write back.
- Write amplification: 1 logical write → several disk I/O operations (read page + write page).
- Good for: Read-heavy workloads, point lookups, systems where reads >> writes.

**LSM Tree (Log-Structured Merge Tree):**
Used in Cassandra, RocksDB, LevelDB, HBase. Optimizes for write throughput.

```
Write path:
  [Write] -> [WAL (durability)] -> [MemTable (in-memory, sorted)] 
  When MemTable full -> flush to [SSTable on disk (immutable, sorted)]
  
  SSTables accumulate on disk:
    L0: [SST1, SST2, SST3]  <- recently flushed, may overlap
    L1: [SST4, SST5, SST6]  <- compacted, no overlap within level
    L2: [SST7, ...]          <- larger, compacted
```

**Read path in LSM:**
```
Read key K:
  1. Check MemTable (in-memory) -> O(1)
  2. Check recent SSTables (L0) -> binary search each (may check multiple, overlap exists)
  3. Check L1 SSTables (no overlap, binary search to find right file)
  4. Check deeper levels if not found
  
Bloom filter per SSTable: quickly skip SSTables that definitely don't contain K
```

**Comparison:**

| Property | B-Tree | LSM Tree |
|----------|--------|---------|
| Write performance | Medium (random I/O) | High (sequential append) |
| Read performance | High (in-place) | Medium (check multiple levels) |
| Write amplification | Low-medium | High (compaction rewrites) |
| Read amplification | Low | Medium-high (multi-level check) |
| Space amplification | Low | Medium (stale data until compaction) |
| Use case | Read-heavy, OLTP | Write-heavy, time-series, logs |
| Examples | MySQL, PostgreSQL | Cassandra, RocksDB, HBase |

**Key insight:** LSM trees turn random writes into sequential writes (append to WAL + MemTable) which are much faster on both SSDs and HDDs. The cost is paid during compaction (background merging of SSTables).

---

### Q5. What is a Write-Ahead Log (WAL), and why is it important?

**Answer:**

A **Write-Ahead Log (WAL)** is a sequential, append-only log that records all database changes *before* they are applied to the main data structures (B-tree pages, MemTable flush, etc.). It provides durability and enables crash recovery.

**The durability problem:**
A database write typically modifies in-memory structures and eventually flushes to disk. If the process crashes between the write and the flush, the data is lost. Updating data in-place on disk is also dangerous — if a crash occurs mid-write, the page is partially written (torn write).

**WAL solution:**
```
Write request: "UPDATE accounts SET balance = 500 WHERE id = 1"

Step 1: Append to WAL (sequential write — fast):
  WAL entry: {LSN: 1042, TXN: T1, type: UPDATE, table: accounts, 
               key: 1, old_val: 400, new_val: 500, checksum: abc}
  fsync() -> guaranteed on disk

Step 2: Update in-memory buffer page (fast)

Step 3: Return success to client (WAL is on disk = durability guaranteed)

Step 4 (async): Checkpoint — flush dirty buffer pages to disk
```

**Crash recovery:**
```
System crashes at step 2 or 3:
  On restart: replay WAL from last checkpoint
  WAL entry shows: accounts.id=1 should be 500
  Apply change to data files
  Data is recovered
```

**WAL enables:**
1. **Durability (D in ACID):** Once WAL entry is fsynced, data is durable.
2. **Crash recovery:** Replay WAL to restore consistent state.
3. **Replication:** Send WAL entries to replicas (PostgreSQL WAL streaming, MySQL binlog).
4. **Change Data Capture (CDC):** Debezium reads WAL/binlog to detect changes.
5. **PITR (Point-in-Time Recovery):** Replay WAL up to any point in time.

**PostgreSQL WAL:**
```
pg_wal/
  000000010000000000000001  <- WAL segment files (16MB each by default)
  000000010000000000000002
  ...

pg_basebackup -> creates a base backup
Apply WAL segments from backup LSN to recovery target -> PITR
```

**Performance consideration:**
`fsync()` on every WAL write ensures durability but limits throughput. Some databases offer `fsync=off` (unsafe) or group commit (batch multiple transactions into one fsync) to improve throughput while maintaining safety.

---

### Q6. What is erasure coding, and how does it compare to replication?

**Answer:**

Both erasure coding and replication achieve data durability — the ability to recover data when some storage nodes fail. They have different space and performance trade-offs.

**Replication:**
Store N complete copies of the data. Survive any N-1 node failures.

```
Data: [Block A] (1MB)
3x Replication:
  Node 1: [Block A] (1MB)
  Node 2: [Block A] (1MB)
  Node 3: [Block A] (1MB)
Total storage: 3MB for 1MB data -> 3x overhead
```

- Read: Read from any single node — fast.
- Write: Write to all N nodes — parallel, but N times the bandwidth.
- Simple to understand and implement.
- High storage overhead (2x, 3x, 6x depending on RF).

**Erasure Coding (Reed-Solomon):**
Data is split into k data chunks and n parity chunks. Can recover from any n failures. Common: (6, 3) — 6 data chunks, 3 parity chunks.

```
Data: [10MB file]
Reed-Solomon (6,3):
  Divide into 6 data chunks: [D1][D2][D3][D4][D5][D6]  (each ~1.67MB)
  Compute 3 parity chunks:   [P1][P2][P3]              (each ~1.67MB)
  Total storage: 9 chunks * 1.67MB = 15MB for 10MB data -> 1.5x overhead

Store each chunk on a different node (9 nodes total).
Any 3 nodes can fail -> reconstruct all data from remaining 6 chunks.
```

**Mathematical basis:**
Reed-Solomon codes treat data as polynomials over finite fields (Galois Field). Given any 6 of the 9 chunks, the original polynomial can be reconstructed — therefore the original data can be recovered.

**Comparison:**

| Aspect | 3x Replication | Erasure Coding (6+3) |
|--------|----------------|----------------------|
| Storage overhead | 3x (200%) | 1.5x (50%) |
| Durability (fault tolerance) | Survives 2 node failures | Survives 3 node failures |
| Read performance | Fast (read from single node) | Slow (may need to reconstruct) |
| Write performance | Simple writes | Compute parity (CPU overhead) |
| Latency | Low | Higher (reconstruction) |
| CPU overhead | Low | High |

**When to use each:**

- **Replication:** Hot data, latency-sensitive reads, databases (OLTP), Kafka.
- **Erasure coding:** Cold/warm storage, large files, archival data, S3 Standard, HDFS for cold data.

**S3 uses erasure coding:** S3 Standard splits objects and uses Reed-Solomon-like erasure coding across Availability Zones. This is how it achieves 11 nines durability at ~1.5x storage cost rather than 3x.

---

### Q7. What is data tiering, and how do lifecycle policies manage it?

**Answer:**

**Data tiering** is the practice of storing data on different types of storage media based on access frequency, with hot (frequently accessed) data on fast expensive storage and cold (rarely accessed) data on slow cheap storage.

**Storage tiers:**
```
HOT (Active/Hot):
  - Access: Daily/hourly
  - Storage: SSD, NVMe, Redis, S3 Standard
  - Latency: Milliseconds
  - Cost: High (~$0.023/GB/month for S3 Standard)
  
WARM (Infrequent):
  - Access: Weekly/monthly
  - Storage: HDD, S3 Standard-IA, S3 One Zone-IA
  - Latency: Milliseconds (S3 IA has retrieval fee)
  - Cost: Medium (~$0.0125/GB/month for S3-IA)

COLD (Archive):
  - Access: Quarterly/yearly
  - Storage: S3 Glacier, S3 Glacier Flexible Retrieval
  - Latency: Minutes to hours for retrieval
  - Cost: Low (~$0.004/GB/month)

FROZEN (Deep Archive):
  - Access: Rarely (compliance, legal)
  - Storage: S3 Glacier Deep Archive, tape
  - Latency: Up to 12 hours
  - Cost: Very low (~$0.00099/GB/month)
```

**S3 Lifecycle Policy example:**
```json
{
  "Rules": [{
    "ID": "archive-old-logs",
    "Filter": {"Prefix": "logs/"},
    "Status": "Enabled",
    "Transitions": [
      {
        "Days": 30,
        "StorageClass": "STANDARD_IA"    // After 30 days -> IA
      },
      {
        "Days": 90,
        "StorageClass": "GLACIER"         // After 90 days -> Glacier
      },
      {
        "Days": 365,
        "StorageClass": "DEEP_ARCHIVE"    // After 1 year -> Deep Archive
      }
    ],
    "Expiration": {
      "Days": 2555                         // Delete after 7 years
    }
  }]
}
```

**Database tiering (PostgreSQL partitioning):**
```sql
-- Partition table by date
CREATE TABLE orders (
    id BIGINT, created_at TIMESTAMPTZ, ...
) PARTITION BY RANGE (created_at);

-- Hot: current month on SSD tablespace
CREATE TABLE orders_2026_05 PARTITION OF orders
    FOR VALUES FROM ('2026-05-01') TO ('2026-06-01')
    TABLESPACE ssd_tablespace;

-- Cold: old months on HDD tablespace
CREATE TABLE orders_2025 PARTITION OF orders
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01')
    TABLESPACE hdd_tablespace;
```

**Cost benefit example:**
An analytics platform storing 100TB of data:
- All on S3 Standard: $2,300/month
- Tiered (10TB hot, 30TB warm, 60TB cold): $650/month (~72% savings)

---

## Medium (Q8–Q15)

---

### Q8. What is the difference between a data lake, data warehouse, and data lakehouse?

**Answer:**

These are architectural patterns for storing and analyzing large-scale data, each designed for different use cases and trade-offs between flexibility and performance.

**Data Warehouse:**
Highly structured, schema-on-write store optimized for SQL analytics. Data is cleaned and transformed (ETL) before loading.

```
Sources -> [ETL Pipeline] -> [Data Warehouse (Redshift/BigQuery/Snowflake)]
              (transform)        (structured, schema enforced, columnar)
              
Schema: predefined star/snowflake schema
Query: SQL, fast aggregations
Users: BI analysts, dashboards
```

Pros: Fast queries, enforced data quality, familiar SQL interface.
Cons: Expensive, schema changes are costly, raw data not accessible, not suitable for ML training.

**Data Lake:**
Raw, schema-on-read store accepting any format (JSON, CSV, Parquet, images, video). No ETL before storage.

```
Sources -> [Data Lake (S3/GCS/ADLS)] (raw formats: JSON, CSV, Parquet, images)
              (no transformation)
              
Schema: schema-on-read (inferred at query time)
Query: Spark, Presto, Athena
Users: Data scientists, ML engineers
```

Pros: Cheap storage, flexible, supports unstructured data, great for ML.
Cons: Can become a "data swamp" (poor governance), slow queries on raw formats, no ACID transactions.

**Data Lakehouse (Delta Lake / Apache Iceberg / Apache Hudi):**
Combines the flexibility and low cost of data lakes with the transactional guarantees and performance of data warehouses.

```
Sources -> [Data Lakehouse (Delta Lake on S3)]
            (open table format: Parquet + metadata layer)
            
Features:
  - ACID transactions on S3/GCS (append, update, delete)
  - Schema enforcement + evolution
  - Time travel (query data as of yesterday)
  - Partitioning, Z-ordering for fast queries
  - Both SQL (via Spark/Presto) and DataFrame APIs
  - Unified batch and streaming
```

**Comparison:**

| Property | Data Warehouse | Data Lake | Data Lakehouse |
|----------|---------------|-----------|----------------|
| Schema | Schema-on-write | Schema-on-read | Both (enforced or flexible) |
| Formats | Proprietary columnar | Any | Open (Parquet + metadata) |
| ACID | Yes | No | Yes (Delta/Iceberg) |
| Cost | High | Low | Low |
| Query speed | Fast | Slow | Fast (with optimizations) |
| Raw data | No | Yes | Yes |
| ML support | Limited | Yes | Yes |
| Examples | Redshift, BigQuery | S3 + Athena | Databricks, Delta Lake |

**Modern architecture (Medallion/Lambda on Lakehouse):**
```
Bronze layer: Raw data (exact copy of source) -> Data Lake
Silver layer: Cleaned, joined, normalized data -> Delta Lake tables
Gold layer: Business aggregates, ML features -> Fast query layer
```

---

### Q9. What is star schema vs snowflake schema in data warehousing?

**Answer:**

These are dimensional modeling techniques for structuring data warehouse tables to optimize analytics queries.

**Star Schema:**
One central **fact table** (events/transactions) surrounded by denormalized **dimension tables**. Named for its star-like shape.

```
            [Date Dim]
                |
[Product Dim] - [Sales Fact] - [Customer Dim]
                |
           [Store Dim]

Sales Fact table:
  date_id FK, product_id FK, customer_id FK, store_id FK,
  quantity, revenue, discount

Product Dim (denormalized — no further normalization):
  product_id PK, product_name, category, subcategory,
  brand, brand_country, brand_founded_year
  (brand_country/founded_year is redundant — repeated for each product of same brand)
```

Pros: Fewer joins (dimension is denormalized, all attributes in one table), faster queries, simpler SQL.
Cons: Data redundancy, dimension table updates require updating all denormalized records.

**Snowflake Schema:**
Dimension tables are normalized — split into multiple related tables.

```
[Sales Fact] - [Product Dim] - [Brand Dim] - [Country Dim]
                              |
                         [Category Dim]
```

Pros: Less data redundancy, smaller storage, easier dimension updates.
Cons: More joins required in queries, slower performance, more complex SQL.

**Comparison:**

| Aspect | Star Schema | Snowflake Schema |
|--------|------------|-----------------|
| Joins | Fewer | More |
| Query speed | Faster | Slower |
| Storage | More (duplication) | Less |
| Complexity | Simple | More complex |
| Update anomalies | Present | Fewer |
| Best for | Query performance (BI) | Storage optimization |

**Practical recommendation:** In modern columnar databases (BigQuery, Redshift, Snowflake), storage cost is low and query engines handle joins efficiently. Star schema is preferred for simplicity and query performance. Snowflake schema is rarely used in modern data warehouses.

**Fact table types:**
- **Transaction facts:** One row per event (order placed, page view).
- **Periodic snapshot:** State at regular intervals (daily balance).
- **Accumulating snapshot:** One row per lifecycle with multiple date columns (order created, shipped, delivered).

---

### Q10. What are the compaction strategies in LSM trees?

**Answer:**

**Compaction** is the process of merging multiple SSTables into fewer, larger SSTables. It's essential to reclaim space from deleted/overwritten records and to improve read performance (fewer files to check).

**Why compaction is needed:**
```
After many writes, Level 0 might have:
  SST1: {a:1, b:2, d:4}
  SST2: {b:3, c:5}        <- b has been updated (SST2 is newer)
  SST3: {a: DELETED}      <- a has been deleted
  SST4: {c:6, d:4}        <- c updated
  
Without compaction: reading key "a" checks all 4 files
With compaction: merged into one SST {b:3, c:6, d:4} (a removed, stale values gone)
```

**Strategy 1: Size-Tiered Compaction (STCS)**
When a level accumulates N SSTables of similar size, merge them into one larger SSTable.

```
4 SSTables of 1MB each -> compact into 1 SSTable of ~4MB
4 SSTables of 4MB each -> compact into 1 SSTable of ~16MB
4 SSTables of 16MB each -> ...
```

Pros: Writes are fast (compact less frequently), good for write-heavy workloads.
Cons: High space amplification (during compaction, old + new files coexist: up to 2x temp space). Reads slower (many overlapping SSTables at each level). Used in Cassandra (default), HBase.

**Strategy 2: Leveled Compaction (LCS)**
Fixed number of levels (L0, L1, L2...). Each level (except L0) has non-overlapping SSTables. SSTables are smaller and evenly distributed.

```
L0: 4 SSTables (newest, may overlap) -> trigger compaction
L1: Many small non-overlapping SSTables (total ~10MB)
L2: Non-overlapping SSTables (total ~100MB, 10x L1)
L3: Non-overlapping (total ~1GB)

Compaction triggers: when L0 has ≥ 4 SSTables:
  Pick one L0 SSTable, find all overlapping L1 SSTables, merge them
  Result goes to L1 (maintains non-overlapping invariant)
```

Pros: Good read performance (few SSTables to check per level), low space amplification.
Cons: High write amplification (data is compacted multiple times as it moves through levels). Used in RocksDB (default), Cassandra (optional), LevelDB.

**Strategy 3: FIFO Compaction**
Oldest SSTables are dropped when total size exceeds a limit. No merging.
Used for: Time-series data where old data expires (IoT sensor readings with TTL).

**Comparison:**

| Strategy | Write Amplification | Read Amplification | Space Amplification | Best For |
|----------|--------------------|--------------------|---------------------|---------|
| STCS | Low | High | High | Write-heavy |
| LCS | High | Low | Low | Read-heavy |
| FIFO | Very Low | High | Low | Time-series with TTL |

**Write Amplification Factor (WAF):**
LCS can have WAF of 10-30x (data written to disk 10-30 times before reaching the deepest level). This is the fundamental trade-off for better read performance.

---

### Q11. What is blob storage design for media, and how do you handle large files?

**Answer:**

Media storage systems (images, videos, audio) require specialized design to handle large files, efficient delivery, and cost optimization.

**Key requirements:**
- Files ranging from KB (thumbnails) to GB (4K videos).
- High read:write ratio (written once, read millions of times).
- Global low-latency access.
- Range request support (video seeking).
- Deduplication to save storage costs.

**Architecture:**

```
Upload flow:
  Client -> [Upload Service] -> [S3/Blob Storage] -> [Processing Queue]
               |                                           |
           Returns         [Processing Workers]           |
         object_id         (thumbnail gen, transcoding) --+
                                    |
                             [Processed files back to S3]
                             [Metadata to DB (object_id -> S3 key)]

Read/Delivery flow:
  Client -> [CDN Edge] -> (cache hit) -> return file
                       -> (cache miss) -> [CDN Origin (S3/Blob)]
```

**Chunking large files:**
For files > 100MB (e.g., videos), use chunked upload:
```
1. Client: POST /upload/initiate -> {upload_id: "abc123", chunk_size: 5MB}
2. Client splits 500MB video into 100 chunks (5MB each)
3. Client uploads chunks in parallel:
   PUT /upload/abc123/chunk/0 (bytes 0-5MB)
   PUT /upload/abc123/chunk/1 (bytes 5-10MB)
   ...
4. Server assembles chunks after all received (or uses S3 multipart)
5. Client: POST /upload/abc123/complete
```

**Range requests for video seeking:**
```
Client: GET /video/movie.mp4
        Range: bytes=104857600-209715199  (seeking to minute 5 in a 2GB video)

Server response:
  HTTP/1.1 206 Partial Content
  Content-Range: bytes 104857600-209715199/2147483648
  [5MB of video data]

S3 and CDNs natively support HTTP Range requests.
```

**Content deduplication:**
```
Before storing, compute hash of file content (SHA-256):
  hash("cat.jpg") = "abc123..."
  
  If "abc123" already exists in storage: don't upload again, just create a new reference.
  Store in metadata: {object_id: "uuid", content_hash: "abc123", s3_key: "content/abc123"}
  
  Result: Even if 1000 users upload the same image, only one copy is stored.
```

**CDN integration:**
```
S3 Key: /videos/original/movie-uuid.mp4
CDN URL: https://cdn.example.com/videos/original/movie-uuid.mp4

CDN caches at edge PoPs globally.
Cache-Control: max-age=31536000, immutable  (content-addressed = never changes)
Signed URLs for private content:
  aws s3 presign s3://bucket/private/file.mp4 --expires-in 3600
```

**Storage cost optimization:**
- Store only one resolution during upload; generate thumbnails on first access (lazy processing).
- Transcode to multiple bitrates (360p, 720p, 1080p) for adaptive streaming (HLS/DASH).
- Use S3-IA or Glacier for unpopular videos (long tail).

---

### Q12. What is the 3-2-1 backup rule, and how do RPO/RTO guide DR design?

**Answer:**

**3-2-1 Backup Rule:**
A widely-adopted best practice for data protection:
- **3** copies of data (1 primary + 2 backups)
- **2** different storage media (e.g., local SSD + cloud object storage)
- **1** offsite copy (different physical location or cloud region)

```
Your data -> [Primary DB on SSD (copy 1)]
          -> [Backup on local NAS / tape (copy 2, different media)]
          -> [Backup in S3 different region (copy 3, offsite)]
```

Modern extension: **3-2-1-1-0**
- +1: One copy offline or immutable (air-gapped, protects against ransomware).
- +0: Zero errors on backups (test restores regularly).

**RPO and RTO:**

**RPO (Recovery Point Objective):** How much data can you afford to lose? Maximum acceptable age of files that must be recovered.
- RPO = 0: No data loss tolerated (synchronous replication, hot standby).
- RPO = 1 hour: Up to 1 hour of transactions may be lost.
- RPO = 24 hours: Daily backups acceptable.

**RTO (Recovery Time Objective):** How long can the system be down? Maximum acceptable downtime.
- RTO = 0: Zero downtime (active-active multi-region).
- RTO = 15 minutes: Automated failover to standby.
- RTO = 4 hours: Manual restore from backup acceptable.

**DR architecture options vs RPO/RTO cost:**

```
RPO/RTO = 0:        Active-Active (most expensive)
  [Region US] <-sync replication-> [Region EU]
  Both serve traffic. Instant failover.

RPO < 1min, RTO < 5min:  Warm Standby
  [Primary] --async replication--> [Standby] (running but not serving traffic)
  Failover: promote standby, update DNS

RPO < 1hr, RTO < 1hr:   Pilot Light
  [Primary] -> [S3 snapshots + WAL] -> [Standby (minimal, scale out on failover)]
  
RPO = 24hr, RTO = 24hr:  Backup + Restore
  [Primary] -> [Daily backup to S3]
  On disaster: spin up new DB, restore from backup
```

**Cost vs RPO/RTO tradeoff:**

| Strategy | RPO | RTO | Relative Cost |
|----------|-----|-----|--------------|
| Active-Active | ~0 | ~0 | 10x |
| Warm Standby | < 1 min | < 10 min | 3x |
| Pilot Light | < 1 hr | < 1 hr | 1.5x |
| Backup/Restore | < 24 hr | Hours | 1x |

**Backup testing:** Backups without tested restores are not backups. Automate a quarterly full restore drill. Netflix "Chaos Kong" includes DR testing at the region level.

---

### Q13. How does columnar storage (Parquet/ORC) enable fast analytics?

**Answer:**

**Row-oriented storage** (traditional RDBMS like PostgreSQL) stores all columns of a row together on disk. **Columnar storage** (Parquet, ORC) stores all values of a column together.

**Row vs columnar layout:**

```
Table: orders (id, customer, amount, status, created_at)

ROW-ORIENTED (layout on disk):
  [1, Alice, 99.99, paid, 2026-01-01]
  [2, Bob, 149.99, pending, 2026-01-02]
  [3, Alice, 29.99, paid, 2026-01-03]
  All columns read even if you only need 2.

COLUMNAR (Parquet layout on disk):
  Column 0 (id):         [1, 2, 3, ...]
  Column 1 (customer):   [Alice, Bob, Alice, ...]
  Column 2 (amount):     [99.99, 149.99, 29.99, ...]
  Column 3 (status):     [paid, pending, paid, ...]
  Column 4 (created_at): [2026-01-01, 2026-01-02, ...]
  
  Read only "amount" and "status" -> only 2 columns read from disk!
```

**Why columnar is faster for analytics:**

**1. Column projection (skip irrelevant columns):**
```sql
SELECT SUM(amount) FROM orders WHERE status = 'paid';
-- Row store: reads id, customer, amount, status, created_at (5 columns)
-- Column store: reads only amount and status (2 of 5 columns)
-- 60% less data read from disk!
```

**2. Compression (same-type data compresses better):**
```
Column: status -> [paid, paid, pending, paid, paid, paid, pending, ...]
Run-length encoding: paid(6), pending(1), paid(5)... -> 10x compression

Column: amount -> [99.99, 149.99, 29.99, ...] -> delta encoding + snappy
```
Analytics queries often read 1% of columns but 100% of rows → column store reads 100x less data.

**3. Vectorized execution (SIMD):**
Columnar data allows query engines to process 256 values of the same type simultaneously using SIMD CPU instructions (AVX-256/AVX-512). Row-oriented data can't do this efficiently.

**4. Predicate pushdown:**
Parquet stores min/max statistics per row group and per column chunk. The query engine skips entire row groups where min/max proves no matching rows exist.
```
Query: WHERE amount > 1000
Parquet row group 0: min=10, max=500 -> skip (no values > 1000)
Parquet row group 1: min=200, max=2000 -> read this group
```

**Parquet file structure:**
```
[Row Group 1] (128MB)
  Column Chunk: id     [dictionary + encoded values + statistics]
  Column Chunk: amount [encoded values + min/max stats]
  ...
[Row Group 2] (128MB)
  ...
[File Footer] -> schema, row group offsets, statistics
```

**When to use row vs columnar:**

| Operation | Row Store | Columnar |
|-----------|----------|---------|
| Single row lookup | Fast | Slow |
| Full table scan aggregation | Slow | Fast |
| Wide queries (many columns) | OK | Slow |
| Narrow queries (few columns) | Wasted IO | Fast |
| OLTP | Yes | No |
| OLAP/Analytics | No | Yes |

---

### Q14. How does RAID work, and when do you use RAID 0, 1, 5, and 10?

**Answer:**

**RAID (Redundant Array of Inexpensive Disks)** combines multiple physical disks to improve performance, capacity, or fault tolerance.

**RAID 0 — Striping (no redundancy):**
Data is split in stripes across all disks. Capacity = sum of all disks. No fault tolerance.

```
Data: [A1][A2][A3][A4]
  Disk 1: [A1][A3]
  Disk 2: [A2][A4]
  
Read/Write: 2x throughput (both disks work in parallel)
Failure: ONE disk fails -> all data LOST
```
Use: Scratch space for high-performance temporary storage (video editing, rendering). Never for production data.

**RAID 1 — Mirroring:**
Exact copy on each disk. Capacity = size of single disk (half efficiency).

```
Data: [A1][A2][A3][A4]
  Disk 1: [A1][A2][A3][A4]  (mirror)
  Disk 2: [A1][A2][A3][A4]  (mirror)
  
Failure: ONE disk fails -> no data loss (read from other disk)
Read: 2x throughput (read from either disk)
Write: Same speed (must write to both)
```
Use: OS boot drives, critical databases. Expensive (50% space efficiency).

**RAID 5 — Striping with distributed parity:**
Data and parity distributed across N disks. Capacity = (N-1) disks. Survives 1 disk failure.

```
3 disks:
  Disk 1: [A1][B1][C1][P_D]   P = parity for that stripe
  Disk 2: [A2][B2][P_C][D2]
  Disk 3: [A3][P_B][C3][D3]
  
  Parity is XOR of data blocks: P_A = A1 XOR A2
  
Failure: Disk 1 fails -> reconstruct A1 = A2 XOR P_A
Read: Good throughput (all disks serve reads)
Write: Slower (must compute and write parity)
Capacity: (N-1)/N = 2/3 = 67% efficiency with 3 disks
```
Use: File servers, NAS storage. Not ideal for databases (parity write penalty).

**RAID 10 — Stripe of mirrors (RAID 1+0):**
Mirror pairs of disks, then stripe across the pairs. Capacity = 50%.

```
4 disks:
  Mirror pair 1: Disk1 <-> Disk2 (Disk1 and Disk2 are mirrors)
  Mirror pair 2: Disk3 <-> Disk4 (Disk3 and Disk4 are mirrors)
  Stripes across pairs: [A1, A2] -> [Mirror Pair 1], [B1, B2] -> [Mirror Pair 2]
  
Failure: Any one disk in each mirror pair -> survive 2 disk failures (if in different pairs)
         Both disks in same pair -> data loss
```
Use: High-performance databases (MySQL InnoDB), applications needing both speed and redundancy.

| RAID Level | Fault Tolerance | Read Speed | Write Speed | Usable Capacity |
|------------|----------------|-----------|------------|-----------------|
| RAID 0 | None | Very Fast | Very Fast | 100% |
| RAID 1 | 1 disk failure | Fast | Same as 1 disk | 50% |
| RAID 5 | 1 disk failure | Fast | Moderate (parity) | (N-1)/N |
| RAID 10 | 1 per mirror pair | Very Fast | Fast | 50% |

**In cloud environments:** RAID is mostly abstracted away. Cloud block storage (EBS, GCP PD) handles replication internally. RAID is still relevant for on-premise and bare-metal deployments.

---

### Q15. What is NVMe, and why does it matter for storage system design?

**Answer:**

**NVMe (Non-Volatile Memory Express)** is a storage protocol designed specifically for SSDs, replacing SATA and SAS which were designed for spinning hard drives decades ago.

**The evolution:**
```
HDD (spinning disk):
  Physical limitations: 7200 RPM, seek time ~8ms, IOPS ~200
  
SATA SSD:
  Same SATA protocol as HDD, but flash memory underneath
  IOPS ~100,000 but SATA interface limits to 600 MB/s bandwidth
  
NVMe SSD (PCIe attached):
  Protocol designed for flash memory parallelism
  Direct PCIe bus connection (no SATA controller bottleneck)
  IOPS: 500,000 - 7,000,000+
  Bandwidth: 3 - 7 GB/s
  Latency: ~100 microseconds (vs ~100ms for HDD)
```

**Key NVMe advantages:**

| Metric | HDD | SATA SSD | NVMe SSD |
|--------|-----|----------|---------|
| Sequential Read | 200 MB/s | 550 MB/s | 7,000 MB/s |
| Sequential Write | 180 MB/s | 520 MB/s | 6,500 MB/s |
| Random Read IOPS | 200 | 100K | 1,000K+ |
| Latency | ~8ms | ~0.1ms | ~0.02ms |
| Queue Depth | 1 (HDD) | 32 (SATA) | 64,000 (NVMe) |

**Queue depth** is a critical difference:
- HDD/SATA: 1 queue, 32 commands max.
- NVMe: 65,535 queues, 65,535 commands per queue.
This allows massive parallelism matching flash memory's internal parallel architecture.

**Impact on storage system design:**

1. **Database design changes:** With NVMe, the bottleneck shifts from I/O to CPU and memory. Previously slow random I/O discouraged operations like non-covering index scans. NVMe makes these cheap.

2. **LSM tree compaction:** Compaction throughput is limited by I/O bandwidth. NVMe 5x-10x bandwidth allows faster compaction, reducing read amplification.

3. **Buffer pool sizing:** With NVMe latency approaching DRAM by ~50x (vs HDD by 50,000x), the importance of the database buffer pool as a caching layer is reduced. Can use smaller buffer pools.

4. **Direct I/O patterns:** NVMe exposes multiple namespaces, allowing databases to bypass OS filesystem for direct PCIe access (io_uring in Linux, SPDK library).

5. **Cloud NVMe instances:** AWS i3en, i4i instance families with local NVMe — used for ElasticSearch, Kafka (local disk for performance), high-IOPs databases where EBS latency (~500μs) is too high.

**When NVMe matters most:**
- OLTP databases with high random I/O.
- Kafka brokers with very high throughput needs.
- Real-time analytics requiring fast segment scans.
- Any system where I/O was previously the bottleneck.

---

## Hard (Q16–Q20)

---

### Q16. How do you design a distributed file system inspired by GFS (Google File System)?

**Answer:**

GFS (2003) introduced the foundational design for distributed file systems at scale. HDFS is directly inspired by it. Key design decisions reflect Google's workload: huge files (100MB+), append-heavy (MapReduce output), infrequent random writes, high throughput over low latency.

**Architecture:**
```
                    [GFS Master (metadata)]
                   /         |         \
                  /          |          \
          [Chunkserver 1] [Chunkserver 2] [Chunkserver 3]
          [Chunk A, B, C] [Chunk A, D]   [Chunk B, D, E]
```

**Master (single, but mirrored):**
- Stores file namespace (directory tree).
- Stores file → chunk handle mapping.
- Stores chunk → chunkserver locations (rebuilt from chunkserver reports on startup, not persisted).
- Manages lease grants for mutations.
- Runs chunk garbage collection, re-replication.

**Chunkserver:**
- Stores 64MB chunks (large to amortize metadata overhead).
- Each chunk identified by a globally unique 64-bit chunk handle.
- Each chunk replicated on 3 chunkservers (default).
- Reports chunk inventory to master on startup.

**Write (mutation) flow:**
```
Client wants to write to file /logs/access.log:

1. Client -> Master: "Give me chunk lease for chunk at offset X"
2. Master: grants lease to one chunkserver (Primary)
           returns: Primary=CS1, Secondaries=[CS2, CS3]

3. Client pushes data to all 3 chunkservers (data flows along a pipeline):
   Client -> CS1 -> CS2 -> CS3 (chained to use network bandwidth efficiently)
   Data stored in each server's buffer (not yet committed)

4. Client -> CS1 (Primary): "WRITE offset=Y, length=Z"
5. CS1 (Primary):
   - Assigns serial number to mutation
   - Applies mutation to own chunk
   - Forwards WRITE command (with serial number) to CS2, CS3
6. CS2, CS3 apply mutation in same serial order
7. CS1 responds to Client: SUCCESS (or collects errors from secondaries)
```

**Why push data first, then WRITE command?**
Decouples data flow from control flow. Data flows along network topology (client to nearest server, then chain). Control flows from client to primary. Allows full-duplex network utilization.

**Consistency model:**
GFS uses a relaxed consistency model:
- **Defined:** Mutation succeeds and all replicas are identical.
- **Inconsistent:** Concurrent writes may cause replicas to have different data (different fragments of different writes).
- Applications (like MapReduce) are designed to tolerate this (write once, verify with checksums).

**Failure handling:**
```
Chunkserver fails:
  Master detects missing heartbeat
  Identifies under-replicated chunks
  Re-replicates from remaining copies

Master fails:
  Shadow masters (read-only replicas) continue serving reads
  New master replays operation log + checkpoint
  Master state rebuilt from chunkserver reports

Data corruption:
  Each 64KB of data has a 32-bit checksum
  Chunkserver verifies checksum on every read
  Corrupted chunk reported to master, re-replicated from healthy copy
```

**Lessons applied to modern designs:**
- Large chunk size → fewer master requests, better sequential throughput.
- Append-optimized → relaxed write consistency, enables concurrent appenders.
- Metadata in RAM → fast operations (master never goes to disk for metadata).

---

### Q17. What is content-addressable storage (CAS), and how is it used?

**Answer:**

**Content-Addressable Storage** (CAS) uses the cryptographic hash of the content as its storage address. Instead of naming files by path or identifier, you name them by what they contain.

```
Traditional storage:
  Store "Hello World" at /files/greeting.txt
  Address: /files/greeting.txt (can contain anything)

Content-Addressable Storage:
  Store "Hello World" -> hash = sha256("Hello World") = "a591a6d..."
  Address: a591a6d...
  -> The address IS the content. Content is immutable by definition.
```

**Properties of CAS:**

1. **Automatic deduplication:** If two different clients store the same content, it maps to the same hash — only stored once.
```
User A uploads photo.jpg (sha256=abc123) -> stored as /cas/abc123
User B uploads same photo.jpg (sha256=abc123) -> already exists, just add reference
Storage: 1 copy, 2 references
```

2. **Integrity verification:** Download content by hash; verify hash after download. If hash matches, content is exactly correct. No MITM corruption or bit rot possible.

3. **Immutability:** Content addressed by hash cannot change (changing content changes the hash, which means a different address). Perfect for versioning.

4. **Cache-forever semantics:** A hash-addressed URL can be cached indefinitely (content-addressed assets in web: `bundle.a3f9b.js`).

**Where CAS is used:**

| System | CAS Usage |
|--------|----------|
| Git | Every blob, tree, commit is a SHA-1/SHA-256 hash of its content |
| Docker/OCI images | Each image layer is a SHA-256 hash |
| IPFS | Distributed hash-addressed storage (content routing) |
| S3 ETag | MD5 hash of object used for integrity check |
| Venti (Plan 9) | Pure CAS store for backup |
| Perforce (VCS) | Content-addressed file storage |
| Bazel build cache | Build outputs addressed by input hashes |

**Git as a CAS example:**
```
git hash-object myfile.txt
# Returns SHA-1: 83baae6...
# Content stored at: .git/objects/83/baae6...
# Same content anywhere in any git repo = same hash = same storage

git cat-file -p 83baae6...
# Returns: original file content
```

**Design pattern: CAS + Metadata DB:**
```
Store content:
  1. Hash content -> content_id = sha256(bytes)
  2. Check if content_id exists in CAS store (S3 or custom) -> skip if yes
  3. Store bytes at CAS[content_id]
  
Retrieve content:
  1. Look up user_file_id in metadata DB -> content_id
  2. Fetch CAS[content_id]
  3. Verify sha256(fetched) == content_id
```

**Hash function choice:**
- SHA-1: Git (historical, but SHA-1 collisions now known — Git migrating to SHA-256).
- SHA-256: Docker, most modern CAS systems.
- BLAKE3: Fast, secure, modern alternative.

---

### Q18. How does Point-in-Time Recovery (PITR) work in databases?

**Answer:**

**PITR (Point-in-Time Recovery)** allows a database to be restored to any specific moment in time, not just the last backup. Essential for recovering from logical corruption (accidental DELETE, bad migration) where the data existed before the error.

**Components:**

1. **Base Backup:** A consistent snapshot of the full database at a specific time. Starting point for recovery.
2. **Write-Ahead Log (WAL) / Transaction Log:** Every transaction recorded sequentially. Replay forward from backup to any target time.

```
Timeline:
  T0: Base backup taken (full DB snapshot to S3)
  T1: 100 transactions
  T2: 200 transactions
  T3: Bad migration runs - deletes 10M rows accidentally
  T4: 50 more transactions
  
PITR to T2 (just before the accident):
  1. Restore T0 base backup to new DB instance
  2. Replay WAL from T0 to T2 (stop before T3)
  3. Database is in state it was at T2 - accident never happened
  4. Export needed data or promote as new primary
```

**PostgreSQL PITR implementation:**
```bash
# Step 1: Regular base backups (via pg_basebackup or snapshot)
pg_basebackup -D /backups/base -Ft -z -P

# Step 2: WAL archiving (continuous)
# postgresql.conf:
archive_mode = on
archive_command = 'aws s3 cp %p s3://my-wal-archive/%f'
# Every WAL segment (16MB, ~few seconds to minutes) is copied to S3

# Recovery (restore to 2026-05-11 10:30:00):
# Create recovery.conf (PostgreSQL 11 and below) or postgresql.conf entries:
restore_command = 'aws s3 cp s3://my-wal-archive/%f %p'
recovery_target_time = '2026-05-11 10:30:00'
recovery_target_action = 'promote'  # make it a primary after reaching target time

# Start PostgreSQL -> it replays WAL until recovery_target_time, then stops
```

**RDS / Aurora PITR:**
AWS handles PITR automatically for RDS and Aurora:
- Continuous automatic backups (snapshot + transaction log upload to S3).
- Any point within the retention window (up to 35 days).
- Restore: creates a new DB instance at target time.
- Typical recovery time: 10-30 minutes.

**PITR vs Snapshot restore:**

| Feature | Snapshot Restore | PITR |
|---------|-----------------|------|
| Granularity | Snapshot time only | Any second within retention |
| Use case | Full environment restore | Undo specific logical error |
| Data loss | Since last snapshot | Seconds (depends on WAL archive lag) |
| Cost | Snapshot storage only | Snapshot + WAL storage |

**WAL archive size:** PostgreSQL generates ~1MB WAL per transaction-heavy second. A busy OLTP system might archive 1GB+ WAL per hour → 720GB per month. Lifecycle policies move old WAL to Glacier for cost optimization.

**Log retention window:** Retention period defines how far back you can recover. Balance: longer retention = more storage cost = longer blast radius protection.

---

### Q19. What is data deduplication at the chunk level, and how does it work?

**Answer:**

**Data deduplication** identifies and eliminates duplicate data blocks to reduce storage consumption. Chunk-level dedup operates below the file level — identical chunks within or across files are stored only once.

**How it works:**

**1. Fixed-size chunking:**
Divide every file into equal-size blocks (e.g., 4KB, 64KB):
```
File A: [Block1][Block2][Block3][Block4]
File B: [Block1'][Block2][Block3'][Block4]
         different   same   different  same

Hash each block:
  File A: [H1][H2][H3][H4]
  File B: [H1'][H2][H3'][H4]

Block H2 and H4 are shared -> store only once
Storage: 6 unique blocks instead of 8 (25% savings)
```

Problem: Fixed boundaries cause "boundary shift problem." If you insert 1 byte at the start of a file, all block boundaries shift, and none of the original hashes match, even though 99% of the content is unchanged.

**2. Variable-size chunking (Content-Defined Chunking — CDC):**
Uses a rolling hash (Rabin fingerprint) to find natural chunk boundaries based on content patterns. Boundaries are stable even when data is inserted/deleted.

```
Rolling hash over sliding window of bytes:
  hash(bytes[i..i+window]) 
  When hash matches a pattern (e.g., last N bits = 0): mark as chunk boundary

Result: 
  Before insert: [chunk1: "Hello Wo"][chunk2: "rld! Foo"][chunk3: "bar"]
  After "INSERTED" at start: [chunk1: "INSERTEDHello Wo"][chunk2: "rld! Foo"][chunk3: "bar"]
  -> chunk2 and chunk3 are identical despite the insert at the start!
```

**3. Deduplication pipeline:**
```
Incoming data stream:
  1. Split into chunks (fixed or CDC)
  2. Hash each chunk (SHA-256)
  3. Check hash against dedup index (hash -> storage location)
  4. If hash exists: store only a reference (pointer), discard duplicate chunk
  5. If hash new: store chunk data, update dedup index

Dedup index:
  {sha256_hash: storage_address}
  Stored in high-performance KV store (RocksDB, Redis)
  For large systems: bloom filter to quickly check "definitely not seen before"
```

**Dedup ratio by data type:**

| Data Type | Dedup Ratio |
|-----------|------------|
| VM images (many similar VMs) | 5:1 - 20:1 |
| Database backups (incremental) | 10:1 - 50:1 |
| Email (many attachments sent to many users) | 3:1 - 10:1 |
| Genomics data | 5:1 - 15:1 |
| Video (unique media) | 1:1 (no benefit) |
| Compressed data | 1:1 (no benefit) |

**Where dedup is used:**
- **Backup systems:** Veeam, NetBackup, Zerto — huge dedup ratios for repeated backups.
- **Dropbox/Google Drive:** CAS + dedup so the same file uploaded by many users is stored once.
- **VMware VSAN:** Deduplicate VM disk images sharing common OS blocks.
- **ZFS filesystem:** Built-in inline dedup at the block level.

**Inline vs Post-process dedup:**
- **Inline:** Dedup happens before writing. No extra storage needed. CPU bottleneck on write path.
- **Post-process:** Write first, dedup in background. Extra storage temporarily. Less impact on write latency.

---

### Q20. How do you design storage for a large-scale video streaming platform?

**Answer:**

Let's design storage for a Netflix/YouTube-scale video platform handling 100M daily active users and 500 hours of video uploaded per minute.

**Requirements:**
- Upload: 500 hours/minute of video → process, store, serve globally.
- Serve: 100M users × average 2 hours/day at various resolutions.
- Storage: petabytes of video content.
- Latency: video playback should start in < 2 seconds.
- Durability: zero video loss.

**Storage architecture:**

```
UPLOAD PIPELINE:
  User uploads raw video
       |
  [Upload Service] -> S3 (raw/original) -> SQS/Kafka
       |                                       |
  Pre-signed URL for                  [Transcoding Workers]
  direct S3 multipart upload          (GPU instances)
                                           |
                          Transcode to multiple formats:
                          - 360p H.264 (audio: AAC 128kbps)
                          - 720p H.264
                          - 1080p H.264
                          - 1080p HEVC (H.265, 40% smaller)
                          - 4K HDR HEVC
                          - HLS segments (2-10s .ts files per quality)
                          - DASH segments (similar)
                               |
                          S3 processed/ (transcoded segments)
                               |
                          CDN origin sync
```

**Adaptive Bitrate Streaming (ABR):**
```
HLS Manifest (master.m3u8):
  #EXT-X-STREAM-INF:BANDWIDTH=500000,RESOLUTION=640x360
  360p/index.m3u8

  #EXT-X-STREAM-INF:BANDWIDTH=2000000,RESOLUTION=1280x720
  720p/index.m3u8

  #EXT-X-STREAM-INF:BANDWIDTH=8000000,RESOLUTION=1920x1080
  1080p/index.m3u8

360p/index.m3u8:
  #EXTINF:6.0,  <- 6 second segment
  segment_001.ts
  segment_002.ts
  ...

Player detects available bandwidth every segment -> switches quality level dynamically
```

**Storage tiers:**
```
Tier 1 (HOT): Top 0.1% content (1000 videos) = 95% of traffic
  - CDN fully cached at all edge PoPs
  - Always warm, zero-origin requests
  - 100 PoPs × 10TB = 1PB CDN cache

Tier 2 (WARM): Top 10% content (100K videos) = 4.9% of traffic
  - Cached at regional CDN PoPs
  - S3 Standard for origin
  - S3 Standard-IA after 90 days

Tier 3 (COLD): Long tail 90% content (900K+ videos) = 0.1% of traffic
  - S3 Standard for on-demand access
  - S3 Glacier after 1 year (rarely accessed)
  - Original raw files: S3 Glacier Deep Archive (keep forever, $0.001/GB/month)
```

**Storage size estimation:**
```
1 hour of video, all qualities:
  4K HEVC: 7 GB
  1080p: 2 GB
  720p: 0.7 GB
  360p: 0.2 GB
  Total per hour: ~10 GB

500 hours/minute × 60 minutes × 24 hours × 365 days × 10 GB/hour:
  = 2.6 PB/day of new content processed
  
Retention: keep all content forever (long tail monetization)
  After 10 years: ~10 EB (exabytes) total
  Cost at $0.023/GB S3: ~$230M/month -> tiering to Glacier critical!
```

**CDN strategy:**
```
Global CDN (Akamai/CloudFront/custom Netflix OpenConnect):
  - Serve HLS/DASH segments from edge nodes (< 20ms latency)
  - Cache hit rate: 95%+ for popular content
  - Cache miss: pull from S3 origin, cache for 24-720 hours
  - Segment size: 2-10 seconds → granular cache control

Custom hardware (Netflix OpenConnect):
  Custom appliances at ISPs holding full copy of top N% of catalog
  Eliminates transit costs entirely for participating ISPs
```

**Metadata storage:**
```
Video metadata (PostgreSQL/Aurora):
  video_id, title, duration, upload_date, uploader_id, status
  
Content manifest (DynamoDB):
  video_id -> {resolutions: [{quality, cdn_url, bitrate}, ...], subtitles: [...]}
  
Search index (Elasticsearch):
  Full-text index on title, description, tags, transcript
  
View counts (Redis + periodic flush to DB):
  Redis sorted set for trending, periodic flush to DB
```

---

## Quick Reference

### Storage Types
| Type | Latency | Scaling | Access | Use Case |
|------|---------|---------|--------|---------|
| Block | < 1ms | Limited | Block I/O | Databases, OS |
| File (NAS) | 1-10ms | Limited | POSIX/NFS | Shared files |
| Object (S3) | 10-100ms | Infinite | HTTP | Media, backups |

### LSM vs B-Tree
| Property | B-Tree | LSM Tree |
|----------|--------|---------|
| Write perf | Medium | High |
| Read perf | High | Medium |
| Write amp | Low | High (compaction) |
| Best for | OLTP reads | Write-heavy |

### Erasure Coding vs Replication
| Aspect | 3x Replication | EC (6+3) |
|--------|---------------|---------|
| Overhead | 3x | 1.5x |
| Read latency | Low | Higher |
| CPU | Low | High |
| Best for | Hot data | Cold/warm |

### RAID Summary
| Level | Fault Tolerance | Speed | Capacity Efficiency |
|-------|----------------|-------|---------------------|
| 0 | None | 2x | 100% |
| 1 | 1 disk | Fast reads | 50% |
| 5 | 1 disk | Good | (N-1)/N |
| 10 | 1 per pair | Very fast | 50% |

### RPO/RTO vs DR Strategy
| Strategy | RPO | RTO | Cost |
|----------|-----|-----|------|
| Active-Active | ~0 | ~0 | 10x |
| Warm Standby | < 1 min | < 10 min | 3x |
| Pilot Light | < 1 hr | < 1 hr | 1.5x |
| Backup/Restore | < 24 hr | Hours | 1x |

### Columnar Storage Benefits
1. Column projection (skip irrelevant columns)
2. Higher compression (same-type data)
3. Vectorized execution (SIMD)
4. Predicate pushdown (min/max statistics)

### PITR Components
```
Base Backup (periodic) + WAL Archive (continuous) = PITR
Restore = Apply Base Backup + Replay WAL up to target time
```

### Data Tiering
```
HOT   -> SSD / S3 Standard     (frequent access)
WARM  -> HDD / S3-IA            (weekly access)
COLD  -> S3 Glacier             (monthly access)
FROZEN -> S3 Deep Archive       (yearly/compliance)
```
