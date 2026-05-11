# Problem 26: Design a Stock Exchange

---

## 1. Problem Statement & Clarifying Questions

### Problem Statement
Design a high-performance stock exchange platform capable of matching buy and sell orders in real time, disseminating market data to participants, and supporting millions of orders per day with microsecond-level matching latency.

### Clarifying Questions
1. **Scale**: How many orders per second at peak? (Assume 500K orders/sec at peak like NYSE)
2. **Order types**: Market, limit, stop, stop-limit only — or also iceberg/reserve orders?
3. **Asset classes**: Equities only, or also options, futures, ETFs?
4. **Latency target**: Co-location required? Sub-microsecond? (Assume < 100 microseconds end-to-end matching)
5. **Market data**: Level 1 (best bid/ask), Level 2 (full depth), Level 3 (full order book with order IDs)?
6. **Participants**: Retail brokers, institutional traders, HFTs — all via FIX protocol or REST/WebSocket too?
7. **Regulatory compliance**: SEC/FINRA rules, trade reporting, audit trail requirements?
8. **Settlement**: T+2 settlement via DTCC/clearing house integration?
9. **Geographic scope**: Single exchange, or multi-venue with smart order routing?
10. **Circuit breakers**: Market-wide halts (LULD rules) or single-stock halts?

---

## 2. Functional & Non-Functional Requirements

### Functional Requirements
- Accept and validate orders (market, limit, stop, stop-limit)
- Match orders using price-time priority (FIFO) algorithm
- Support partial fills, fill-or-kill (FOK), immediate-or-cancel (IOC)
- Cancel and modify outstanding orders
- Disseminate real-time market data (L1, L2, L3 feeds)
- Track positions and P&L per participant
- Enforce pre-trade risk checks (buying power, position limits, price collars)
- Circuit breaker mechanism to halt trading on extreme moves
- Append-only trade record for audit and settlement
- Generate end-of-day reports for clearing and settlement (T+2)

### Non-Functional Requirements
- **Latency**: < 100 microseconds order-to-acknowledgement; < 10 microseconds matching engine tick
- **Throughput**: 500K orders/sec, 2M market data updates/sec
- **Availability**: 99.999% during trading hours (< 5 min downtime/year)
- **Durability**: Zero trade loss; all orders persisted before matching
- **Fairness**: Strict price-time priority; no front-running
- **Auditability**: Full order lifecycle log with nanosecond timestamps
- **Disaster Recovery**: Hot standby failover < 1 second
- **Scalability**: Horizontal scaling per instrument symbol

---

## 3. Capacity Estimation

### Traffic Estimates
- NYSE handles ~2B shares/day across ~8,000 symbols
- Peak: 500K new orders/sec during market open/close
- Cancel/modify ratio: ~90% of orders are cancelled (HFT behavior)
- Trades executed: ~50K trades/sec (10% fill rate)

### Market Data
- L1 update: ~200 bytes per quote update
- L2 update: ~1 KB per depth snapshot
- Market data bandwidth: 2M updates/sec × 200 bytes = 400 MB/s outbound
- FIX message inbound: ~200 bytes/order × 500K/sec = 100 MB/s inbound

### Storage
- Order records: 500K/sec × 200 bytes × 6.5 hrs = ~2.3 TB/day
- Trade records: 50K/sec × 300 bytes × 6.5 hrs = ~350 GB/day
- Tick data (time-series): ~5 TB/day compressed
- Annual storage: ~1 PB/year (7-year regulatory retention = 7 PB)

### Memory (Order Book)
- 8,000 symbols × 10,000 price levels × 64 bytes = ~5 GB in RAM
- Each matching engine instance holds 1 symbol's book in L3 cache for max speed

---

## 4. High-Level Architecture (ASCII Diagram)

```
                          ┌──────────────────────────────────────────────────┐
                          │                  PARTICIPANTS                     │
                          │  Retail Brokers  │  Institutions  │  HFT Firms   │
                          └────────┬─────────┴───────┬────────┴──────┬───────┘
                                   │  FIX/WebSocket  │               │
                          ┌────────▼─────────────────▼───────────────▼───────┐
                          │              ORDER GATEWAY (Load Balanced)        │
                          │   FIX Engine │ REST API │ WebSocket Server        │
                          │   Session Mgmt │ Auth & Rate Limiting             │
                          └────────────────────────┬──────────────────────────┘
                                                   │
                          ┌────────────────────────▼──────────────────────────┐
                          │                ORDER ROUTER                        │
                          │   Pre-Trade Risk Checks │ Symbol Routing           │
                          │   Buying Power │ Position Limits │ Price Collars   │
                          └────────────────────────┬──────────────────────────┘
                                                   │
                          ┌────────────────────────▼──────────────────────────┐
                          │              SEQUENCER (Single Writer)             │
                          │   Assigns monotonic sequence numbers               │
                          │   Persists to WAL before forwarding                │
                          │   Guarantees strict ordering per symbol            │
                          └────────┬───────────────┬──────────────────────────┘
                                   │               │
              ┌────────────────────▼───┐   ┌───────▼──────────────────────────┐
              │  MATCHING ENGINE       │   │  MATCHING ENGINE                  │
              │  Symbol: AAPL          │   │  Symbol: GOOGL                    │
              │  Order Book (Bid/Ask)  │   │  Order Book (Bid/Ask)             │
              │  Price-Time Priority   │   │  Price-Time Priority              │
              │  Circuit Breaker       │   │  Circuit Breaker                  │
              └────────────┬───────────┘   └────────────┬─────────────────────┘
                           │                            │
              ┌────────────▼────────────────────────────▼─────────────────────┐
              │                   TRADE EXECUTION ENGINE                       │
              │   Trade Recording │ Position Updates │ P&L Calculation         │
              └────────┬──────────────────────┬────────────────────────────────┘
                       │                      │
        ┌──────────────▼──────┐   ┌───────────▼──────────────────────────────┐
        │  MARKET DATA        │   │  CLEARING & SETTLEMENT                    │
        │  PUBLISHER          │   │  T+2 Settlement │ DTCC Integration        │
        │  L1/L2/L3 Feeds     │   │  Net Positions │ Cash Movement            │
        │  WebSocket Push     │   └──────────────────────────────────────────┘
        └──────────────┬──────┘
                       │
        ┌──────────────▼──────────────────────────────────────────────────────┐
        │                          STORAGE LAYER                              │
        │  Orders DB (PostgreSQL)  │  Trades DB (Cassandra)                  │
        │  Tick DB (InfluxDB/kdb+) │  Positions DB (Redis)                   │
        └─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Component Deep-Dive

### 5.1 Order Gateway
- Terminates FIX 4.2/4.4 sessions from brokers; REST and WebSocket for retail
- **Session management**: Sequence number tracking, heartbeat, session-level recovery
- **Authentication**: TLS mutual auth for FIX; JWT for REST/WebSocket
- **Rate limiting**: Per-participant order rate limits (e.g., 10K orders/sec max)
- Normalizes all order formats into internal canonical representation
- Stateless horizontally scalable; routes to Order Router via consistent hash on participant ID

### 5.2 Order Router (Pre-Trade Risk)
- **Buying power check**: `available_cash >= order_qty × limit_price × margin_factor`
- **Position limit check**: `abs(current_position + order_qty) <= max_position_limit`
- **Price collar check**: Reject orders > 5% away from last trade price
- **Symbol validation**: Verify symbol is listed and not halted
- **Duplicate order detection**: Idempotency via client_order_id deduplication
- Adds internal routing metadata (sequence number slot request, symbol shard)

### 5.3 Sequencer
The sequencer is the most critical component for fairness. It is a **single-threaded, single-writer** process per exchange:
- Assigns a globally monotonic sequence number to every accepted order event
- Writes to a Write-Ahead Log (WAL / persistent ring buffer) before forwarding
- Implemented using kernel bypass networking (DPDK) or RDMA for sub-microsecond latency
- Hot standby sequencer maintains replicated WAL; failover < 1 ms
- Partitioned per symbol group (not truly global) to allow parallelism

### 5.4 Matching Engine
The heart of the exchange — one engine instance per symbol (or symbol group):

**Order Book Data Structures:**
- **Bids side**: Max-heap (or red-black BST) keyed by (price DESC, time ASC)
- **Asks side**: Min-heap (or red-black BST) keyed by (price ASC, time ASC)
- **Price level aggregation**: Doubly-linked list of orders at each price level
- **Order lookup map**: Hash map of order_id → order node for O(1) cancel
- **Skip list alternative**: O(log n) insert/delete with O(1) best price access

**Matching Algorithm (Price-Time Priority):**
```
For each incoming order:
  1. If MARKET order: match against best asks/bids until filled or book empty
  2. If LIMIT order:
     a. Check if crossing (limit buy >= best ask, or limit sell <= best bid)
     b. If crossing: match against opposing side at their prices
     c. Remaining quantity: rest in book at limit price
  3. On match: generate Trade event (aggressor_order_id, passive_order_id, price, qty)
  4. Partial fill: reduce quantities; fully filled orders removed from book
```

**Order Type Handling:**
- **FOK (Fill-or-Kill)**: Check if entire quantity can be filled; if not, reject without matching
- **IOC (Immediate-or-Cancel)**: Match what can be filled immediately; cancel remainder
- **Stop orders**: Held in stop order book; triggered when last trade price crosses stop price
- **Stop-limit**: Triggered like stop, then behaves as limit order

### 5.5 Market Data Dissemination
- **Level 1 (NBBO)**: Best bid price/size + best ask price/size; every quote change triggers update
- **Level 2 (Market Depth)**: Top 5-10 price levels with aggregated sizes
- **Level 3 (Full Book)**: Every individual order with order IDs (used by HFTs)
- **Multicast**: Market data broadcast via UDP multicast to co-located participants (latency < 1 µs)
- **WebSocket**: For retail/institutional clients not co-located
- **Sequence numbers on market data**: Clients detect gaps and can request retransmission

### 5.6 Circuit Breakers
Implementing SEC's LULD (Limit Up-Limit Down) rules:
- Calculate price bands: ±5% from reference price for Tier 1 NMS stocks
- If last sale price moves outside band: enter 15-second trading pause
- If price moves > 10% in 5 minutes: Level 1 market-wide circuit breaker (15-min halt)
- If price moves > 20%: Level 2 circuit breaker (15-min halt)
- If price moves > 20% after 3:25 PM: Level 3 (halt for rest of day)

### 5.7 Clearing and Settlement
- **Trade capture**: All trades reported to clearing house in real time
- **Netting**: End-of-day net position calculation per participant per symbol
- **T+2 settlement**: Cash and shares exchanged 2 business days after trade date
- **DTCC integration**: National Securities Clearing Corporation (NSCC) for US equities
- **Margin calls**: Real-time monitoring of margin requirements during volatile markets

---

## 6. Database Design

### Orders Table (PostgreSQL with partitioning)
```sql
CREATE TABLE orders (
    order_id        UUID PRIMARY KEY,
    client_order_id VARCHAR(64) NOT NULL,
    participant_id  VARCHAR(32) NOT NULL,
    symbol          VARCHAR(16) NOT NULL,
    side            CHAR(1) NOT NULL,         -- 'B' buy, 'S' sell
    order_type      VARCHAR(16) NOT NULL,     -- MARKET, LIMIT, STOP, STOP_LIMIT
    quantity        BIGINT NOT NULL,
    filled_qty      BIGINT DEFAULT 0,
    limit_price     DECIMAL(18,4),
    stop_price      DECIMAL(18,4),
    status          VARCHAR(16) NOT NULL,     -- NEW, PARTIAL, FILLED, CANCELLED, REJECTED
    time_in_force   VARCHAR(8) NOT NULL,      -- DAY, GTC, IOC, FOK
    sequence_num    BIGINT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL,
    UNIQUE(participant_id, client_order_id, created_at::DATE)
) PARTITION BY RANGE (created_at);

CREATE INDEX idx_orders_symbol_status ON orders(symbol, status, sequence_num);
CREATE INDEX idx_orders_participant ON orders(participant_id, created_at);
```

### Trades Table (Cassandra for write-heavy append)
```
CREATE TABLE trades (
    trade_id        UUID,
    symbol          TEXT,
    trade_date      DATE,
    trade_time      TIMESTAMP,
    price           DECIMAL,
    quantity        BIGINT,
    aggressor_order UUID,
    passive_order   UUID,
    aggressor_side  TEXT,
    sequence_num    BIGINT,
    PRIMARY KEY ((symbol, trade_date), trade_time, trade_id)
) WITH CLUSTERING ORDER BY (trade_time ASC);
```

### Positions Table (Redis for real-time, PostgreSQL for EOD)
```
positions:{participant_id}:{symbol} → HASH {
    quantity: long (positive=long, negative=short),
    avg_cost: decimal,
    realized_pnl: decimal,
    unrealized_pnl: decimal,
    last_updated: timestamp
}
```

### Instruments Table
```sql
CREATE TABLE instruments (
    symbol          VARCHAR(16) PRIMARY KEY,
    name            VARCHAR(256),
    exchange        VARCHAR(8),
    lot_size        INT DEFAULT 1,
    tick_size       DECIMAL(10,6),
    circuit_breaker_pct DECIMAL(5,2) DEFAULT 5.0,
    is_active       BOOLEAN DEFAULT TRUE,
    listing_date    DATE,
    sector          VARCHAR(64)
);
```

---

## 7. API Design

### Order Management (FIX 4.4 / REST)
```
POST   /v1/orders                    # Place new order
DELETE /v1/orders/{order_id}         # Cancel order
PUT    /v1/orders/{order_id}         # Modify order (cancel-replace)
GET    /v1/orders/{order_id}         # Get order status
GET    /v1/orders?symbol=AAPL&status=OPEN  # List orders

POST /v1/orders Body:
{
  "client_order_id": "client-123",
  "symbol": "AAPL",
  "side": "BUY",
  "order_type": "LIMIT",
  "quantity": 100,
  "limit_price": 150.25,
  "time_in_force": "DAY"
}
```

### Market Data (WebSocket)
```
WS /v1/market-data?symbols=AAPL,GOOGL&level=2

# Server pushes:
{
  "type": "quote",
  "symbol": "AAPL",
  "bid_price": 150.20, "bid_size": 500,
  "ask_price": 150.25, "ask_size": 300,
  "sequence": 98765432,
  "timestamp": "2024-01-15T14:30:00.000000123Z"
}
```

### Position & Account (REST)
```
GET /v1/positions                    # All positions
GET /v1/positions/{symbol}           # Position for symbol
GET /v1/account/buying-power         # Available buying power
GET /v1/trades?symbol=AAPL&date=today  # Trade history
```

---

## 8. Scalability & Bottlenecks

### Bottleneck 1: Matching Engine (Single-Threaded per Symbol)
- **Problem**: By design single-threaded for fairness; can't parallelize matching for one symbol
- **Solution**: Each symbol runs on a dedicated CPU core with thread pinning; NUMA-aware memory allocation
- **Scale**: 8,000 symbols across 128 CPU cores = ~62 symbols/core (grouped by liquidity)

### Bottleneck 2: Sequencer Throughput
- **Problem**: Single writer bottleneck; maximum ~2M messages/sec on commodity hardware
- **Solution**: Kernel bypass (DPDK), busy-spin instead of OS scheduler, pre-allocated ring buffers
- **Partition**: Separate sequencer per symbol group; each handles ~500 symbols

### Bottleneck 3: Market Data Fan-Out
- **Problem**: 400 MB/s outbound to thousands of subscribers
- **Solution**: UDP multicast to co-located clients; CDN/WebSocket for remote clients
- **Architecture**: Dedicated market data servers separate from matching engines

### Bottleneck 4: Order Persistence Latency
- **Problem**: Must persist order before matching; disk I/O adds latency
- **Solution**: Persist to RAM-backed append-only log (Aeron/Chronicle Queue); async flush to SSD
- **Durability**: RAID 10 NVMe SSDs with battery-backed write cache (BBWC)

### Bottleneck 5: Position Updates at Scale
- **Problem**: 50K trades/sec each requiring position update for 2 participants
- **Solution**: Redis cluster with pipelining; async batch updates to PostgreSQL

---

## 9. Trade-offs & Design Decisions

### Decision 1: Price-Time Priority vs Pro-Rata
- **Price-Time (FIFO)**: First order at a price level gets first fill; simpler, fairer for time-conscious traders
- **Pro-Rata**: Fill proportionally to order size at same price; used in options/futures markets
- **Choice**: Price-time for equities (NYSE/NASDAQ model); pro-rata configurable per instrument

### Decision 2: Order Book Data Structure
- **Sorted Array/List**: Simple, cache-friendly for small books; O(n) insert/delete
- **Skip List**: O(log n) operations, cache-friendly, easy range queries; used in Redis
- **Red-Black BST**: O(log n) guaranteed; complex implementation
- **Hash + Linked List per price level**: O(1) add/remove within price level; used by most exchanges
- **Choice**: Hash map of price → FIFO queue; separate sorted set for price discovery

### Decision 3: Synchronous vs Asynchronous Risk Checks
- **Synchronous**: Risk check in critical path; adds 1-5 µs latency but guarantees compliance
- **Asynchronous**: Post-hoc rejection is too late (order already matched)
- **Choice**: Synchronous pre-trade checks; hardware-accelerated where possible (FPGA for largest participants)

### Decision 4: ZooKeeper vs Raft for Leader Election
- **ZooKeeper**: Battle-tested, used by older systems; external dependency
- **Raft (embedded)**: Self-contained, faster failover, simpler ops
- **Choice**: Raft for sequencer and matching engine leader election (like Nasdaq's approach)

### Decision 5: Tick Data Storage
- **PostgreSQL with TimescaleDB**: SQL queries; moderate write performance
- **InfluxDB**: Purpose-built time-series; excellent compression
- **kdb+**: Industry standard for high-frequency tick data; fastest query performance
- **Choice**: kdb+ for real-time tick data (industry standard); InfluxDB for operational metrics

---

## 10. Key Interview Talking Points

### 1. Why the Sequencer is Critical
Explain that without a strict sequencer, two matching engines could process the same order simultaneously, causing double-fills or inconsistent state. The sequencer is the "single source of truth" for event ordering — analogous to a Kafka partition.

### 2. Price-Time Priority and Fairness
Walk through: incoming limit buy at $150.25 when best ask is $150.20 — it will match at $150.20 (price improvement). Multiple asks at $150.20 are filled in time order. Demonstrate understanding of maker/taker model.

### 3. The Latency Stack
- Network (kernel bypass DPDK): 1-2 µs
- Sequencer (memory-mapped ring buffer): < 1 µs
- Matching engine (in-memory, cache-warm): 1-5 µs
- Market data publish (multicast): 1-2 µs
- Total: < 10 µs for co-located participants

### 4. Circuit Breaker Design
Discuss stock-level (LULD) vs market-wide halts. Mention that the circuit breaker needs a reference price (calculated as volume-weighted average of last 5 minutes) and must be replicated across all matching engine instances for the same symbol.

### 5. At-Least-Once vs Exactly-Once for Trades
- Orders: At-least-once delivery to matching engine (idempotent via sequence numbers)
- Trades: Exactly-once recording is critical (duplicate trade = double settlement)
- Use idempotency key = (sequence_num_of_aggressor + sequence_num_of_passive)

### 6. HFT Considerations
- Co-location: HFT firms rent rack space within exchange data center
- Market access: Direct Market Access (DMA) vs Sponsored Access
- Order types: Internalization, dark pools, hidden/iceberg orders
- Speed: 10GbE → 100GbE → InfiniBand → microwave/laser towers for inter-exchange arbitrage

### 7. Regulatory Requirements
- **Rule 611 (Order Protection Rule)**: Must trade at best available price across all venues
- **Reg NMS**: National Market System — mandates intermarket sweep orders (ISO)
- **Consolidated Tape**: All trades reported to SIP within 10 milliseconds
- **Audit Trail**: FINRA CAT (Consolidated Audit Trail) — nanosecond timestamps for all order events

### 8. Disaster Recovery
- Active-passive: Hot standby sequencer with WAL replication < 1 ms lag
- Geographic redundancy: Primary in NY4 (Mahwah/Carteret), DR in Chicago
- Recovery Time Objective (RTO): < 1 second (automated failover)
- Recovery Point Objective (RPO): Zero (synchronous WAL replication)
