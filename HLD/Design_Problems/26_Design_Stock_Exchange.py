"""
Problem 26: Design a Stock Exchange
====================================
Working simulation of a stock exchange with:
- OrderBook (bid max-heap, ask min-heap)
- MatchingEngine (price-time priority, partial fills)
- OrderRouter (pre-trade risk checks)
- MarketDataPublisher (broadcast order book changes)
- TradeRecorder (append-only trade history)
- CircuitBreaker (halt on rapid price moves)
- PositionTracker (per-user P&L)
"""

import heapq
import uuid
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional
from enum import Enum


# ─── Enums & Constants ───────────────────────────────────────────────────────

class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"

class OrderStatus(Enum):
    NEW = "NEW"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class TimeInForce(Enum):
    DAY = "DAY"   # Good for day
    GTC = "GTC"   # Good till cancelled
    IOC = "IOC"   # Immediate-or-cancel
    FOK = "FOK"   # Fill-or-kill


# ─── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class Order:
    order_id: str
    participant_id: str
    symbol: str
    side: Side
    order_type: OrderType
    quantity: int
    limit_price: Optional[float]
    time_in_force: TimeInForce
    sequence_num: int
    timestamp: float = field(default_factory=time.time)
    filled_qty: int = 0
    status: OrderStatus = OrderStatus.NEW

    @property
    def remaining_qty(self) -> int:
        return self.quantity - self.filled_qty

    def fill(self, qty: int) -> None:
        self.filled_qty += qty
        self.status = OrderStatus.FILLED if self.filled_qty >= self.quantity else OrderStatus.PARTIAL


@dataclass
class Trade:
    trade_id: str
    symbol: str
    price: float
    quantity: int
    aggressor_order_id: str
    passive_order_id: str
    aggressor_side: Side
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        return (f"TRADE {self.symbol}: {self.aggressor_side.value} {self.quantity} "
                f"@ ${self.price:.2f} | agg={self.aggressor_order_id[:8]} "
                f"pass={self.passive_order_id[:8]}")


# ─── Order Book ───────────────────────────────────────────────────────────────

class OrderBook:
    """
    Maintains bid (max-heap) and ask (min-heap) order queues.
    Bids heap: negate price for max-heap using Python's min-heap.
    Heap entries: (price_key, sequence_num, order) for price-time priority.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self._bids: list = []   # max-heap: (-price, seq, order)
        self._asks: list = []   # min-heap: (+price, seq, order)
        self._order_map: dict[str, Order] = {}  # order_id → Order for O(1) cancel

    def add_order(self, order: Order) -> None:
        self._order_map[order.order_id] = order
        if order.side == Side.BUY:
            heapq.heappush(self._bids, (-order.limit_price, order.sequence_num, order))
        else:
            heapq.heappush(self._asks, (order.limit_price, order.sequence_num, order))

    def cancel_order(self, order_id: str) -> Optional[Order]:
        order = self._order_map.pop(order_id, None)
        if order:
            order.status = OrderStatus.CANCELLED
        return order

    def best_bid(self) -> Optional[Order]:
        self._cleanup(self._bids, is_bid=True)
        if self._bids:
            return self._bids[0][2]
        return None

    def best_ask(self) -> Optional[Order]:
        self._cleanup(self._asks, is_bid=False)
        if self._asks:
            return self._asks[0][2]
        return None

    def pop_best_bid(self) -> Optional[Order]:
        self._cleanup(self._bids, is_bid=True)
        if self._bids:
            order = heapq.heappop(self._bids)[2]
            self._order_map.pop(order.order_id, None)
            return order
        return None

    def pop_best_ask(self) -> Optional[Order]:
        self._cleanup(self._asks, is_bid=False)
        if self._asks:
            order = heapq.heappop(self._asks)[2]
            self._order_map.pop(order.order_id, None)
            return order
        return None

    def _cleanup(self, heap: list, is_bid: bool) -> None:
        """Remove stale (cancelled/filled) entries from top of heap."""
        while heap:
            order = heap[0][2]
            if order.status in (OrderStatus.CANCELLED, OrderStatus.FILLED) or order.remaining_qty == 0:
                heapq.heappop(heap)
            else:
                break

    def get_depth(self, levels: int = 5) -> dict:
        """Return top N bid/ask levels for market data."""
        bids_snapshot = {}
        asks_snapshot = {}
        seen_orders = set()

        for entry in sorted(self._bids):
            order = entry[2]
            if order.status not in (OrderStatus.CANCELLED, OrderStatus.FILLED) and order.order_id not in seen_orders:
                price = -entry[0]
                bids_snapshot[price] = bids_snapshot.get(price, 0) + order.remaining_qty
                seen_orders.add(order.order_id)

        seen_orders.clear()
        for entry in sorted(self._asks):
            order = entry[2]
            if order.status not in (OrderStatus.CANCELLED, OrderStatus.FILLED) and order.order_id not in seen_orders:
                price = entry[0]
                asks_snapshot[price] = asks_snapshot.get(price, 0) + order.remaining_qty
                seen_orders.add(order.order_id)

        sorted_bids = sorted(bids_snapshot.items(), reverse=True)[:levels]
        sorted_asks = sorted(asks_snapshot.items())[:levels]
        return {"bids": sorted_bids, "asks": sorted_asks}


# ─── Circuit Breaker ──────────────────────────────────────────────────────────

class CircuitBreaker:
    """Halt trading if price moves > threshold% within window_seconds."""

    def __init__(self, threshold_pct: float = 5.0, window_seconds: float = 300.0):
        self.threshold_pct = threshold_pct
        self.window_seconds = window_seconds
        self._price_history: list[tuple[float, float]] = []  # (timestamp, price)
        self.is_halted: bool = False
        self.halt_reason: str = ""

    def record_trade(self, price: float, timestamp: float) -> bool:
        """Record a trade price. Returns True if circuit breaker tripped."""
        now = timestamp
        self._price_history.append((now, price))
        # Remove old entries outside window
        cutoff = now - self.window_seconds
        self._price_history = [(t, p) for t, p in self._price_history if t >= cutoff]

        if len(self._price_history) >= 2:
            oldest_price = self._price_history[0][1]
            pct_move = abs(price - oldest_price) / oldest_price * 100
            if pct_move >= self.threshold_pct:
                self.is_halted = True
                self.halt_reason = f"Price moved {pct_move:.1f}% in {self.window_seconds}s (threshold: {self.threshold_pct}%)"
                return True
        return False

    def reset(self) -> None:
        self.is_halted = False
        self.halt_reason = ""
        self._price_history.clear()


# ─── Position Tracker ─────────────────────────────────────────────────────────

class PositionTracker:
    """Track per-user positions and running P&L."""

    def __init__(self):
        self._positions: dict = defaultdict(lambda: defaultdict(lambda: {"qty": 0, "avg_cost": 0.0, "realized_pnl": 0.0}))
        self._cash: dict[str, float] = defaultdict(lambda: 100_000.0)  # $100K starting cash

    def record_fill(self, participant_id: str, symbol: str, side: Side, qty: int, price: float) -> None:
        pos = self._positions[participant_id][symbol]
        if side == Side.BUY:
            total_cost = pos["avg_cost"] * pos["qty"] + price * qty
            pos["qty"] += qty
            pos["avg_cost"] = total_cost / pos["qty"] if pos["qty"] > 0 else 0.0
            self._cash[participant_id] -= price * qty
        else:  # SELL
            if pos["qty"] > 0:
                realized = (price - pos["avg_cost"]) * qty
                pos["realized_pnl"] += realized
            pos["qty"] -= qty
            self._cash[participant_id] += price * qty

    def get_unrealized_pnl(self, participant_id: str, symbol: str, current_price: float) -> float:
        pos = self._positions[participant_id][symbol]
        return (current_price - pos["avg_cost"]) * pos["qty"]

    def get_position(self, participant_id: str, symbol: str) -> dict:
        return dict(self._positions[participant_id][symbol])

    def get_buying_power(self, participant_id: str) -> float:
        return self._cash[participant_id]


# ─── Trade Recorder ───────────────────────────────────────────────────────────

class TradeRecorder:
    """Append-only trade history — immutable audit log."""

    def __init__(self):
        self._trades: list[Trade] = []

    def record(self, trade: Trade) -> None:
        self._trades.append(trade)

    def get_trades(self, symbol: Optional[str] = None) -> list[Trade]:
        if symbol:
            return [t for t in self._trades if t.symbol == symbol]
        return list(self._trades)

    def last_price(self, symbol: str) -> Optional[float]:
        for trade in reversed(self._trades):
            if trade.symbol == symbol:
                return trade.price
        return None


# ─── Market Data Publisher ────────────────────────────────────────────────────

class MarketDataPublisher:
    """Broadcast order book and trade updates to subscribers."""

    def __init__(self):
        self._subscribers: list = []
        self._updates: list[dict] = []

    def subscribe(self, callback) -> None:
        self._subscribers.append(callback)

    def publish_trade(self, trade: Trade) -> None:
        event = {"type": "TRADE", "symbol": trade.symbol, "price": trade.price,
                 "qty": trade.quantity, "ts": trade.timestamp}
        self._updates.append(event)
        for cb in self._subscribers:
            cb(event)

    def publish_quote(self, symbol: str, best_bid: Optional[float],
                      best_ask: Optional[float], bid_size: int, ask_size: int) -> None:
        event = {"type": "QUOTE", "symbol": symbol, "bid": best_bid,
                 "ask": best_ask, "bid_size": bid_size, "ask_size": ask_size}
        self._updates.append(event)

    def get_feed(self) -> list[dict]:
        return list(self._updates)


# ─── Order Router (Pre-Trade Risk) ───────────────────────────────────────────

class OrderRouter:
    """Pre-trade risk checks before order reaches matching engine."""

    MAX_POSITION = 10_000
    PRICE_COLLAR_PCT = 0.10  # Reject orders > 10% from last price

    def __init__(self, position_tracker: PositionTracker, trade_recorder: TradeRecorder):
        self.position_tracker = position_tracker
        self.trade_recorder = trade_recorder

    def validate(self, order: Order) -> tuple[bool, str]:
        # Buying power check
        if order.side == Side.BUY and order.limit_price:
            cost = order.quantity * order.limit_price
            buying_power = self.position_tracker.get_buying_power(order.participant_id)
            if cost > buying_power:
                return False, f"Insufficient buying power: need ${cost:.2f}, have ${buying_power:.2f}"

        # Position limit check
        pos = self.position_tracker.get_position(order.participant_id, order.symbol)
        current_qty = pos.get("qty", 0)
        new_qty = current_qty + (order.quantity if order.side == Side.BUY else -order.quantity)
        if abs(new_qty) > self.MAX_POSITION:
            return False, f"Position limit exceeded: {abs(new_qty)} > {self.MAX_POSITION}"

        # Price collar check
        last_price = self.trade_recorder.last_price(order.symbol)
        if last_price and order.limit_price:
            pct_away = abs(order.limit_price - last_price) / last_price
            if pct_away > self.PRICE_COLLAR_PCT:
                return False, f"Price ${order.limit_price} is {pct_away*100:.1f}% from last trade ${last_price}"

        return True, "OK"


# ─── Matching Engine ──────────────────────────────────────────────────────────

class MatchingEngine:
    """Core price-time priority matching engine for one symbol."""

    def __init__(self, symbol: str, trade_recorder: TradeRecorder,
                 market_data: MarketDataPublisher, position_tracker: PositionTracker,
                 circuit_breaker: CircuitBreaker):
        self.symbol = symbol
        self.order_book = OrderBook(symbol)
        self.trade_recorder = trade_recorder
        self.market_data = market_data
        self.position_tracker = position_tracker
        self.circuit_breaker = circuit_breaker
        self._sequence = 0

    def _next_seq(self) -> int:
        self._sequence += 1
        return self._sequence

    def match(self, incoming: Order) -> list[Trade]:
        """Match incoming order against the order book."""
        if self.circuit_breaker.is_halted:
            incoming.status = OrderStatus.REJECTED
            print(f"  [HALTED] Order rejected: {self.circuit_breaker.halt_reason}")
            return []

        trades: list[Trade] = []
        incoming.sequence_num = self._next_seq()

        # FOK check: verify full fill possible before matching
        if incoming.time_in_force == TimeInForce.FOK:
            if not self._can_fully_fill(incoming):
                incoming.status = OrderStatus.CANCELLED
                print(f"  [FOK] Order {incoming.order_id[:8]} rejected — cannot fill completely")
                return []

        # Match against opposing side
        while incoming.remaining_qty > 0:
            if incoming.side == Side.BUY:
                passive = self.order_book.best_ask()
                if passive is None:
                    break
                if incoming.order_type == OrderType.LIMIT and incoming.limit_price < passive.limit_price:
                    break
            else:
                passive = self.order_book.best_bid()
                if passive is None:
                    break
                if incoming.order_type == OrderType.LIMIT and incoming.limit_price > passive.limit_price:
                    break

            # Execute fill
            fill_qty = min(incoming.remaining_qty, passive.remaining_qty)
            fill_price = passive.limit_price  # Passive order's price wins

            trade = Trade(
                trade_id=str(uuid.uuid4()),
                symbol=self.symbol,
                price=fill_price,
                quantity=fill_qty,
                aggressor_order_id=incoming.order_id,
                passive_order_id=passive.order_id,
                aggressor_side=incoming.side
            )

            incoming.fill(fill_qty)
            passive.fill(fill_qty)

            # Update positions
            self.position_tracker.record_fill(incoming.participant_id, self.symbol, incoming.side, fill_qty, fill_price)
            self.position_tracker.record_fill(passive.participant_id, self.symbol,
                                               Side.SELL if incoming.side == Side.BUY else Side.BUY,
                                               fill_qty, fill_price)

            # Remove fully filled passive order from book
            if passive.remaining_qty == 0:
                if passive.side == Side.SELL:
                    self.order_book.pop_best_ask()
                else:
                    self.order_book.pop_best_bid()

            self.trade_recorder.record(trade)
            self.market_data.publish_trade(trade)
            trades.append(trade)
            print(f"  {trade}")

            # Check circuit breaker after each trade
            if self.circuit_breaker.record_trade(fill_price, trade.timestamp):
                print(f"  [CIRCUIT BREAKER TRIPPED] {self.circuit_breaker.halt_reason}")
                break

        # Rest limit order in book if not fully filled (unless IOC)
        if incoming.remaining_qty > 0 and incoming.time_in_force not in (TimeInForce.IOC, TimeInForce.FOK):
            if incoming.order_type == OrderType.LIMIT:
                self.order_book.add_order(incoming)
        elif incoming.remaining_qty > 0 and incoming.time_in_force == TimeInForce.IOC:
            incoming.status = OrderStatus.CANCELLED
            print(f"  [IOC] Remaining {incoming.remaining_qty} shares of {incoming.order_id[:8]} cancelled")

        self._publish_quote()
        return trades

    def _can_fully_fill(self, order: Order) -> bool:
        """Check if FOK order can be fully filled."""
        remaining = order.quantity
        if order.side == Side.BUY:
            for _, _, ask in sorted(self.order_book._asks):
                if ask.status in (OrderStatus.CANCELLED, OrderStatus.FILLED):
                    continue
                if order.limit_price and ask.limit_price > order.limit_price:
                    break
                remaining -= min(remaining, ask.remaining_qty)
                if remaining == 0:
                    return True
        else:
            for _, _, bid in sorted(self.order_book._bids, reverse=True):
                if bid.status in (OrderStatus.CANCELLED, OrderStatus.FILLED):
                    continue
                if order.limit_price and bid.limit_price < order.limit_price:
                    break
                remaining -= min(remaining, bid.remaining_qty)
                if remaining == 0:
                    return True
        return remaining == 0

    def _publish_quote(self) -> None:
        best_bid = self.order_book.best_bid()
        best_ask = self.order_book.best_ask()
        self.market_data.publish_quote(
            self.symbol,
            best_bid.limit_price if best_bid else None,
            best_ask.limit_price if best_ask else None,
            best_bid.remaining_qty if best_bid else 0,
            best_ask.remaining_qty if best_ask else 0
        )


# ─── Stock Exchange (Orchestrator) ───────────────────────────────────────────

class StockExchange:
    """Top-level exchange: routes orders to per-symbol matching engines."""

    def __init__(self):
        self.trade_recorder = TradeRecorder()
        self.market_data = MarketDataPublisher()
        self.position_tracker = PositionTracker()
        self.order_router = OrderRouter(self.position_tracker, self.trade_recorder)
        self._engines: dict[str, MatchingEngine] = {}
        self._orders: dict[str, Order] = {}
        self._global_seq = 0

    def list_symbol(self, symbol: str) -> None:
        cb = CircuitBreaker(threshold_pct=5.0, window_seconds=300.0)
        self._engines[symbol] = MatchingEngine(
            symbol, self.trade_recorder, self.market_data, self.position_tracker, cb
        )
        print(f"Listed {symbol} on exchange")

    def place_order(self, participant_id: str, symbol: str, side: Side,
                    order_type: OrderType, quantity: int, limit_price: Optional[float] = None,
                    time_in_force: TimeInForce = TimeInForce.DAY) -> Optional[Order]:
        if symbol not in self._engines:
            print(f"  [ERROR] Symbol {symbol} not listed")
            return None

        self._global_seq += 1
        order = Order(
            order_id=str(uuid.uuid4()),
            participant_id=participant_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            time_in_force=time_in_force,
            sequence_num=self._global_seq
        )

        valid, reason = self.order_router.validate(order)
        if not valid:
            order.status = OrderStatus.REJECTED
            print(f"  [RISK REJECTED] {reason}")
            return order

        self._orders[order.order_id] = order
        print(f"\nOrder [{order.order_id[:8]}] {participant_id}: {side.value} {quantity} {symbol}"
              f" @ {'MKT' if limit_price is None else f'${limit_price:.2f}'} {time_in_force.value}")
        self._engines[symbol].match(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if not order:
            return False
        engine = self._engines.get(order.symbol)
        if engine:
            engine.order_book.cancel_order(order_id)
        return True

    def get_order_book(self, symbol: str, levels: int = 5) -> dict:
        engine = self._engines.get(symbol)
        return engine.order_book.get_depth(levels) if engine else {}

    def get_best_quote(self, symbol: str) -> dict:
        engine = self._engines.get(symbol)
        if not engine:
            return {}
        best_bid = engine.order_book.best_bid()
        best_ask = engine.order_book.best_ask()
        return {
            "symbol": symbol,
            "bid": best_bid.limit_price if best_bid else None,
            "ask": best_ask.limit_price if best_ask else None,
            "spread": round((best_ask.limit_price - best_bid.limit_price), 4)
                      if best_bid and best_ask else None
        }

    def get_position(self, participant_id: str, symbol: str, current_price: float) -> dict:
        pos = self.position_tracker.get_position(participant_id, symbol)
        upnl = self.position_tracker.get_unrealized_pnl(participant_id, symbol, current_price)
        return {**pos, "unrealized_pnl": round(upnl, 2),
                "buying_power": round(self.position_tracker.get_buying_power(participant_id), 2)}


# ─── Demo / Simulation ────────────────────────────────────────────────────────

def run_simulation():
    print("=" * 60)
    print("STOCK EXCHANGE SIMULATION")
    print("=" * 60)

    exchange = StockExchange()
    exchange.list_symbol("AAPL")
    exchange.list_symbol("GOOGL")

    # Seed order book with resting limit orders
    print("\n--- Seeding Order Book with Limit Orders ---")
    exchange.place_order("market_maker_1", "AAPL", Side.BUY,  OrderType.LIMIT, 500, 149.90)
    exchange.place_order("market_maker_1", "AAPL", Side.BUY,  OrderType.LIMIT, 300, 149.80)
    exchange.place_order("market_maker_1", "AAPL", Side.SELL, OrderType.LIMIT, 400, 150.10)
    exchange.place_order("market_maker_1", "AAPL", Side.SELL, OrderType.LIMIT, 600, 150.20)

    print("\n--- Initial Order Book for AAPL ---")
    book = exchange.get_order_book("AAPL")
    print(f"  Bids: {book['bids']}")
    print(f"  Asks: {book['asks']}")
    print(f"  Best Quote: {exchange.get_best_quote('AAPL')}")

    # Aggressive buy — partial fill then rest in book
    print("\n--- Aggressive Buy (Partial Fill) ---")
    exchange.place_order("trader_alice", "AAPL", Side.BUY, OrderType.LIMIT, 600, 150.15)

    # Market sell order
    print("\n--- Market Sell Order ---")
    exchange.place_order("trader_bob", "AAPL", Side.SELL, OrderType.MARKET, 200, None)

    # IOC Order
    print("\n--- IOC Order (Immediate or Cancel) ---")
    exchange.place_order("trader_carol", "AAPL", Side.BUY, OrderType.LIMIT, 1000, 150.10,
                         TimeInForce.IOC)

    # FOK Order
    print("\n--- FOK Order (Fill or Kill) ---")
    exchange.place_order("trader_dave", "AAPL", Side.BUY, OrderType.LIMIT, 5000, 150.10,
                         TimeInForce.FOK)

    # Risk rejection test
    print("\n--- Risk Check: Insufficient Buying Power ---")
    exchange.position_tracker._cash["broke_trader"] = 100.0
    exchange.place_order("broke_trader", "AAPL", Side.BUY, OrderType.LIMIT, 100, 150.00)

    # Position summary
    print("\n--- Position Summary for trader_alice ---")
    last_price = exchange.trade_recorder.last_price("AAPL") or 150.00
    pos = exchange.get_position("trader_alice", "AAPL", last_price)
    print(f"  AAPL position: qty={pos.get('qty')}, avg_cost=${pos.get('avg_cost', 0):.2f}, "
          f"realized_pnl=${pos.get('realized_pnl', 0):.2f}, "
          f"unrealized_pnl=${pos.get('unrealized_pnl', 0):.2f}, "
          f"buying_power=${pos.get('buying_power', 0):.2f}")

    # Circuit breaker simulation
    print("\n--- Circuit Breaker Simulation ---")
    exchange.list_symbol("TSLA")
    # Build up a price history first
    engine_tsla = exchange._engines["TSLA"]
    exchange.place_order("market_maker_2", "TSLA", Side.SELL, OrderType.LIMIT, 1000, 200.00)
    for i in range(5):
        exchange.place_order("market_maker_2", "TSLA", Side.BUY, OrderType.LIMIT, 100, 200.00 + i * 2.5)
        engine_tsla.circuit_breaker.record_trade(200.00 + i * 2.5, time.time() - (4 - i) * 30)
    # Force a large price move
    engine_tsla.circuit_breaker.record_trade(215.00, time.time())
    print(f"  Circuit Breaker halted: {engine_tsla.circuit_breaker.is_halted}")
    print(f"  Reason: {engine_tsla.circuit_breaker.halt_reason}")
    exchange.place_order("trader_alice", "TSLA", Side.BUY, OrderType.LIMIT, 100, 215.00)

    # Trade history
    print("\n--- Trade History (AAPL) ---")
    for trade in exchange.trade_recorder.get_trades("AAPL"):
        print(f"  {trade}")

    print("\n--- Market Data Feed (Last 5 Events) ---")
    for event in exchange.market_data.get_feed()[-5:]:
        print(f"  {event}")

    print("\n" + "=" * 60)
    print("Simulation complete.")


if __name__ == "__main__":
    run_simulation()
