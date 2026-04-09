"""
STOCK EXCHANGE — Financial Trading Platform
============================================

FUNCTIONAL REQUIREMENTS:
- Place orders: market, limit, stop-loss, stop-limit
- Order matching: continuous trading with price-time priority
- Order book: real-time bids and asks
- Trade execution: fill orders, partial fills
- Market data: real-time quotes, OHLCV, trade history
- Account management: positions, P&L, margin
- Cancel/amend orders

NON-FUNCTIONAL REQUIREMENTS:
- Throughput: 1 M orders/second (NSE/NYSE peak)
- Latency: < 10 microseconds for matching engine (co-located traders)
- Consistency: absolutely no duplicate trades or phantom orders
- Audit trail: every order, amendment, cancellation permanently logged
- Market integrity: circuit breakers, price bands

ARCHITECTURE:
  Order Gateway ──▶ Risk Engine ──▶ Matching Engine ──▶ Trade Reporting
                                          │
                                    Order Book (in-memory)
                                    (sorted data structures)

KEY DESIGN DECISIONS:
1. ORDER BOOK — two priority queues:
   - Bids: max-heap sorted by price DESC, then time ASC (FIFO)
   - Asks: min-heap sorted by price ASC, then time ASC
   Implemented with sorted dict (price → deque of orders).

2. MATCHING ALGORITHM — Price-time priority (FIFO):
   When order arrives, match against opposite side at best price.
   Partial fills: remainder stays in book.
   Fully filled: remove from book.

3. ORDER TYPES:
   - Market: match immediately at best available price
   - Limit: match if price satisfies condition, else rest in book
   - Stop: resting order that activates when price crosses stop price
   - IOC (Immediate or Cancel): fill what you can, cancel rest
   - FOK (Fill or Kill): fill entirely or cancel

4. MATCHING ENGINE ARCHITECTURE:
   Single-threaded per instrument (avoids locks on hot path).
   Multiple instruments handled by independent engines.
   Sequence numbers ensure deterministic replay from event log.

5. CIRCUIT BREAKERS:
   - L1: ±10% move in 5 min → 15-minute trading halt
   - L2: ±15% move from open → 30-minute halt
   Market-wide: NYSE triggers at -7%, -13%, -20%

6. MARKET DATA:
   - Level 1: best bid/ask + last trade
   - Level 2: full order book depth
   - Tick data: every trade event with timestamp
"""

from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque
from enum import Enum
from collections import defaultdict, deque
import threading
from decimal import Decimal, ROUND_HALF_UP


# ---------------------------------------------------------------------------
# Enums and Types
# ---------------------------------------------------------------------------

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(Enum):
    GTC = "good_till_cancel"
    IOC = "immediate_or_cancel"
    FOK = "fill_or_kill"
    DAY = "day"


Price = Decimal


# ---------------------------------------------------------------------------
# Order Model
# ---------------------------------------------------------------------------

@dataclass
class Order:
    order_id: str
    symbol: str
    trader_id: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[Price]       # None for market orders
    stop_price: Optional[Price]  # For stop / stop-limit orders
    time_in_force: TimeInForce
    status: OrderStatus = OrderStatus.NEW
    filled_qty: int = 0
    avg_fill_price: Optional[Price] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    sequence_num: int = 0        # Exchange-assigned sequence number

    @property
    def remaining_qty(self) -> int:
        return self.quantity - self.filled_qty

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED)

    def __lt__(self, other: "Order") -> bool:
        """Used for priority queue comparison (price-time priority)."""
        if self.side == OrderSide.BUY:
            # Higher price = higher priority; same price → earlier time
            if self.price != other.price:
                return self.price > other.price
        else:
            # Lower price = higher priority; same price → earlier time
            if self.price != other.price:
                return self.price < other.price
        return self.sequence_num < other.sequence_num


# ---------------------------------------------------------------------------
# Trade / Fill
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    trade_id: str
    symbol: str
    buy_order_id: str
    sell_order_id: str
    price: Price
    quantity: int
    buyer_id: str
    seller_id: str
    executed_at: float = field(default_factory=time.time)
    sequence_num: int = 0

    @property
    def notional(self) -> Price:
        return self.price * self.quantity


# ---------------------------------------------------------------------------
# Order Book — price-time priority
# ---------------------------------------------------------------------------

class OrderBook:
    """
    Bids: sorted price DESC → orders at same price sorted by sequence_num ASC
    Asks: sorted price ASC → orders at same price sorted by sequence_num ASC
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        # price → deque of orders (FIFO at same price level)
        self._bids: Dict[Price, Deque[Order]] = {}
        self._asks: Dict[Price, Deque[Order]] = {}
        self._order_index: Dict[str, Order] = {}

    def add(self, order: Order) -> None:
        book = self._bids if order.side == OrderSide.BUY else self._asks
        price = order.price
        if price not in book:
            book[price] = deque()
        book[price].append(order)
        self._order_index[order.order_id] = order

    def remove(self, order_id: str) -> Optional[Order]:
        order = self._order_index.pop(order_id, None)
        if not order or not order.price:
            return order
        book = self._bids if order.side == OrderSide.BUY else self._asks
        price_level = book.get(order.price)
        if price_level:
            # Remove from deque (O(n) but acceptable for cancel rates)
            remaining = deque(o for o in price_level if o.order_id != order_id)
            if remaining:
                book[order.price] = remaining
            else:
                del book[order.price]
        return order

    def best_bid(self) -> Optional[Price]:
        return max(self._bids.keys()) if self._bids else None

    def best_ask(self) -> Optional[Price]:
        return min(self._asks.keys()) if self._asks else None

    def top_bid_orders(self) -> Deque[Order]:
        best = self.best_bid()
        return self._bids[best] if best else deque()

    def top_ask_orders(self) -> Deque[Order]:
        best = self.best_ask()
        return self._asks[best] if best else deque()

    def spread(self) -> Optional[Price]:
        bid = self.best_bid()
        ask = self.best_ask()
        if bid and ask:
            return ask - bid
        return None

    def depth(self, levels: int = 5) -> Dict[str, List[Tuple[Price, int]]]:
        """Returns top N price levels for bids and asks."""
        bid_prices = sorted(self._bids.keys(), reverse=True)[:levels]
        ask_prices = sorted(self._asks.keys())[:levels]

        bids = [(p, sum(o.remaining_qty for o in self._bids[p])) for p in bid_prices]
        asks = [(p, sum(o.remaining_qty for o in self._asks[p])) for p in ask_prices]
        return {"bids": bids, "asks": asks}


# ---------------------------------------------------------------------------
# Matching Engine
# ---------------------------------------------------------------------------

class MatchingEngine:
    """
    Single-threaded per instrument.
    Processes orders sequentially to ensure deterministic matching.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self._book = OrderBook(symbol)
        self._trades: List[Trade] = []
        self._sequence = 0
        self._last_price: Optional[Price] = None

    def _next_seq(self) -> int:
        self._sequence += 1
        return self._sequence

    def submit(self, order: Order) -> Tuple[List[Trade], Order]:
        """Submit order for matching. Returns (trades, updated_order)."""
        order.sequence_num = self._next_seq()

        if order.order_type == OrderType.MARKET:
            return self._match_market(order)
        elif order.order_type == OrderType.LIMIT:
            return self._match_limit(order)
        else:
            # Stop/stop-limit: add to stop list, activate on price cross
            order.status = OrderStatus.NEW
            return [], order

    def _match_market(self, order: Order) -> Tuple[List[Trade], Order]:
        trades = []
        while order.remaining_qty > 0:
            if order.side == OrderSide.BUY:
                opposite = self._book.top_ask_orders()
            else:
                opposite = self._book.top_bid_orders()

            if not opposite:
                break  # No liquidity

            resting = opposite[0]
            trade = self._execute(order, resting)
            trades.append(trade)

            if resting.remaining_qty == 0:
                self._book.remove(resting.order_id)

        if order.remaining_qty > 0:
            if order.time_in_force == TimeInForce.IOC:
                order.status = OrderStatus.CANCELLED
            elif order.filled_qty > 0:
                order.status = OrderStatus.PARTIALLY_FILLED
            # For market orders: partially fill is unusual (circuit breaker would halt)

        return trades, order

    def _match_limit(self, order: Order) -> Tuple[List[Trade], Order]:
        trades = []
        while order.remaining_qty > 0:
            if order.side == OrderSide.BUY:
                best_ask = self._book.best_ask()
                if best_ask is None or order.price < best_ask:
                    break  # No price match
                opposite_orders = self._book.top_ask_orders()
            else:
                best_bid = self._book.best_bid()
                if best_bid is None or order.price > best_bid:
                    break  # No price match
                opposite_orders = self._book.top_bid_orders()

            if not opposite_orders:
                break

            resting = opposite_orders[0]
            trade = self._execute(order, resting)
            trades.append(trade)

            if resting.remaining_qty == 0:
                self._book.remove(resting.order_id)

        # Handle time-in-force
        if order.remaining_qty > 0:
            if order.time_in_force == TimeInForce.FOK:
                # FOK: cancel entirely if not fully filled
                for trade in trades:
                    self._undo_trade(trade)
                trades = []
                order.status = OrderStatus.CANCELLED
            elif order.time_in_force == TimeInForce.IOC:
                order.status = (OrderStatus.PARTIALLY_FILLED if order.filled_qty > 0
                                 else OrderStatus.CANCELLED)
            elif order.is_active:
                # GTC/DAY: rest remaining qty in book
                self._book.add(order)

        return trades, order

    def _execute(self, aggressor: Order, resting: Order) -> Trade:
        qty = min(aggressor.remaining_qty, resting.remaining_qty)
        price = resting.price  # Price improvement: resting order price

        buy_order = aggressor if aggressor.side == OrderSide.BUY else resting
        sell_order = aggressor if aggressor.side == OrderSide.SELL else resting

        trade = Trade(
            trade_id=str(uuid.uuid4())[:10],
            symbol=self.symbol,
            buy_order_id=buy_order.order_id,
            sell_order_id=sell_order.order_id,
            price=price,
            quantity=qty,
            buyer_id=buy_order.trader_id,
            seller_id=sell_order.trader_id,
            sequence_num=self._next_seq(),
        )
        self._trades.append(trade)

        # Update orders
        for order in (aggressor, resting):
            order.filled_qty += qty
            if order.avg_fill_price is None:
                order.avg_fill_price = price
            else:
                # Volume-weighted average
                prev_notional = order.avg_fill_price * (order.filled_qty - qty)
                curr_notional = price * qty
                order.avg_fill_price = (prev_notional + curr_notional) / order.filled_qty
            order.status = (OrderStatus.FILLED if order.remaining_qty == 0
                             else OrderStatus.PARTIALLY_FILLED)
            order.updated_at = time.time()

        self._last_price = price
        return trade

    def _undo_trade(self, trade: Trade) -> None:
        """Rollback a trade (for FOK cancellation)."""
        for order_id in (trade.buy_order_id, trade.sell_order_id):
            order = self._book._order_index.get(order_id)
            if order:
                order.filled_qty -= trade.quantity
                if order.filled_qty == 0:
                    order.status = OrderStatus.NEW
                    order.avg_fill_price = None
                else:
                    order.status = OrderStatus.PARTIALLY_FILLED
        self._trades.pop()

    def cancel(self, order_id: str, trader_id: str) -> Optional[Order]:
        order = self._book._order_index.get(order_id)
        if not order or order.trader_id != trader_id:
            return None
        if not order.is_active:
            return None
        self._book.remove(order_id)
        order.status = OrderStatus.CANCELLED
        order.updated_at = time.time()
        return order

    @property
    def last_price(self) -> Optional[Price]:
        return self._last_price

    @property
    def order_book(self) -> OrderBook:
        return self._book

    @property
    def trade_history(self) -> List[Trade]:
        return list(self._trades)


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Halts trading when price moves exceed thresholds."""

    def __init__(self, l1_pct: float = 10.0, l2_pct: float = 15.0):
        self._l1_pct = l1_pct
        self._l2_pct = l2_pct
        self._open_price: Dict[str, Price] = {}
        self._halted: Dict[str, float] = {}    # symbol → halt_until timestamp

    def set_open_price(self, symbol: str, price: Price):
        self._open_price[symbol] = price

    def check_and_halt(self, symbol: str, current_price: Price) -> Optional[str]:
        open_price = self._open_price.get(symbol)
        if not open_price:
            return None
        move_pct = abs(float(current_price - open_price) / float(open_price)) * 100

        halt_duration = None
        if move_pct >= self._l2_pct:
            halt_duration = 1800  # 30 minutes
            level = "L2"
        elif move_pct >= self._l1_pct:
            halt_duration = 900   # 15 minutes
            level = "L1"

        if halt_duration:
            self._halted[symbol] = time.time() + halt_duration
            return f"HALT {level}: {symbol} moved {move_pct:.1f}% from open"
        return None

    def is_halted(self, symbol: str) -> bool:
        until = self._halted.get(symbol)
        return until is not None and time.time() < until


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def make_order(symbol, trader, side, order_type, qty, price=None,
               tif=TimeInForce.GTC) -> Order:
    return Order(
        order_id=str(uuid.uuid4())[:8],
        symbol=symbol,
        trader_id=trader,
        side=side,
        order_type=order_type,
        quantity=qty,
        price=Price(str(price)) if price else None,
        stop_price=None,
        time_in_force=tif,
    )


def demonstrate_1_order_book_matching():
    print("\n=== 1. Limit Order Book & Matching ===")
    engine = MatchingEngine("AAPL")

    # Resting sell orders (asks)
    sell_orders = [
        make_order("AAPL", "mm_a", OrderSide.SELL, OrderType.LIMIT, 100, 150.50),
        make_order("AAPL", "mm_b", OrderSide.SELL, OrderType.LIMIT, 200, 151.00),
        make_order("AAPL", "mm_c", OrderSide.SELL, OrderType.LIMIT, 150, 150.50),  # same price, later
    ]
    for o in sell_orders:
        trades, _ = engine.submit(o)

    # Resting buy orders (bids)
    buy_orders = [
        make_order("AAPL", "mm_d", OrderSide.BUY, OrderType.LIMIT, 300, 149.00),
        make_order("AAPL", "mm_e", OrderSide.BUY, OrderType.LIMIT, 100, 149.50),
    ]
    for o in buy_orders:
        engine.submit(o)

    # Display order book
    depth = engine.order_book.depth(3)
    print(f"Order Book (AAPL):")
    print(f"  ASKS:")
    for price, qty in depth["asks"]:
        print(f"    ${price} × {qty}")
    print(f"  Spread: ${engine.order_book.spread()}")
    print(f"  BIDS:")
    for price, qty in depth["bids"]:
        print(f"    ${price} × {qty}")

    # Aggressive buy order — crosses the spread
    aggressive_buy = make_order("AAPL", "trader_x", OrderSide.BUY,
                                  OrderType.LIMIT, 250, 151.00)
    trades, order = engine.submit(aggressive_buy)
    print(f"\nAggressive buy 250 @ $151.00:")
    print(f"  Trades executed: {len(trades)}")
    for t in trades:
        print(f"    Filled {t.quantity} @ ${t.price} (trade_id={t.trade_id[:6]})")
    print(f"  Order status: {order.status.value}, filled={order.filled_qty}, "
          f"remaining={order.remaining_qty}")
    print(f"  Avg fill price: ${order.avg_fill_price}")


def demonstrate_2_market_order():
    print("\n=== 2. Market Order — Best Available Price ===")
    engine = MatchingEngine("TSLA")

    # Seed the book
    for price, qty in [(200.00, 100), (200.50, 200), (201.00, 500)]:
        o = make_order("TSLA", "mm", OrderSide.SELL, OrderType.LIMIT, qty, price)
        engine.submit(o)

    print(f"Best ask before market order: ${engine.order_book.best_ask()}")

    # Market order sweeps multiple price levels
    market_buy = make_order("TSLA", "buyer", OrderSide.BUY,
                              OrderType.MARKET, 350)
    trades, order = engine.submit(market_buy)

    print(f"Market buy 350 shares:")
    for t in trades:
        print(f"  Filled {t.quantity} @ ${t.price}")
    print(f"Avg fill: ${order.avg_fill_price:.2f}")
    print(f"New best ask: ${engine.order_book.best_ask()}")


def demonstrate_3_ioc_and_fok():
    print("\n=== 3. IOC and FOK Time-in-Force ===")
    engine = MatchingEngine("GOOG")

    # Only 100 shares available
    o = make_order("GOOG", "mm", OrderSide.SELL, OrderType.LIMIT, 100, 170.00)
    engine.submit(o)

    # IOC buy for 300 — fill 100, cancel 200
    ioc_order = make_order("GOOG", "buyer_a", OrderSide.BUY, OrderType.LIMIT,
                             300, 170.00, tif=TimeInForce.IOC)
    trades, order = engine.submit(ioc_order)
    print(f"IOC buy 300 @ $170:")
    print(f"  Filled: {order.filled_qty}, Status: {order.status.value}")

    # Seed fresh book
    o2 = make_order("GOOG", "mm2", OrderSide.SELL, OrderType.LIMIT, 50, 170.00)
    engine.submit(o2)

    # FOK buy for 300 — not enough shares → cancel entirely
    fok_order = make_order("GOOG", "buyer_b", OrderSide.BUY, OrderType.LIMIT,
                             300, 170.00, tif=TimeInForce.FOK)
    trades_fok, order_fok = engine.submit(fok_order)
    print(f"\nFOK buy 300 @ $170 (only 50 available):")
    print(f"  Trades: {len(trades_fok)}, Status: {order_fok.status.value}")


def demonstrate_4_circuit_breaker():
    print("\n=== 4. Circuit Breaker ===")
    breaker = CircuitBreaker(l1_pct=10.0, l2_pct=15.0)

    symbol = "MEME"
    open_price = Price("100.00")
    breaker.set_open_price(symbol, open_price)

    test_prices = [105, 109, 111, 115, 116]
    for current in test_prices:
        halt_msg = breaker.check_and_halt(symbol, Price(str(current)))
        is_halted = breaker.is_halted(symbol)
        pct = (current - 100) / 100 * 100
        print(f"  ${current} (+{pct:.0f}%) → "
              f"{'HALTED: ' + halt_msg if halt_msg else ('Trading halted' if is_halted else 'Trading continues')}")


def demonstrate_5_partial_fill_book():
    print("\n=== 5. Partial Fill & Remaining in Book ===")
    engine = MatchingEngine("NVDA")

    # 100 shares available at $500
    sell = make_order("NVDA", "seller", OrderSide.SELL, OrderType.LIMIT, 100, 500.00)
    engine.submit(sell)

    # Buy 300 GTC — only 100 fills, 200 rests in book
    buy = make_order("NVDA", "buyer", OrderSide.BUY, OrderType.LIMIT, 300, 500.00)
    trades, order = engine.submit(buy)

    print(f"Buy 300 GTC @ $500 (only 100 available):")
    print(f"  Filled: {order.filled_qty}, Remaining in book: {order.remaining_qty}")
    print(f"  Status: {order.status.value}")

    # New seller arrives — fills remaining
    sell2 = make_order("NVDA", "seller2", OrderSide.SELL, OrderType.LIMIT, 250, 499.50)
    trades2, sell_order = engine.submit(sell2)
    print(f"\nNew sell 250 @ $499.50 (aggressor):")
    for t in trades2:
        print(f"  Filled {t.quantity} @ ${t.price} (buyer got price improvement!)")

    # Check buyer's order is now fully filled
    book_bids = engine.order_book.depth(3)["bids"]
    print(f"  Remaining bids in book: {book_bids}")


if __name__ == "__main__":
    demonstrate_1_order_book_matching()
    demonstrate_2_market_order()
    demonstrate_3_ioc_and_fok()
    demonstrate_4_circuit_breaker()
    demonstrate_5_partial_fill_book()
