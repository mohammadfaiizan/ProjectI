#!/usr/bin/env python3
"""
Market-Based Coordination: Resource Allocation Through Economic Principles
=========================================================================

WHAT IS THE PROBLEM?
==================
Centrally planning resource allocation becomes impossible as systems grow complex. Market mechanisms provide efficient, decentralized coordination.

Example: Soviet vs Market Economy
BAD APPROACH (Central Planning):
- Government decides what to produce, how much, for whom
- No price signals to indicate demand/supply
- Massive inefficiencies and shortages
- Slow adaptation to changing needs
- Information bottlenecks at the top

REAL WORLD EXAMPLE:
=================
How does Amazon's pricing work?

MARKET-BASED PRICING:
- Sellers set prices based on supply/demand
- Buyers choose based on price/value
- High demand → prices rise → more sellers enter
- Oversupply → prices fall → weak sellers exit
- Automatic resource allocation without central control

MARKET MECHANISMS:
1. Price discovery through bidding
2. Supply and demand balancing
3. Competition drives efficiency
4. Automatic resource reallocation
5. Information aggregation through prices

THE ALGORITHM:
=============
1. SETUP: Create market with agents who can buy/sell
2. DISCOVER: Agents announce what they want/offer
3. NEGOTIATE: Buyers and sellers negotiate prices
4. MATCH: Match buyers with sellers at agreed prices
5. EXECUTE: Complete transactions
6. ADAPT: Prices adjust based on market feedback

WHY IS THIS POWERFUL?
===================
- Efficient resource allocation without central planning
- Automatic price discovery and optimization
- Self-organizing system that adapts to changes
- Scales to massive numbers of participants
- Robust to individual agent failures
"""

import asyncio
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

class ResourceType(Enum):
    CPU_TIME = "cpu_time"
    MEMORY = "memory"
    STORAGE = "storage"
    BANDWIDTH = "bandwidth"
    DATA_PROCESSING = "data_processing"
    ANALYSIS_SERVICE = "analysis_service"

class TransactionType(Enum):
    BUY = "buy"
    SELL = "sell"

class AuctionType(Enum):
    ENGLISH = "english"      # Ascending price
    DUTCH = "dutch"          # Descending price
    SEALED_BID = "sealed_bid"  # Single sealed bid
    DOUBLE = "double"        # Both buyers and sellers bid

@dataclass
class MarketOrder:
    """Order to buy or sell resources"""
    id: str
    agent_id: str
    resource_type: ResourceType
    quantity: int
    price_limit: float  # Max willing to pay (buy) or min willing to accept (sell)
    transaction_type: TransactionType
    timestamp: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    filled: bool = False

@dataclass
class Transaction:
    """Completed market transaction"""
    id: str
    buyer_id: str
    seller_id: str
    resource_type: ResourceType
    quantity: int
    price: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class MarketData:
    """Current market state for a resource"""
    resource_type: ResourceType
    current_price: float
    bid_prices: List[float]  # Buy orders
    ask_prices: List[float]  # Sell orders
    recent_transactions: List[Transaction]
    total_volume: int
    price_history: List[Tuple[float, float]]  # (timestamp, price)

class MarketAgent:
    """Agent that participates in market-based coordination"""
    
    def __init__(self, agent_id: str, initial_cash: float, initial_resources: Dict[ResourceType, int]):
        self.agent_id = agent_id
        self.cash = initial_cash
        self.resources = initial_resources.copy()
        
        # Market behavior
        self.active_orders: Dict[str, MarketOrder] = {}
        self.transaction_history: List[Transaction] = []
        self.market_knowledge: Dict[ResourceType, MarketData] = {}
        
        # Agent preferences and strategy
        self.risk_tolerance = random.uniform(0.3, 0.9)
        self.price_sensitivity = random.uniform(0.5, 1.5)
        self.trading_strategy = random.choice(["conservative", "aggressive", "balanced"])
        
        # Resource needs and production capacity
        self.resource_needs: Dict[ResourceType, int] = {}
        self.production_capacity: Dict[ResourceType, int] = {}
        
    def assess_resource_needs(self) -> Dict[ResourceType, int]:
        """Assess current resource needs"""
        needs = {}
        
        # Simulate dynamic resource needs
        for resource_type in ResourceType:
            current_amount = self.resources.get(resource_type, 0)
            desired_amount = random.randint(50, 200)
            
            if current_amount < desired_amount:
                needs[resource_type] = desired_amount - current_amount
        
        self.resource_needs = needs
        return needs
    
    def assess_excess_resources(self) -> Dict[ResourceType, int]:
        """Assess resources available for sale"""
        excess = {}
        
        for resource_type, amount in self.resources.items():
            min_reserve = random.randint(20, 50)  # Keep some reserve
            if amount > min_reserve:
                excess[resource_type] = amount - min_reserve
        
        return excess
    
    async def create_buy_order(self, resource_type: ResourceType, quantity: int, 
                             market: 'MarketSystem') -> MarketOrder:
        """Create order to buy resources"""
        
        # Determine price limit based on market data and strategy
        current_price = market.get_current_price(resource_type)
        
        if self.trading_strategy == "aggressive":
            price_limit = current_price * random.uniform(1.1, 1.3)  # Pay premium
        elif self.trading_strategy == "conservative":
            price_limit = current_price * random.uniform(0.8, 0.95)  # Pay less
        else:  # balanced
            price_limit = current_price * random.uniform(0.95, 1.1)
        
        order = MarketOrder(
            id=f"buy_{self.agent_id}_{int(time.time())}_{random.randint(1000, 9999)}",
            agent_id=self.agent_id,
            resource_type=resource_type,
            quantity=quantity,
            price_limit=price_limit,
            transaction_type=TransactionType.BUY,
            expires_at=time.time() + 300  # 5 minutes
        )
        
        self.active_orders[order.id] = order
        return order
    
    async def create_sell_order(self, resource_type: ResourceType, quantity: int, 
                              market: 'MarketSystem') -> MarketOrder:
        """Create order to sell resources"""
        
        # Check if we have enough to sell
        available = self.resources.get(resource_type, 0)
        if available < quantity:
            quantity = available
        
        if quantity <= 0:
            return None
        
        # Determine price limit
        current_price = market.get_current_price(resource_type)
        
        if self.trading_strategy == "aggressive":
            price_limit = current_price * random.uniform(1.0, 1.2)  # Ask higher
        elif self.trading_strategy == "conservative":
            price_limit = current_price * random.uniform(0.9, 1.0)  # Quick sale
        else:  # balanced
            price_limit = current_price * random.uniform(0.95, 1.1)
        
        order = MarketOrder(
            id=f"sell_{self.agent_id}_{int(time.time())}_{random.randint(1000, 9999)}",
            agent_id=self.agent_id,
            resource_type=resource_type,
            quantity=quantity,
            price_limit=price_limit,
            transaction_type=TransactionType.SELL,
            expires_at=time.time() + 300
        )
        
        self.active_orders[order.id] = order
        return order
    
    async def participate_in_market(self, market: 'MarketSystem') -> Dict[str, Any]:
        """Actively participate in market trading"""
        
        print(f"  {self.agent_id} entering market (Cash: ${self.cash:.0f})")
        
        participation_results = {
            "orders_created": 0,
            "transactions_completed": 0,
            "profit_loss": 0.0
        }
        
        initial_cash = self.cash
        
        # Assess needs and create buy orders
        needs = self.assess_resource_needs()
        for resource_type, needed_quantity in needs.items():
            if self.cash > 100:  # Only if we have cash
                buy_order = await self.create_buy_order(resource_type, needed_quantity, market)
                if buy_order:
                    await market.submit_order(buy_order)
                    participation_results["orders_created"] += 1
        
        # Assess excess and create sell orders
        excess = self.assess_excess_resources()
        for resource_type, excess_quantity in excess.items():
            sell_order = await self.create_sell_order(resource_type, excess_quantity, market)
            if sell_order:
                await market.submit_order(sell_order)
                participation_results["orders_created"] += 1
        
        # Wait for some trading to happen
        await asyncio.sleep(0.5)
        
        # Update profit/loss
        participation_results["profit_loss"] = self.cash - initial_cash
        participation_results["transactions_completed"] = len(self.transaction_history)
        
        return participation_results
    
    def execute_transaction(self, transaction: Transaction, is_buyer: bool) -> None:
        """Execute a completed transaction"""
        
        if is_buyer:
            # Buying: lose cash, gain resources
            self.cash -= transaction.price * transaction.quantity
            current_amount = self.resources.get(transaction.resource_type, 0)
            self.resources[transaction.resource_type] = current_amount + transaction.quantity
        else:
            # Selling: gain cash, lose resources
            self.cash += transaction.price * transaction.quantity
            current_amount = self.resources.get(transaction.resource_type, 0)
            self.resources[transaction.resource_type] = max(0, current_amount - transaction.quantity)
        
        self.transaction_history.append(transaction)
        print(f"    {self.agent_id} {'bought' if is_buyer else 'sold'} "
              f"{transaction.quantity} {transaction.resource_type.value} "
              f"@ ${transaction.price:.2f} each")
    
    def get_portfolio_value(self, market: 'MarketSystem') -> float:
        """Calculate total portfolio value"""
        
        total_value = self.cash
        
        for resource_type, quantity in self.resources.items():
            price = market.get_current_price(resource_type)
            total_value += price * quantity
        
        return total_value
    
    def get_agent_summary(self) -> Dict[str, Any]:
        """Get summary of agent's market activity"""
        
        return {
            "agent_id": self.agent_id,
            "cash": self.cash,
            "resources": dict(self.resources),
            "active_orders": len(self.active_orders),
            "total_transactions": len(self.transaction_history),
            "trading_strategy": self.trading_strategy,
            "risk_tolerance": self.risk_tolerance
        }

class MarketSystem:
    """
    Market-based coordination system
    
    EXAMPLE USAGE:
    =============
    # Create market system
    market = MarketSystem("resource_market")
    
    # Add market participants
    for i in range(10):
        agent = MarketAgent(f"agent_{i}", 1000.0, {ResourceType.CPU_TIME: 100})
        market.add_participant(agent)
    
    # Run market trading session
    result = await market.run_trading_session(duration=60)
    """
    
    def __init__(self, market_id: str):
        self.market_id = market_id
        self.participants: Dict[str, MarketAgent] = {}
        
        # Market state
        self.order_book: Dict[ResourceType, Dict[str, MarketOrder]] = {}
        self.completed_transactions: List[Transaction] = []
        self.market_data: Dict[ResourceType, MarketData] = {}
        
        # Initialize market data for each resource type
        for resource_type in ResourceType:
            self.market_data[resource_type] = MarketData(
                resource_type=resource_type,
                current_price=random.uniform(10.0, 50.0),  # Initial random prices
                bid_prices=[],
                ask_prices=[],
                recent_transactions=[],
                total_volume=0,
                price_history=[]
            )
            
            self.order_book[resource_type] = {}
    
    def add_participant(self, agent: MarketAgent) -> None:
        """Add participant to market"""
        self.participants[agent.agent_id] = agent
        print(f"Added market participant: {agent.agent_id}")
    
    async def submit_order(self, order: MarketOrder) -> bool:
        """Submit order to market"""
        
        self.order_book[order.resource_type][order.id] = order
        
        # Try to match order immediately
        await self.match_orders(order.resource_type)
        
        return True
    
    async def match_orders(self, resource_type: ResourceType) -> List[Transaction]:
        """Match buy and sell orders for a resource type"""
        
        orders = self.order_book[resource_type]
        
        # Separate buy and sell orders
        buy_orders = [o for o in orders.values() 
                     if o.transaction_type == TransactionType.BUY and not o.filled]
        sell_orders = [o for o in orders.values() 
                      if o.transaction_type == TransactionType.SELL and not o.filled]
        
        # Sort orders: buy orders by price (highest first), sell orders by price (lowest first)
        buy_orders.sort(key=lambda x: x.price_limit, reverse=True)
        sell_orders.sort(key=lambda x: x.price_limit)
        
        transactions = []
        
        # Match orders
        i, j = 0, 0
        while i < len(buy_orders) and j < len(sell_orders):
            buy_order = buy_orders[i]
            sell_order = sell_orders[j]
            
            # Check if orders can be matched
            if buy_order.price_limit >= sell_order.price_limit:
                # Orders match! Execute transaction
                
                # Transaction price is typically the average or the earlier order's price
                transaction_price = (buy_order.price_limit + sell_order.price_limit) / 2
                
                # Transaction quantity is minimum of both orders
                transaction_quantity = min(buy_order.quantity, sell_order.quantity)
                
                # Create transaction
                transaction = Transaction(
                    id=f"txn_{int(time.time())}_{random.randint(1000, 9999)}",
                    buyer_id=buy_order.agent_id,
                    seller_id=sell_order.agent_id,
                    resource_type=resource_type,
                    quantity=transaction_quantity,
                    price=transaction_price
                )
                
                # Execute transaction with agents
                buyer = self.participants[buy_order.agent_id]
                seller = self.participants[sell_order.agent_id]
                
                buyer.execute_transaction(transaction, True)
                seller.execute_transaction(transaction, False)
                
                # Update orders
                buy_order.quantity -= transaction_quantity
                sell_order.quantity -= transaction_quantity
                
                if buy_order.quantity <= 0:
                    buy_order.filled = True
                    i += 1
                
                if sell_order.quantity <= 0:
                    sell_order.filled = True
                    j += 1
                
                # Record transaction
                transactions.append(transaction)
                self.completed_transactions.append(transaction)
                
                # Update market data
                await self.update_market_data(resource_type, transaction)
                
            else:
                # No more matches possible
                break
        
        return transactions
    
    async def update_market_data(self, resource_type: ResourceType, transaction: Transaction) -> None:
        """Update market data after transaction"""
        
        market_data = self.market_data[resource_type]
        
        # Update current price
        market_data.current_price = transaction.price
        
        # Add to recent transactions
        market_data.recent_transactions.append(transaction)
        if len(market_data.recent_transactions) > 10:
            market_data.recent_transactions.pop(0)
        
        # Update volume
        market_data.total_volume += transaction.quantity
        
        # Update price history
        market_data.price_history.append((transaction.timestamp, transaction.price))
        if len(market_data.price_history) > 50:
            market_data.price_history.pop(0)
        
        # Update bid/ask prices from current orders
        orders = self.order_book[resource_type]
        
        market_data.bid_prices = [o.price_limit for o in orders.values() 
                                 if o.transaction_type == TransactionType.BUY and not o.filled]
        market_data.ask_prices = [o.price_limit for o in orders.values() 
                                 if o.transaction_type == TransactionType.SELL and not o.filled]
    
    def get_current_price(self, resource_type: ResourceType) -> float:
        """Get current market price for resource"""
        return self.market_data[resource_type].current_price
    
    async def run_trading_session(self, duration: float = 30.0) -> Dict[str, Any]:
        """Run a trading session for specified duration"""
        
        print(f"\nRUNNING MARKET TRADING SESSION")
        print(f"Duration: {duration} seconds")
        print("=" * 50)
        
        start_time = time.time()
        session_transactions = []
        
        # Initial market state
        initial_prices = {rt: self.get_current_price(rt) for rt in ResourceType}
        
        # Run trading session
        while time.time() - start_time < duration:
            # Have agents participate in market
            for agent in self.participants.values():
                if random.random() < 0.3:  # 30% chance each round
                    await agent.participate_in_market(self)
            
            # Match orders for all resource types
            for resource_type in ResourceType:
                transactions = await self.match_orders(resource_type)
                session_transactions.extend(transactions)
            
            # Clean up expired orders
            await self.clean_expired_orders()
            
            # Brief pause
            await asyncio.sleep(0.2)
        
        session_time = time.time() - start_time
        
        # Final market state
        final_prices = {rt: self.get_current_price(rt) for rt in ResourceType}
        
        # Calculate results
        price_changes = {rt: final_prices[rt] - initial_prices[rt] for rt in ResourceType}
        
        print(f"\nTrading session completed in {session_time:.2f} seconds")
        print(f"Total transactions: {len(session_transactions)}")
        
        return {
            "session_duration": session_time,
            "total_transactions": len(session_transactions),
            "transactions": session_transactions,
            "initial_prices": initial_prices,
            "final_prices": final_prices,
            "price_changes": price_changes,
            "market_summary": self.get_market_summary()
        }
    
    async def clean_expired_orders(self) -> None:
        """Remove expired orders from order book"""
        
        current_time = time.time()
        
        for resource_type in ResourceType:
            orders = self.order_book[resource_type]
            expired_orders = [order_id for order_id, order in orders.items() 
                            if order.expires_at and current_time > order.expires_at]
            
            for order_id in expired_orders:
                del orders[order_id]
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get comprehensive market summary"""
        
        summary = {
            "market_id": self.market_id,
            "total_participants": len(self.participants),
            "total_transactions": len(self.completed_transactions),
            "resource_markets": {}
        }
        
        for resource_type, market_data in self.market_data.items():
            summary["resource_markets"][resource_type.value] = {
                "current_price": market_data.current_price,
                "total_volume": market_data.total_volume,
                "active_buy_orders": len(market_data.bid_prices),
                "active_sell_orders": len(market_data.ask_prices),
                "recent_transactions": len(market_data.recent_transactions)
            }
        
        # Participant performance
        participant_summaries = []
        for agent in self.participants.values():
            portfolio_value = agent.get_portfolio_value(self)
            participant_summaries.append({
                "agent_id": agent.agent_id,
                "portfolio_value": portfolio_value,
                "transactions": len(agent.transaction_history),
                "strategy": agent.trading_strategy
            })
        
        summary["participants"] = participant_summaries
        
        return summary

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_cloud_resource_market():
    """Demo: Cloud computing resource marketplace"""
    print("\nDEMO 1: CLOUD RESOURCE MARKETPLACE")
    print("=" * 50)
    
    market = MarketSystem("cloud_market")
    
    # Add different types of participants
    # Big companies (high cash, high needs)
    for i in range(3):
        resources = {ResourceType.CPU_TIME: random.randint(200, 500)}
        agent = MarketAgent(f"big_corp_{i}", 5000.0, resources)
        agent.trading_strategy = "aggressive"
        market.add_participant(agent)
    
    # Startups (low cash, moderate needs)
    for i in range(5):
        resources = {ResourceType.MEMORY: random.randint(50, 150)}
        agent = MarketAgent(f"startup_{i}", 1000.0, resources)
        agent.trading_strategy = "conservative"
        market.add_participant(agent)
    
    # Cloud providers (high resources, need cash)
    for i in range(2):
        resources = {
            ResourceType.CPU_TIME: random.randint(1000, 2000),
            ResourceType.MEMORY: random.randint(800, 1500),
            ResourceType.STORAGE: random.randint(5000, 10000)
        }
        agent = MarketAgent(f"cloud_provider_{i}", 2000.0, resources)
        agent.trading_strategy = "balanced"
        market.add_participant(agent)
    
    # Run trading session
    result = await market.run_trading_session(duration=10.0)
    
    print(f"\nMarket Results:")
    print(f"- Transactions completed: {result['total_transactions']}")
    
    # Show price movements
    for resource, change in result['price_changes'].items():
        direction = "↑" if change > 0 else "↓" if change < 0 else "→"
        print(f"- {resource.value}: {direction} ${change:+.2f}")

async def demo_freelancer_service_market():
    """Demo: Freelancer service marketplace"""
    print("\nDEMO 2: FREELANCER SERVICE MARKETPLACE")
    print("=" * 50)
    
    market = MarketSystem("freelancer_market")
    
    # Add clients needing services
    for i in range(4):
        resources = {}  # Clients start with no services
        agent = MarketAgent(f"client_{i}", 2000.0, resources)
        market.add_participant(agent)
    
    # Add freelancers offering services
    for i in range(6):
        resources = {
            ResourceType.DATA_PROCESSING: random.randint(20, 50),
            ResourceType.ANALYSIS_SERVICE: random.randint(10, 30)
        }
        agent = MarketAgent(f"freelancer_{i}", 500.0, resources)
        market.add_participant(agent)
    
    # Run market session
    result = await market.run_trading_session(duration=8.0)
    
    # Show final market state
    summary = result['market_summary']
    print(f"\nFreelancer Market Summary:")
    print(f"- Active participants: {summary['total_participants']}")
    print(f"- Service transactions: {summary['total_transactions']}")

async def demo_energy_trading_market():
    """Demo: Energy trading marketplace"""
    print("\nDEMO 3: ENERGY TRADING MARKETPLACE")
    print("=" * 50)
    
    market = MarketSystem("energy_market")
    
    # Energy producers (power plants, solar farms)
    for i in range(3):
        resources = {ResourceType.BANDWIDTH: random.randint(500, 1000)}  # Using bandwidth as energy proxy
        agent = MarketAgent(f"energy_producer_{i}", 3000.0, resources)
        agent.trading_strategy = "balanced"
        market.add_participant(agent)
    
    # Energy consumers (factories, data centers)
    for i in range(5):
        resources = {ResourceType.BANDWIDTH: random.randint(10, 50)}
        agent = MarketAgent(f"energy_consumer_{i}", 4000.0, resources)
        agent.trading_strategy = "conservative"
        market.add_participant(agent)
    
    # Energy traders (buy low, sell high)
    for i in range(2):
        resources = {ResourceType.BANDWIDTH: random.randint(100, 200)}
        agent = MarketAgent(f"energy_trader_{i}", 1500.0, resources)
        agent.trading_strategy = "aggressive"
        market.add_participant(agent)
    
    # Run energy trading session
    result = await market.run_trading_session(duration=12.0)
    
    print(f"\nEnergy Market Results:")
    participants = result['market_summary']['participants']
    
    # Show top performers
    top_performers = sorted(participants, key=lambda x: x['portfolio_value'], reverse=True)[:3]
    for i, performer in enumerate(top_performers):
        print(f"  {i+1}. {performer['agent_id']}: ${performer['portfolio_value']:.0f} portfolio value")

async def main():
    """
    Demonstrate Market-Based Coordination for resource allocation
    
    WHAT YOU'LL LEARN:
    ================
    1. How market mechanisms enable efficient resource allocation
    2. How price discovery works through supply and demand
    3. How to implement order matching and transaction processing
    4. How market-based systems scale and adapt automatically
    5. How economic incentives drive system optimization
    
    REAL WORLD APPLICATIONS:
    =======================
    - Cloud computing resource allocation (AWS Spot Instances)
    - Electricity grid management and energy trading
    - Freelancer and gig economy platforms
    - Supply chain and logistics optimization
    - Advertising auction systems (Google Ads)
    - Financial and commodity trading markets
    """
    
    print("MARKET-BASED COORDINATION DEMONSTRATION")
    print("This shows how economic principles enable efficient resource allocation!")
    
    await demo_cloud_resource_market()
    await demo_freelancer_service_market()
    await demo_energy_trading_market()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Market mechanisms provide efficient decentralized coordination")
    print("✓ Price discovery automatically balances supply and demand")
    print("✓ Economic incentives drive optimal resource allocation")
    print("✓ Market systems scale and adapt without central planning")
    print("✓ Competition and trading improve overall system efficiency")

if __name__ == "__main__":
    asyncio.run(main())
