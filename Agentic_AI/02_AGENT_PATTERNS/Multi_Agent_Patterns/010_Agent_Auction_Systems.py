#!/usr/bin/env python3
"""
Agent Auction Systems: Competitive Resource Allocation
======================================================

WHAT IS THE PROBLEM?
==================
How do you fairly allocate limited resources when multiple agents want them? Auctions provide competitive, efficient allocation.

Example: Concert Ticket Chaos
BAD APPROACH:
- First-come-first-served leads to camping out
- Fixed pricing doesn't match true demand
- Scalpers exploit the system
- Fans who value tickets most don't get them

REAL WORLD EXAMPLE:
=================
How does Google Ads auction work?

GOOGLE ADS AUCTION:
1. User searches for "pizza delivery"
2. Google identifies relevant advertisers
3. Each advertiser automatically bids
4. Highest bid gets top position
5. Advertiser pays second-highest bid price
6. System automatically optimizes for revenue

AUCTION BENEFITS:
- Fair competition for resources
- Price discovery through bidding
- Efficient allocation to highest bidders
- Automatic market clearing

THE ALGORITHM:
=============
1. ANNOUNCE: Auction item/resource announced
2. BID: Agents submit bids based on their valuation
3. EVALUATE: Auctioneer compares all bids
4. ALLOCATE: Highest bidder wins the resource
5. PAYMENT: Winner pays according to auction rules
6. DELIVER: Resource is transferred to winner

WHY IS THIS POWERFUL?
===================
- Discovers true market value through competition
- Allocates resources to agents who value them most
- Prevents unfair advantage or manipulation
- Scales to many participants and items
- Automatically adapts to changing demand
"""

import asyncio
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

class AuctionType(Enum):
    ENGLISH = "english"          # Ascending price auction
    DUTCH = "dutch"              # Descending price auction
    SEALED_BID = "sealed_bid"    # Single round sealed bids
    VICKREY = "vickrey"          # Second-price sealed bid
    REVERSE = "reverse"          # Buyers announce what they want, sellers bid

class AuctionStatus(Enum):
    SCHEDULED = "scheduled"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

@dataclass
class AuctionItem:
    """Item being auctioned"""
    id: str
    name: str
    description: str
    starting_price: float
    reserve_price: Optional[float] = None  # Minimum acceptable price
    quantity: int = 1

@dataclass
class Bid:
    """A bid submitted by an agent"""
    id: str
    bidder_id: str
    auction_id: str
    amount: float
    quantity: int = 1
    timestamp: float = field(default_factory=time.time)
    is_active: bool = True

@dataclass
class AuctionResult:
    """Result of completed auction"""
    auction_id: str
    item: AuctionItem
    winning_bid: Optional[Bid]
    all_bids: List[Bid]
    final_price: float
    revenue: float
    completed_at: float = field(default_factory=time.time)

class BiddingAgent:
    """Agent that participates in auctions"""
    
    def __init__(self, agent_id: str, budget: float, bidding_strategy: str = "rational"):
        self.agent_id = agent_id
        self.budget = budget
        self.bidding_strategy = bidding_strategy  # rational, aggressive, conservative, random
        
        # Bidding history and preferences
        self.bid_history: List[Bid] = []
        self.won_auctions: List[AuctionResult] = []
        self.preferences: Dict[str, float] = {}  # item_type -> preference_score
        
        # Strategy parameters
        self.max_budget_per_item = budget * 0.3  # Don't spend more than 30% on one item
        self.risk_tolerance = random.uniform(0.4, 0.9)
        self.patience = random.uniform(0.3, 0.8)  # How long to wait in English auctions
    
    def assess_item_value(self, item: AuctionItem) -> float:
        """Assess personal value of an auction item"""
        
        # Base value assessment
        base_value = item.starting_price * random.uniform(0.8, 2.0)
        
        # Adjust based on preferences
        preference_multiplier = 1.0
        for keyword, preference in self.preferences.items():
            if keyword.lower() in item.name.lower() or keyword.lower() in item.description.lower():
                preference_multiplier *= preference
        
        personal_value = base_value * preference_multiplier
        
        # Strategy adjustments
        if self.bidding_strategy == "aggressive":
            personal_value *= random.uniform(1.1, 1.3)
        elif self.bidding_strategy == "conservative":
            personal_value *= random.uniform(0.7, 0.9)
        
        return min(personal_value, self.max_budget_per_item)
    
    async def participate_in_english_auction(self, auction: 'Auction') -> List[Bid]:
        """Participate in English (ascending) auction"""
        
        item_value = self.assess_item_value(auction.item)
        print(f"  {self.agent_id} values {auction.item.name} at ${item_value:.2f}")
        
        bids_placed = []
        current_price = auction.current_price
        
        while auction.status == AuctionStatus.ACTIVE and current_price < item_value:
            # Decide whether to bid
            should_bid = await self.decide_to_bid(auction, item_value)
            
            if should_bid and current_price < item_value:
                # Calculate bid amount
                if self.bidding_strategy == "aggressive":
                    # Bid aggressively to discourage competition
                    bid_amount = min(current_price * 1.15, item_value)
                elif self.bidding_strategy == "conservative":
                    # Minimal increment
                    bid_amount = current_price + auction.bid_increment
                else:
                    # Rational bidding
                    bid_amount = min(current_price * 1.05, item_value * 0.9)
                
                if bid_amount <= self.budget and bid_amount <= item_value:
                    bid = await self.place_bid(auction, bid_amount)
                    if bid:
                        bids_placed.append(bid)
                        current_price = bid_amount
            
            # Wait before next bid attempt
            await asyncio.sleep(random.uniform(0.1, 0.3))
        
        return bids_placed
    
    async def participate_in_sealed_bid_auction(self, auction: 'Auction') -> Optional[Bid]:
        """Participate in sealed bid auction"""
        
        item_value = self.assess_item_value(auction.item)
        
        # Strategic bid calculation for sealed bid
        if self.bidding_strategy == "aggressive":
            # Bid closer to full value
            bid_amount = item_value * random.uniform(0.85, 0.95)
        elif self.bidding_strategy == "conservative":
            # Bid conservatively
            bid_amount = item_value * random.uniform(0.6, 0.75)
        else:
            # Rational strategy - shade bid below value
            bid_amount = item_value * random.uniform(0.7, 0.85)
        
        if bid_amount <= self.budget and bid_amount >= auction.item.starting_price:
            return await self.place_bid(auction, bid_amount)
        
        return None
    
    async def decide_to_bid(self, auction: 'Auction', item_value: float) -> bool:
        """Decide whether to place a bid"""
        
        # Don't bid if current price exceeds our value
        if auction.current_price >= item_value:
            return False
        
        # Don't bid if we don't have budget
        if auction.current_price > self.budget:
            return False
        
        # Strategy-based decision
        if self.bidding_strategy == "aggressive":
            return random.random() < 0.8  # Usually bid
        elif self.bidding_strategy == "conservative":
            return random.random() < 0.4  # Rarely bid
        else:
            # Rational: bid if price is reasonable
            price_ratio = auction.current_price / item_value
            return price_ratio < self.risk_tolerance
    
    async def place_bid(self, auction: 'Auction', amount: float) -> Optional[Bid]:
        """Place a bid in an auction"""
        
        bid = Bid(
            id=f"bid_{self.agent_id}_{int(time.time())}",
            bidder_id=self.agent_id,
            auction_id=auction.id,
            amount=amount
        )
        
        success = await auction.submit_bid(bid)
        if success:
            self.bid_history.append(bid)
            print(f"    {self.agent_id} bids ${amount:.2f}")
            return bid
        
        return None
    
    def handle_auction_win(self, result: AuctionResult) -> None:
        """Handle winning an auction"""
        self.budget -= result.final_price
        self.won_auctions.append(result)
        print(f"    {self.agent_id} WON {result.item.name} for ${result.final_price:.2f}")
    
    def get_bidding_summary(self) -> Dict[str, Any]:
        """Get summary of bidding activity"""
        
        total_spent = sum(result.final_price for result in self.won_auctions)
        win_rate = len(self.won_auctions) / len(self.bid_history) if self.bid_history else 0
        
        return {
            "agent_id": self.agent_id,
            "strategy": self.bidding_strategy,
            "budget_remaining": self.budget,
            "total_bids": len(self.bid_history),
            "auctions_won": len(self.won_auctions),
            "win_rate": win_rate,
            "total_spent": total_spent
        }

class Auction:
    """Individual auction for a specific item"""
    
    def __init__(self, auction_id: str, item: AuctionItem, auction_type: AuctionType):
        self.id = auction_id
        self.item = item
        self.auction_type = auction_type
        self.status = AuctionStatus.SCHEDULED
        
        # Auction state
        self.current_price = item.starting_price
        self.bid_increment = max(1.0, item.starting_price * 0.05)  # 5% increment
        self.bids: List[Bid] = []
        self.winning_bid: Optional[Bid] = None
        
        # Timing
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration = 30.0  # Default 30 seconds
    
    async def submit_bid(self, bid: Bid) -> bool:
        """Submit a bid to the auction"""
        
        if self.status != AuctionStatus.ACTIVE:
            return False
        
        # Validate bid
        if self.auction_type == AuctionType.ENGLISH:
            if bid.amount <= self.current_price:
                return False  # Bid must be higher than current price
        
        # Accept bid
        self.bids.append(bid)
        
        if self.auction_type == AuctionType.ENGLISH:
            self.current_price = bid.amount
            self.winning_bid = bid
        
        return True
    
    async def run_auction(self, participants: List[BiddingAgent]) -> AuctionResult:
        """Run the complete auction process"""
        
        print(f"\nRUNNING AUCTION: {self.item.name}")
        print(f"Type: {self.auction_type.value}, Starting: ${self.item.starting_price}")
        print("-" * 40)
        
        self.status = AuctionStatus.ACTIVE
        self.start_time = time.time()
        
        if self.auction_type == AuctionType.ENGLISH:
            result = await self.run_english_auction(participants)
        elif self.auction_type == AuctionType.SEALED_BID:
            result = await self.run_sealed_bid_auction(participants)
        elif self.auction_type == AuctionType.VICKREY:
            result = await self.run_vickrey_auction(participants)
        else:
            result = await self.run_dutch_auction(participants)
        
        self.status = AuctionStatus.COMPLETED
        self.end_time = time.time()
        
        # Notify winner
        if result.winning_bid:
            winner = next((p for p in participants if p.agent_id == result.winning_bid.bidder_id), None)
            if winner:
                winner.handle_auction_win(result)
        
        return result
    
    async def run_english_auction(self, participants: List[BiddingAgent]) -> AuctionResult:
        """Run English (ascending) auction"""
        
        # All participants can bid simultaneously
        bidding_tasks = []
        for participant in participants:
            task = participant.participate_in_english_auction(self)
            bidding_tasks.append(task)
        
        # Run auction for specified duration
        auction_task = asyncio.create_task(asyncio.sleep(self.duration))
        bidding_task = asyncio.create_task(asyncio.gather(*bidding_tasks))
        
        # Wait for auction to complete
        done, pending = await asyncio.wait([auction_task, bidding_task], return_when=asyncio.FIRST_COMPLETED)
        
        # Cancel remaining tasks
        for task in pending:
            task.cancel()
        
        # Determine winner
        if self.bids and self.winning_bid:
            final_price = self.winning_bid.amount
        else:
            final_price = 0.0
        
        return AuctionResult(
            auction_id=self.id,
            item=self.item,
            winning_bid=self.winning_bid,
            all_bids=self.bids.copy(),
            final_price=final_price,
            revenue=final_price
        )
    
    async def run_sealed_bid_auction(self, participants: List[BiddingAgent]) -> AuctionResult:
        """Run sealed bid auction"""
        
        # Collect sealed bids
        bidding_tasks = []
        for participant in participants:
            task = participant.participate_in_sealed_bid_auction(self)
            bidding_tasks.append(task)
        
        # Wait for all bids
        submitted_bids = await asyncio.gather(*bidding_tasks)
        
        # Filter valid bids
        valid_bids = [bid for bid in submitted_bids if bid is not None]
        self.bids.extend(valid_bids)
        
        # Find highest bid
        if valid_bids:
            winning_bid = max(valid_bids, key=lambda b: b.amount)
            final_price = winning_bid.amount
        else:
            winning_bid = None
            final_price = 0.0
        
        self.winning_bid = winning_bid
        
        return AuctionResult(
            auction_id=self.id,
            item=self.item,
            winning_bid=winning_bid,
            all_bids=self.bids.copy(),
            final_price=final_price,
            revenue=final_price
        )
    
    async def run_vickrey_auction(self, participants: List[BiddingAgent]) -> AuctionResult:
        """Run Vickrey (second-price) auction"""
        
        # Run like sealed bid auction
        result = await self.run_sealed_bid_auction(participants)
        
        # But winner pays second-highest price
        if len(self.bids) >= 2:
            sorted_bids = sorted(self.bids, key=lambda b: b.amount, reverse=True)
            second_price = sorted_bids[1].amount
            result.final_price = second_price
            result.revenue = second_price
        
        return result
    
    async def run_dutch_auction(self, participants: List[BiddingAgent]) -> AuctionResult:
        """Run Dutch (descending) auction"""
        
        # Start at high price and decrease
        current_price = self.item.starting_price * 2.0
        price_decrement = self.item.starting_price * 0.1
        
        winning_bid = None
        
        # Decrease price until someone bids
        while current_price >= self.item.starting_price and not winning_bid:
            self.current_price = current_price
            
            # Check if any participant wants to bid at current price
            for participant in participants:
                item_value = participant.assess_item_value(self.item)
                if current_price <= item_value and current_price <= participant.budget:
                    # First to accept wins
                    bid = Bid(
                        id=f"dutch_bid_{participant.agent_id}",
                        bidder_id=participant.agent_id,
                        auction_id=self.id,
                        amount=current_price
                    )
                    winning_bid = bid
                    break
            
            if not winning_bid:
                current_price -= price_decrement
                await asyncio.sleep(0.1)  # Brief pause for price decrease
        
        self.winning_bid = winning_bid
        if winning_bid:
            self.bids.append(winning_bid)
        
        return AuctionResult(
            auction_id=self.id,
            item=self.item,
            winning_bid=winning_bid,
            all_bids=self.bids.copy(),
            final_price=winning_bid.amount if winning_bid else 0.0,
            revenue=winning_bid.amount if winning_bid else 0.0
        )

class AuctionSystem:
    """
    System managing multiple auctions
    
    EXAMPLE USAGE:
    =============
    # Create auction system
    system = AuctionSystem("art_auctions")
    
    # Add bidding agents
    for i in range(10):
        agent = BiddingAgent(f"bidder_{i}", 1000.0, "rational")
        system.add_bidder(agent)
    
    # Run auction for valuable item
    result = await system.run_auction("Rare Painting", 500.0, AuctionType.ENGLISH)
    """
    
    def __init__(self, system_id: str):
        self.system_id = system_id
        self.bidders: Dict[str, BiddingAgent] = {}
        self.auction_history: List[AuctionResult] = []
        self.active_auctions: Dict[str, Auction] = {}
    
    def add_bidder(self, agent: BiddingAgent) -> None:
        """Add bidding agent to the system"""
        self.bidders[agent.agent_id] = agent
        print(f"Added bidder: {agent.agent_id} (${agent.budget} budget)")
    
    async def run_auction(self, item_name: str, starting_price: float, 
                         auction_type: AuctionType, description: str = "") -> AuctionResult:
        """Run a single auction"""
        
        # Create auction item
        item = AuctionItem(
            id=f"item_{int(time.time())}",
            name=item_name,
            description=description,
            starting_price=starting_price
        )
        
        # Create auction
        auction = Auction(
            auction_id=f"auction_{int(time.time())}",
            item=item,
            auction_type=auction_type
        )
        
        # Run auction with all bidders
        participants = list(self.bidders.values())
        result = await auction.run_auction(participants)
        
        # Record result
        self.auction_history.append(result)
        
        return result
    
    async def run_auction_series(self, items: List[Tuple[str, float, str]], 
                               auction_type: AuctionType) -> List[AuctionResult]:
        """Run series of auctions"""
        
        print(f"\nRUNNING AUCTION SERIES: {len(items)} items")
        print("=" * 50)
        
        results = []
        
        for item_name, starting_price, description in items:
            result = await self.run_auction(item_name, starting_price, auction_type, description)
            results.append(result)
            
            # Brief pause between auctions
            await asyncio.sleep(0.5)
        
        return results
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive auction system summary"""
        
        total_revenue = sum(result.revenue for result in self.auction_history)
        successful_auctions = len([r for r in self.auction_history if r.winning_bid])
        
        # Bidder performance
        bidder_summaries = {}
        for bidder in self.bidders.values():
            bidder_summaries[bidder.agent_id] = bidder.get_bidding_summary()
        
        return {
            "system_id": self.system_id,
            "total_auctions": len(self.auction_history),
            "successful_auctions": successful_auctions,
            "total_revenue": total_revenue,
            "average_sale_price": total_revenue / successful_auctions if successful_auctions > 0 else 0,
            "bidder_summaries": bidder_summaries
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_art_auction_house():
    """Demo: Art auction house with different auction types"""
    print("\nDEMO 1: ART AUCTION HOUSE")
    print("=" * 50)
    
    auction_house = AuctionSystem("fine_arts")
    
    # Add diverse bidders
    bidder_types = [
        ("museum_curator", 5000.0, "conservative"),
        ("private_collector", 8000.0, "aggressive"),
        ("art_dealer", 6000.0, "rational"),
        ("investment_fund", 12000.0, "aggressive"),
        ("gallery_owner", 4000.0, "rational"),
        ("art_enthusiast", 2000.0, "conservative")
    ]
    
    for bidder_id, budget, strategy in bidder_types:
        agent = BiddingAgent(bidder_id, budget, strategy)
        # Add art preferences
        agent.preferences = {"painting": 1.2, "sculpture": 0.9, "rare": 1.5}
        auction_house.add_bidder(agent)
    
    # Auction valuable artworks
    artworks = [
        ("Rare Van Gogh Painting", 3000.0, "19th century masterpiece"),
        ("Modern Sculpture", 1500.0, "Contemporary bronze sculpture"),
        ("Ancient Artifact", 2500.0, "Rare archaeological find")
    ]
    
    # Run English auctions
    results = await auction_house.run_auction_series(artworks, AuctionType.ENGLISH)
    
    # Show results
    summary = auction_house.get_system_summary()
    print(f"\nAuction House Results:")
    print(f"- Total revenue: ${summary['total_revenue']:.2f}")
    print(f"- Average sale price: ${summary['average_sale_price']:.2f}")
    
    # Top bidders
    top_bidders = sorted(summary['bidder_summaries'].items(), 
                        key=lambda x: x[1]['auctions_won'], reverse=True)[:3]
    for i, (bidder_id, stats) in enumerate(top_bidders):
        print(f"- Top bidder {i+1}: {bidder_id} ({stats['auctions_won']} wins)")

async def demo_cloud_resource_auctions():
    """Demo: Cloud computing resource auctions"""
    print("\nDEMO 2: CLOUD RESOURCE AUCTIONS")
    print("=" * 50)
    
    cloud_market = AuctionSystem("cloud_resources")
    
    # Add different types of cloud users
    users = [
        ("startup_a", 500.0, "conservative"),
        ("enterprise_b", 2000.0, "rational"),
        ("research_lab", 800.0, "aggressive"),
        ("gaming_company", 1500.0, "aggressive"),
        ("data_analytics", 1200.0, "rational")
    ]
    
    for user_id, budget, strategy in users:
        agent = BiddingAgent(user_id, budget, strategy)
        agent.preferences = {"gpu": 1.3, "cpu": 1.0, "memory": 1.1}
        cloud_market.add_bidder(agent)
    
    # Auction cloud resources using Vickrey (second-price) auctions
    resources = [
        ("High-Performance GPU Cluster", 300.0, "NVIDIA A100 cluster for 24 hours"),
        ("Large Memory Instance", 150.0, "1TB RAM instance for machine learning"),
        ("CPU Intensive Compute", 200.0, "128-core CPU cluster for data processing")
    ]
    
    results = await cloud_market.run_auction_series(resources, AuctionType.VICKREY)
    
    print(f"\nCloud Resource Auction Results:")
    for result in results:
        if result.winning_bid:
            print(f"- {result.item.name}: Won by {result.winning_bid.bidder_id} "
                  f"for ${result.final_price:.2f}")

async def demo_government_spectrum_auction():
    """Demo: Government spectrum licensing auction"""
    print("\nDEMO 3: GOVERNMENT SPECTRUM AUCTION")
    print("=" * 50)
    
    spectrum_auction = AuctionSystem("spectrum_licensing")
    
    # Add telecom companies
    telecoms = [
        ("verizon", 50000.0, "aggressive"),
        ("att", 45000.0, "aggressive"),
        ("tmobile", 35000.0, "rational"),
        ("sprint", 25000.0, "conservative"),
        ("regional_carrier", 15000.0, "conservative")
    ]
    
    for telecom_id, budget, strategy in telecoms:
        agent = BiddingAgent(telecom_id, budget, strategy)
        agent.preferences = {"5g": 1.4, "urban": 1.3, "nationwide": 1.5}
        spectrum_auction.add_bidder(agent)
    
    # Auction spectrum licenses using sealed bid
    spectrum_lots = [
        ("5G Urban Spectrum Block A", 8000.0, "Prime 5G frequencies for major cities"),
        ("Nationwide Coverage Block B", 12000.0, "Nationwide spectrum license"),
        ("Regional Rural Block C", 4000.0, "Rural area coverage spectrum")
    ]
    
    results = await spectrum_auction.run_auction_series(spectrum_lots, AuctionType.SEALED_BID)
    
    print(f"\nSpectrum Auction Results:")
    total_government_revenue = sum(r.revenue for r in results)
    print(f"- Total government revenue: ${total_government_revenue:,.2f}")
    
    for result in results:
        if result.winning_bid:
            print(f"- {result.item.name}: ${result.final_price:,.2f} "
                  f"({result.winning_bid.bidder_id})")

async def main():
    """
    Demonstrate Agent Auction Systems for competitive resource allocation
    
    WHAT YOU'LL LEARN:
    ================
    1. How auctions enable fair and efficient resource allocation
    2. How different auction types (English, Dutch, Sealed bid, Vickrey) work
    3. How bidding strategies affect auction outcomes
    4. How to implement competitive price discovery mechanisms
    5. How auction systems scale to many participants and items
    
    REAL WORLD APPLICATIONS:
    =======================
    - Online advertising auctions (Google Ads, Facebook Ads)
    - Cloud computing resource allocation (AWS Spot Instances)
    - Government spectrum licensing and asset sales
    - Art and collectibles auctions (Sotheby's, Christie's)
    - Financial markets and treasury bond auctions
    - Procurement and reverse auctions for suppliers
    """
    
    print("AGENT AUCTION SYSTEMS DEMONSTRATION")
    print("This shows how auctions enable competitive and fair resource allocation!")
    
    await demo_art_auction_house()
    await demo_cloud_resource_auctions()
    await demo_government_spectrum_auction()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Auctions discover true market value through competitive bidding")
    print("✓ Different auction types suit different allocation scenarios")
    print("✓ Strategic bidding behavior affects auction outcomes")
    print("✓ Auction systems provide fair and transparent resource allocation")
    print("✓ Competitive mechanisms maximize efficiency and revenue")

if __name__ == "__main__":
    asyncio.run(main())
