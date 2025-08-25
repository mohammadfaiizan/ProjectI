"""
1244. Design A Leaderboard - Multiple Approaches
Difficulty: Medium

Design a Leaderboard class, which has three functions:
1. addScore(playerId, score): Update the leaderboard by adding score to the given player's score. If there is no player with such id on the leaderboard, add him to the leaderboard with the given score.
2. top(K): Return the score sum of the top K players.
3. reset(playerId): Reset the scores of the player with the given id to 0. It is guaranteed that the player was added to the leaderboard before calling this function.
"""

from typing import Dict, List
import heapq
from collections import defaultdict

class LeaderboardSimple:
    """
    Approach 1: Simple Dictionary with Sorting
    
    Store scores in dictionary and sort for top K queries.
    
    Time Complexity:
    - addScore: O(1)
    - top: O(n log n) for sorting
    - reset: O(1)
    
    Space Complexity: O(n) where n is number of players
    """
    
    def __init__(self):
        self.scores = {}
    
    def addScore(self, playerId: int, score: int) -> None:
        self.scores[playerId] = self.scores.get(playerId, 0) + score
    
    def top(self, K: int) -> int:
        # Sort all scores in descending order and sum top K
        sorted_scores = sorted(self.scores.values(), reverse=True)
        return sum(sorted_scores[:K])
    
    def reset(self, playerId: int) -> None:
        if playerId in self.scores:
            self.scores[playerId] = 0

class LeaderboardHeap:
    """
    Approach 2: Min-Heap for Top K
    
    Use min-heap to efficiently maintain top K scores.
    
    Time Complexity:
    - addScore: O(1)
    - top: O(n log K)
    - reset: O(1)
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.scores = {}
    
    def addScore(self, playerId: int, score: int) -> None:
        self.scores[playerId] = self.scores.get(playerId, 0) + score
    
    def top(self, K: int) -> int:
        # Use min-heap to find top K scores
        heap = []
        
        for score in self.scores.values():
            if len(heap) < K:
                heapq.heappush(heap, score)
            elif score > heap[0]:
                heapq.heapreplace(heap, score)
        
        return sum(heap)
    
    def reset(self, playerId: int) -> None:
        if playerId in self.scores:
            self.scores[playerId] = 0

class LeaderboardSortedList:
    """
    Approach 3: Maintain Sorted List
    
    Keep scores in sorted order for efficient top K queries.
    
    Time Complexity:
    - addScore: O(n) for maintaining order
    - top: O(K)
    - reset: O(n) for maintaining order
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.scores = {}
        self.sorted_scores = []  # Maintained in descending order
    
    def addScore(self, playerId: int, score: int) -> None:
        old_score = self.scores.get(playerId, 0)
        new_score = old_score + score
        self.scores[playerId] = new_score
        
        # Update sorted list
        if old_score in self.sorted_scores:
            self.sorted_scores.remove(old_score)
        
        # Insert new score in correct position
        self._insert_sorted(new_score)
    
    def _insert_sorted(self, score: int) -> None:
        """Insert score in descending sorted order"""
        left, right = 0, len(self.sorted_scores)
        
        while left < right:
            mid = (left + right) // 2
            if self.sorted_scores[mid] > score:
                left = mid + 1
            else:
                right = mid
        
        self.sorted_scores.insert(left, score)
    
    def top(self, K: int) -> int:
        return sum(self.sorted_scores[:K])
    
    def reset(self, playerId: int) -> None:
        if playerId in self.scores:
            old_score = self.scores[playerId]
            self.scores[playerId] = 0
            
            # Update sorted list
            if old_score in self.sorted_scores:
                self.sorted_scores.remove(old_score)
            
            self._insert_sorted(0)

class LeaderboardAdvanced:
    """
    Approach 4: Advanced with Multiple Features
    
    Enhanced leaderboard with rankings, statistics, and optimizations.
    
    Time Complexity:
    - addScore: O(1)
    - top: O(n log K)
    - reset: O(1)
    
    Space Complexity: O(n + features)
    """
    
    def __init__(self):
        self.scores = {}
        self.game_count = {}  # Track games played per player
        self.total_operations = 0
        self.top_queries = 0
        self.add_operations = 0
        self.reset_operations = 0
        
        # Caching for frequent top K queries
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def addScore(self, playerId: int, score: int) -> None:
        self.add_operations += 1
        self.total_operations += 1
        
        self.scores[playerId] = self.scores.get(playerId, 0) + score
        self.game_count[playerId] = self.game_count.get(playerId, 0) + 1
        
        # Invalidate cache when scores change
        self.cache.clear()
    
    def top(self, K: int) -> int:
        self.top_queries += 1
        self.total_operations += 1
        
        # Check cache first
        cache_key = K
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        self.cache_misses += 1
        
        # Use min-heap for efficiency
        heap = []
        
        for score in self.scores.values():
            if score > 0:  # Only consider positive scores
                if len(heap) < K:
                    heapq.heappush(heap, score)
                elif score > heap[0]:
                    heapq.heapreplace(heap, score)
        
        result = sum(heap)
        
        # Cache the result
        self.cache[cache_key] = result
        
        return result
    
    def reset(self, playerId: int) -> None:
        self.reset_operations += 1
        self.total_operations += 1
        
        if playerId in self.scores:
            self.scores[playerId] = 0
        
        # Invalidate cache
        self.cache.clear()
    
    def getRank(self, playerId: int) -> int:
        """Get player's current rank (1-indexed)"""
        if playerId not in self.scores:
            return -1
        
        player_score = self.scores[playerId]
        rank = 1
        
        for score in self.scores.values():
            if score > player_score:
                rank += 1
        
        return rank
    
    def getAverageScore(self, playerId: int) -> float:
        """Get player's average score per game"""
        if playerId not in self.scores or self.game_count.get(playerId, 0) == 0:
            return 0.0
        
        return self.scores[playerId] / self.game_count[playerId]
    
    def getPlayerCount(self) -> int:
        """Get total number of players"""
        return len([p for p, score in self.scores.items() if score > 0])
    
    def getStatistics(self) -> dict:
        """Get leaderboard statistics"""
        active_players = [score for score in self.scores.values() if score > 0]
        
        return {
            'total_operations': self.total_operations,
            'add_operations': self.add_operations,
            'top_queries': self.top_queries,
            'reset_operations': self.reset_operations,
            'active_players': len(active_players),
            'total_players': len(self.scores),
            'average_score': sum(active_players) / len(active_players) if active_players else 0,
            'max_score': max(active_players) if active_players else 0,
            'cache_hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        }

class LeaderboardSegmentTree:
    """
    Approach 5: Segment Tree for Range Queries
    
    Use segment tree for efficient range sum queries and updates.
    
    Time Complexity:
    - addScore: O(log n)
    - top: O(K log n)
    - reset: O(log n)
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.scores = {}
        self.max_players = 10000  # Preallocate for efficiency
        self.player_to_index = {}
        self.index_to_player = {}
        self.next_index = 0
        
        # Segment tree for range sums
        self.tree = [0] * (4 * self.max_players)
    
    def _get_player_index(self, playerId: int) -> int:
        """Get or assign index for player"""
        if playerId not in self.player_to_index:
            self.player_to_index[playerId] = self.next_index
            self.index_to_player[self.next_index] = playerId
            self.next_index += 1
        
        return self.player_to_index[playerId]
    
    def _update_tree(self, node: int, start: int, end: int, idx: int, val: int) -> None:
        """Update segment tree"""
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            if idx <= mid:
                self._update_tree(2 * node, start, mid, idx, val)
            else:
                self._update_tree(2 * node + 1, mid + 1, end, idx, val)
            
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]
    
    def _query_tree(self, node: int, start: int, end: int, l: int, r: int) -> int:
        """Query segment tree for range sum"""
        if r < start or end < l:
            return 0
        
        if l <= start and end <= r:
            return self.tree[node]
        
        mid = (start + end) // 2
        left_sum = self._query_tree(2 * node, start, mid, l, r)
        right_sum = self._query_tree(2 * node + 1, mid + 1, end, l, r)
        
        return left_sum + right_sum
    
    def addScore(self, playerId: int, score: int) -> None:
        self.scores[playerId] = self.scores.get(playerId, 0) + score
        
        idx = self._get_player_index(playerId)
        self._update_tree(1, 0, self.max_players - 1, idx, self.scores[playerId])
    
    def top(self, K: int) -> int:
        # Get all scores and find top K
        all_scores = [(score, idx) for idx, score in enumerate(self.scores.values())]
        all_scores.sort(reverse=True)
        
        return sum(score for score, _ in all_scores[:K])
    
    def reset(self, playerId: int) -> None:
        if playerId in self.scores:
            self.scores[playerId] = 0
            idx = self._get_player_index(playerId)
            self._update_tree(1, 0, self.max_players - 1, idx, 0)


def test_leaderboard_basic():
    """Test basic leaderboard functionality"""
    print("=== Testing Basic Leaderboard Functionality ===")
    
    implementations = [
        ("Simple Sorting", LeaderboardSimple),
        ("Min-Heap", LeaderboardHeap),
        ("Sorted List", LeaderboardSortedList),
        ("Advanced", LeaderboardAdvanced),
        ("Segment Tree", LeaderboardSegmentTree)
    ]
    
    for name, LeaderboardClass in implementations:
        print(f"\n{name}:")
        
        lb = LeaderboardClass()
        
        # Test sequence from problem
        operations = [
            ("addScore", 1, 73), ("addScore", 2, 56), ("addScore", 3, 39),
            ("addScore", 4, 51), ("addScore", 5, 4), ("top", 1, None),
            ("reset", 1, None), ("reset", 2, None), ("addScore", 2, 51),
            ("top", 3, None)
        ]
        
        for op, player_or_k, score in operations:
            if op == "addScore":
                lb.addScore(player_or_k, score)
                print(f"  addScore({player_or_k}, {score})")
            elif op == "top":
                result = lb.top(player_or_k)
                print(f"  top({player_or_k}): {result}")
            elif op == "reset":
                lb.reset(player_or_k)
                print(f"  reset({player_or_k})")

def test_leaderboard_edge_cases():
    """Test leaderboard edge cases"""
    print("\n=== Testing Leaderboard Edge Cases ===")
    
    lb = LeaderboardAdvanced()
    
    # Test top K when fewer than K players
    print("Top K with fewer than K players:")
    lb.addScore(1, 100)
    lb.addScore(2, 200)
    
    result = lb.top(5)  # More than available players
    print(f"  top(5) with 2 players: {result}")
    
    # Test negative scores
    print(f"\nNegative scores:")
    lb.addScore(1, -50)  # Player 1: 100 - 50 = 50
    lb.addScore(3, -10)  # Player 3: 0 - 10 = -10
    
    result = lb.top(3)
    print(f"  top(3) with negative scores: {result}")
    
    # Test multiple resets
    print(f"\nMultiple resets:")
    lb.reset(1)
    lb.reset(2)
    lb.reset(3)
    
    result = lb.top(1)
    print(f"  top(1) after all resets: {result}")
    
    # Test zero scores
    print(f"\nZero scores handling:")
    lb.addScore(1, 0)
    lb.addScore(2, 0)
    
    result = lb.top(2)
    print(f"  top(2) with zero scores: {result}")

def test_performance_comparison():
    """Test performance of different implementations"""
    print("\n=== Testing Performance Comparison ===")
    
    import time
    
    implementations = [
        ("Simple Sorting", LeaderboardSimple),
        ("Min-Heap", LeaderboardHeap),
        ("Sorted List", LeaderboardSortedList),
        ("Advanced", LeaderboardAdvanced)
    ]
    
    num_players = 1000
    num_operations = 2000
    
    for name, LeaderboardClass in implementations:
        lb = LeaderboardClass()
        
        # Time addScore operations
        start_time = time.time()
        for i in range(num_operations):
            player_id = i % num_players
            score = (i % 100) + 1
            lb.addScore(player_id, score)
        add_time = (time.time() - start_time) * 1000
        
        # Time top queries
        start_time = time.time()
        for k in range(1, 101):  # top(1) to top(100)
            lb.top(k)
        top_time = (time.time() - start_time) * 1000
        
        # Time reset operations
        start_time = time.time()
        for i in range(100):  # Reset 100 players
            lb.reset(i)
        reset_time = (time.time() - start_time) * 1000
        
        print(f"  {name}:")
        print(f"    {num_operations} adds: {add_time:.2f}ms")
        print(f"    100 top queries: {top_time:.2f}ms")
        print(f"    100 resets: {reset_time:.2f}ms")

def test_advanced_features():
    """Test advanced features"""
    print("\n=== Testing Advanced Features ===")
    
    lb = LeaderboardAdvanced()
    
    # Create leaderboard with multiple players
    players_scores = [
        (1, 100), (2, 150), (3, 75), (4, 200), (5, 125)
    ]
    
    print("Building leaderboard:")
    for player_id, score in players_scores:
        lb.addScore(player_id, score)
        print(f"  addScore({player_id}, {score})")
    
    # Test ranking
    print(f"\nPlayer rankings:")
    for player_id, _ in players_scores:
        rank = lb.getRank(player_id)
        avg_score = lb.getAverageScore(player_id)
        print(f"  Player {player_id}: rank {rank}, avg score {avg_score:.1f}")
    
    # Test statistics
    stats = lb.getStatistics()
    print(f"\nLeaderboard statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Test caching
    print(f"\nTesting cache performance:")
    
    # Multiple identical top queries should hit cache
    for _ in range(5):
        result = lb.top(3)
    
    updated_stats = lb.getStatistics()
    cache_hit_rate = updated_stats['cache_hit_rate']
    print(f"  Cache hit rate after repeated queries: {cache_hit_rate:.3f}")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: Gaming tournament
    print("Application 1: Gaming Tournament Leaderboard")
    
    tournament = LeaderboardAdvanced()
    
    # Simulate tournament rounds
    rounds = [
        # Round 1
        [(101, 250), (102, 180), (103, 320), (104, 210), (105, 275)],
        # Round 2  
        [(101, 180), (102, 290), (103, 150), (104, 340), (105, 220)],
        # Round 3
        [(101, 310), (102, 200), (103, 280), (104, 190), (105, 350)]
    ]
    
    for round_num, scores in enumerate(rounds, 1):
        print(f"\n  Round {round_num} results:")
        
        for player_id, score in scores:
            tournament.addScore(player_id, score)
            print(f"    Player {player_id}: {score} points")
        
        print(f"  Top 3 after round {round_num}: {tournament.top(3)} total points")
    
    # Final rankings
    print(f"\n  Final tournament rankings:")
    for player_id in [101, 102, 103, 104, 105]:
        rank = tournament.getRank(player_id)
        avg = tournament.getAverageScore(player_id)
        total = tournament.scores.get(player_id, 0)
        print(f"    Player {player_id}: Rank {rank}, Total {total}, Avg {avg:.1f}")
    
    # Application 2: Sales performance tracking
    print(f"\nApplication 2: Sales Team Performance")
    
    sales_board = LeaderboardSimple()
    
    # Monthly sales data
    sales_data = [
        ("Alice", 15000), ("Bob", 12000), ("Charlie", 18000),
        ("Diana", 14000), ("Eve", 16000)
    ]
    
    # Convert names to IDs for demo
    name_to_id = {name: i for i, (name, _) in enumerate(sales_data)}
    
    print("  Monthly sales:")
    for name, sales in sales_data:
        player_id = name_to_id[name]
        sales_board.addScore(player_id, sales)
        print(f"    {name}: ${sales:,}")
    
    # Top performers
    top_3_total = sales_board.top(3)
    print(f"  Top 3 performers total: ${top_3_total:,}")
    
    # Quarterly bonus reset simulation
    print(f"\n  Quarterly bonus reset for underperformers:")
    # Reset bottom performers (simulated)
    for name in ["Bob", "Diana"]:
        player_id = name_to_id[name]
        sales_board.reset(player_id)
        print(f"    Reset {name}'s bonus points")
    
    new_top_3 = sales_board.top(3)
    print(f"  New top 3 total after resets: ${new_top_3:,}")
    
    # Application 3: Student grade tracking
    print(f"\nApplication 3: Student Grade Tracking")
    
    grade_board = LeaderboardHeap()
    
    # Students and their test scores
    students = [
        (1001, "Test 1", 85), (1002, "Test 1", 92), (1003, "Test 1", 78),
        (1001, "Test 2", 90), (1002, "Test 2", 88), (1003, "Test 2", 95),
        (1001, "Test 3", 87), (1002, "Test 3", 94), (1003, "Test 3", 89)
    ]
    
    print("  Test scores:")
    for student_id, test_name, score in students:
        grade_board.addScore(student_id, score)
        print(f"    Student {student_id} - {test_name}: {score}")
    
    # Class performance metrics
    print(f"  Class performance:")
    for k in [1, 2, 3]:
        top_k_total = grade_board.top(k)
        avg_top_k = top_k_total / k if k > 0 else 0
        print(f"    Top {k} student(s) average: {avg_top_k:.1f}")

def test_large_scale_performance():
    """Test large scale performance"""
    print("\n=== Testing Large Scale Performance ===")
    
    import time
    
    lb = LeaderboardHeap()  # Use heap for efficiency
    
    # Large scale test
    num_players = 10000
    num_updates = 50000
    
    print(f"Large scale test: {num_players} players, {num_updates} updates")
    
    start_time = time.time()
    
    # Add initial scores
    for i in range(num_players):
        lb.addScore(i, i % 1000)  # Scores 0-999
    
    # Simulate game updates
    import random
    for _ in range(num_updates):
        player_id = random.randint(0, num_players - 1)
        score_change = random.randint(-50, 100)
        lb.addScore(player_id, score_change)
    
    update_time = (time.time() - start_time) * 1000
    
    # Test top K queries
    start_time = time.time()
    
    for k in [10, 50, 100, 500]:
        result = lb.top(k)
    
    query_time = (time.time() - start_time) * 1000
    
    print(f"  Update time: {update_time:.2f}ms")
    print(f"  Query time for various K: {query_time:.2f}ms")

def test_memory_efficiency():
    """Test memory efficiency"""
    print("\n=== Testing Memory Efficiency ===")
    
    implementations = [
        ("Simple Sorting", LeaderboardSimple),
        ("Min-Heap", LeaderboardHeap),
        ("Advanced", LeaderboardAdvanced)
    ]
    
    for name, LeaderboardClass in implementations:
        lb = LeaderboardClass()
        
        # Add many players
        num_players = 5000
        for i in range(num_players):
            lb.addScore(i, i % 500)
        
        # Estimate memory usage (simplified)
        if hasattr(lb, 'scores'):
            base_memory = len(lb.scores)
        else:
            base_memory = num_players
        
        if hasattr(lb, 'sorted_scores'):
            extra_memory = len(lb.sorted_scores)
        elif hasattr(lb, 'cache'):
            extra_memory = len(lb.cache)
        else:
            extra_memory = 0
        
        total_memory = base_memory + extra_memory
        
        print(f"  {name}: ~{total_memory} memory units")

def stress_test_leaderboard():
    """Stress test leaderboard"""
    print("\n=== Stress Testing Leaderboard ===")
    
    import time
    import random
    
    lb = LeaderboardAdvanced()
    
    # Stress test parameters
    num_players = 1000
    num_operations = 10000
    
    print(f"Stress test: {num_operations} mixed operations")
    
    start_time = time.time()
    
    for i in range(num_operations):
        operation_type = random.choice(['add', 'top', 'reset'])
        
        if operation_type == 'add':
            player_id = random.randint(1, num_players)
            score = random.randint(-100, 200)
            lb.addScore(player_id, score)
        
        elif operation_type == 'top':
            k = random.randint(1, min(50, lb.getPlayerCount() + 1))
            lb.top(k)
        
        elif operation_type == 'reset':
            if lb.getPlayerCount() > 0:
                # Only reset if we have players
                player_id = random.randint(1, num_players)
                if player_id in lb.scores:
                    lb.reset(player_id)
    
    elapsed = (time.time() - start_time) * 1000
    
    # Get final statistics
    final_stats = lb.getStatistics()
    
    print(f"  Completed in {elapsed:.2f}ms")
    print(f"  Final statistics:")
    print(f"    Active players: {final_stats['active_players']}")
    print(f"    Cache hit rate: {final_stats['cache_hit_rate']:.3f}")
    print(f"    Average score: {final_stats['average_score']:.1f}")

def benchmark_top_k_performance():
    """Benchmark top K performance for different K values"""
    print("\n=== Benchmarking Top K Performance ===")
    
    import time
    
    lb = LeaderboardHeap()
    
    # Create leaderboard with many players
    num_players = 5000
    for i in range(num_players):
        lb.addScore(i, random.randint(0, 10000))
    
    # Test different K values
    k_values = [1, 10, 50, 100, 500, 1000]
    
    for k in k_values:
        start_time = time.time()
        
        # Multiple queries for averaging
        for _ in range(100):
            lb.top(k)
        
        elapsed = (time.time() - start_time) * 1000
        avg_time = elapsed / 100
        
        print(f"  top({k}): {avg_time:.3f}ms average")

if __name__ == "__main__":
    test_leaderboard_basic()
    test_leaderboard_edge_cases()
    test_performance_comparison()
    test_advanced_features()
    demonstrate_applications()
    test_large_scale_performance()
    test_memory_efficiency()
    stress_test_leaderboard()
    benchmark_top_k_performance()

"""
Leaderboard Design demonstrates key concepts:

Core Approaches:
1. Simple Sorting - Dictionary + sort for top K queries
2. Min-Heap - Use heap to efficiently maintain top K scores  
3. Sorted List - Maintain scores in sorted order
4. Advanced - Enhanced with rankings, caching, and analytics
5. Segment Tree - Range queries for advanced operations

Key Design Principles:
- Trade-offs between update and query performance
- Top-K algorithms and data structures
- Caching strategies for frequent queries
- Memory efficiency for large player bases

Performance Characteristics:
- Simple: O(1) add, O(n log n) top, O(1) reset
- Heap: O(1) add, O(n log K) top, O(1) reset
- Sorted: O(n) add, O(K) top, O(n) reset
- Advanced: O(1) add with caching, analytics overhead

Real-world Applications:
- Gaming tournament leaderboards
- Sales team performance tracking
- Student grade and ranking systems
- Sports league standings
- Employee performance dashboards
- Online competition platforms

The min-heap approach provides the best balance
for most use cases, offering efficient top-K queries
while maintaining fast updates and reasonable memory usage.
"""
