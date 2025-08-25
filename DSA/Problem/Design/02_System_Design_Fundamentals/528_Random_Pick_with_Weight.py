"""
528. Random Pick with Weight - Multiple Approaches
Difficulty: Medium

You are given a 0-indexed array of positive integers w where w[i] describes the weight of the ith index.

You need to implement the function pickIndex(), which randomly picks an index in the range [0, w.length - 1] (inclusive) and returns it. The probability of picking an index i is w[i] / sum(w).

For example, if w = [1, 3], the probability of picking index 0 is 1 / (1 + 3) = 1/4, and the probability of picking index 1 is 3 / (1 + 3) = 3/4.
"""

import random
import bisect
from typing import List

class SolutionCumulative:
    """
    Approach 1: Cumulative Sum with Binary Search
    
    Build cumulative sum array and use binary search to find index.
    
    Time Complexity:
    - __init__: O(n)
    - pickIndex: O(log n)
    
    Space Complexity: O(n)
    """
    
    def __init__(self, w: List[int]):
        self.cumulative = []
        cumulative_sum = 0
        
        for weight in w:
            cumulative_sum += weight
            self.cumulative.append(cumulative_sum)
        
        self.total_sum = cumulative_sum
    
    def pickIndex(self) -> int:
        # Generate random number in [1, total_sum]
        target = random.randint(1, self.total_sum)
        
        # Binary search for the target
        left, right = 0, len(self.cumulative) - 1
        
        while left < right:
            mid = (left + right) // 2
            
            if self.cumulative[mid] < target:
                left = mid + 1
            else:
                right = mid
        
        return left

class SolutionBisect:
    """
    Approach 2: Using Python's bisect module
    
    Simplified version using built-in binary search.
    
    Time Complexity:
    - __init__: O(n)
    - pickIndex: O(log n)
    
    Space Complexity: O(n)
    """
    
    def __init__(self, w: List[int]):
        self.cumulative = []
        cumulative_sum = 0
        
        for weight in w:
            cumulative_sum += weight
            self.cumulative.append(cumulative_sum)
        
        self.total_sum = cumulative_sum
    
    def pickIndex(self) -> int:
        target = random.randint(1, self.total_sum)
        return bisect.bisect_left(self.cumulative, target)

class SolutionLinearScan:
    """
    Approach 3: Linear Scan (Simple but Slow)
    
    Generate random number and scan linearly to find index.
    
    Time Complexity:
    - __init__: O(n)
    - pickIndex: O(n)
    
    Space Complexity: O(n)
    """
    
    def __init__(self, w: List[int]):
        self.cumulative = []
        cumulative_sum = 0
        
        for weight in w:
            cumulative_sum += weight
            self.cumulative.append(cumulative_sum)
        
        self.total_sum = cumulative_sum
    
    def pickIndex(self) -> int:
        target = random.randint(1, self.total_sum)
        
        for i, cumulative_weight in enumerate(self.cumulative):
            if target <= cumulative_weight:
                return i
        
        return len(self.cumulative) - 1  # Should never reach here

class SolutionAliasMethod:
    """
    Approach 4: Alias Method (Advanced)
    
    O(1) picking time with O(n) preprocessing.
    
    Time Complexity:
    - __init__: O(n)
    - pickIndex: O(1)
    
    Space Complexity: O(n)
    """
    
    def __init__(self, w: List[int]):
        n = len(w)
        self.n = n
        
        # Normalize weights to probabilities
        total_weight = sum(w)
        probabilities = [weight / total_weight for weight in w]
        
        # Scale probabilities by n
        scaled_probs = [p * n for p in probabilities]
        
        # Initialize alias table
        self.prob = [0.0] * n
        self.alias = [0] * n
        
        # Separate into small and large groups
        small = []
        large = []
        
        for i, scaled_prob in enumerate(scaled_probs):
            if scaled_prob < 1.0:
                small.append(i)
            else:
                large.append(i)
        
        # Build alias table
        while small and large:
            small_idx = small.pop()
            large_idx = large.pop()
            
            self.prob[small_idx] = scaled_probs[small_idx]
            self.alias[small_idx] = large_idx
            
            # Update the large item
            scaled_probs[large_idx] = (scaled_probs[large_idx] + scaled_probs[small_idx]) - 1.0
            
            if scaled_probs[large_idx] < 1.0:
                small.append(large_idx)
            else:
                large.append(large_idx)
        
        # Handle remaining items
        while large:
            self.prob[large.pop()] = 1.0
        
        while small:
            self.prob[small.pop()] = 1.0
    
    def pickIndex(self) -> int:
        # Generate random column
        column = random.randint(0, self.n - 1)
        
        # Generate random probability
        coin_toss = random.random()
        
        if coin_toss < self.prob[column]:
            return column
        else:
            return self.alias[column]

class SolutionWithStatistics:
    """
    Approach 5: Enhanced with Statistics Tracking
    
    Track statistics about picks for analysis.
    
    Time Complexity:
    - __init__: O(n)
    - pickIndex: O(log n)
    
    Space Complexity: O(n)
    """
    
    def __init__(self, w: List[int]):
        self.weights = w[:]
        self.cumulative = []
        cumulative_sum = 0
        
        for weight in w:
            cumulative_sum += weight
            self.cumulative.append(cumulative_sum)
        
        self.total_sum = cumulative_sum
        
        # Statistics tracking
        self.pick_counts = [0] * len(w)
        self.total_picks = 0
    
    def pickIndex(self) -> int:
        target = random.randint(1, self.total_sum)
        index = bisect.bisect_left(self.cumulative, target)
        
        # Update statistics
        self.pick_counts[index] += 1
        self.total_picks += 1
        
        return index
    
    def get_statistics(self) -> dict:
        """Get picking statistics"""
        if self.total_picks == 0:
            return {"error": "No picks made yet"}
        
        statistics = {
            "total_picks": self.total_picks,
            "expected_probabilities": [w / self.total_sum for w in self.weights],
            "actual_frequencies": [count / self.total_picks for count in self.pick_counts],
            "pick_counts": self.pick_counts[:]
        }
        
        # Calculate deviations
        deviations = []
        for i in range(len(self.weights)):
            expected = self.weights[i] / self.total_sum
            actual = self.pick_counts[i] / self.total_picks if self.total_picks > 0 else 0
            deviation = abs(expected - actual)
            deviations.append(deviation)
        
        statistics["deviations"] = deviations
        statistics["max_deviation"] = max(deviations) if deviations else 0
        
        return statistics
    
    def reset_statistics(self) -> None:
        """Reset picking statistics"""
        self.pick_counts = [0] * len(self.weights)
        self.total_picks = 0


def test_random_pick_basic():
    """Test basic random pick functionality"""
    print("=== Testing Basic Random Pick Functionality ===")
    
    implementations = [
        ("Cumulative Sum", SolutionCumulative),
        ("Bisect Module", SolutionBisect),
        ("Linear Scan", SolutionLinearScan),
        ("Alias Method", SolutionAliasMethod),
        ("With Statistics", SolutionWithStatistics)
    ]
    
    test_weights = [1, 3, 2, 4]
    num_picks = 20
    
    for name, SolutionClass in implementations:
        print(f"\n{name}:")
        
        solution = SolutionClass(test_weights)
        
        picks = []
        for _ in range(num_picks):
            picks.append(solution.pickIndex())
        
        print(f"  Weights: {test_weights}")
        print(f"  {num_picks} picks: {picks}")
        
        # Count frequency
        frequency = [0] * len(test_weights)
        for pick in picks:
            frequency[pick] += 1
        
        print(f"  Frequency: {frequency}")

def test_distribution_quality():
    """Test quality of weight distribution"""
    print("\n=== Testing Distribution Quality ===")
    
    test_weights = [1, 3, 2, 4]  # Total weight: 10
    expected_probs = [w / sum(test_weights) for w in test_weights]
    
    solution = SolutionWithStatistics(test_weights)
    
    # Make many picks
    num_picks = 10000
    
    print(f"Making {num_picks} picks...")
    for _ in range(num_picks):
        solution.pickIndex()
    
    # Analyze statistics
    stats = solution.get_statistics()
    
    print(f"\nDistribution Analysis:")
    print(f"  Weights: {test_weights}")
    print(f"  Expected probabilities: {[f'{p:.3f}' for p in expected_probs]}")
    print(f"  Actual frequencies: {[f'{f:.3f}' for f in stats['actual_frequencies']]}")
    print(f"  Pick counts: {stats['pick_counts']}")
    print(f"  Max deviation: {stats['max_deviation']:.4f}")

def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    # Single element
    print("Single element:")
    solution = SolutionCumulative([5])
    picks = [solution.pickIndex() for _ in range(10)]
    print(f"  Weight: [5], Picks: {picks}")
    
    # Two elements with very different weights
    print(f"\nVery different weights:")
    solution = SolutionBisect([1, 1000])
    picks = [solution.pickIndex() for _ in range(20)]
    frequency = [0, 0]
    for pick in picks:
        frequency[pick] += 1
    
    print(f"  Weights: [1, 1000]")
    print(f"  Frequency: {frequency}")
    print(f"  Expected ratio: ~1:1000, Actual ratio: {frequency[0]}:{frequency[1]}")
    
    # All equal weights
    print(f"\nAll equal weights:")
    solution = SolutionCumulative([2, 2, 2, 2])
    picks = [solution.pickIndex() for _ in range(40)]
    frequency = [0, 0, 0, 0]
    for pick in picks:
        frequency[pick] += 1
    
    print(f"  Weights: [2, 2, 2, 2]")
    print(f"  Frequency: {frequency}")

def test_performance_comparison():
    """Test performance of different implementations"""
    print("\n=== Testing Performance Comparison ===")
    
    import time
    
    # Large weights array
    large_weights = [i + 1 for i in range(1000)]  # [1, 2, 3, ..., 1000]
    
    implementations = [
        ("Cumulative Sum", SolutionCumulative),
        ("Bisect Module", SolutionBisect),
        ("Linear Scan", SolutionLinearScan),
        ("Alias Method", SolutionAliasMethod)
    ]
    
    num_picks = 10000
    
    for name, SolutionClass in implementations:
        # Time initialization
        start_time = time.time()
        solution = SolutionClass(large_weights)
        init_time = (time.time() - start_time) * 1000
        
        # Time picking
        start_time = time.time()
        for _ in range(num_picks):
            solution.pickIndex()
        pick_time = (time.time() - start_time) * 1000
        
        avg_pick_time = pick_time / num_picks
        
        print(f"  {name}:")
        print(f"    Init: {init_time:.2f}ms")
        print(f"    {num_picks} picks: {pick_time:.2f}ms ({avg_pick_time:.4f}ms per pick)")

def test_alias_method_correctness():
    """Test Alias Method correctness"""
    print("\n=== Testing Alias Method Correctness ===")
    
    test_weights = [1, 2, 3, 4, 5]
    solution = SolutionAliasMethod(test_weights)
    
    # Make many picks to test distribution
    num_picks = 15000  # Total weight is 15, so 1000 picks per unit weight
    picks = [solution.pickIndex() for _ in range(num_picks)]
    
    # Count frequency
    frequency = [0] * len(test_weights)
    for pick in picks:
        frequency[pick] += 1
    
    total_weight = sum(test_weights)
    expected_counts = [w * num_picks // total_weight for w in test_weights]
    
    print(f"Alias Method Distribution Test:")
    print(f"  Weights: {test_weights}")
    print(f"  Expected counts: {expected_counts}")
    print(f"  Actual counts: {frequency}")
    
    # Calculate chi-square test statistic
    chi_square = sum((actual - expected) ** 2 / expected 
                    for actual, expected in zip(frequency, expected_counts))
    
    print(f"  Chi-square statistic: {chi_square:.2f}")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: Weighted random sampling for A/B testing
    print("Application 1: A/B Testing with Weighted Groups")
    
    # Group weights: Control=50%, Variant A=30%, Variant B=20%
    group_weights = [50, 30, 20]
    group_names = ["Control", "Variant A", "Variant B"]
    
    ab_picker = SolutionBisect(group_weights)
    
    # Assign 20 users
    assignments = []
    for user_id in range(1, 21):
        group_index = ab_picker.pickIndex()
        group_name = group_names[group_index]
        assignments.append((user_id, group_name))
    
    print(f"  Group weights: {dict(zip(group_names, group_weights))}")
    print(f"  User assignments:")
    
    group_counts = {"Control": 0, "Variant A": 0, "Variant B": 0}
    for user_id, group in assignments:
        group_counts[group] += 1
        if user_id <= 10:  # Show first 10
            print(f"    User {user_id:2d}: {group}")
    
    print(f"  Final distribution: {group_counts}")
    
    # Application 2: Weighted item drops in games
    print(f"\nApplication 2: Game Item Drop System")
    
    # Item rarities with weights (higher = more common)
    item_weights = [50, 30, 15, 4, 1]  # Common, Uncommon, Rare, Epic, Legendary
    item_names = ["Common", "Uncommon", "Rare", "Epic", "Legendary"]
    
    loot_picker = SolutionCumulative(item_weights)
    
    print(f"  Drop rates:")
    total_weight = sum(item_weights)
    for name, weight in zip(item_names, item_weights):
        percentage = (weight / total_weight) * 100
        print(f"    {name}: {percentage:.1f}%")
    
    # Simulate 10 item drops
    print(f"  10 random drops:")
    drops = []
    for drop_num in range(10):
        item_index = loot_picker.pickIndex()
        item_name = item_names[item_index]
        drops.append(item_name)
        print(f"    Drop {drop_num + 1}: {item_name}")
    
    # Application 3: Weighted server selection for load balancing
    print(f"\nApplication 3: Weighted Load Balancing")
    
    # Server capacities (weights)
    server_weights = [10, 15, 8, 12, 20]  # Different server capacities
    server_names = [f"Server-{i+1}" for i in range(len(server_weights))]
    
    load_balancer = SolutionAliasMethod(server_weights)
    
    print(f"  Server capacities:")
    for name, weight in zip(server_names, server_weights):
        print(f"    {name}: {weight} units")
    
    # Route 15 requests
    print(f"  Routing 15 requests:")
    server_loads = [0] * len(server_names)
    
    for request_id in range(1, 16):
        server_index = load_balancer.pickIndex()
        server_name = server_names[server_index]
        server_loads[server_index] += 1
        
        if request_id <= 8:  # Show first 8
            print(f"    Request {request_id:2d} -> {server_name}")
    
    print(f"  Final load distribution:")
    for name, load in zip(server_names, server_loads):
        print(f"    {name}: {load} requests")

def test_large_scale_distribution():
    """Test distribution with large number of picks"""
    print("\n=== Testing Large Scale Distribution ===")
    
    # Test with Fibonacci-like weights
    weights = [1, 1, 2, 3, 5, 8, 13, 21]
    solution = SolutionWithStatistics(weights)
    
    # Make very large number of picks
    num_picks = 100000
    
    print(f"Making {num_picks} picks with Fibonacci weights...")
    print(f"Weights: {weights}")
    
    for _ in range(num_picks):
        solution.pickIndex()
    
    # Analyze results
    stats = solution.get_statistics()
    
    print(f"\nLarge Scale Analysis:")
    print(f"  Expected vs Actual frequencies:")
    
    for i in range(len(weights)):
        expected = stats['expected_probabilities'][i]
        actual = stats['actual_frequencies'][i]
        deviation = abs(expected - actual)
        
        print(f"    Index {i} (weight {weights[i]:2d}): "
              f"expected {expected:.4f}, actual {actual:.4f}, "
              f"deviation {deviation:.4f}")
    
    print(f"  Maximum deviation: {stats['max_deviation']:.6f}")

def benchmark_initialization():
    """Benchmark initialization time for different array sizes"""
    print("\n=== Benchmarking Initialization Time ===")
    
    import time
    
    implementations = [
        ("Cumulative Sum", SolutionCumulative),
        ("Alias Method", SolutionAliasMethod)
    ]
    
    sizes = [100, 1000, 10000, 50000]
    
    for size in sizes:
        print(f"\nArray size: {size}")
        
        # Create weights array
        weights = [i + 1 for i in range(size)]
        
        for name, SolutionClass in implementations:
            start_time = time.time()
            solution = SolutionClass(weights)
            init_time = (time.time() - start_time) * 1000
            
            print(f"  {name}: {init_time:.2f}ms")

def test_memory_usage():
    """Test memory usage of different approaches"""
    print("\n=== Testing Memory Usage ===")
    
    sizes = [1000, 10000, 100000]
    
    for size in sizes:
        print(f"\nArray size: {size}")
        
        weights = [1] * size  # All equal weights
        
        implementations = [
            ("Cumulative Sum", SolutionCumulative),
            ("Alias Method", SolutionAliasMethod),
            ("With Statistics", SolutionWithStatistics)
        ]
        
        for name, SolutionClass in implementations:
            solution = SolutionClass(weights)
            
            # Estimate memory usage (simplified)
            memory_estimate = 0
            
            if hasattr(solution, 'cumulative'):
                memory_estimate += len(solution.cumulative)
            
            if hasattr(solution, 'prob') and hasattr(solution, 'alias'):
                memory_estimate += len(solution.prob) + len(solution.alias)
            
            if hasattr(solution, 'pick_counts'):
                memory_estimate += len(solution.pick_counts)
            
            print(f"  {name}: ~{memory_estimate} memory units")

def stress_test_picking():
    """Stress test picking operations"""
    print("\n=== Stress Testing Picking Operations ===")
    
    import time
    
    # Create challenging weight distribution
    weights = [1] * 1000 + [1000] * 10  # Many small weights + few large weights
    
    solution = SolutionBisect(weights)
    
    print(f"Stress testing with {len(weights)} weights...")
    print(f"Weight distribution: {len([w for w in weights if w == 1])} small, "
          f"{len([w for w in weights if w == 1000])} large")
    
    # Perform many picks
    num_picks = 100000
    start_time = time.time()
    
    picks = [solution.pickIndex() for _ in range(num_picks)]
    
    elapsed = time.time() - start_time
    
    # Analyze results
    small_picks = len([p for p in picks if p < 1000])
    large_picks = len([p for p in picks if p >= 1000])
    
    print(f"Results:")
    print(f"  {num_picks} picks in {elapsed:.2f}s ({num_picks/elapsed:.0f} picks/sec)")
    print(f"  Small weight picks: {small_picks}")
    print(f"  Large weight picks: {large_picks}")
    print(f"  Ratio: {large_picks/small_picks:.2f} (expected: ~1000)")

if __name__ == "__main__":
    test_random_pick_basic()
    test_distribution_quality()
    test_edge_cases()
    test_performance_comparison()
    test_alias_method_correctness()
    demonstrate_applications()
    test_large_scale_distribution()
    benchmark_initialization()
    test_memory_usage()
    stress_test_picking()

"""
Random Pick with Weight Design demonstrates key concepts:

Core Approaches:
1. Cumulative Sum + Binary Search - Standard approach O(log n) pick
2. Bisect Module - Simplified using built-in binary search
3. Linear Scan - Simple but inefficient O(n) pick
4. Alias Method - Advanced O(1) pick with O(n) preprocessing
5. With Statistics - Enhanced with distribution analysis

Key Design Principles:
- Weighted probability distribution
- Efficient random selection algorithms
- Cumulative probability calculations
- Trade-offs between preprocessing and query time

Performance Characteristics:
- Cumulative + Binary Search: O(n) init, O(log n) pick
- Linear Scan: O(n) init, O(n) pick
- Alias Method: O(n) init, O(1) pick
- Space complexity: O(n) for all approaches

Real-world Applications:
- A/B testing with weighted group assignment
- Game item drop systems with rarity weights
- Load balancing with server capacity weights
- Recommendation systems with preference weights
- Sampling from non-uniform distributions
- Monte Carlo simulations

The cumulative sum with binary search approach is most
commonly used due to its simplicity and good performance
characteristics for most use cases.
"""
