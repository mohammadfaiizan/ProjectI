"""
384. Shuffle an Array - Multiple Approaches
Difficulty: Medium

Given an integer array nums, design an algorithm to randomly shuffle the array. All permutations of the array should be equally likely as a result of the shuffling.

Implement the Solution class:
- Solution(int[] nums) Initializes the object with the integer array nums.
- int[] reset() Resets the array to its original configuration and returns it.
- int[] shuffle() Returns a random shuffling of the array.
"""

import random
from typing import List
import copy

class Solution:
    """
    Approach 1: Fisher-Yates Shuffle (Knuth Shuffle)
    
    The gold standard for array shuffling with uniform distribution.
    
    Time Complexity: 
    - shuffle: O(n)
    - reset: O(n) for copying
    
    Space Complexity: O(n) for storing original array
    """
    
    def __init__(self, nums: List[int]):
        self.original = nums[:]  # Keep original copy
        self.array = nums[:]     # Working copy
    
    def reset(self) -> List[int]:
        self.array = self.original[:]
        return self.array
    
    def shuffle(self) -> List[int]:
        # Fisher-Yates shuffle
        for i in range(len(self.array) - 1, 0, -1):
            # Pick random index from 0 to i (inclusive)
            j = random.randint(0, i)
            # Swap elements
            self.array[i], self.array[j] = self.array[j], self.array[i]
        
        return self.array

class SolutionBruteForce:
    """
    Approach 2: Brute Force Random Selection
    
    Build shuffled array by randomly selecting from remaining elements.
    
    Time Complexity: 
    - shuffle: O(n²) due to list operations
    - reset: O(n)
    
    Space Complexity: O(n)
    """
    
    def __init__(self, nums: List[int]):
        self.original = nums[:]
        self.array = nums[:]
    
    def reset(self) -> List[int]:
        self.array = self.original[:]
        return self.array
    
    def shuffle(self) -> List[int]:
        shuffled = []
        remaining = self.array[:]
        
        while remaining:
            # Pick random element
            index = random.randint(0, len(remaining) - 1)
            shuffled.append(remaining.pop(index))
        
        self.array = shuffled
        return self.array

class SolutionSortBased:
    """
    Approach 3: Sort with Random Keys
    
    Assign random keys and sort - not uniformly random but simple.
    
    Time Complexity: 
    - shuffle: O(n log n)
    - reset: O(n)
    
    Space Complexity: O(n)
    """
    
    def __init__(self, nums: List[int]):
        self.original = nums[:]
        self.array = nums[:]
    
    def reset(self) -> List[int]:
        self.array = self.original[:]
        return self.array
    
    def shuffle(self) -> List[int]:
        # Create pairs of (random_key, value)
        paired = [(random.random(), num) for num in self.array]
        
        # Sort by random key
        paired.sort(key=lambda x: x[0])
        
        # Extract values
        self.array = [num for _, num in paired]
        return self.array

class SolutionInsideOut:
    """
    Approach 4: Inside-Out Fisher-Yates
    
    Build shuffled array from scratch without modifying original.
    
    Time Complexity: 
    - shuffle: O(n)
    - reset: O(n)
    
    Space Complexity: O(n)
    """
    
    def __init__(self, nums: List[int]):
        self.original = nums[:]
        self.array = nums[:]
    
    def reset(self) -> List[int]:
        self.array = self.original[:]
        return self.array
    
    def shuffle(self) -> List[int]:
        # Inside-out Fisher-Yates
        shuffled = [0] * len(self.original)
        
        for i in range(len(self.original)):
            j = random.randint(0, i)
            shuffled[i] = shuffled[j]
            shuffled[j] = self.original[i]
        
        self.array = shuffled
        return self.array

class SolutionAdvanced:
    """
    Approach 5: Advanced with Statistics and Analysis
    
    Enhanced version with shuffle quality analysis and statistics.
    
    Time Complexity: 
    - shuffle: O(n)
    - reset: O(n)
    
    Space Complexity: O(n + k) where k is history size
    """
    
    def __init__(self, nums: List[int], track_history: bool = False):
        self.original = nums[:]
        self.array = nums[:]
        self.track_history = track_history
        self.shuffle_history = [] if track_history else None
        self.shuffle_count = 0
        self.reset_count = 0
    
    def reset(self) -> List[int]:
        self.array = self.original[:]
        self.reset_count += 1
        return self.array
    
    def shuffle(self) -> List[int]:
        # Fisher-Yates with tracking
        swaps_made = []
        
        for i in range(len(self.array) - 1, 0, -1):
            j = random.randint(0, i)
            
            if self.track_history:
                swaps_made.append((i, j, self.array[i], self.array[j]))
            
            self.array[i], self.array[j] = self.array[j], self.array[i]
        
        self.shuffle_count += 1
        
        if self.track_history:
            self.shuffle_history.append({
                'shuffle_id': self.shuffle_count,
                'swaps': swaps_made,
                'result': self.array[:]
            })
        
        return self.array
    
    def get_statistics(self) -> dict:
        """Get shuffle statistics"""
        return {
            'shuffle_count': self.shuffle_count,
            'reset_count': self.reset_count,
            'original_length': len(self.original),
            'current_array': self.array[:],
            'history_size': len(self.shuffle_history) if self.shuffle_history else 0
        }
    
    def analyze_randomness(self, num_shuffles: int = 1000) -> dict:
        """Analyze randomness quality of shuffles"""
        if len(self.original) > 10:  # Don't analyze large arrays
            return {"error": "Array too large for analysis"}
        
        position_counts = {}
        
        # Initialize position tracking
        for i, val in enumerate(self.original):
            position_counts[val] = [0] * len(self.original)
        
        # Perform multiple shuffles
        for _ in range(num_shuffles):
            self.shuffle()
            
            for pos, val in enumerate(self.array):
                position_counts[val][pos] += 1
        
        # Calculate uniformity
        expected_count = num_shuffles / len(self.original)
        max_deviation = 0
        
        for val, counts in position_counts.items():
            for count in counts:
                deviation = abs(count - expected_count) / expected_count
                max_deviation = max(max_deviation, deviation)
        
        return {
            'num_shuffles': num_shuffles,
            'expected_count_per_position': expected_count,
            'max_deviation_percentage': max_deviation * 100,
            'position_counts': position_counts
        }


def test_shuffle_basic():
    """Test basic shuffle functionality"""
    print("=== Testing Basic Shuffle Functionality ===")
    
    implementations = [
        ("Fisher-Yates", Solution),
        ("Brute Force", SolutionBruteForce),
        ("Sort-based", SolutionSortBased),
        ("Inside-Out", SolutionInsideOut),
        ("Advanced", SolutionAdvanced)
    ]
    
    test_array = [1, 2, 3, 4, 5]
    
    for name, SolutionClass in implementations:
        print(f"\n{name}:")
        
        solution = SolutionClass(test_array)
        
        # Test multiple shuffles
        print(f"  Original: {test_array}")
        
        for i in range(3):
            shuffled = solution.shuffle()
            print(f"  Shuffle {i+1}: {shuffled}")
        
        # Test reset
        reset_array = solution.reset()
        print(f"  After reset: {reset_array}")

def test_shuffle_edge_cases():
    """Test shuffle edge cases"""
    print("\n=== Testing Shuffle Edge Cases ===")
    
    # Empty array
    print("Empty array:")
    solution = Solution([])
    print(f"  shuffle(): {solution.shuffle()}")
    print(f"  reset(): {solution.reset()}")
    
    # Single element
    print(f"\nSingle element:")
    solution = Solution([42])
    print(f"  Original: {solution.reset()}")
    print(f"  Shuffle: {solution.shuffle()}")
    print(f"  Reset: {solution.reset()}")
    
    # Two elements
    print(f"\nTwo elements:")
    solution = Solution([1, 2])
    
    results = []
    for i in range(10):
        shuffled = solution.shuffle()
        results.append(tuple(shuffled))
        solution.reset()
    
    print(f"  10 shuffles of [1, 2]: {results}")
    
    # All same elements
    print(f"\nAll same elements:")
    solution = Solution([5, 5, 5, 5])
    print(f"  Original: {solution.reset()}")
    print(f"  Shuffle: {solution.shuffle()}")

def test_shuffle_distribution():
    """Test shuffle distribution quality"""
    print("\n=== Testing Shuffle Distribution Quality ===")
    
    test_array = [1, 2, 3]
    solution = Solution(test_array)
    
    # Count how often each element appears in each position
    position_counts = {}
    for val in test_array:
        position_counts[val] = [0, 0, 0]
    
    num_trials = 6000  # Should be multiple of 6 for [1,2,3]
    
    for _ in range(num_trials):
        shuffled = solution.shuffle()
        
        for pos, val in enumerate(shuffled):
            position_counts[val][pos] += 1
        
        solution.reset()
    
    print(f"Distribution analysis over {num_trials} shuffles:")
    print(f"Expected count per position: {num_trials // 3}")
    
    for val in test_array:
        print(f"  Element {val}: {position_counts[val]}")
        
        # Calculate deviation from expected
        expected = num_trials // 3
        deviations = [abs(count - expected) / expected for count in position_counts[val]]
        max_deviation = max(deviations) * 100
        
        print(f"    Max deviation: {max_deviation:.1f}%")

def test_shuffle_performance():
    """Test shuffle performance"""
    print("\n=== Testing Shuffle Performance ===")
    
    import time
    
    implementations = [
        ("Fisher-Yates", Solution),
        ("Brute Force", SolutionBruteForce),
        ("Sort-based", SolutionSortBased),
        ("Inside-Out", SolutionInsideOut)
    ]
    
    array_sizes = [100, 1000, 10000]
    
    for size in array_sizes:
        print(f"\nArray size: {size}")
        test_array = list(range(size))
        
        for name, SolutionClass in implementations:
            solution = SolutionClass(test_array)
            
            # Time shuffle operations
            start_time = time.time()
            
            for _ in range(100):  # 100 shuffles
                solution.shuffle()
                solution.reset()
            
            elapsed = (time.time() - start_time) * 1000
            avg_time = elapsed / 100
            
            print(f"  {name}: {elapsed:.2f}ms total, {avg_time:.3f}ms per shuffle")

def test_advanced_features():
    """Test advanced features"""
    print("\n=== Testing Advanced Features ===")
    
    test_array = [1, 2, 3, 4]
    solution = SolutionAdvanced(test_array, track_history=True)
    
    # Perform several operations
    print("Performing shuffles and resets:")
    
    for i in range(3):
        shuffled = solution.shuffle()
        print(f"  Shuffle {i+1}: {shuffled}")
    
    solution.reset()
    print(f"  Reset: {solution.reset()}")
    
    solution.shuffle()
    print(f"  Final shuffle: {solution.shuffle()}")
    
    # Get statistics
    stats = solution.get_statistics()
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

def test_randomness_analysis():
    """Test randomness analysis"""
    print("\n=== Testing Randomness Analysis ===")
    
    test_array = [1, 2, 3, 4]
    solution = SolutionAdvanced(test_array)
    
    print("Analyzing randomness quality...")
    analysis = solution.analyze_randomness(num_shuffles=2400)  # 600 expected per position
    
    print(f"Analysis results:")
    for key, value in analysis.items():
        if key == 'position_counts':
            print(f"  {key}:")
            for val, counts in value.items():
                print(f"    Element {val}: {counts}")
        else:
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: Card deck shuffling
    print("Application 1: Card Deck Shuffling")
    
    # Create a deck of cards (simplified)
    suits = ['♠', '♥', '♦', '♣']
    ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    deck = [f"{rank}{suit}" for suit in suits for rank in ranks]
    
    card_shuffler = Solution(deck)
    
    print(f"  Original deck (first 10): {deck[:10]}")
    
    shuffled_deck = card_shuffler.shuffle()
    print(f"  Shuffled deck (first 10): {shuffled_deck[:10]}")
    
    # Deal some hands
    print(f"  Dealing 4 hands of 5 cards each:")
    for player in range(4):
        hand = shuffled_deck[player*5:(player+1)*5]
        print(f"    Player {player+1}: {hand}")
    
    # Application 2: Playlist shuffling
    print(f"\nApplication 2: Music Playlist Shuffling")
    
    playlist = [
        "Song A - Artist 1", "Song B - Artist 2", "Song C - Artist 1",
        "Song D - Artist 3", "Song E - Artist 2", "Song F - Artist 4",
        "Song G - Artist 1", "Song H - Artist 3"
    ]
    
    music_shuffler = Solution(playlist)
    
    print(f"  Original playlist:")
    for i, song in enumerate(playlist):
        print(f"    {i+1}. {song}")
    
    shuffled_playlist = music_shuffler.shuffle()
    print(f"  Shuffled playlist:")
    for i, song in enumerate(shuffled_playlist):
        print(f"    {i+1}. {song}")
    
    # Application 3: A/B testing user assignment
    print(f"\nApplication 3: A/B Testing User Assignment")
    
    user_ids = [f"user_{i:04d}" for i in range(1, 101)]  # 100 users
    ab_shuffler = Solution(user_ids)
    
    shuffled_users = ab_shuffler.shuffle()
    
    # Assign to groups
    group_a = shuffled_users[:50]  # First 50
    group_b = shuffled_users[50:]  # Last 50
    
    print(f"  Total users: {len(user_ids)}")
    print(f"  Group A (first 10): {group_a[:10]}")
    print(f"  Group B (first 10): {group_b[:10]}")
    print(f"  Group sizes: A={len(group_a)}, B={len(group_b)}")

def test_bias_detection():
    """Test for potential bias in shuffle algorithms"""
    print("\n=== Testing for Bias Detection ===")
    
    # Simple test for obvious bias
    test_array = [1, 2, 3]
    implementations = [
        ("Fisher-Yates", Solution),
        ("Sort-based", SolutionSortBased)
    ]
    
    for name, SolutionClass in implementations:
        print(f"\n{name} bias test:")
        
        solution = SolutionClass(test_array)
        
        # Count first element frequency
        first_element_counts = {1: 0, 2: 0, 3: 0}
        
        num_tests = 3000
        for _ in range(num_tests):
            shuffled = solution.shuffle()
            first_element_counts[shuffled[0]] += 1
            solution.reset()
        
        expected_count = num_tests // 3
        print(f"  First element frequency (expected ~{expected_count}):")
        
        for element, count in first_element_counts.items():
            percentage = (count / num_tests) * 100
            deviation = abs(count - expected_count) / expected_count * 100
            print(f"    Element {element}: {count} times ({percentage:.1f}%, deviation: {deviation:.1f}%)")

def test_shuffle_invariants():
    """Test shuffle invariants"""
    print("\n=== Testing Shuffle Invariants ===")
    
    test_array = [1, 2, 3, 4, 5, 6]
    solution = Solution(test_array)
    
    print("Testing shuffle invariants:")
    
    for test_num in range(5):
        original = solution.reset()
        shuffled = solution.shuffle()
        
        # Check 1: Same length
        length_preserved = len(original) == len(shuffled)
        
        # Check 2: Same elements (sorted)
        elements_preserved = sorted(original) == sorted(shuffled)
        
        # Check 3: Valid permutation
        is_permutation = set(original) == set(shuffled)
        
        print(f"  Test {test_num + 1}:")
        print(f"    Original: {original}")
        print(f"    Shuffled: {shuffled}")
        print(f"    Length preserved: {length_preserved}")
        print(f"    Elements preserved: {elements_preserved}")
        print(f"    Valid permutation: {is_permutation}")
        
        if not (length_preserved and elements_preserved and is_permutation):
            print("    ❌ INVARIANT VIOLATION!")
        else:
            print("    ✅ All invariants satisfied")

def benchmark_memory_usage():
    """Benchmark memory usage"""
    print("\n=== Benchmarking Memory Usage ===")
    
    import sys
    
    array_sizes = [1000, 10000, 100000]
    
    for size in array_sizes:
        print(f"\nArray size: {size}")
        
        test_array = list(range(size))
        
        implementations = [
            ("Fisher-Yates", Solution),
            ("Inside-Out", SolutionInsideOut),
            ("Advanced (no history)", lambda arr: SolutionAdvanced(arr, track_history=False)),
            ("Advanced (with history)", lambda arr: SolutionAdvanced(arr, track_history=True))
        ]
        
        for name, solution_factory in implementations:
            solution = solution_factory(test_array)
            
            # Rough memory estimation
            original_size = len(solution.original) if hasattr(solution, 'original') else size
            array_size = len(solution.array) if hasattr(solution, 'array') else size
            
            base_memory = original_size + array_size
            
            if hasattr(solution, 'shuffle_history') and solution.shuffle_history is not None:
                history_memory = 0  # Would grow with shuffles
            else:
                history_memory = 0
            
            total_memory = base_memory + history_memory
            
            print(f"  {name}: ~{total_memory} elements ({base_memory} base + {history_memory} history)")

def test_concurrent_shuffling():
    """Simulate concurrent shuffling scenarios"""
    print("\n=== Testing Concurrent Shuffling Scenarios ===")
    
    # Simulate multiple independent shufflers
    base_array = [1, 2, 3, 4, 5]
    
    shufflers = [Solution(base_array) for _ in range(3)]
    
    print("Multiple independent shufflers:")
    
    for round_num in range(3):
        print(f"\nRound {round_num + 1}:")
        
        for i, shuffler in enumerate(shufflers):
            shuffled = shuffler.shuffle()
            print(f"  Shuffler {i+1}: {shuffled}")
        
        # Reset all
        for i, shuffler in enumerate(shufflers):
            reset_result = shuffler.reset()
            print(f"  Shuffler {i+1} reset: {reset_result}")

def stress_test_shuffle():
    """Stress test shuffle operations"""
    print("\n=== Stress Testing Shuffle Operations ===")
    
    import time
    
    test_array = list(range(1000))
    solution = Solution(test_array)
    
    print("Performing stress test...")
    
    start_time = time.time()
    
    # Many shuffle/reset cycles
    for cycle in range(1000):
        solution.shuffle()
        
        if cycle % 100 == 0:
            solution.reset()
    
    elapsed = time.time() - start_time
    
    print(f"Completed 1000 shuffle operations with resets in {elapsed:.2f}s")
    print(f"Average time per operation: {(elapsed/1000)*1000:.3f}ms")

if __name__ == "__main__":
    test_shuffle_basic()
    test_shuffle_edge_cases()
    test_shuffle_distribution()
    test_shuffle_performance()
    test_advanced_features()
    test_randomness_analysis()
    demonstrate_applications()
    test_bias_detection()
    test_shuffle_invariants()
    benchmark_memory_usage()
    test_concurrent_shuffling()
    stress_test_shuffle()

"""
Array Shuffle Design demonstrates key concepts:

Core Approaches:
1. Fisher-Yates - Gold standard with uniform distribution O(n)
2. Brute Force - Simple but inefficient O(n²)
3. Sort-based - Using random keys O(n log n) 
4. Inside-Out - Build shuffled array from scratch O(n)
5. Advanced - Enhanced with analytics and tracking

Key Design Principles:
- Uniform random distribution
- Preserving original array for reset
- Efficiency considerations for large arrays
- Statistical analysis of randomness quality

Algorithm Properties:
- Fisher-Yates: Optimal time/space, proven uniform distribution
- Brute Force: Simple to understand but inefficient
- Sort-based: Not uniformly random due to sorting stability
- Inside-Out: Good for scenarios where original shouldn't be modified

Real-world Applications:
- Card game shuffling
- Music playlist randomization
- A/B testing user assignment
- Random sampling for data analysis
- Cryptographic applications requiring randomness
- Game development (random events, loot drops)

The Fisher-Yates shuffle is the industry standard
due to its proven uniform distribution and optimal
O(n) time complexity.
"""
