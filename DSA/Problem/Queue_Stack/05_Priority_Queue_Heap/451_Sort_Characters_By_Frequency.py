"""
451. Sort Characters By Frequency - Multiple Approaches
Difficulty: Medium

Given a string s, sort it in decreasing order based on the frequency of the characters. The frequency of a character is the number of times it appears in the string.

Return the sorted string. If there are multiple answers, return any of them.
"""

from typing import Dict, List
import heapq
from collections import Counter, defaultdict

class SortCharactersByFrequency:
    """Multiple approaches to sort characters by frequency"""
    
    def frequencySort_heap_approach(self, s: str) -> str:
        """
        Approach 1: Max Heap (Optimal)
        
        Use max heap to sort by frequency.
        
        Time: O(n log k), Space: O(k) where k is unique characters
        """
        # Count frequencies
        freq = Counter(s)
        
        # Build max heap (negate frequencies)
        heap = [(-count, char) for char, count in freq.items()]
        heapq.heapify(heap)
        
        # Build result
        result = []
        while heap:
            neg_count, char = heapq.heappop(heap)
            count = -neg_count
            result.append(char * count)
        
        return ''.join(result)
    
    def frequencySort_sorting(self, s: str) -> str:
        """
        Approach 2: Sorting
        
        Sort characters by frequency using built-in sort.
        
        Time: O(k log k + n), Space: O(k)
        """
        freq = Counter(s)
        
        # Sort by frequency (descending)
        sorted_chars = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        # Build result
        result = []
        for char, count in sorted_chars:
            result.append(char * count)
        
        return ''.join(result)
    
    def frequencySort_bucket_sort(self, s: str) -> str:
        """
        Approach 3: Bucket Sort
        
        Use bucket sort for O(n) time complexity.
        
        Time: O(n), Space: O(n)
        """
        freq = Counter(s)
        
        # Create buckets for each frequency
        buckets = [[] for _ in range(len(s) + 1)]
        
        # Place characters in buckets by frequency
        for char, count in freq.items():
            buckets[count].append(char)
        
        # Build result from highest frequency to lowest
        result = []
        for count in range(len(buckets) - 1, 0, -1):
            for char in buckets[count]:
                result.append(char * count)
        
        return ''.join(result)
    
    def frequencySort_priority_queue(self, s: str) -> str:
        """
        Approach 4: Priority Queue with Custom Comparator
        
        Use priority queue with frequency-based comparison.
        
        Time: O(n log k), Space: O(k)
        """
        freq = Counter(s)
        
        # Create list of (frequency, character) tuples
        freq_list = [(count, char) for char, count in freq.items()]
        
        # Sort by frequency (descending)
        freq_list.sort(reverse=True)
        
        # Build result
        result = []
        for count, char in freq_list:
            result.append(char * count)
        
        return ''.join(result)
    
    def frequencySort_counting_sort(self, s: str) -> str:
        """
        Approach 5: Counting Sort
        
        Use counting sort for ASCII characters.
        
        Time: O(n), Space: O(1) for ASCII
        """
        # Count frequencies (assuming ASCII)
        freq = [0] * 128
        for char in s:
            freq[ord(char)] += 1
        
        # Create list of (frequency, character) pairs
        char_freq = []
        for i in range(128):
            if freq[i] > 0:
                char_freq.append((freq[i], chr(i)))
        
        # Sort by frequency (descending)
        char_freq.sort(reverse=True)
        
        # Build result
        result = []
        for count, char in char_freq:
            result.append(char * count)
        
        return ''.join(result)
    
    def frequencySort_multiset_simulation(self, s: str) -> str:
        """
        Approach 6: Multiset Simulation
        
        Simulate multiset behavior with sorted operations.
        
        Time: O(n log k), Space: O(k)
        """
        freq = Counter(s)
        
        # Convert to list and sort by frequency
        items = list(freq.items())
        items.sort(key=lambda x: (-x[1], x[0]))  # Sort by freq desc, then char asc
        
        # Build result
        result = []
        for char, count in items:
            result.append(char * count)
        
        return ''.join(result)


def test_sort_characters_by_frequency():
    """Test sort characters by frequency algorithms"""
    solver = SortCharactersByFrequency()
    
    test_cases = [
        ("tree", ["eert", "eetr"], "Example 1"),
        ("cccaaa", ["aaaccc", "cccaaa"], "Example 2"),
        ("Aabb", ["bbAa", "bbaA"], "Example 3"),
        ("a", ["a"], "Single character"),
        ("aa", ["aa"], "Two same characters"),
        ("ab", ["ab", "ba"], "Two different characters"),
        ("abcabc", ["aabbcc", "bbaacc", "ccaabb", "ccbbaa", "aaccbb", "bbccaa"], "Multiple same frequencies"),
        ("programming", ["mmrrggaiopn", "rrmmggaiopn"], "Complex string"),
    ]
    
    algorithms = [
        ("Heap Approach", solver.frequencySort_heap_approach),
        ("Sorting", solver.frequencySort_sorting),
        ("Bucket Sort", solver.frequencySort_bucket_sort),
        ("Priority Queue", solver.frequencySort_priority_queue),
        ("Counting Sort", solver.frequencySort_counting_sort),
        ("Multiset Simulation", solver.frequencySort_multiset_simulation),
    ]
    
    print("=== Testing Sort Characters By Frequency ===")
    
    for s, expected_list, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Input: '{s}'")
        print(f"Expected (any of): {expected_list}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(s)
                status = "✓" if result in expected_list else "✗"
                print(f"{alg_name:20} | {status} | Result: '{result}'")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_heap_approach():
    """Demonstrate heap approach step by step"""
    print("\n=== Heap Approach Step-by-Step Demo ===")
    
    s = "tree"
    print(f"Input string: '{s}'")
    
    # Count frequencies
    freq = Counter(s)
    print(f"Character frequencies: {dict(freq)}")
    
    # Build max heap
    heap = [(-count, char) for char, count in freq.items()]
    heapq.heapify(heap)
    
    print(f"Max heap (negated frequencies): {heap}")
    
    # Extract characters by frequency
    result_parts = []
    step = 1
    
    while heap:
        neg_count, char = heapq.heappop(heap)
        count = -neg_count
        part = char * count
        result_parts.append(part)
        
        print(f"Step {step}: Extract '{char}' (freq={count}) -> '{part}'")
        step += 1
    
    result = ''.join(result_parts)
    print(f"\nFinal result: '{result}'")


def demonstrate_bucket_sort_approach():
    """Demonstrate bucket sort approach"""
    print("\n=== Bucket Sort Approach Demonstration ===")
    
    s = "programming"
    print(f"Input string: '{s}'")
    
    # Count frequencies
    freq = Counter(s)
    print(f"Character frequencies: {dict(freq)}")
    
    # Create buckets
    buckets = [[] for _ in range(len(s) + 1)]
    print(f"Created {len(buckets)} buckets (indices 0 to {len(s)})")
    
    # Place characters in buckets
    for char, count in freq.items():
        buckets[count].append(char)
        print(f"  Placed '{char}' in bucket {count}")
    
    print(f"\nBuckets after placement:")
    for i, bucket in enumerate(buckets):
        if bucket:
            print(f"  Bucket {i}: {bucket}")
    
    # Build result from highest frequency
    result_parts = []
    print(f"\nBuilding result from highest to lowest frequency:")
    
    for count in range(len(buckets) - 1, 0, -1):
        if buckets[count]:
            for char in buckets[count]:
                part = char * count
                result_parts.append(part)
                print(f"  Frequency {count}: '{char}' -> '{part}'")
    
    result = ''.join(result_parts)
    print(f"\nFinal result: '{result}'")


def visualize_frequency_distribution():
    """Visualize frequency distribution"""
    print("\n=== Frequency Distribution Visualization ===")
    
    s = "aabbbbcccc"
    print(f"Input string: '{s}'")
    
    freq = Counter(s)
    
    print(f"\nFrequency distribution:")
    max_freq = max(freq.values())
    
    for char in sorted(freq.keys()):
        count = freq[char]
        bar = "█" * count + "░" * (max_freq - count)
        print(f"  '{char}': {bar} ({count})")
    
    # Show sorted result
    sorted_chars = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    result_parts = [char * count for char, count in sorted_chars]
    result = ''.join(result_parts)
    
    print(f"\nSorted by frequency: '{result}'")


def benchmark_sort_characters_by_frequency():
    """Benchmark different approaches"""
    import time
    import random
    import string
    
    algorithms = [
        ("Heap Approach", SortCharactersByFrequency().frequencySort_heap_approach),
        ("Sorting", SortCharactersByFrequency().frequencySort_sorting),
        ("Bucket Sort", SortCharactersByFrequency().frequencySort_bucket_sort),
        ("Counting Sort", SortCharactersByFrequency().frequencySort_counting_sort),
    ]
    
    # Test with different string lengths
    test_sizes = [1000, 10000, 50000]
    
    print("\n=== Sort Characters By Frequency Performance Benchmark ===")
    
    for size in test_sizes:
        print(f"\n--- String Length: {size} ---")
        
        # Generate random string
        test_string = ''.join(random.choices(string.ascii_lowercase, k=size))
        
        for alg_name, alg_func in algorithms:
            start_time = time.time()
            
            try:
                result = alg_func(test_string)
                end_time = time.time()
                print(f"{alg_name:20} | Time: {end_time - start_time:.4f}s | Length: {len(result)}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:30]}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = SortCharactersByFrequency()
    
    edge_cases = [
        ("", [""], "Empty string"),
        ("a", ["a"], "Single character"),
        ("aa", ["aa"], "Two same characters"),
        ("ab", ["ab", "ba"], "Two different characters"),
        ("ABC", ["ABC", "ACB", "BAC", "BCA", "CAB", "CBA"], "All different frequencies"),
        ("aaa", ["aaa"], "All same characters"),
        ("123", ["123", "132", "213", "231", "312", "321"], "Numbers"),
        ("a1B2c3", ["a1B2c3"], "Mixed alphanumeric"),
    ]
    
    for s, expected_list, description in edge_cases:
        try:
            result = solver.frequencySort_heap_approach(s)
            status = "✓" if result in expected_list else "✗"
            print(f"{description:25} | {status} | input: '{s}' -> '{result}'")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def compare_approaches():
    """Compare different approaches"""
    print("\n=== Approach Comparison ===")
    
    test_cases = [
        "tree",
        "cccaaa",
        "programming",
        "abcdefg",
    ]
    
    solver = SortCharactersByFrequency()
    
    approaches = [
        ("Heap", solver.frequencySort_heap_approach),
        ("Sorting", solver.frequencySort_sorting),
        ("Bucket Sort", solver.frequencySort_bucket_sort),
        ("Counting Sort", solver.frequencySort_counting_sort),
    ]
    
    for i, s in enumerate(test_cases):
        print(f"\nTest case {i+1}: '{s}'")
        
        results = {}
        
        for name, func in approaches:
            try:
                result = func(s)
                results[name] = result
                print(f"{name:15} | Result: '{result}'")
            except Exception as e:
                print(f"{name:15} | ERROR: {str(e)[:40]}")
        
        # Check if all results are valid (same character counts)
        if results:
            first_result = list(results.values())[0]
            first_counter = Counter(first_result)
            
            all_valid = all(Counter(result) == first_counter for result in results.values())
            print(f"All results have same character counts: {'✓' if all_valid else '✗'}")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Heap Approach", "O(n + k log k)", "O(k)", "Count + heap operations"),
        ("Sorting", "O(n + k log k)", "O(k)", "Count + sort unique characters"),
        ("Bucket Sort", "O(n)", "O(n)", "Linear time, linear space"),
        ("Priority Queue", "O(n + k log k)", "O(k)", "Count + sort by frequency"),
        ("Counting Sort", "O(n)", "O(1)", "For ASCII characters"),
        ("Multiset Simulation", "O(n + k log k)", "O(k)", "Count + sort operations"),
    ]
    
    print(f"{'Approach':<20} | {'Time':<15} | {'Space':<8} | {'Notes'}")
    print("-" * 70)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<20} | {time_comp:<15} | {space_comp:<8} | {notes}")
    
    print(f"\nWhere n = string length, k = unique characters")


if __name__ == "__main__":
    test_sort_characters_by_frequency()
    demonstrate_heap_approach()
    demonstrate_bucket_sort_approach()
    visualize_frequency_distribution()
    test_edge_cases()
    compare_approaches()
    analyze_time_complexity()
    benchmark_sort_characters_by_frequency()

"""
Sort Characters By Frequency demonstrates heap and sorting applications
for frequency-based problems, including bucket sort optimization and
multiple approaches for character frequency analysis.
"""
