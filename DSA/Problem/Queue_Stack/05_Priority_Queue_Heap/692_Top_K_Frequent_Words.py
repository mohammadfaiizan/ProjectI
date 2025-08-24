"""
692. Top K Frequent Words - Multiple Approaches
Difficulty: Medium

Given an array of strings words and an integer k, return the k most frequent strings.

Return the answer sorted by the frequency from highest to lowest. Sort the words with the same frequency by their lexicographical order.
"""

from typing import List
import heapq
from collections import Counter

class TopKFrequentWords:
    """Multiple approaches to find top k frequent words"""
    
    def topKFrequent_heap_approach(self, words: List[str], k: int) -> List[str]:
        """
        Approach 1: Min Heap (Optimal)
        
        Use min heap of size k with custom comparison.
        
        Time: O(n log k), Space: O(k)
        """
        freq = Counter(words)
        
        # Min heap: (frequency, word)
        # For same frequency, we want lexicographically larger word at top
        heap = []
        
        for word, count in freq.items():
            # Push (-count, word) for max heap behavior on frequency
            # But we want min heap for lexicographical order when frequencies are same
            heapq.heappush(heap, (count, word))
            
            if len(heap) > k:
                heapq.heappop(heap)
        
        # Extract results and reverse for correct order
        result = []
        while heap:
            count, word = heapq.heappop(heap)
            result.append(word)
        
        # Sort by frequency (desc) then lexicographically (asc)
        result.sort(key=lambda x: (-freq[x], x))
        
        return result
    
    def topKFrequent_sorting(self, words: List[str], k: int) -> List[str]:
        """
        Approach 2: Sorting
        
        Sort all words by frequency and lexicographical order.
        
        Time: O(n log n), Space: O(n)
        """
        freq = Counter(words)
        
        # Sort by frequency (desc) then lexicographically (asc)
        sorted_words = sorted(freq.keys(), key=lambda x: (-freq[x], x))
        
        return sorted_words[:k]
    
    def topKFrequent_custom_heap(self, words: List[str], k: int) -> List[str]:
        """
        Approach 3: Custom Heap with Proper Comparison
        
        Use heap with custom comparison for frequency and lexicographical order.
        
        Time: O(n log k), Space: O(k)
        """
        freq = Counter(words)
        
        # Min heap of size k
        # Store (frequency, word) where we want:
        # - Smallest frequency at top (for removal)
        # - Lexicographically largest word at top (for removal when freq same)
        heap = []
        
        for word, count in freq.items():
            if len(heap) < k:
                # For min heap: smaller frequency and larger lexicographical word should be at top
                heapq.heappush(heap, (count, word))
            elif count > heap[0][0] or (count == heap[0][0] and word < heap[0][1]):
                heapq.heappop(heap)
                heapq.heappush(heap, (count, word))
        
        # Extract and sort properly
        result = [word for _, word in heap]
        result.sort(key=lambda x: (-freq[x], x))
        
        return result
    
    def topKFrequent_bucket_sort(self, words: List[str], k: int) -> List[str]:
        """
        Approach 4: Bucket Sort
        
        Use bucket sort for frequency-based sorting.
        
        Time: O(n), Space: O(n)
        """
        freq = Counter(words)
        
        # Create buckets for each frequency
        buckets = [[] for _ in range(len(words) + 1)]
        
        # Place words in buckets by frequency
        for word, count in freq.items():
            buckets[count].append(word)
        
        # Sort words within each bucket lexicographically
        for bucket in buckets:
            bucket.sort()
        
        # Collect top k words from highest frequency to lowest
        result = []
        for count in range(len(buckets) - 1, 0, -1):
            for word in buckets[count]:
                if len(result) < k:
                    result.append(word)
                else:
                    return result
        
        return result
    
    def topKFrequent_priority_queue_simulation(self, words: List[str], k: int) -> List[str]:
        """
        Approach 5: Priority Queue Simulation
        
        Simulate priority queue with list operations.
        
        Time: O(n^2), Space: O(n)
        """
        freq = Counter(words)
        
        # Create list of (frequency, word) tuples
        word_freq = [(count, word) for word, count in freq.items()]
        
        # Sort by frequency (desc) then lexicographically (asc)
        word_freq.sort(key=lambda x: (-x[0], x[1]))
        
        return [word for _, word in word_freq[:k]]


def test_top_k_frequent_words():
    """Test top k frequent words algorithms"""
    solver = TopKFrequentWords()
    
    test_cases = [
        (["i","love","leetcode","i","love","coding"], 2, ["i","love"], "Example 1"),
        (["the","day","is","sunny","the","the","the","sunny","is","is"], 4, ["the","is","sunny","day"], "Example 2"),
        (["a","aa","aaa"], 2, ["a","aa"], "Different lengths"),
        (["a"], 1, ["a"], "Single word"),
        (["a","b","c"], 2, ["a","b"], "All same frequency"),
        (["apple","banana","apple","cherry","banana","apple"], 2, ["apple","banana"], "Fruits"),
        (["word","word","word"], 1, ["word"], "All same word"),
    ]
    
    algorithms = [
        ("Heap Approach", solver.topKFrequent_heap_approach),
        ("Sorting", solver.topKFrequent_sorting),
        ("Custom Heap", solver.topKFrequent_custom_heap),
        ("Bucket Sort", solver.topKFrequent_bucket_sort),
        ("Priority Queue Sim", solver.topKFrequent_priority_queue_simulation),
    ]
    
    print("=== Testing Top K Frequent Words ===")
    
    for words, k, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Words: {words}")
        print(f"k: {k}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(words, k)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_sorting_criteria():
    """Demonstrate sorting criteria"""
    print("\n=== Sorting Criteria Demonstration ===")
    
    words = ["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"]
    k = 4
    
    print(f"Words: {words}")
    print(f"k: {k}")
    
    # Count frequencies
    freq = Counter(words)
    print(f"\nWord frequencies: {dict(freq)}")
    
    # Show sorting criteria
    print(f"\nSorting criteria:")
    print("1. By frequency (descending)")
    print("2. By lexicographical order (ascending) for same frequency")
    
    # Sort and show step by step
    word_freq = [(word, count) for word, count in freq.items()]
    
    print(f"\nBefore sorting: {word_freq}")
    
    # Sort by frequency (desc) then lexicographically (asc)
    word_freq.sort(key=lambda x: (-x[1], x[0]))
    
    print(f"After sorting: {word_freq}")
    
    result = [word for word, _ in word_freq[:k]]
    print(f"\nTop {k} frequent words: {result}")


if __name__ == "__main__":
    test_top_k_frequent_words()
    demonstrate_sorting_criteria()

"""
Top K Frequent Words demonstrates heap applications with custom comparison
criteria, including frequency-based sorting with lexicographical tie-breaking
and multiple optimization approaches.
"""
