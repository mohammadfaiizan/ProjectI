"""
LeetCode 300: Longest Increasing Subsequence
Difficulty: Medium
Category: Longest Subsequence Problems (Classic LIS)

PROBLEM DESCRIPTION:
===================
Given an integer array nums, return the length of the longest strictly increasing subsequence.

A subsequence is a sequence that can be derived from an array by deleting some or no elements 
without changing the order of the remaining elements. For example, [3,6,2,7] is a subsequence 
of the array [0,3,1,6,2,2,7].

Example 1:
Input: nums = [10,9,2,5,3,7,101,18]
Output: 4
Explanation: The longest increasing subsequence is [2,3,7,18].

Example 2:
Input: nums = [0,1,0,3,2,3]
Output: 4

Example 3:
Input: nums = [7,7,7,7,7,7,7]
Output: 1

Constraints:
- 1 <= nums.length <= 2500
- -10^4 <= nums[i] <= 10^4
"""

def length_of_lis_brute_force(nums):
    """
    BRUTE FORCE APPROACH - RECURSION:
    ================================
    Try all possible subsequences.
    
    Time Complexity: O(2^n) - exponential subsequences
    Space Complexity: O(n) - recursion stack depth
    """
    def lis_from_index(index, prev_value):
        if index >= len(nums):
            return 0
        
        # Option 1: Include current element if it's increasing
        include = 0
        if nums[index] > prev_value:
            include = 1 + lis_from_index(index + 1, nums[index])
        
        # Option 2: Skip current element
        skip = lis_from_index(index + 1, prev_value)
        
        return max(include, skip)
    
    return lis_from_index(0, float('-inf'))


def length_of_lis_memoization(nums):
    """
    MEMOIZATION APPROACH:
    ====================
    Cache results to avoid recomputation.
    
    Time Complexity: O(n^2) - n*n states
    Space Complexity: O(n^2) - memoization table
    """
    memo = {}
    
    def lis_from_index(index, prev_index):
        if index >= len(nums):
            return 0
        
        if (index, prev_index) in memo:
            return memo[(index, prev_index)]
        
        # Option 1: Include current element if it's increasing
        include = 0
        if prev_index == -1 or nums[index] > nums[prev_index]:
            include = 1 + lis_from_index(index + 1, index)
        
        # Option 2: Skip current element
        skip = lis_from_index(index + 1, prev_index)
        
        result = max(include, skip)
        memo[(index, prev_index)] = result
        return result
    
    return lis_from_index(0, -1)


def length_of_lis_dp_quadratic(nums):
    """
    DYNAMIC PROGRAMMING O(n^2):
    ===========================
    Classic DP approach with nested loops.
    
    Time Complexity: O(n^2) - nested loops
    Space Complexity: O(n) - DP array
    """
    if not nums:
        return 0
    
    n = len(nums)
    dp = [1] * n  # dp[i] = length of LIS ending at index i
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)


def length_of_lis_binary_search(nums):
    """
    BINARY SEARCH APPROACH (OPTIMAL):
    =================================
    Use binary search with patience sorting algorithm.
    
    Time Complexity: O(n log n) - n elements, log n binary search
    Space Complexity: O(n) - tails array
    """
    if not nums:
        return 0
    
    # tails[i] = smallest ending element of all increasing subsequences of length i+1
    tails = []
    
    for num in nums:
        # Binary search for the position to insert/replace
        left, right = 0, len(tails)
        
        while left < right:
            mid = (left + right) // 2
            if tails[mid] < num:
                left = mid + 1
            else:
                right = mid
        
        # If left == len(tails), append; otherwise replace
        if left == len(tails):
            tails.append(num)
        else:
            tails[left] = num
    
    return len(tails)


def length_of_lis_binary_search_builtin(nums):
    """
    BINARY SEARCH WITH BUILT-IN BISECT:
    ===================================
    Use Python's bisect module for cleaner code.
    
    Time Complexity: O(n log n) - same as above
    Space Complexity: O(n) - tails array
    """
    import bisect
    
    if not nums:
        return 0
    
    tails = []
    
    for num in nums:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    
    return len(tails)


def length_of_lis_with_sequence(nums):
    """
    FIND ACTUAL LIS SEQUENCE:
    ========================
    Return both length and one possible LIS.
    
    Time Complexity: O(n^2) - DP + reconstruction
    Space Complexity: O(n) - DP array + parent tracking
    """
    if not nums:
        return 0, []
    
    n = len(nums)
    dp = [1] * n
    parent = [-1] * n  # To reconstruct the sequence
    
    max_length = 1
    max_index = 0
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
        
        if dp[i] > max_length:
            max_length = dp[i]
            max_index = i
    
    # Reconstruct the sequence
    lis = []
    current = max_index
    
    while current != -1:
        lis.append(nums[current])
        current = parent[current]
    
    lis.reverse()
    return max_length, lis


def length_of_lis_with_all_sequences(nums):
    """
    FIND ALL LIS OF MAXIMUM LENGTH:
    ==============================
    Return all possible LIS sequences.
    
    Time Complexity: O(n^2 * k) where k is number of LIS
    Space Complexity: O(n * k) - store all sequences
    """
    if not nums:
        return 0, []
    
    n = len(nums)
    dp = [1] * n
    sequences = [[[nums[i]]] for i in range(n)]  # All sequences ending at each index
    
    max_length = 1
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                if dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    sequences[i] = [seq + [nums[i]] for seq in sequences[j]]
                elif dp[j] + 1 == dp[i]:
                    sequences[i].extend([seq + [nums[i]] for seq in sequences[j]])
        
        max_length = max(max_length, dp[i])
    
    # Collect all sequences of maximum length
    all_lis = []
    for i in range(n):
        if dp[i] == max_length:
            all_lis.extend(sequences[i])
    
    # Remove duplicates
    unique_lis = []
    seen = set()
    for seq in all_lis:
        seq_tuple = tuple(seq)
        if seq_tuple not in seen:
            seen.add(seq_tuple)
            unique_lis.append(seq)
    
    return max_length, unique_lis


def length_of_lis_segment_tree(nums):
    """
    SEGMENT TREE APPROACH:
    =====================
    Advanced approach using segment tree for range maximum queries.
    
    Time Complexity: O(n log n) - coordinate compression + segment tree
    Space Complexity: O(n) - segment tree
    """
    if not nums:
        return 0
    
    # Coordinate compression
    sorted_nums = sorted(set(nums))
    coord_map = {v: i for i, v in enumerate(sorted_nums)}
    
    # Segment tree for range maximum queries
    class SegmentTree:
        def __init__(self, size):
            self.size = size
            self.tree = [0] * (4 * size)
        
        def update(self, node, start, end, idx, val):
            if start == end:
                self.tree[node] = max(self.tree[node], val)
            else:
                mid = (start + end) // 2
                if idx <= mid:
                    self.update(2 * node, start, mid, idx, val)
                else:
                    self.update(2 * node + 1, mid + 1, end, idx, val)
                self.tree[node] = max(self.tree[2 * node], self.tree[2 * node + 1])
        
        def query(self, node, start, end, l, r):
            if r < start or end < l:
                return 0
            if l <= start and end <= r:
                return self.tree[node]
            
            mid = (start + end) // 2
            left_max = self.query(2 * node, start, mid, l, r)
            right_max = self.query(2 * node + 1, mid + 1, end, l, r)
            return max(left_max, right_max)
    
    seg_tree = SegmentTree(len(sorted_nums))
    max_length = 0
    
    for num in nums:
        coord = coord_map[num]
        # Query for maximum LIS length ending with value < num
        current_max = seg_tree.query(1, 0, len(sorted_nums) - 1, 0, coord - 1)
        new_length = current_max + 1
        
        # Update segment tree
        seg_tree.update(1, 0, len(sorted_nums) - 1, coord, new_length)
        max_length = max(max_length, new_length)
    
    return max_length


def length_of_lis_patience_sorting_detailed(nums):
    """
    PATIENCE SORTING DETAILED EXPLANATION:
    =====================================
    Detailed implementation showing the patience sorting intuition.
    
    Time Complexity: O(n log n) - binary search for each element
    Space Complexity: O(n) - piles array
    """
    if not nums:
        return 0
    
    # Think of this as patience sorting (card game)
    # Each pile maintains cards in decreasing order from bottom to top
    # We want to minimize the number of piles
    
    piles = []  # piles[i] = top card of pile i
    
    for num in nums:
        # Find the leftmost pile where we can place this card
        # (binary search for smallest element >= num)
        left, right = 0, len(piles)
        
        while left < right:
            mid = (left + right) // 2
            if piles[mid] < num:
                left = mid + 1
            else:
                right = mid
        
        # Place the card
        if left == len(piles):
            piles.append(num)  # Start a new pile
        else:
            piles[left] = num  # Place on existing pile
        
        print(f"After placing {num}: piles = {piles}")
    
    print(f"Final number of piles (LIS length): {len(piles)}")
    return len(piles)


# Test cases
def test_length_of_lis():
    """Test all implementations with various inputs"""
    test_cases = [
        ([10,9,2,5,3,7,101,18], 4),
        ([0,1,0,3,2,3], 4),
        ([7,7,7,7,7,7,7], 1),
        ([1,3,6,7,9,4,10,5,6], 6),
        ([1], 1),
        ([1,2,3,4,5], 5),
        ([5,4,3,2,1], 1),
        ([2,2], 1),
        ([1,3,2,4,5], 4),
        ([10,22,9,33,21,50,41,60], 5)
    ]
    
    print("Testing Longest Increasing Subsequence Solutions:")
    print("=" * 70)
    
    for i, (nums, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: nums = {nums}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for large inputs)
        if len(nums) <= 8:
            brute = length_of_lis_brute_force(nums.copy())
            print(f"Brute Force:      {brute:>3} {'✓' if brute == expected else '✗'}")
        
        memo = length_of_lis_memoization(nums.copy())
        dp_quad = length_of_lis_dp_quadratic(nums.copy())
        binary = length_of_lis_binary_search(nums.copy())
        builtin = length_of_lis_binary_search_builtin(nums.copy())
        
        print(f"Memoization:      {memo:>3} {'✓' if memo == expected else '✗'}")
        print(f"DP O(n²):         {dp_quad:>3} {'✓' if dp_quad == expected else '✗'}")
        print(f"Binary Search:    {binary:>3} {'✓' if binary == expected else '✗'}")
        print(f"Builtin Bisect:   {builtin:>3} {'✓' if builtin == expected else '✗'}")
        
        if len(nums) <= 15:
            seg_tree = length_of_lis_segment_tree(nums.copy())
            print(f"Segment Tree:     {seg_tree:>3} {'✓' if seg_tree == expected else '✗'}")
        
        # Show actual LIS for small cases
        if expected > 1 and len(nums) <= 10:
            length, lis_seq = length_of_lis_with_sequence(nums.copy())
            print(f"One LIS: {lis_seq}")
            
            if len(nums) <= 8:
                max_len, all_lis = length_of_lis_with_all_sequences(nums.copy())
                if len(all_lis) <= 5:  # Don't show too many
                    print(f"All LIS: {all_lis}")
    
    # Demonstrate patience sorting
    print(f"\n" + "=" * 70)
    print("PATIENCE SORTING DEMONSTRATION:")
    print("-" * 40)
    length_of_lis_patience_sorting_detailed([10,9,2,5,3,7,101,18])
    
    print("\n" + "=" * 70)
    print("Algorithm Comparison:")
    print("Brute Force:      Try all 2^n subsequences")
    print("Memoization:      Cache recursive results")
    print("DP O(n²):         Classic DP with nested loops")
    print("Binary Search:    Patience sorting algorithm")
    print("Segment Tree:     Advanced range query approach")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(2^n),     Space: O(n)")
    print("Memoization:      Time: O(n²),      Space: O(n²)")
    print("DP O(n²):         Time: O(n²),      Space: O(n)")
    print("Binary Search:    Time: O(n log n), Space: O(n)  ← OPTIMAL")
    print("Segment Tree:     Time: O(n log n), Space: O(n)")


if __name__ == "__main__":
    test_length_of_lis()


"""
PATTERN RECOGNITION:
==================
This is THE classic subsequence problem:
- Foundation for many other subsequence problems
- Multiple solution approaches with different trade-offs
- Showcases evolution from brute force to optimal algorithms

KEY INSIGHTS:
============

1. **SUBSEQUENCE vs SUBARRAY**:
   - Subsequence: Can skip elements (this problem)
   - Subarray: Must be contiguous (LCIS problem)

2. **PATIENCE SORTING ALGORITHM**:
   - Think of card game: minimize number of piles
   - Each pile has decreasing cards from bottom to top
   - Binary search finds correct pile for each card

3. **DP STATE DEFINITION**:
   - dp[i] = length of LIS ending at index i
   - Not dp[i] = length of LIS in first i elements!

ALGORITHM PROGRESSION:
=====================

1. **Brute Force**: O(2^n)
   - Try all possible subsequences
   - Exponential time complexity

2. **Memoization**: O(n²)
   - Cache recursive results
   - (index, prev_index) state space

3. **DP Tabulation**: O(n²)
   - Bottom-up DP with nested loops
   - Most intuitive approach

4. **Binary Search**: O(n log n) ← OPTIMAL
   - Patience sorting algorithm
   - Maintain array of "smallest tail" elements

5. **Segment Tree**: O(n log n)
   - Advanced approach using range queries
   - Good for variants with additional constraints

PATIENCE SORTING INTUITION:
==========================
```
nums = [10,9,2,5,3,7,101,18]

After 10: piles = [10]
After 9:  piles = [9]     (replace 10)
After 2:  piles = [2]     (replace 9)
After 5:  piles = [2,5]   (new pile)
After 3:  piles = [2,3]   (replace 5)
After 7:  piles = [2,3,7] (new pile)
After 101:piles = [2,3,7,101] (new pile)
After 18: piles = [2,3,7,18]  (replace 101)

LIS length = 4 piles
```

STATE DEFINITION (DP):
=====================
dp[i] = length of LIS ending exactly at index i

RECURRENCE RELATION:
===================
dp[i] = max(dp[j] + 1) for all j < i where nums[j] < nums[i]
Base case: dp[i] = 1 (single element)

BINARY SEARCH INSIGHT:
=====================
- tails[i] = smallest ending element of all LIS of length i+1
- For each new element, find position using binary search
- Either extend array or replace existing element

VARIANTS TO PRACTICE:
====================
- Longest Decreasing Subsequence
- Number of LIS (673)
- Russian Doll Envelopes (354)
- Maximum Length of Pair Chain (646)
- Largest Divisible Subset (368)

INTERVIEW TIPS:
==============
1. **Start with DP O(n²)**: Most intuitive approach
2. **Explain patience sorting**: Key insight for O(n log n)
3. **Show binary search logic**: Why it works
4. **Handle edge cases**: Empty array, single element
5. **Follow-up questions**: 
   - Find actual LIS sequence?
   - Count number of LIS?
   - What if we need all LIS?
6. **Optimization discussion**: When to use which approach
7. **Related problems**: Connect to other subsequence problems
"""
