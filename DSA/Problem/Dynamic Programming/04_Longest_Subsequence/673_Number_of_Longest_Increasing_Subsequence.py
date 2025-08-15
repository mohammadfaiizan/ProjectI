"""
LeetCode 673: Number of Longest Increasing Subsequence
Difficulty: Medium
Category: Longest Subsequence Problems (LIS Counting)

PROBLEM DESCRIPTION:
===================
Given an integer array nums, return the number of longest increasing subsequences.

Notice that the sequence has to be strictly increasing.

Example 1:
Input: nums = [1,3,6,7,9,4,10,5,6]
Output: 20
Explanation: The 6 longest increasing subsequences are:
[1,3,6,7,9,10], [1,3,6,7,9,4,10], [1,3,6,7,9,4,5,6], [1,3,6,9,10], [1,3,6,9,4,10], [1,3,6,9,4,5,6], etc.

Example 2:
Input: nums = [2,2,2,2,2]
Output: 5

Example 3:
Input: nums = [1,2,3]
Output: 1

Constraints:
- 1 <= nums.length <= 2000
- -10^6 <= nums[i] <= 10^6
"""

def find_number_of_lis_brute_force(nums):
    """
    BRUTE FORCE APPROACH:
    ====================
    Generate all increasing subsequences and count those with maximum length.
    
    Time Complexity: O(2^n) - exponential subsequences
    Space Complexity: O(n) - recursion stack
    """
    if not nums:
        return 0
    
    max_length = 0
    all_sequences = []
    
    def generate_sequences(index, current_seq):
        nonlocal max_length
        
        if index >= len(nums):
            if current_seq:
                length = len(current_seq)
                if length > max_length:
                    max_length = length
                    all_sequences.clear()
                    all_sequences.append(current_seq[:])
                elif length == max_length:
                    all_sequences.append(current_seq[:])
            return
        
        # Option 1: Include current element if it maintains increasing order
        if not current_seq or nums[index] > current_seq[-1]:
            current_seq.append(nums[index])
            generate_sequences(index + 1, current_seq)
            current_seq.pop()
        
        # Option 2: Skip current element
        generate_sequences(index + 1, current_seq)
    
    generate_sequences(0, [])
    
    # Count sequences with maximum length
    return len([seq for seq in all_sequences if len(seq) == max_length])


def find_number_of_lis_dp(nums):
    """
    DYNAMIC PROGRAMMING APPROACH:
    ============================
    Use two DP arrays: lengths and counts.
    
    Time Complexity: O(n^2) - nested loops
    Space Complexity: O(n) - DP arrays
    """
    if not nums:
        return 0
    
    n = len(nums)
    lengths = [1] * n  # lengths[i] = length of LIS ending at index i
    counts = [1] * n   # counts[i] = number of LIS ending at index i
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                if lengths[j] + 1 > lengths[i]:
                    # Found a longer LIS ending at i
                    lengths[i] = lengths[j] + 1
                    counts[i] = counts[j]
                elif lengths[j] + 1 == lengths[i]:
                    # Found another LIS of same length ending at i
                    counts[i] += counts[j]
    
    # Find maximum length
    max_length = max(lengths)
    
    # Count all LIS with maximum length
    result = 0
    for i in range(n):
        if lengths[i] == max_length:
            result += counts[i]
    
    return result


def find_number_of_lis_segment_tree(nums):
    """
    SEGMENT TREE APPROACH:
    =====================
    Use segment tree to efficiently find maximum length and count.
    
    Time Complexity: O(n log n) - coordinate compression + segment tree
    Space Complexity: O(n) - segment tree
    """
    if not nums:
        return 0
    
    # Coordinate compression
    sorted_vals = sorted(set(nums))
    coord_map = {v: i for i, v in enumerate(sorted_vals)}
    
    class Node:
        def __init__(self):
            self.max_length = 0
            self.count = 0
    
    class SegmentTree:
        def __init__(self, size):
            self.size = size
            self.tree = [Node() for _ in range(4 * size)]
        
        def merge(self, left, right):
            result = Node()
            if left.max_length > right.max_length:
                result.max_length = left.max_length
                result.count = left.count
            elif left.max_length < right.max_length:
                result.max_length = right.max_length
                result.count = right.count
            else:
                result.max_length = left.max_length
                result.count = left.count + right.count
            return result
        
        def update(self, node, start, end, idx, length, count):
            if start == end:
                if length > self.tree[node].max_length:
                    self.tree[node].max_length = length
                    self.tree[node].count = count
                elif length == self.tree[node].max_length:
                    self.tree[node].count += count
            else:
                mid = (start + end) // 2
                if idx <= mid:
                    self.update(2 * node, start, mid, idx, length, count)
                else:
                    self.update(2 * node + 1, mid + 1, end, idx, length, count)
                
                left_child = self.tree[2 * node]
                right_child = self.tree[2 * node + 1]
                self.tree[node] = self.merge(left_child, right_child)
        
        def query(self, node, start, end, l, r):
            if r < start or end < l:
                return Node()
            if l <= start and end <= r:
                return self.tree[node]
            
            mid = (start + end) // 2
            left_result = self.query(2 * node, start, mid, l, r)
            right_result = self.query(2 * node + 1, mid + 1, end, l, r)
            return self.merge(left_result, right_result)
    
    seg_tree = SegmentTree(len(sorted_vals))
    
    for num in nums:
        coord = coord_map[num]
        
        # Query for maximum LIS length with values < num
        if coord > 0:
            query_result = seg_tree.query(1, 0, len(sorted_vals) - 1, 0, coord - 1)
            max_prev_length = query_result.max_length
            prev_count = query_result.count if max_prev_length > 0 else 1
        else:
            max_prev_length = 0
            prev_count = 1
        
        new_length = max_prev_length + 1
        new_count = prev_count if max_prev_length > 0 else 1
        
        # Update segment tree
        seg_tree.update(1, 0, len(sorted_vals) - 1, coord, new_length, new_count)
    
    # Get final result
    final_result = seg_tree.query(1, 0, len(sorted_vals) - 1, 0, len(sorted_vals) - 1)
    return final_result.count


def find_number_of_lis_optimized_dp(nums):
    """
    OPTIMIZED DP WITH EARLY TERMINATION:
    ===================================
    Add optimizations to standard DP approach.
    
    Time Complexity: O(n^2) - same as basic DP, but with optimizations
    Space Complexity: O(n) - DP arrays
    """
    if not nums:
        return 0
    
    n = len(nums)
    lengths = [1] * n
    counts = [1] * n
    max_length_so_far = 1
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                if lengths[j] + 1 > lengths[i]:
                    lengths[i] = lengths[j] + 1
                    counts[i] = counts[j]
                    max_length_so_far = max(max_length_so_far, lengths[i])
                elif lengths[j] + 1 == lengths[i]:
                    counts[i] += counts[j]
    
    # Count all positions with maximum length
    result = 0
    for i in range(n):
        if lengths[i] == max_length_so_far:
            result += counts[i]
    
    return result


def find_number_of_lis_with_sequences(nums):
    """
    FIND ACTUAL LIS SEQUENCES:
    =========================
    Return count and some example LIS sequences.
    
    Time Complexity: O(n^2 + k) where k is number of LIS
    Space Complexity: O(n * k) - store sequences
    """
    if not nums:
        return 0, []
    
    n = len(nums)
    lengths = [1] * n
    counts = [1] * n
    sequences = [[[nums[i]]] for i in range(n)]  # sequences[i] = all LIS ending at i
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                if lengths[j] + 1 > lengths[i]:
                    lengths[i] = lengths[j] + 1
                    counts[i] = counts[j]
                    sequences[i] = [seq + [nums[i]] for seq in sequences[j]]
                elif lengths[j] + 1 == lengths[i]:
                    counts[i] += counts[j]
                    sequences[i].extend([seq + [nums[i]] for seq in sequences[j]])
    
    # Find maximum length
    max_length = max(lengths)
    
    # Collect all LIS
    all_lis = []
    for i in range(n):
        if lengths[i] == max_length:
            all_lis.extend(sequences[i])
    
    return len(all_lis), all_lis[:10]  # Return up to 10 examples


def find_number_of_lis_memoization(nums):
    """
    MEMOIZATION APPROACH:
    ====================
    Use recursive approach with memoization.
    
    Time Complexity: O(n^2) - memoized states
    Space Complexity: O(n^2) - memoization table
    """
    if not nums:
        return 0
    
    memo_length = {}
    memo_count = {}
    
    def lis_length(index, prev_index):
        if index >= len(nums):
            return 0
        
        if (index, prev_index) in memo_length:
            return memo_length[(index, prev_index)]
        
        # Option 1: Include current element if valid
        include = 0
        if prev_index == -1 or nums[index] > nums[prev_index]:
            include = 1 + lis_length(index + 1, index)
        
        # Option 2: Skip current element
        skip = lis_length(index + 1, prev_index)
        
        result = max(include, skip)
        memo_length[(index, prev_index)] = result
        return result
    
    def lis_count(index, prev_index, target_length):
        if target_length == 0:
            return 1
        if index >= len(nums) or target_length < 0:
            return 0
        
        if (index, prev_index, target_length) in memo_count:
            return memo_count[(index, prev_index, target_length)]
        
        result = 0
        
        # Option 1: Include current element if valid
        if prev_index == -1 or nums[index] > nums[prev_index]:
            if lis_length(index + 1, index) == target_length - 1:
                result += lis_count(index + 1, index, target_length - 1)
        
        # Option 2: Skip current element
        if lis_length(index + 1, prev_index) == target_length:
            result += lis_count(index + 1, prev_index, target_length)
        
        memo_count[(index, prev_index, target_length)] = result
        return result
    
    max_length = lis_length(0, -1)
    return lis_count(0, -1, max_length)


# Test cases
def test_find_number_of_lis():
    """Test all implementations with various inputs"""
    test_cases = [
        ([1,3,6,7,9,4,10,5,6], 20),
        ([2,2,2,2,2], 5),
        ([1,2,3], 1),
        ([1,3,2,4], 2),
        ([1], 1),
        ([1,2,1,3], 2),
        ([10,9,2,5,3,7,101,18], 4),
        ([1,2,3,4,5], 1),
        ([5,4,3,2,1], 5),
        ([1,3,2,4,5], 2)
    ]
    
    print("Testing Number of Longest Increasing Subsequence Solutions:")
    print("=" * 70)
    
    for i, (nums, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: nums = {nums}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for large inputs)
        if len(nums) <= 8:
            brute = find_number_of_lis_brute_force(nums.copy())
            print(f"Brute Force:      {brute:>4} {'✓' if brute == expected else '✗'}")
        
        dp = find_number_of_lis_dp(nums.copy())
        optimized = find_number_of_lis_optimized_dp(nums.copy())
        
        print(f"DP:               {dp:>4} {'✓' if dp == expected else '✗'}")
        print(f"Optimized DP:     {optimized:>4} {'✓' if optimized == expected else '✗'}")
        
        if len(nums) <= 10:
            memo = find_number_of_lis_memoization(nums.copy())
            print(f"Memoization:      {memo:>4} {'✓' if memo == expected else '✗'}")
        
        if len(nums) <= 15:
            seg_tree = find_number_of_lis_segment_tree(nums.copy())
            print(f"Segment Tree:     {seg_tree:>4} {'✓' if seg_tree == expected else '✗'}")
        
        # Show actual sequences for small cases
        if expected <= 10 and len(nums) <= 8:
            count, sequences = find_number_of_lis_with_sequences(nums.copy())
            print(f"Example LIS: {sequences[:3]}...")  # Show first 3
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. TWO DP ARRAYS: lengths[i] and counts[i]")
    print("2. UPDATE LOGIC: If new length > current, reset count")
    print("                 If new length = current, add to count")
    print("3. FINAL COUNT: Sum counts[i] for all i with max length")
    print("4. SEGMENT TREE: O(n log n) approach for large inputs")
    
    print("\n" + "=" * 70)
    print("Algorithm Comparison:")
    print("Brute Force:      Generate all subsequences")
    print("DP:               Two arrays for length and count")
    print("Optimized DP:     Early termination optimizations")
    print("Memoization:      Recursive with caching")
    print("Segment Tree:     Advanced O(n log n) approach")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(2^n),     Space: O(n)")
    print("DP:               Time: O(n²),      Space: O(n)")
    print("Optimized DP:     Time: O(n²),      Space: O(n)")
    print("Memoization:      Time: O(n²),      Space: O(n²)")
    print("Segment Tree:     Time: O(n log n), Space: O(n)")


if __name__ == "__main__":
    test_find_number_of_lis()


"""
PATTERN RECOGNITION:
==================
This extends classic LIS to counting problem:
- Not just find length of LIS, but COUNT how many LIS exist
- Requires tracking both length and count for each position
- More complex than basic LIS but follows similar DP pattern

KEY INSIGHT - DUAL DP ARRAYS:
=============================
Use two DP arrays simultaneously:
1. **lengths[i]**: Length of LIS ending at index i
2. **counts[i]**: Number of LIS ending at index i

**Update Logic**:
- If extending gives LONGER LIS: Update length, RESET count
- If extending gives SAME length LIS: Keep length, ADD to count

ALGORITHM APPROACHES:
====================

1. **Brute Force**: O(2^n)
   - Generate all increasing subsequences
   - Count those with maximum length

2. **DP O(n²)**: O(n²)
   - Two DP arrays: lengths and counts
   - Nested loops to compute both arrays

3. **Segment Tree**: O(n log n)
   - Advanced approach using range queries
   - Each node stores (max_length, count)

4. **Memoization**: O(n²)
   - Recursive approach with caching
   - Separate functions for length and count

DP STATE TRANSITIONS:
====================
```python
for i in range(1, n):
    for j in range(i):
        if nums[j] < nums[i]:
            if lengths[j] + 1 > lengths[i]:
                lengths[i] = lengths[j] + 1
                counts[i] = counts[j]        # Reset count
            elif lengths[j] + 1 == lengths[i]:
                counts[i] += counts[j]       # Add to count
```

SEGMENT TREE APPROACH:
=====================
Each node stores:
- `max_length`: Maximum LIS length in range
- `count`: Number of LIS with max_length

**Merge Operation**:
```python
if left.max_length > right.max_length:
    result = (left.max_length, left.count)
elif left.max_length < right.max_length:
    result = (right.max_length, right.count)
else:
    result = (left.max_length, left.count + right.count)
```

EXAMPLE WALKTHROUGH:
===================
```
nums = [1,3,2,4]

After processing:
lengths = [1,2,2,3]
counts =  [1,1,1,2]

Max length = 3
Count at positions with length 3: counts[3] = 2
Answer: 2 (sequences [1,3,4] and [1,2,4])
```

EDGE CASES:
==========
1. **All equal elements**: Each forms LIS of length 1
2. **Strictly increasing**: Only one LIS of length n
3. **Strictly decreasing**: n different LIS of length 1
4. **Single element**: One LIS of length 1

MATHEMATICAL INSIGHT:
====================
The number of LIS ending at position i is the sum of:
- Number of LIS ending at all positions j < i where:
  - nums[j] < nums[i] 
  - lengths[j] + 1 = lengths[i]

VARIANTS TO PRACTICE:
====================
- Longest Increasing Subsequence (300) - find length only
- Russian Doll Envelopes (354) - 2D LIS
- Maximum Length of Pair Chain (646) - interval version
- Largest Divisible Subset (368) - divisibility constraint

INTERVIEW TIPS:
==============
1. **Start with LIS**: Understand basic LIS first
2. **Dual arrays**: Explain need for both length and count
3. **Update logic**: Critical to get the count updates right
4. **Edge cases**: All equal, strictly increasing/decreasing
5. **Optimization**: Segment tree for O(n log n) solution
6. **Follow-up**: What if we need actual sequences?
7. **Verification**: Show examples with small inputs
8. **Complexity**: Compare different approaches
9. **Extensions**: How to handle equal elements differently?
10. **Applications**: When would we need count vs just length?
"""
