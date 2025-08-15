"""
LeetCode 674: Longest Continuous Increasing Subsequence
Difficulty: Easy
Category: Longest Subsequence Problems

PROBLEM DESCRIPTION:
===================
Given an unsorted array of integers nums, return the length of the longest continuous increasing subsequence (i.e. subarray). The subsequence must be strictly increasing.

A continuous increasing subsequence is defined as a subarray where nums[i] < nums[i+1] for all i in the range [left, right).

Example 1:
Input: nums = [1,3,2,3,4,7]
Output: 4
Explanation: The longest continuous increasing subsequence is [1,3] then [2,3,4,7], so its length is 4.

Example 2:
Input: nums = [2,2,2,2,2]
Output: 1
Explanation: The longest continuous increasing subsequence is [2], so its length is 1.

Constraints:
- 1 <= nums.length <= 10^4
- -10^9 <= nums[i] <= 10^9
"""

def find_length_of_lcis_brute_force(nums):
    """
    BRUTE FORCE APPROACH:
    ====================
    Check all possible continuous subarrays.
    
    Time Complexity: O(n^2) - nested loops to check all subarrays
    Space Complexity: O(1) - constant extra space
    """
    if not nums:
        return 0
    
    n = len(nums)
    max_length = 1
    
    # Try every starting position
    for i in range(n):
        current_length = 1
        
        # Extend as far as possible from position i
        for j in range(i + 1, n):
            if nums[j] > nums[j - 1]:
                current_length += 1
            else:
                break
        
        max_length = max(max_length, current_length)
    
    return max_length


def find_length_of_lcis_one_pass(nums):
    """
    ONE PASS APPROACH (OPTIMAL):
    ===========================
    Track current increasing sequence length in single pass.
    
    Time Complexity: O(n) - single pass through array
    Space Complexity: O(1) - constant extra space
    """
    if not nums:
        return 0
    
    max_length = 1
    current_length = 1
    
    for i in range(1, len(nums)):
        if nums[i] > nums[i - 1]:
            current_length += 1
        else:
            max_length = max(max_length, current_length)
            current_length = 1
    
    # Don't forget to check the last sequence
    max_length = max(max_length, current_length)
    
    return max_length


def find_length_of_lcis_sliding_window(nums):
    """
    SLIDING WINDOW APPROACH:
    =======================
    Use sliding window technique to track increasing sequences.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if not nums:
        return 0
    
    left = 0
    max_length = 1
    
    for right in range(1, len(nums)):
        # If sequence breaks, move left pointer
        if nums[right] <= nums[right - 1]:
            left = right
        
        # Update maximum length
        max_length = max(max_length, right - left + 1)
    
    return max_length


def find_length_of_lcis_dp(nums):
    """
    DYNAMIC PROGRAMMING APPROACH:
    ============================
    Use DP array to track lengths ending at each position.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(n) - DP array
    """
    if not nums:
        return 0
    
    n = len(nums)
    dp = [1] * n  # dp[i] = length of LCIS ending at index i
    
    for i in range(1, n):
        if nums[i] > nums[i - 1]:
            dp[i] = dp[i - 1] + 1
    
    return max(dp)


def find_length_of_lcis_with_indices(nums):
    """
    FIND LCIS WITH STARTING AND ENDING INDICES:
    ==========================================
    Return length and the actual indices of the longest sequence.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if not nums:
        return 0, -1, -1
    
    max_length = 1
    current_length = 1
    best_start = 0
    best_end = 0
    current_start = 0
    
    for i in range(1, len(nums)):
        if nums[i] > nums[i - 1]:
            current_length += 1
        else:
            # Check if current sequence is the best so far
            if current_length > max_length:
                max_length = current_length
                best_start = current_start
                best_end = i - 1
            
            # Start new sequence
            current_length = 1
            current_start = i
    
    # Check the last sequence
    if current_length > max_length:
        max_length = current_length
        best_start = current_start
        best_end = len(nums) - 1
    
    return max_length, best_start, best_end


def find_length_of_lcis_with_sequences(nums):
    """
    FIND ALL LCIS OF MAXIMUM LENGTH:
    ===============================
    Return all continuous increasing subsequences of maximum length.
    
    Time Complexity: O(n) - single pass + result construction
    Space Complexity: O(n) - store sequences
    """
    if not nums:
        return 0, []
    
    max_length = 1
    sequences = []
    current_start = 0
    
    for i in range(1, len(nums)):
        if nums[i] <= nums[i - 1]:
            # End of current sequence
            current_length = i - current_start
            
            if current_length > max_length:
                max_length = current_length
                sequences = [nums[current_start:i]]
            elif current_length == max_length:
                sequences.append(nums[current_start:i])
            
            current_start = i
    
    # Handle the last sequence
    current_length = len(nums) - current_start
    if current_length > max_length:
        max_length = current_length
        sequences = [nums[current_start:]]
    elif current_length == max_length:
        sequences.append(nums[current_start:])
    
    return max_length, sequences


def find_length_of_lcis_recursive(nums):
    """
    RECURSIVE APPROACH:
    ==================
    Solve using recursion (for educational purposes).
    
    Time Complexity: O(n^2) - recursive calls
    Space Complexity: O(n) - recursion stack
    """
    if not nums:
        return 0
    
    def lcis_from_index(index, prev_value):
        if index >= len(nums):
            return 0
        
        # Option 1: Include current element if it's increasing
        include = 0
        if nums[index] > prev_value:
            include = 1 + lcis_from_index(index + 1, nums[index])
        
        # Option 2: Start new sequence from current element
        start_new = lcis_from_index(index + 1, float('-inf'))
        
        return max(include, start_new)
    
    return lcis_from_index(0, float('-inf'))


def find_length_of_lcis_memoized(nums):
    """
    MEMOIZED RECURSIVE APPROACH:
    ===========================
    Add memoization to recursive solution.
    
    Time Complexity: O(n^2) - memoized states
    Space Complexity: O(n^2) - memoization table
    """
    if not nums:
        return 0
    
    memo = {}
    
    def lcis_from_index(index, prev_index):
        if index >= len(nums):
            return 0
        
        if (index, prev_index) in memo:
            return memo[(index, prev_index)]
        
        # Option 1: Include current element if it's increasing
        include = 0
        if prev_index == -1 or nums[index] > nums[prev_index]:
            include = 1 + lcis_from_index(index + 1, index)
        
        # Option 2: Skip current element
        skip = lcis_from_index(index + 1, prev_index)
        
        result = max(include, skip)
        memo[(index, prev_index)] = result
        return result
    
    return lcis_from_index(0, -1)


# Test cases
def test_find_length_of_lcis():
    """Test all implementations with various inputs"""
    test_cases = [
        ([1,3,2,3,4,7], 4),
        ([2,2,2,2,2], 1),
        ([1,2,3,4,5], 5),
        ([5,4,3,2,1], 1),
        ([1], 1),
        ([], 0),
        ([1,3,2,4,5,6], 4),
        ([1,2,1,3,4,5,6], 5),
        ([10,9,2,5,3,7,101,18], 2),
        ([0,1,2,2,3,4], 2),
        ([1,2,3,2,3,4,5], 4)
    ]
    
    print("Testing Longest Continuous Increasing Subsequence Solutions:")
    print("=" * 70)
    
    for i, (nums, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: nums = {nums}")
        print(f"Expected: {expected}")
        
        if nums:  # Skip empty arrays for some methods
            brute = find_length_of_lcis_brute_force(nums.copy())
            one_pass = find_length_of_lcis_one_pass(nums.copy())
            sliding = find_length_of_lcis_sliding_window(nums.copy())
            dp = find_length_of_lcis_dp(nums.copy())
            
            print(f"Brute Force:      {brute:>3} {'✓' if brute == expected else '✗'}")
            print(f"One Pass:         {one_pass:>3} {'✓' if one_pass == expected else '✗'}")
            print(f"Sliding Window:   {sliding:>3} {'✓' if sliding == expected else '✗'}")
            print(f"DP:               {dp:>3} {'✓' if dp == expected else '✗'}")
            
            if len(nums) <= 10:
                recursive = find_length_of_lcis_recursive(nums.copy())
                memoized = find_length_of_lcis_memoized(nums.copy())
                print(f"Recursive:        {recursive:>3} {'✓' if recursive == expected else '✗'}")
                print(f"Memoized:         {memoized:>3} {'✓' if memoized == expected else '✗'}")
            
            # Show detailed results for interesting cases
            if expected > 1 and len(nums) <= 10:
                length, start, end = find_length_of_lcis_with_indices(nums.copy())
                print(f"LCIS: {nums[start:end+1]} (indices {start}-{end})")
                
                max_len, sequences = find_length_of_lcis_with_sequences(nums.copy())
                if len(sequences) <= 3:  # Don't show too many
                    print(f"All max sequences: {sequences}")
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. CONTINUOUS: Must be contiguous subarray (unlike LIS)")
    print("2. OPTIMAL: O(n) time, O(1) space with one-pass approach")
    print("3. SLIDING WINDOW: Natural fit for contiguous constraint")
    print("4. SIMPLE LOGIC: Reset counter when sequence breaks")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(n²),  Space: O(1)")
    print("One Pass:         Time: O(n),   Space: O(1)  ← OPTIMAL")
    print("Sliding Window:   Time: O(n),   Space: O(1)")
    print("DP:               Time: O(n),   Space: O(n)")
    print("Recursive:        Time: O(n²),  Space: O(n)")
    print("Memoized:         Time: O(n²),  Space: O(n²)")


if __name__ == "__main__":
    test_find_length_of_lcis()


"""
PATTERN RECOGNITION:
==================
This is the simplest form of subsequence problems:
- CONTINUOUS constraint makes it much easier than general LIS
- Single pass solution with O(1) space is possible
- Foundation for understanding more complex subsequence problems

KEY INSIGHT - CONTINUOUS CONSTRAINT:
===================================
Unlike Longest Increasing Subsequence (LIS), this requires:
- CONTIGUOUS elements (subarray, not subsequence)
- Can be solved optimally in O(n) time, O(1) space
- Much simpler than general LIS which needs O(n log n) or O(n²)

ALGORITHM APPROACHES:
====================

1. **One Pass (Optimal)**: O(n) time, O(1) space
   - Track current sequence length
   - Reset when sequence breaks
   - Update maximum continuously

2. **Sliding Window**: O(n) time, O(1) space
   - Maintain window of increasing elements
   - Shrink window when constraint violated

3. **Dynamic Programming**: O(n) time, O(n) space
   - dp[i] = length of LCIS ending at index i
   - Good for understanding DP patterns

4. **Brute Force**: O(n²) time, O(1) space
   - Check all possible starting positions
   - Extend each as far as possible

COMPARISON WITH LIS:
===================
**LCIS (this problem)**:
- Continuous/contiguous requirement
- O(n) time, O(1) space solution
- Simpler algorithm

**LIS (general subsequence)**:
- Can skip elements
- O(n log n) optimal time complexity
- More complex algorithm (binary search + patience sorting)

STATE TRANSITIONS:
=================
```python
if nums[i] > nums[i-1]:
    current_length += 1
else:
    max_length = max(max_length, current_length)
    current_length = 1
```

EDGE CASES:
==========
1. **Empty array**: Return 0
2. **Single element**: Return 1
3. **All decreasing**: Return 1
4. **All increasing**: Return n
5. **All equal**: Return 1

VARIANTS TO PRACTICE:
====================
- Longest Increasing Subsequence (300) - general case
- Longest Decreasing Subsequence - mirror problem  
- Longest Bitonic Subsequence - increasing then decreasing
- Maximum Length of Pair Chain (646) - interval version

INTERVIEW TIPS:
==============
1. **Clarify**: Continuous vs general subsequence
2. **Start simple**: Show one-pass O(n) solution
3. **Explain logic**: Reset counter when sequence breaks
4. **Edge cases**: Empty array, single element
5. **Follow-up**: What if we want actual sequence?
6. **Compare**: How this differs from general LIS
7. **Optimize**: From O(n²) brute force to O(n) optimal
8. **Variants**: Discuss related problems (LIS, etc.)
"""
