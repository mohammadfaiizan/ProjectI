"""
LeetCode 805: Split Array With Same Average
Difficulty: Hard
Category: Knapsack Problems (Advanced Subset Partition)

PROBLEM DESCRIPTION:
===================
You are given an integer array nums.

You should move each element of nums into one of the two arrays A and B such that A and B are non-empty, 
and the average of elements in array A equals to the average of elements in array B.

Return true if it is possible to achieve that and false otherwise.

Note: For an array arr, average(arr) is the sum of all the elements over the length of the array.

Example 1:
Input: nums = [1,2,3,4,5,6,7,8]
Output: true
Explanation: We can split the array into [1,4,5,8] and [2,3,6,7].
Both arrays have the same average of 4.5.

Example 2:
Input: nums = [3,1]
Output: false

Constraints:
- 1 <= nums.length <= 30
- 0 <= nums[i] <= 10^4
"""

def split_array_same_average_bruteforce(nums):
    """
    BRUTE FORCE APPROACH - TRY ALL SUBSETS:
    ======================================
    Try all possible ways to split array into two non-empty subsets.
    
    Time Complexity: O(2^n) - try all 2^n subsets
    Space Complexity: O(n) - recursion stack depth
    """
    n = len(nums)
    total_sum = sum(nums)
    
    def check_subset(index, subset_a, sum_a, count_a):
        if index == n:
            if count_a == 0 or count_a == n:  # Both subsets must be non-empty
                return False
            
            count_b = n - count_a
            sum_b = total_sum - sum_a
            
            # Check if averages are equal: sum_a/count_a == sum_b/count_b
            # Equivalent to: sum_a * count_b == sum_b * count_a
            return sum_a * count_b == sum_b * count_a
        
        # Try putting current element in subset A
        if check_subset(index + 1, subset_a + [nums[index]], 
                       sum_a + nums[index], count_a + 1):
            return True
        
        # Try putting current element in subset B (not explicitly tracked)
        if check_subset(index + 1, subset_a, sum_a, count_a):
            return True
        
        return False
    
    return check_subset(0, [], 0, 0)


def split_array_same_average_mathematical_insight(nums):
    """
    MATHEMATICAL INSIGHT APPROACH:
    =============================
    Use mathematical properties to reduce the problem.
    
    Time Complexity: O(n * sum * 2^(n/2)) - optimized subset search
    Space Complexity: O(2^(n/2)) - subset storage
    """
    n = len(nums)
    total_sum = sum(nums)
    
    # Mathematical insight: If averages are equal, then:
    # sum(A) / len(A) = sum(B) / len(B) = total_sum / n
    # This means: sum(A) = len(A) * total_sum / n
    
    # For valid split, len(A) * total_sum must be divisible by n
    possible_sizes = []
    for size in range(1, n):  # Both subsets must be non-empty
        if (size * total_sum) % n == 0:
            possible_sizes.append(size)
    
    if not possible_sizes:
        return False
    
    # For each possible size, check if subset with required sum exists
    for size in possible_sizes:
        required_sum = (size * total_sum) // n
        
        if subset_sum_exists(nums, size, required_sum):
            return True
    
    return False


def subset_sum_exists(nums, target_size, target_sum):
    """
    Check if subset of given size with given sum exists.
    
    Time Complexity: O(target_size * target_sum * n) - DP
    Space Complexity: O(target_size * target_sum) - DP table
    """
    n = len(nums)
    
    # dp[i][j][k] = True if using first i elements, 
    # we can form subset of size j with sum k
    # Optimize to 2D: dp[j][k] for current iteration
    
    # Use set for memory efficiency
    prev_states = {(0, 0)}  # (size, sum) pairs
    
    for num in nums:
        new_states = set(prev_states)
        
        for size, current_sum in prev_states:
            if size < target_size and current_sum + num <= target_sum:
                new_states.add((size + 1, current_sum + num))
        
        prev_states = new_states
        
        # Early termination
        if (target_size, target_sum) in prev_states:
            return True
    
    return (target_size, target_sum) in prev_states


def split_array_same_average_dp(nums):
    """
    DYNAMIC PROGRAMMING APPROACH:
    ============================
    Use DP to check all possible subset sums and sizes.
    
    Time Complexity: O(n^2 * sum) - DP states
    Space Complexity: O(n * sum) - DP table
    """
    n = len(nums)
    total_sum = sum(nums)
    
    # Check mathematical feasibility first
    possible = False
    for size in range(1, n):
        if (size * total_sum) % n == 0:
            possible = True
            break
    
    if not possible:
        return False
    
    # dp[i][j] = set of possible sums using exactly i elements
    dp = [set() for _ in range(n + 1)]
    dp[0].add(0)  # 0 elements, sum 0
    
    for num in nums:
        # Process in reverse to avoid using updated values
        for i in range(min(n - 1, len([s for s in dp if s])), 0, -1):
            for prev_sum in list(dp[i]):
                dp[i + 1].add(prev_sum + num)
    
    # Check if any valid split exists
    for size in range(1, n):
        if (size * total_sum) % n == 0:
            required_sum = (size * total_sum) // n
            if required_sum in dp[size]:
                return True
    
    return False


def split_array_same_average_optimized(nums):
    """
    OPTIMIZED APPROACH:
    ==================
    Multiple optimizations for better performance.
    
    Time Complexity: O(n * sum * n) - optimized DP
    Space Complexity: O(sum * n) - DP table
    """
    n = len(nums)
    total_sum = sum(nums)
    
    # Early termination: check if split is mathematically possible
    valid_sizes = []
    for size in range(1, n // 2 + 1):  # Only check up to half
        if (size * total_sum) % n == 0:
            valid_sizes.append(size)
    
    if not valid_sizes:
        return False
    
    # Sort numbers for potential optimizations
    nums.sort()
    
    # Use bitmask DP for small n
    if n <= 20:
        return split_array_bitmask_dp(nums, total_sum, valid_sizes)
    
    # Use subset sum DP for larger n
    return split_array_subset_dp(nums, total_sum, valid_sizes)


def split_array_bitmask_dp(nums, total_sum, valid_sizes):
    """
    Bitmask DP for small arrays.
    
    Time Complexity: O(2^n) - bitmask enumeration
    Space Complexity: O(2^n) - bitmask storage
    """
    n = len(nums)
    
    # Precompute all subset sums and sizes
    subset_info = {}  # mask -> (size, sum)
    
    for mask in range(1, 1 << n):
        size = bin(mask).count('1')
        subset_sum = sum(nums[i] for i in range(n) if mask & (1 << i))
        subset_info[mask] = (size, subset_sum)
    
    # Check each valid size
    for target_size in valid_sizes:
        target_sum = (target_size * total_sum) // n
        
        for mask, (size, subset_sum) in subset_info.items():
            if size == target_size and subset_sum == target_sum:
                return True
    
    return False


def split_array_subset_dp(nums, total_sum, valid_sizes):
    """
    Subset sum DP for larger arrays.
    
    Time Complexity: O(n * sum * max_size) - DP computation
    Space Complexity: O(sum * max_size) - DP table
    """
    n = len(nums)
    max_size = max(valid_sizes)
    
    # dp[i][j] = True if we can achieve sum j with exactly i elements
    dp = [[False] * (total_sum + 1) for _ in range(max_size + 1)]
    dp[0][0] = True
    
    for num in nums:
        # Process in reverse to avoid using updated values
        for size in range(min(max_size, n - 1), 0, -1):
            for s in range(total_sum, num - 1, -1):
                if dp[size - 1][s - num]:
                    dp[size][s] = True
    
    # Check each valid configuration
    for size in valid_sizes:
        target_sum = (size * total_sum) // n
        if target_sum <= total_sum and dp[size][target_sum]:
            return True
    
    return False


def split_array_same_average_with_partition(nums):
    """
    FIND ACTUAL PARTITION:
    =====================
    Return whether split is possible and show actual partition.
    
    Time Complexity: O(n * sum * n) - DP + reconstruction
    Space Complexity: O(n * sum * n) - store choices
    """
    n = len(nums)
    total_sum = sum(nums)
    
    # Find valid sizes
    valid_configs = []
    for size in range(1, n):
        if (size * total_sum) % n == 0:
            target_sum = (size * total_sum) // n
            valid_configs.append((size, target_sum))
    
    if not valid_configs:
        return False, [], []
    
    # Use backtracking to find actual partition
    def find_partition(index, subset_a, sum_a, target_size, target_sum):
        if len(subset_a) == target_size:
            return sum_a == target_sum
        
        if index >= n or len(subset_a) > target_size:
            return False
        
        # Try including current element
        subset_a.append(nums[index])
        if find_partition(index + 1, subset_a, sum_a + nums[index], target_size, target_sum):
            return True
        subset_a.pop()
        
        # Try not including current element
        if find_partition(index + 1, subset_a, sum_a, target_size, target_sum):
            return True
        
        return False
    
    for target_size, target_sum in valid_configs:
        subset_a = []
        if find_partition(0, subset_a, 0, target_size, target_sum):
            subset_b = [nums[i] for i in range(n) if nums[i] not in subset_a or 
                       subset_a.count(nums[i]) < nums[:i].count(nums[i]) + 1]
            # Handle duplicates properly
            used = [False] * n
            for val in subset_a:
                for i in range(n):
                    if not used[i] and nums[i] == val:
                        used[i] = True
                        break
            subset_b = [nums[i] for i in range(n) if not used[i]]
            
            return True, subset_a, subset_b
    
    return False, [], []


# Test cases
def test_split_array_same_average():
    """Test all implementations with various inputs"""
    test_cases = [
        ([1,2,3,4,5,6,7,8], True),
        ([3,1], False),
        ([1,2,3,4,5], False),
        ([2,2,2,2], True),
        ([1,6,1], False),
        ([1,1], True),
        ([60,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30], False),
        ([0,1,2,3], True),
        ([4,4,4,4,4,4], True),
        ([1,3,5,7,9,2,4,6,8], True)
    ]
    
    print("Testing Split Array Same Average Solutions:")
    print("=" * 70)
    
    for i, (nums, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: nums = {nums}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip brute force for large inputs)
        if len(nums) <= 12:
            brute = split_array_same_average_bruteforce(nums.copy())
            print(f"Brute Force:      {brute} {'✓' if brute == expected else '✗'}")
        
        math_insight = split_array_same_average_mathematical_insight(nums.copy())
        dp_approach = split_array_same_average_dp(nums.copy())
        optimized = split_array_same_average_optimized(nums.copy())
        
        print(f"Math Insight:     {math_insight} {'✓' if math_insight == expected else '✗'}")
        print(f"DP Approach:      {dp_approach} {'✓' if dp_approach == expected else '✗'}")
        print(f"Optimized:        {optimized} {'✓' if optimized == expected else '✗'}")
        
        # Show actual partition for positive small cases
        if expected and len(nums) <= 10:
            possible, subset_a, subset_b = split_array_same_average_with_partition(nums.copy())
            if possible and subset_a and subset_b:
                avg_a = sum(subset_a) / len(subset_a)
                avg_b = sum(subset_b) / len(subset_b)
                print(f"Partition A: {subset_a} (avg: {avg_a:.2f})")
                print(f"Partition B: {subset_b} (avg: {avg_b:.2f})")
    
    print("\n" + "=" * 70)
    print("Key Mathematical Insight:")
    print("If avg(A) = avg(B), then both equal total_avg = total_sum / n")
    print("So: sum(A) = len(A) * total_sum / n")
    print("For integer solutions: len(A) * total_sum must be divisible by n")
    print("This drastically reduces the search space!")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(2^n),           Space: O(n)")
    print("Math Insight:     Time: O(n*sum*2^(n/2)), Space: O(2^(n/2))")
    print("DP Approach:      Time: O(n^2*sum),       Space: O(n*sum)")
    print("Optimized:        Time: O(n*sum*n),       Space: O(sum*n)")
    print("Bitmask DP:       Time: O(2^n),           Space: O(2^n)")
    print("Subset DP:        Time: O(n*sum*max_size), Space: O(sum*max_size)")


if __name__ == "__main__":
    test_split_array_same_average()


"""
PATTERN RECOGNITION:
==================
This is an advanced subset partition problem:
- Split array into two non-empty subsets with equal averages
- Mathematical constraint: avg(A) = avg(B) = total_avg
- Reduces to: find subset with specific size and sum
- Combines number theory with subset sum DP

KEY MATHEMATICAL INSIGHT:
========================
If avg(A) = avg(B), then both must equal total_average:

avg(A) = sum(A) / len(A) = total_sum / n
Therefore: sum(A) = len(A) * total_sum / n

**Critical constraint**: len(A) * total_sum must be divisible by n

This reduces the problem from checking all 2^n subsets to checking only 
valid (size, sum) combinations where size * total_sum ≡ 0 (mod n).

MATHEMATICAL PROOF:
==================
Let A and B be the two subsets, |A| = k, |B| = n-k

If avg(A) = avg(B):
- sum(A) / k = sum(B) / (n-k)
- sum(A) / k = (total_sum - sum(A)) / (n-k)
- sum(A) * (n-k) = k * (total_sum - sum(A))
- sum(A) * (n-k) = k * total_sum - k * sum(A)
- sum(A) * (n-k) + k * sum(A) = k * total_sum
- sum(A) * n = k * total_sum
- sum(A) = k * total_sum / n

For integer solution: k * total_sum ≡ 0 (mod n)

ALGORITHM APPROACHES:
====================

1. **Brute Force**: O(2^n)
   - Try all possible subsets
   - Check if averages are equal

2. **Mathematical Filtering**: O(n × sum × 2^(n/2))
   - Filter valid sizes using divisibility
   - Use subset sum DP for each valid configuration

3. **DP Approach**: O(n² × sum)
   - Build DP table for all (size, sum) combinations
   - Check valid configurations

4. **Optimized**: O(n × sum × max_size)
   - Combine mathematical filtering with optimized DP
   - Use bitmask for small n, subset DP for large n

OPTIMIZATION TECHNIQUES:
=======================

1. **Mathematical Filtering**: Only check sizes where size × total_sum ≡ 0 (mod n)
2. **Symmetry**: Only check sizes ≤ n/2 (by symmetry)
3. **Early Termination**: Stop when valid configuration found
4. **Algorithm Selection**: Bitmask DP for small n, subset DP for large n
5. **Space Optimization**: Use rolling arrays for DP

STATE DEFINITION:
================
dp[i][j] = True if we can achieve sum j using exactly i elements

SUBSET SUM WITH SIZE CONSTRAINT:
===============================
This problem extends classic subset sum to include size constraint:
- Classic: "Can we achieve sum S?"
- This problem: "Can we achieve sum S using exactly K elements?"

EDGE CASES:
==========
1. **n = 2**: Always false (can't split into two non-empty with equal averages)
2. **All elements equal**: Always true (any split works)
3. **Single valid size**: Check only that configuration
4. **No valid sizes**: Immediately return false

VARIANTS TO PRACTICE:
====================
- Partition Equal Subset Sum (416) - simpler version
- Split Array With Equal Sum (548) - similar but different constraints
- Subset sum problems - foundation techniques
- Average-based partitioning problems

INTERVIEW TIPS:
==============
1. **Start with mathematical analysis** - derive the constraint
2. Show brute force approach first
3. **Critical insight**: len(A) × total_sum ≡ 0 (mod n)
4. Explain how this reduces search space dramatically
5. Show subset sum DP with size constraint
6. Discuss algorithm selection based on n
7. Handle edge cases (n=2, all equal elements)
8. **Emphasize**: This transforms from exponential to polynomial!
9. Mention space optimizations
10. Compare with related partition problems
"""
