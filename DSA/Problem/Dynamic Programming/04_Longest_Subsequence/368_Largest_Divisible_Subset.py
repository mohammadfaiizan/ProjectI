"""
LeetCode 368: Largest Divisible Subset
Difficulty: Medium
Category: Longest Subsequence Problems (LIS with Divisibility)

PROBLEM DESCRIPTION:
===================
Given a set of distinct positive integers nums, return the largest subset answer such that every pair 
(answer[i], answer[j]) of elements in this subset satisfies:

answer[i] % answer[j] == 0, or
answer[j] % answer[i] == 0

If there are multiple solutions, return any of them.

Example 1:
Input: nums = [1,2,3]
Output: [1,2]
Explanation: [1,3] is also accepted.

Example 2:
Input: nums = [1,2,4,8]
Output: [1,2,4,8]

Constraints:
- 1 <= nums.length <= 1000
- 1 <= nums[i] <= 2 * 10^9
- All the integers in nums are unique.
"""

def largest_divisible_subset_brute_force(nums):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible subsets and check divisibility property.
    
    Time Complexity: O(2^n * n^2) - 2^n subsets, n^2 to check each
    Space Complexity: O(n) - recursion stack
    """
    def is_divisible_subset(subset):
        """Check if subset satisfies divisibility property"""
        n = len(subset)
        for i in range(n):
            for j in range(i + 1, n):
                if subset[i] % subset[j] != 0 and subset[j] % subset[i] != 0:
                    return False
        return True
    
    def generate_subsets(index, current_subset):
        if index >= len(nums):
            if is_divisible_subset(current_subset):
                return current_subset[:]
            return []
        
        # Skip current element
        skip = generate_subsets(index + 1, current_subset)
        
        # Include current element
        current_subset.append(nums[index])
        include = generate_subsets(index + 1, current_subset)
        current_subset.pop()
        
        # Return longer subset
        return include if len(include) > len(skip) else skip
    
    return generate_subsets(0, [])


def largest_divisible_subset_dp(nums):
    """
    DYNAMIC PROGRAMMING APPROACH:
    ============================
    Sort array and use LIS-like DP with divisibility check.
    
    Time Complexity: O(n^2) - nested loops after sorting
    Space Complexity: O(n) - DP array
    """
    if not nums:
        return []
    
    # Sort the array
    nums.sort()
    n = len(nums)
    
    # dp[i] = length of largest divisible subset ending at index i
    dp = [1] * n
    parent = [-1] * n  # For reconstruction
    
    max_length = 1
    max_index = 0
    
    for i in range(1, n):
        for j in range(i):
            # Check if nums[i] is divisible by nums[j]
            if nums[i] % nums[j] == 0 and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
        
        # Update maximum
        if dp[i] > max_length:
            max_length = dp[i]
            max_index = i
    
    # Reconstruct the subset
    result = []
    current = max_index
    
    while current != -1:
        result.append(nums[current])
        current = parent[current]
    
    result.reverse()
    return result


def largest_divisible_subset_memoization(nums):
    """
    MEMOIZATION APPROACH:
    ====================
    Use recursive approach with memoization.
    
    Time Complexity: O(n^2) - memoized states
    Space Complexity: O(n^2) - memoization table
    """
    if not nums:
        return []
    
    nums.sort()
    memo = {}
    
    def dp(index, last_index):
        if index >= len(nums):
            return []
        
        if (index, last_index) in memo:
            return memo[(index, last_index)]
        
        # Skip current element
        skip = dp(index + 1, last_index)
        
        # Include current element if divisible
        include = []
        if last_index == -1 or nums[index] % nums[last_index] == 0:
            include = [nums[index]] + dp(index + 1, index)
        
        # Choose longer subset
        result = include if len(include) > len(skip) else skip
        memo[(index, last_index)] = result
        return result
    
    return dp(0, -1)


def largest_divisible_subset_optimized(nums):
    """
    OPTIMIZED DP APPROACH:
    =====================
    Optimizations for better practical performance.
    
    Time Complexity: O(n^2) - same as basic DP
    Space Complexity: O(n) - DP array
    """
    if not nums:
        return []
    
    nums.sort()
    n = len(nums)
    
    dp = [1] * n
    parent = [-1] * n
    
    max_length = 1
    max_index = 0
    
    for i in range(1, n):
        for j in range(i - 1, -1, -1):  # Reverse order for early termination
            if nums[i] % nums[j] == 0:
                if dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j
                
                # Early termination: if we found a good predecessor, 
                # earlier elements will be smaller and potentially better
                if dp[i] >= max_length:
                    break
        
        if dp[i] > max_length:
            max_length = dp[i]
            max_index = i
    
    # Reconstruct
    result = []
    current = max_index
    
    while current != -1:
        result.append(nums[current])
        current = parent[current]
    
    result.reverse()
    return result


def largest_divisible_subset_with_analysis(nums):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step DP computation and analysis.
    
    Time Complexity: O(n^2) - DP computation
    Space Complexity: O(n) - DP array
    """
    if not nums:
        return []
    
    print(f"Original array: {nums}")
    nums.sort()
    print(f"Sorted array: {nums}")
    
    n = len(nums)
    dp = [1] * n
    parent = [-1] * n
    
    print(f"\nDP computation:")
    print(f"Initial dp: {dp}")
    
    for i in range(1, n):
        print(f"\nProcessing nums[{i}] = {nums[i]}:")
        
        for j in range(i):
            if nums[i] % nums[j] == 0:
                if dp[j] + 1 > dp[i]:
                    old_dp = dp[i]
                    dp[i] = dp[j] + 1
                    parent[i] = j
                    print(f"  nums[{i}] % nums[{j}] == 0: {nums[i]} % {nums[j]} = 0")
                    print(f"  Updated dp[{i}]: {old_dp} -> {dp[i]}, parent[{i}] = {j}")
        
        print(f"  Final dp[{i}] = {dp[i]}")
    
    print(f"\nFinal dp array: {dp}")
    print(f"Parent array: {parent}")
    
    # Find maximum
    max_length = max(dp)
    max_index = dp.index(max_length)
    
    print(f"Maximum length: {max_length} at index {max_index}")
    
    # Reconstruct
    result = []
    current = max_index
    
    print(f"\nReconstruction:")
    while current != -1:
        result.append(nums[current])
        print(f"Add nums[{current}] = {nums[current]}")
        current = parent[current]
    
    result.reverse()
    print(f"Final subset: {result}")
    
    return result


def largest_divisible_subset_all_solutions(nums):
    """
    FIND ALL MAXIMUM DIVISIBLE SUBSETS:
    ==================================
    Return all subsets with maximum length.
    
    Time Complexity: O(n^2 + k*n) where k is number of solutions
    Space Complexity: O(k*n) - store all solutions
    """
    if not nums:
        return []
    
    nums.sort()
    n = len(nums)
    
    dp = [1] * n
    predecessors = [[] for _ in range(n)]  # All possible predecessors
    
    for i in range(1, n):
        for j in range(i):
            if nums[i] % nums[j] == 0:
                if dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    predecessors[i] = [j]
                elif dp[j] + 1 == dp[i]:
                    predecessors[i].append(j)
    
    max_length = max(dp)
    
    # Find all indices with maximum length
    max_indices = [i for i in range(n) if dp[i] == max_length]
    
    # Reconstruct all solutions
    all_solutions = []
    
    def reconstruct_all(index, current_path):
        current_path.append(nums[index])
        
        if not predecessors[index]:  # No predecessors
            all_solutions.append(current_path[::-1])  # Reverse to get correct order
        else:
            for pred in predecessors[index]:
                reconstruct_all(pred, current_path[:])
        
        current_path.pop()
    
    for idx in max_indices:
        reconstruct_all(idx, [])
    
    # Remove duplicates
    unique_solutions = []
    seen = set()
    
    for sol in all_solutions:
        sol_tuple = tuple(sol)
        if sol_tuple not in seen:
            seen.add(sol_tuple)
            unique_solutions.append(sol)
    
    return unique_solutions


def largest_divisible_subset_mathematical_insight(nums):
    """
    MATHEMATICAL INSIGHT:
    ====================
    Explain why sorting and LIS approach works.
    
    Time Complexity: O(n^2) - DP computation
    Space Complexity: O(n) - DP array
    """
    print("Mathematical Insight:")
    print("=" * 50)
    print("1. KEY PROPERTY: If a divides b and b divides c, then a divides c")
    print("2. TRANSITIVITY: Divisibility is transitive")
    print("3. SORTING: After sorting, if nums[i] divides nums[j] where i < j,")
    print("   then nums[i] < nums[j] (since all numbers are distinct)")
    print("4. CHAIN PROPERTY: Any divisible subset forms a chain when sorted")
    print("5. LIS ANALOGY: Find longest chain where each element divides the next")
    print()
    
    # Example demonstration
    nums_example = [1, 2, 4, 8, 16]
    print(f"Example: {nums_example}")
    print("Divisibility chain: 1 | 2 | 4 | 8 | 16")
    print("Each element divides all elements after it in this chain")
    print()
    
    result = largest_divisible_subset_dp(nums)
    
    # Verify divisibility property
    print(f"Result for input {nums}: {result}")
    if len(result) > 1:
        print("Verification:")
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                print(f"  {result[j]} % {result[i]} = {result[j] % result[i]}")
    
    return result


# Test cases
def test_largest_divisible_subset():
    """Test all implementations with various inputs"""
    test_cases = [
        ([1,2,3], [1,2]),  # One possible answer
        ([1,2,4,8], [1,2,4,8]),
        ([1], [1]),
        ([1,2], [1,2]),
        ([2,3,4,9,8], [2,4,8]),  # One possible answer
        ([4,8,10,240], [4,8,240]),  # One possible answer
        ([1,4,8,13,26,52], [1,4,8]),  # One possible answer
        ([5,9,18,54,108,540,90,180,360,720], [5,90,180,360,720]),  # One possible answer
        ([3,4,16,8], [4,8,16]),  # One possible answer
        ([1,3,6,24], [1,3,6,24])
    ]
    
    print("Testing Largest Divisible Subset Solutions:")
    print("=" * 70)
    
    for i, (nums, one_expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: nums = {nums}")
        print(f"One valid answer: {one_expected} (length: {len(one_expected)})")
        
        # Test approaches (skip expensive ones for large inputs)
        if len(nums) <= 6:
            try:
                brute = largest_divisible_subset_brute_force(nums.copy())
                print(f"Brute Force:      {brute} (len: {len(brute)}) {'✓' if len(brute) == len(one_expected) else '✗'}")
            except:
                print(f"Brute Force:      Timeout/Error")
        
        dp_result = largest_divisible_subset_dp(nums.copy())
        memo_result = largest_divisible_subset_memoization(nums.copy())
        opt_result = largest_divisible_subset_optimized(nums.copy())
        
        print(f"DP:               {dp_result} (len: {len(dp_result)}) {'✓' if len(dp_result) == len(one_expected) else '✗'}")
        print(f"Memoization:      {memo_result} (len: {len(memo_result)}) {'✓' if len(memo_result) == len(one_expected) else '✗'}")
        print(f"Optimized:        {opt_result} (len: {len(opt_result)}) {'✓' if len(opt_result) == len(one_expected) else '✗'}")
        
        # Verify divisibility property
        def verify_divisible_subset(subset):
            for i in range(len(subset)):
                for j in range(i + 1, len(subset)):
                    if subset[i] % subset[j] != 0 and subset[j] % subset[i] != 0:
                        return False
            return True
        
        print(f"DP result valid:  {verify_divisible_subset(dp_result)}")
        
        # Show all solutions for small cases
        if len(nums) <= 6:
            try:
                all_solutions = largest_divisible_subset_all_solutions(nums.copy())
                if len(all_solutions) <= 5:
                    print(f"All max solutions: {all_solutions}")
            except:
                pass
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    largest_divisible_subset_with_analysis([1,2,4,8])
    
    # Mathematical insight
    print(f"\n" + "=" * 70)
    print("MATHEMATICAL INSIGHT:")
    print("-" * 40)
    largest_divisible_subset_mathematical_insight([4,8,10,240])
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. SORTING IS CRUCIAL: Enables LIS-like DP approach")
    print("2. TRANSITIVITY: If a|b and b|c, then a|c")
    print("3. CHAIN PROPERTY: Divisible subset forms a chain when sorted")
    print("4. LIS VARIANT: Find longest chain with divisibility constraint")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(2^n * n²), Space: O(n)")
    print("DP:               Time: O(n²),       Space: O(n)")
    print("Memoization:      Time: O(n²),       Space: O(n²)")
    print("Optimized:        Time: O(n²),       Space: O(n)")


if __name__ == "__main__":
    test_largest_divisible_subset()


"""
PATTERN RECOGNITION:
==================
This is LIS with a divisibility constraint:
- Instead of nums[i] < nums[j], we need nums[i] | nums[j] (divisibility)
- After sorting, divisibility creates a chain structure
- Classic DP approach with constraint modification

KEY INSIGHT - DIVISIBILITY CHAIN:
================================
**Mathematical Property**: 
If a|b and b|c, then a|c (transitivity of divisibility)

**Chain Structure**:
After sorting, any divisible subset forms a chain:
a₁ | a₂ | a₃ | ... | aₖ where a₁ < a₂ < a₃ < ... < aₖ

**Why Sorting Works**:
- All numbers are distinct and positive
- If a|b and a ≠ b, then a < b
- So sorted divisible subset maintains divisibility order

ALGORITHM APPROACHES:
====================

1. **DP (Optimal)**: O(n²) after sorting
   - Sort array first
   - Apply LIS logic with divisibility check
   - dp[i] = max length ending at index i

2. **Memoization**: O(n²) with caching
   - Recursive formulation with memo
   - Natural top-down approach

3. **Brute Force**: O(2^n × n²)
   - Generate all subsets
   - Check divisibility property for each

DP STATE DEFINITION:
===================
dp[i] = length of largest divisible subset ending at index i

RECURRENCE RELATION:
===================
```
dp[i] = max(dp[j] + 1) for all j < i where nums[i] % nums[j] == 0
```

Base case: dp[i] = 1 (each element forms subset of size 1)

RECONSTRUCTION:
==============
Use parent array to track predecessors:
```python
if nums[i] % nums[j] == 0 and dp[j] + 1 > dp[i]:
    dp[i] = dp[j] + 1
    parent[i] = j
```

SORTING STRATEGY:
================
**Must sort the array first**:
- Enables simple DP formulation
- Ensures that if nums[i] divides nums[j] (i < j), then we can extend
- Without sorting, would need complex 2D state

MATHEMATICAL PROPERTIES:
=======================
1. **Transitivity**: a|b ∧ b|c ⟹ a|c
2. **Chain Structure**: Divisible subset forms totally ordered chain
3. **Uniqueness**: For distinct positive integers, divisibility chain is unique
4. **Optimal Substructure**: Removing last element gives optimal subproblem

COMPARISON WITH LIS:
===================
- **LIS**: nums[i] < nums[j] (ordering constraint)
- **This Problem**: nums[j] % nums[i] == 0 (divisibility constraint)
- **Solution**: Same DP structure, different constraint

APPLICATIONS:
============
1. **Number Theory**: Finding chains of divisors
2. **Factor Trees**: Building factor hierarchies
3. **Mathematical Structures**: Partially ordered sets
4. **Optimization**: Resource allocation with dependencies

VARIANTS TO PRACTICE:
====================
- Longest Increasing Subsequence (300) - ordering version
- Russian Doll Envelopes (354) - 2D ordering
- Chain of Pairs (646) - interval version
- Number of LIS (673) - counting version

EDGE CASES:
==========
1. **Single element**: Return [element]
2. **All coprime**: Return any single element
3. **Powers of same number**: Full chain possible
4. **Large numbers**: Algorithm works for any range

INTERVIEW TIPS:
==============
1. **Identify as LIS variant**: Key insight
2. **Explain sorting necessity**: Why we need sorted order
3. **Show divisibility property**: Transitivity proof
4. **Trace DP computation**: Step-by-step example
5. **Handle reconstruction**: Build actual subset
6. **Discuss optimizations**: Early termination possibilities
7. **Edge cases**: Single element, coprime numbers
8. **Mathematical insight**: Chain structure explanation
9. **Complexity analysis**: Why O(n²) is necessary
10. **Related problems**: Connect to other LIS variants
"""
