"""
LeetCode 740: Delete and Earn
Difficulty: Medium
Category: Fibonacci & Linear DP

PROBLEM DESCRIPTION:
===================
You are given an integer array nums. You want to maximize the number of points you get by 
performing the following operation any number of times:

Pick any nums[i] and delete it to earn nums[i] points. Afterwards, you must delete every 
element equal to nums[i] - 1 and nums[i] + 1.

Return the maximum number of points you can earn.

Example 1:
Input: nums = [3,4,2]
Output: 6
Explanation: You can perform the following operations:
- Delete 4 to earn 4 points. Consequently, 3 is also deleted, and nums becomes [2].
- Delete 2 to earn 2 points. nums becomes [].
You earn a total of 6 points.

Example 2:
Input: nums = [2,2,3,3,3,4]
Output: 9
Explanation: You can perform the following operations:
- Delete a 3 to earn 3 points. All 2's and 4's are also deleted. nums becomes [3,3].
- Delete a 3 again to earn 3 points. nums becomes [3].
- Delete a 3 once more to earn 3 points. nums becomes [].
You earn a total of 9 points.

Constraints:
- 1 <= nums.length <= 2 * 10^4
- 1 <= nums[i] <= 10^4
"""

def delete_and_earn_bruteforce(nums):
    """
    BRUTE FORCE APPROACH - RECURSION:
    ================================
    Try all possible deletion sequences.
    
    Time Complexity: O(2^n) - exponential choices
    Space Complexity: O(n) - recursion stack
    """
    from collections import Counter
    
    def max_points(remaining_nums):
        if not remaining_nums:
            return 0
        
        max_pts = 0
        counter = Counter(remaining_nums)
        
        for num in set(remaining_nums):
            # Calculate points from deleting all instances of num
            points = num * counter[num]
            
            # Create new array after deletion
            new_nums = []
            for x in remaining_nums:
                if x != num and x != num - 1 and x != num + 1:
                    new_nums.append(x)
            
            total_points = points + max_points(new_nums)
            max_pts = max(max_pts, total_points)
        
        return max_pts
    
    return max_points(nums)


def delete_and_earn_transform_to_house_robber(nums):
    """
    TRANSFORM TO HOUSE ROBBER PROBLEM:
    =================================
    Key insight: Transform to house robber problem.
    Create array where index represents number value and value represents total points.
    
    Time Complexity: O(n + k) where k is max(nums)
    Space Complexity: O(k) - points array
    """
    if not nums:
        return 0
    
    # Count total points for each number
    max_num = max(nums)
    points = [0] * (max_num + 1)
    
    for num in nums:
        points[num] += num
    
    # Now solve house robber on points array
    def rob_house(houses):
        if len(houses) <= 2:
            return max(houses) if houses else 0
        
        prev2 = houses[0]
        prev1 = max(houses[0], houses[1])
        
        for i in range(2, len(houses)):
            current = max(prev1, prev2 + houses[i])
            prev2 = prev1
            prev1 = current
        
        return prev1
    
    return rob_house(points)


def delete_and_earn_memoization(nums):
    """
    DYNAMIC PROGRAMMING - TOP DOWN (MEMOIZATION):
    ============================================
    Use memoization with number frequency approach.
    
    Time Complexity: O(n + k) where k is range of numbers
    Space Complexity: O(k) - memoization + frequency map
    """
    from collections import Counter
    
    if not nums:
        return 0
    
    counter = Counter(nums)
    unique_nums = sorted(counter.keys())
    memo = {}
    
    def max_points(index):
        if index >= len(unique_nums):
            return 0
        
        if index in memo:
            return memo[index]
        
        num = unique_nums[index]
        
        # Option 1: Don't take current number
        skip = max_points(index + 1)
        
        # Option 2: Take current number
        take = num * counter[num]
        next_index = index + 1
        
        # Skip next number if it's adjacent (num + 1)
        if (next_index < len(unique_nums) and 
            unique_nums[next_index] == num + 1):
            take += max_points(index + 2)
        else:
            take += max_points(index + 1)
        
        memo[index] = max(skip, take)
        return memo[index]
    
    return max_points(0)


def delete_and_earn_tabulation(nums):
    """
    DYNAMIC PROGRAMMING - BOTTOM UP (TABULATION):
    =============================================
    Build solution using tabulation with grouped numbers.
    
    Time Complexity: O(n + k log k) - counting + sorting
    Space Complexity: O(k) - DP array for unique numbers
    """
    from collections import Counter
    
    if not nums:
        return 0
    
    counter = Counter(nums)
    unique_nums = sorted(counter.keys())
    n = len(unique_nums)
    
    if n == 1:
        return unique_nums[0] * counter[unique_nums[0]]
    
    # dp[i] = max points using numbers from index i onwards
    dp = [0] * (n + 2)  # Extra space for boundary
    
    # Fill DP table from right to left
    for i in range(n - 1, -1, -1):
        num = unique_nums[i]
        
        # Don't take current number
        skip = dp[i + 1]
        
        # Take current number
        take = num * counter[num]
        if i + 1 < n and unique_nums[i + 1] == num + 1:
            take += dp[i + 2]  # Skip adjacent number
        else:
            take += dp[i + 1]
        
        dp[i] = max(skip, take)
    
    return dp[0]


def delete_and_earn_space_optimized(nums):
    """
    SPACE OPTIMIZED DYNAMIC PROGRAMMING:
    ===================================
    Optimize space using only necessary variables.
    
    Time Complexity: O(n + k log k) - counting + sorting
    Space Complexity: O(k) - only for counting, O(1) for DP
    """
    from collections import Counter
    
    if not nums:
        return 0
    
    counter = Counter(nums)
    unique_nums = sorted(counter.keys())
    n = len(unique_nums)
    
    if n == 1:
        return unique_nums[0] * counter[unique_nums[0]]
    
    # Variables for DP states
    next2 = 0  # dp[i+2]
    next1 = 0  # dp[i+1]
    
    # Process from right to left
    for i in range(n - 1, -1, -1):
        num = unique_nums[i]
        
        # Don't take current number
        skip = next1
        
        # Take current number
        take = num * counter[num]
        if i + 1 < n and unique_nums[i + 1] == num + 1:
            take += next2
        else:
            take += next1
        
        current = max(skip, take)
        next2 = next1
        next1 = current
    
    return next1


def delete_and_earn_range_optimization(nums):
    """
    RANGE-BASED OPTIMIZATION:
    ========================
    Optimize for sparse arrays by processing consecutive ranges.
    
    Time Complexity: O(n + k log k) - counting + sorting
    Space Complexity: O(k) - for counting and grouping
    """
    from collections import Counter
    
    if not nums:
        return 0
    
    counter = Counter(nums)
    unique_nums = sorted(counter.keys())
    
    def solve_consecutive_range(start, end):
        """Solve house robber for consecutive range [start, end]"""
        length = end - start + 1
        if length == 1:
            return start * counter[start]
        
        prev2 = start * counter[start]
        prev1 = max(start * counter[start], (start + 1) * counter[start + 1])
        
        for num in range(start + 2, end + 1):
            current = max(prev1, prev2 + num * counter[num])
            prev2 = prev1
            prev1 = current
        
        return prev1
    
    total_points = 0
    i = 0
    
    while i < len(unique_nums):
        start = i
        
        # Find end of consecutive sequence
        while (i + 1 < len(unique_nums) and 
               unique_nums[i + 1] == unique_nums[i] + 1):
            i += 1
        
        end = i
        
        # Solve for this consecutive range
        if start == end:
            total_points += unique_nums[start] * counter[unique_nums[start]]
        else:
            points = solve_consecutive_range(unique_nums[start], unique_nums[end])
            total_points += points
        
        i += 1
    
    return total_points


def delete_and_earn_bucket_sort_optimization(nums):
    """
    BUCKET SORT OPTIMIZATION:
    ========================
    Use bucket sort since numbers are bounded (1 <= nums[i] <= 10^4).
    
    Time Complexity: O(n + k) where k = 10^4
    Space Complexity: O(k) - bucket array
    """
    if not nums:
        return 0
    
    # Use bucket sort approach
    max_num = max(nums)
    points = [0] * (max_num + 1)
    
    # Count total points for each number
    for num in nums:
        points[num] += num
    
    # Apply house robber logic
    prev2 = 0
    prev1 = points[0] if max_num >= 0 else 0
    
    for i in range(1, max_num + 1):
        current = max(prev1, prev2 + points[i])
        prev2 = prev1
        prev1 = current
    
    return prev1


def delete_and_earn_with_sequence(nums):
    """
    FIND OPTIMAL DELETION SEQUENCE:
    ==============================
    Return both maximum points and the deletion sequence.
    
    Time Complexity: O(n + k log k) - DP + sequence reconstruction
    Space Complexity: O(k) - DP table and sequence storage
    """
    from collections import Counter
    
    if not nums:
        return 0, []
    
    counter = Counter(nums)
    unique_nums = sorted(counter.keys())
    n = len(unique_nums)
    
    # DP with choice tracking
    dp = [0] * (n + 2)
    choice = [False] * n  # True if we take number at index i
    
    # Fill DP table from right to left
    for i in range(n - 1, -1, -1):
        num = unique_nums[i]
        
        skip = dp[i + 1]
        take = num * counter[num]
        
        if i + 1 < n and unique_nums[i + 1] == num + 1:
            take += dp[i + 2]
        else:
            take += dp[i + 1]
        
        if take >= skip:
            dp[i] = take
            choice[i] = True
        else:
            dp[i] = skip
            choice[i] = False
    
    # Reconstruct sequence
    deletion_sequence = []
    i = 0
    while i < n:
        if choice[i]:
            num = unique_nums[i]
            deletion_sequence.extend([num] * counter[num])
            # Skip next number if adjacent
            if i + 1 < n and unique_nums[i + 1] == num + 1:
                i += 2
            else:
                i += 1
        else:
            i += 1
    
    return dp[0], deletion_sequence


# Test cases
def test_delete_and_earn():
    """Test all implementations with various inputs"""
    test_cases = [
        ([3, 4, 2], 6),
        ([2, 2, 3, 3, 3, 4], 9),
        ([1], 1),
        ([1, 1, 1, 1], 4),
        ([1, 2, 3, 4, 5], 9),  # 1+3+5 = 9
        ([5, 2, 3, 4, 1], 9),  # Same as above, different order
        ([10, 8, 4, 2, 1, 3, 4], 18),  # 10+8 = 18
        ([8, 10, 4, 9, 1, 3, 5, 6, 4], 27)  # 8+10+9 = 27
    ]
    
    print("Testing Delete and Earn Solutions:")
    print("=" * 70)
    
    for i, (nums, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: nums = {nums}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip brute force for large inputs)
        if len(nums) <= 6:
            brute = delete_and_earn_bruteforce(nums.copy())
            print(f"Brute Force:      {brute:>3} {'✓' if brute == expected else '✗'}")
        
        house_robber = delete_and_earn_transform_to_house_robber(nums.copy())
        memo = delete_and_earn_memoization(nums.copy())
        tab = delete_and_earn_tabulation(nums.copy())
        space_opt = delete_and_earn_space_optimized(nums.copy())
        range_opt = delete_and_earn_range_optimization(nums.copy())
        bucket = delete_and_earn_bucket_sort_optimization(nums.copy())
        
        print(f"House Robber:     {house_robber:>3} {'✓' if house_robber == expected else '✗'}")
        print(f"Memoization:      {memo:>3} {'✓' if memo == expected else '✗'}")
        print(f"Tabulation:       {tab:>3} {'✓' if tab == expected else '✗'}")
        print(f"Space Optimized:  {space_opt:>3} {'✓' if space_opt == expected else '✗'}")
        print(f"Range Optimized:  {range_opt:>3} {'✓' if range_opt == expected else '✗'}")
        print(f"Bucket Sort:      {bucket:>3} {'✓' if bucket == expected else '✗'}")
        
        # Show deletion sequence for small examples
        if len(nums) <= 8:
            points, sequence = delete_and_earn_with_sequence(nums.copy())
            print(f"Deletion sequence: {sequence} (points: {points})")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(2^n),      Space: O(n)")
    print("House Robber:     Time: O(n+k),      Space: O(k)")
    print("Memoization:      Time: O(n+k),      Space: O(k)")
    print("Tabulation:       Time: O(n+k logk), Space: O(k)")
    print("Space Optimized:  Time: O(n+k logk), Space: O(k)")
    print("Range Optimized:  Time: O(n+k logk), Space: O(k)")
    print("Bucket Sort:      Time: O(n+k),      Space: O(k)")


if __name__ == "__main__":
    test_delete_and_earn()


"""
PATTERN RECOGNITION:
==================
This is a House Robber variant with number grouping:
- Deleting a number deletes all adjacent numbers (±1)
- Goal: maximize total points
- Key insight: Group same numbers, transform to House Robber

KEY INSIGHT - TRANSFORMATION:
============================
1. Count frequency of each number: num → total_points
2. Problem becomes: select non-adjacent numbers to maximize sum
3. This is exactly the House Robber problem!

Example: [2,2,3,3,3,4] → points[2]=4, points[3]=9, points[4]=4
Can't take both 2 and 3 (adjacent), or both 3 and 4 (adjacent)
Optimal: take 3 only → 9 points

STATE DEFINITION:
================
dp[i] = maximum points using numbers from index i onwards in sorted unique numbers

RECURRENCE RELATION:
===================
For each unique number at index i:
- Skip: dp[i] = dp[i+1]
- Take: dp[i] = points[num] + dp[i+2] (if next number is adjacent)
                or points[num] + dp[i+1] (if next number is not adjacent)

dp[i] = max(skip, take)

OPTIMIZATION TECHNIQUES:
=======================
1. House Robber transformation: O(n + k) where k = max(nums)
2. Sparse optimization: process consecutive ranges separately
3. Bucket sort: since numbers bounded by 10^4
4. Space optimization: O(k) → O(1) for DP variables

VARIANTS TO PRACTICE:
====================
- House Robber (198) - direct application after transformation
- House Robber II (213) - circular version
- House Robber III (337) - binary tree version
- Maximum Product Subarray (152) - similar constraint patterns

INTERVIEW TIPS:
==============
1. Recognize constraint: deleting num eliminates num±1
2. Transform to House Robber by grouping same numbers
3. Explain why adjacent numbers cannot be selected together
4. Show frequency counting + sorting approach
5. Optimize with bucket sort for bounded input
6. Handle edge cases (single number, all same numbers)
7. Discuss sparse array optimization for large ranges
"""
