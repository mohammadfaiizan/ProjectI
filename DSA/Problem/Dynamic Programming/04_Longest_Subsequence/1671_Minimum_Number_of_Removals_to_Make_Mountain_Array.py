"""
LeetCode 1671: Minimum Number of Removals to Make Mountain Array
Difficulty: Hard
Category: Longest Subsequence Problems (Bitonic LIS)

PROBLEM DESCRIPTION:
===================
You may recall that an array arr is a mountain array if and only if:

- arr.length >= 3
- There exists some index i (0-indexed) with 0 < i < arr.length - 1 such that:
  - arr[0] < arr[1] < ... < arr[i - 1] < arr[i]
  - arr[i] > arr[i + 1] > ... > arr[arr.length - 1]

Given an integer array nums, return the minimum number of elements to remove to make nums a mountain array.

Example 1:
Input: nums = [1,3,1]
Output: 0
Explanation: The array itself is a mountain array so we do not need to remove any elements.

Example 2:
Input: nums = [2,1,1,5,6,2,3,1]
Output: 3
Explanation: One solution is to remove the elements at indices 0, 1, and 5, making the array [1,5,6,2,3,1] → [5,6,2,3] → [5,6,3].

Example 3:
Input: nums = [4,3,2,1,1,2,3,1]
Output: 4

Example 4:
Input: nums = [1,2,3,4,4,3,2,1]
Output: 1

Constraints:
- 3 <= nums.length <= 1000
- 1 <= nums[i] <= 10^9
- It's guaranteed that you can make a mountain array out of nums.
"""

def minimum_mountain_removals_brute_force(nums):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible mountain arrays and find the longest one.
    
    Time Complexity: O(2^n * n) - 2^n subsequences, O(n) to validate each
    Space Complexity: O(n) - recursion stack
    """
    def is_mountain(arr):
        if len(arr) < 3:
            return False
        
        # Find peak
        peak_idx = -1
        for i in range(1, len(arr) - 1):
            if arr[i-1] < arr[i] > arr[i+1]:
                if peak_idx == -1:
                    peak_idx = i
                else:
                    return False  # Multiple peaks
        
        if peak_idx == -1:
            return False
        
        # Check strictly increasing before peak
        for i in range(peak_idx):
            if arr[i] >= arr[i+1]:
                return False
        
        # Check strictly decreasing after peak
        for i in range(peak_idx, len(arr) - 1):
            if arr[i] <= arr[i+1]:
                return False
        
        return True
    
    def generate_subsequences(index, current):
        if index >= len(nums):
            if is_mountain(current):
                return len(current)
            return 0
        
        # Skip current element
        skip = generate_subsequences(index + 1, current)
        
        # Include current element
        current.append(nums[index])
        include = generate_subsequences(index + 1, current)
        current.pop()
        
        return max(skip, include)
    
    max_mountain_length = generate_subsequences(0, [])
    return len(nums) - max_mountain_length


def minimum_mountain_removals_lis_based(nums):
    """
    LIS-BASED APPROACH (OPTIMAL):
    ============================
    Find LIS from left and LDS from right, then find best mountain.
    
    Time Complexity: O(n^2) - two LIS computations
    Space Complexity: O(n) - DP arrays
    """
    n = len(nums)
    
    # LIS ending at each position (left to right)
    lis_left = [1] * n
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                lis_left[i] = max(lis_left[i], lis_left[j] + 1)
    
    # LIS starting from each position (right to left) = LDS from left to right
    lis_right = [1] * n
    for i in range(n - 2, -1, -1):
        for j in range(i + 1, n):
            if nums[i] > nums[j]:
                lis_right[i] = max(lis_right[i], lis_right[j] + 1)
    
    # Find the maximum mountain length
    max_mountain_length = 0
    for i in range(1, n - 1):  # Peak must be in the middle
        # Valid mountain: both sides have length >= 2
        if lis_left[i] >= 2 and lis_right[i] >= 2:
            mountain_length = lis_left[i] + lis_right[i] - 1
            max_mountain_length = max(max_mountain_length, mountain_length)
    
    return n - max_mountain_length


def minimum_mountain_removals_optimized_lis(nums):
    """
    OPTIMIZED LIS APPROACH:
    ======================
    Use binary search for O(n log n) LIS computation.
    
    Time Complexity: O(n log n) - optimized LIS
    Space Complexity: O(n) - LIS arrays
    """
    def lis_binary_search(arr):
        """Compute LIS lengths for each position using binary search"""
        n = len(arr)
        lis_length = [0] * n
        tails = []
        
        for i in range(n):
            # Binary search for position
            left, right = 0, len(tails)
            while left < right:
                mid = (left + right) // 2
                if tails[mid] < arr[i]:
                    left = mid + 1
                else:
                    right = mid
            
            # Update tails array
            if left == len(tails):
                tails.append(arr[i])
            else:
                tails[left] = arr[i]
            
            lis_length[i] = left + 1
        
        return lis_length
    
    n = len(nums)
    
    # LIS from left
    lis_left = lis_binary_search(nums)
    
    # LIS from right (reverse array, then reverse result)
    lis_right = lis_binary_search(nums[::-1])[::-1]
    
    # Find maximum mountain
    max_mountain_length = 0
    for i in range(1, n - 1):
        if lis_left[i] >= 2 and lis_right[i] >= 2:
            mountain_length = lis_left[i] + lis_right[i] - 1
            max_mountain_length = max(max_mountain_length, mountain_length)
    
    return n - max_mountain_length


def minimum_mountain_removals_with_mountain(nums):
    """
    FIND ACTUAL MOUNTAIN:
    ====================
    Return minimum removals and one possible mountain array.
    
    Time Complexity: O(n^2) - LIS + reconstruction
    Space Complexity: O(n) - arrays for reconstruction
    """
    n = len(nums)
    
    # Compute LIS from left with parent tracking
    lis_left = [1] * n
    parent_left = [-1] * n
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i] and lis_left[j] + 1 > lis_left[i]:
                lis_left[i] = lis_left[j] + 1
                parent_left[i] = j
    
    # Compute LIS from right with parent tracking
    lis_right = [1] * n
    parent_right = [-1] * n
    
    for i in range(n - 2, -1, -1):
        for j in range(i + 1, n):
            if nums[i] > nums[j] and lis_right[j] + 1 > lis_right[i]:
                lis_right[i] = lis_right[j] + 1
                parent_right[i] = j
    
    # Find best peak
    max_mountain_length = 0
    best_peak = -1
    
    for i in range(1, n - 1):
        if lis_left[i] >= 2 and lis_right[i] >= 2:
            mountain_length = lis_left[i] + lis_right[i] - 1
            if mountain_length > max_mountain_length:
                max_mountain_length = mountain_length
                best_peak = i
    
    if best_peak == -1:
        return n - 3, []  # Minimum valid mountain
    
    # Reconstruct mountain
    mountain = []
    
    # Reconstruct left side
    left_side = []
    current = best_peak
    while current != -1:
        left_side.append(nums[current])
        current = parent_left[current]
    left_side.reverse()
    
    # Reconstruct right side
    right_side = []
    current = parent_right[best_peak]
    while current != -1:
        right_side.append(nums[current])
        current = parent_right[current]
    
    mountain = left_side + right_side
    
    return n - max_mountain_length, mountain


def minimum_mountain_removals_detailed_analysis(nums):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step computation and analysis.
    
    Time Complexity: O(n^2) - LIS computations
    Space Complexity: O(n) - DP arrays
    """
    n = len(nums)
    
    print(f"Array: {nums}")
    print(f"Length: {n}")
    
    # Compute LIS from left
    lis_left = [1] * n
    print(f"\nComputing LIS from left:")
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                lis_left[i] = max(lis_left[i], lis_left[j] + 1)
        print(f"  lis_left[{i}] = {lis_left[i]} (ending at nums[{i}]={nums[i]})")
    
    # Compute LIS from right
    lis_right = [1] * n
    print(f"\nComputing LIS from right (LDS from left):")
    
    for i in range(n - 2, -1, -1):
        for j in range(i + 1, n):
            if nums[i] > nums[j]:
                lis_right[i] = max(lis_right[i], lis_right[j] + 1)
        print(f"  lis_right[{i}] = {lis_right[i]} (starting at nums[{i}]={nums[i]})")
    
    print(f"\nFinal arrays:")
    print(f"lis_left:  {lis_left}")
    print(f"lis_right: {lis_right}")
    
    # Analyze each potential peak
    print(f"\nAnalyzing potential peaks:")
    max_mountain_length = 0
    best_peak = -1
    
    for i in range(1, n - 1):
        left_len = lis_left[i]
        right_len = lis_right[i]
        
        if left_len >= 2 and right_len >= 2:
            mountain_length = left_len + right_len - 1
            print(f"  Peak at index {i} (value={nums[i]}): left={left_len}, right={right_len}, total={mountain_length}")
            
            if mountain_length > max_mountain_length:
                max_mountain_length = mountain_length
                best_peak = i
        else:
            print(f"  Index {i} (value={nums[i]}): invalid peak (left={left_len}, right={right_len})")
    
    print(f"\nBest mountain:")
    print(f"  Peak at index {best_peak} (value={nums[best_peak] if best_peak != -1 else 'None'})")
    print(f"  Mountain length: {max_mountain_length}")
    print(f"  Minimum removals: {n - max_mountain_length}")
    
    return n - max_mountain_length


def minimum_mountain_removals_all_mountains(nums):
    """
    FIND ALL POSSIBLE MOUNTAINS:
    ===========================
    Find all possible mountain arrays of maximum length.
    
    Time Complexity: O(n^2) - LIS computations
    Space Complexity: O(n) - DP arrays
    """
    n = len(nums)
    
    # Compute LIS arrays
    lis_left = [1] * n
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                lis_left[i] = max(lis_left[i], lis_left[j] + 1)
    
    lis_right = [1] * n
    for i in range(n - 2, -1, -1):
        for j in range(i + 1, n):
            if nums[i] > nums[j]:
                lis_right[i] = max(lis_right[i], lis_right[j] + 1)
    
    # Find maximum mountain length
    max_mountain_length = 0
    for i in range(1, n - 1):
        if lis_left[i] >= 2 and lis_right[i] >= 2:
            mountain_length = lis_left[i] + lis_right[i] - 1
            max_mountain_length = max(max_mountain_length, mountain_length)
    
    # Find all peaks that achieve maximum length
    optimal_peaks = []
    for i in range(1, n - 1):
        if lis_left[i] >= 2 and lis_right[i] >= 2:
            mountain_length = lis_left[i] + lis_right[i] - 1
            if mountain_length == max_mountain_length:
                optimal_peaks.append(i)
    
    return n - max_mountain_length, optimal_peaks


# Test cases
def test_minimum_mountain_removals():
    """Test all implementations with various inputs"""
    test_cases = [
        ([1,3,1], 0),
        ([2,1,1,5,6,2,3,1], 3),
        ([4,3,2,1,1,2,3,1], 4),
        ([1,2,3,4,4,3,2,1], 1),
        ([100,92,89,77,74,66,64,66,64], 6),
        ([1,2,1,3,1], 2),
        ([2,1,1,5,6,2,3,1], 3),
        ([9,8,1,7,6,5,4,3,2,1], 2),
        ([1,2,3,4,5,4,3,2,1], 0),
        ([23,47,63,72,81,99,88,55,21,33,32], 1)
    ]
    
    print("Testing Minimum Mountain Removals Solutions:")
    print("=" * 70)
    
    for i, (nums, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: nums = {nums[:8]}{'...' if len(nums) > 8 else ''}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for large inputs)
        if len(nums) <= 8:
            try:
                brute = minimum_mountain_removals_brute_force(nums.copy())
                print(f"Brute Force:      {brute:>3} {'✓' if brute == expected else '✗'}")
            except:
                print(f"Brute Force:      Timeout")
        
        lis_based = minimum_mountain_removals_lis_based(nums.copy())
        optimized = minimum_mountain_removals_optimized_lis(nums.copy())
        
        print(f"LIS-based:        {lis_based:>3} {'✓' if lis_based == expected else '✗'}")
        print(f"Optimized:        {optimized:>3} {'✓' if optimized == expected else '✗'}")
        
        # Show actual mountain for small cases
        if len(nums) <= 10:
            removals, mountain = minimum_mountain_removals_with_mountain(nums.copy())
            if mountain:
                print(f"One mountain: {mountain}")
            
            # Show all optimal peaks
            removals, peaks = minimum_mountain_removals_all_mountains(nums.copy())
            if peaks:
                peak_values = [nums[p] for p in peaks]
                print(f"Optimal peaks at indices {peaks} (values: {peak_values})")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    minimum_mountain_removals_detailed_analysis([2,1,1,5,6,2,3,1])
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. MOUNTAIN = LIS (left) + LDS (right) with common peak")
    print("2. LDS = LIS on reversed array")
    print("3. VALID PEAK: Both LIS_left[i] >= 2 and LIS_right[i] >= 2")
    print("4. MOUNTAIN LENGTH: LIS_left[i] + LIS_right[i] - 1")
    print("5. REMOVALS: n - max_mountain_length")
    
    print("\n" + "=" * 70)
    print("Algorithm Comparison:")
    print("Brute Force:      Try all mountain subsequences")
    print("LIS-based:        LIS from both directions")
    print("Optimized:        Binary search LIS (O(n log n))")
    print("With Mountain:    LIS + reconstruction")
    print("All Mountains:    Find all optimal peaks")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(2^n * n), Space: O(n)")
    print("LIS-based:        Time: O(n²),      Space: O(n)")
    print("Optimized:        Time: O(n log n), Space: O(n)")
    print("With Mountain:    Time: O(n²),      Space: O(n)")
    print("All Mountains:    Time: O(n²),      Space: O(n)")


if __name__ == "__main__":
    test_minimum_mountain_removals()


"""
PATTERN RECOGNITION:
==================
This is a Bitonic LIS problem:
- Mountain array = Bitonic sequence (increasing then decreasing)
- Maximum mountain = Longest Bitonic Subsequence
- Minimum removals = n - longest_mountain_length
- Combines LIS from both directions

KEY INSIGHT - BITONIC DECOMPOSITION:
===================================
**Mountain Structure**:
- Strictly increasing part: LIS ending at peak
- Strictly decreasing part: LDS starting from peak  
- Peak: Element that participates in both parts

**Mathematical Formulation**:
Mountain_length[i] = LIS_left[i] + LIS_right[i] - 1

**Validity Condition**:
Valid peak: LIS_left[i] ≥ 2 AND LIS_right[i] ≥ 2

ALGORITHM APPROACHES:
====================

1. **LIS-based (Standard)**: O(n²)
   - Compute LIS from left to right
   - Compute LIS from right to left (= LDS from left)
   - Find best peak combining both

2. **Optimized LIS**: O(n log n)
   - Use binary search for LIS computation
   - Same logic but faster LIS

3. **Brute Force**: O(2^n × n)
   - Generate all subsequences
   - Check mountain property for each

DETAILED ALGORITHM:
==================
```python
# Step 1: LIS from left
lis_left[i] = length of LIS ending at position i

# Step 2: LIS from right (equivalent to LDS from left)
lis_right[i] = length of LDS starting at position i

# Step 3: Find best mountain
for each position i in [1, n-2]:  # Peak can't be at edges
    if lis_left[i] >= 2 and lis_right[i] >= 2:
        mountain_length = lis_left[i] + lis_right[i] - 1
        max_mountain_length = max(max_mountain_length, mountain_length)

# Step 4: Calculate removals
return n - max_mountain_length
```

MOUNTAIN VALIDITY:
=================
**Why LIS_left[i] ≥ 2 and LIS_right[i] ≥ 2?**
- Mountain needs at least 3 elements: a < b > c
- LIS_left[i] ≥ 2: at least one element before peak
- LIS_right[i] ≥ 2: at least one element after peak
- Peak itself contributes to both sides

LIS COMPUTATION DETAILS:
========================
**From Left (Standard LIS)**:
```python
for i in range(1, n):
    for j in range(i):
        if nums[j] < nums[i]:
            lis_left[i] = max(lis_left[i], lis_left[j] + 1)
```

**From Right (LDS from Left)**:
```python
for i in range(n-2, -1, -1):
    for j in range(i+1, n):
        if nums[i] > nums[j]:  # Decreasing
            lis_right[i] = max(lis_right[i], lis_right[j] + 1)
```

OPTIMIZATION OPPORTUNITIES:
==========================
1. **Binary Search LIS**: Reduce O(n²) to O(n log n)
2. **Early Termination**: Skip if no valid mountain possible
3. **Space Optimization**: Can be done with O(1) extra space (complex)

APPLICATIONS:
============
1. **Signal Processing**: Peak detection in time series
2. **Stock Analysis**: Buy-sell pattern identification  
3. **Topography**: Mountain/hill detection in elevation data
4. **Pattern Recognition**: Bitonic pattern in sequences

VARIANTS TO PRACTICE:
====================
- Longest Bitonic Subsequence - direct version
- Peak Index in Mountain Array (852) - find peak
- Valid Mountain Array (941) - validation problem
- Longest Mountain in Array (845) - contiguous version

EDGE CASES:
==========
1. **Already mountain**: Return 0 removals
2. **No valid mountain**: Not possible per constraints
3. **Multiple peaks**: Choose the one giving longest mountain
4. **Flat regions**: Handle equal elements carefully

INTERVIEW TIPS:
==============
1. **Recognize as bitonic LIS**: Key insight
2. **Explain two-direction LIS**: Why we need both
3. **Show validity condition**: Why ≥ 2 requirement
4. **Trace through example**: Step-by-step computation
5. **Optimization discussion**: Binary search LIS
6. **Handle edge cases**: Boundary conditions
7. **Mathematical proof**: Why formula works
8. **Reconstruction**: How to find actual mountain
9. **Complexity analysis**: O(n²) vs O(n log n)
10. **Related problems**: Connect to other bitonic problems

MATHEMATICAL INSIGHT:
====================
This problem demonstrates the connection between:
- **Longest Bitonic Subsequence** (maximize)
- **Minimum Mountain Removals** (minimize)

The transformation: minimize removals = maximize mountain length
Shows the duality between optimization problems.
"""
