"""
LeetCode 174: Dungeon Game
Difficulty: Hard
Category: Grid/Matrix DP

PROBLEM DESCRIPTION:
===================
The demons had captured the princess and imprisoned her in the bottom-right corner of a dungeon. 
The dungeon consists of m x n rooms laid out in a 2D grid. Our valiant knight starts in the 
top-left corner and must fight his way through dungeon to rescue the princess.

The knight has an initial health point represented by a positive integer. If at any point his 
health point drops to 0 or below, he dies immediately.

Some of the rooms are guarded by demons (represented by negative integers), so the knight loses 
health upon entering such rooms; other rooms are either empty (represented as 0) or contain magic 
orbs that increase the knight's health (represented by positive integers).

To reach the princess as quickly as possible, the knight decides to move only rightward or downward 
in each step.

Write a function to determine the knight's minimum initial health so that he can rescue the princess.

Note:
- The knight's health has no upper bound.
- Any room can contain threats or power-ups, even the first room the knight enters and the 
  bottom-right room where the princess is.

Example 1:
Input: dungeon = [[-3,5]]
Output: 4
Explanation: The initial health of the knight must be at least 4 if he follows the optimal path 
RIGHT-> STOP.

Example 2:
Input: dungeon = [[-3,5],[-1,3]]
Output: 3

Example 3:
Input: dungeon = [[1,-3,3],[0,-2,0],[-3,-3,-3]]
Output: 3

Constraints:
- m == dungeon.length
- n == dungeon[i].length
- 1 <= m, n <= 200
- -1000 <= dungeon[i][j] <= 1000
"""

def calculate_minimum_hp_brute_force(dungeon):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible paths and find minimum initial health needed.
    
    Time Complexity: O(2^(m+n)) - exponential paths
    Space Complexity: O(m+n) - recursion depth
    """
    if not dungeon or not dungeon[0]:
        return 1
    
    m, n = len(dungeon), len(dungeon[0])
    min_initial_health = float('inf')
    
    def dfs(row, col, current_health, min_health_needed):
        nonlocal min_initial_health
        
        # Update health after entering current room
        current_health += dungeon[row][col]
        
        # Track minimum health needed so far
        min_health_needed = max(min_health_needed, 1 - current_health)
        
        # If reached destination
        if row == m - 1 and col == n - 1:
            min_initial_health = min(min_initial_health, min_health_needed)
            return
        
        # Try going down
        if row + 1 < m:
            dfs(row + 1, col, current_health, min_health_needed)
        
        # Try going right
        if col + 1 < n:
            dfs(row, col + 1, current_health, min_health_needed)
    
    dfs(0, 0, 0, 1)
    return min_initial_health


def calculate_minimum_hp_memoization(dungeon):
    """
    MEMOIZATION APPROACH:
    ====================
    Use memoization to avoid recomputing subproblems.
    
    Time Complexity: O(m*n) - each cell computed once
    Space Complexity: O(m*n) - memoization table
    """
    if not dungeon or not dungeon[0]:
        return 1
    
    m, n = len(dungeon), len(dungeon[0])
    memo = {}
    
    def min_health_needed(row, col):
        # Base case: beyond the grid
        if row >= m or col >= n:
            return float('inf')
        
        # Base case: reached destination
        if row == m - 1 and col == n - 1:
            return max(1, 1 - dungeon[row][col])
        
        if (row, col) in memo:
            return memo[(row, col)]
        
        # Try going right and down
        right = min_health_needed(row, col + 1)
        down = min_health_needed(row + 1, col)
        
        # Choose the path requiring less health
        min_health_next = min(right, down)
        
        # Calculate health needed at current position
        current_requirement = max(1, min_health_next - dungeon[row][col])
        
        memo[(row, col)] = current_requirement
        return current_requirement
    
    return min_health_needed(0, 0)


def calculate_minimum_hp_dp(dungeon):
    """
    DYNAMIC PROGRAMMING APPROACH (BOTTOM-UP):
    =========================================
    Build solution from bottom-right to top-left.
    
    Time Complexity: O(m*n) - process each cell once
    Space Complexity: O(m*n) - DP table
    """
    if not dungeon or not dungeon[0]:
        return 1
    
    m, n = len(dungeon), len(dungeon[0])
    
    # DP table: dp[i][j] = minimum health needed at position (i,j)
    dp = [[0] * n for _ in range(m)]
    
    # Initialize destination (bottom-right)
    dp[m-1][n-1] = max(1, 1 - dungeon[m-1][n-1])
    
    # Fill last row (can only move right)
    for j in range(n - 2, -1, -1):
        dp[m-1][j] = max(1, dp[m-1][j+1] - dungeon[m-1][j])
    
    # Fill last column (can only move down)
    for i in range(m - 2, -1, -1):
        dp[i][n-1] = max(1, dp[i+1][n-1] - dungeon[i][n-1])
    
    # Fill rest of the table
    for i in range(m - 2, -1, -1):
        for j in range(n - 2, -1, -1):
            min_health_next = min(dp[i+1][j], dp[i][j+1])
            dp[i][j] = max(1, min_health_next - dungeon[i][j])
    
    return dp[0][0]


def calculate_minimum_hp_space_optimized(dungeon):
    """
    SPACE OPTIMIZED APPROACH:
    =========================
    Use only O(n) space by processing row by row from bottom.
    
    Time Complexity: O(m*n) - process each cell once
    Space Complexity: O(n) - single row array
    """
    if not dungeon or not dungeon[0]:
        return 1
    
    m, n = len(dungeon), len(dungeon[0])
    
    # Use single array to represent current row
    dp = [0] * n
    
    # Initialize last row
    dp[n-1] = max(1, 1 - dungeon[m-1][n-1])
    for j in range(n - 2, -1, -1):
        dp[j] = max(1, dp[j+1] - dungeon[m-1][j])
    
    # Process from second last row to top
    for i in range(m - 2, -1, -1):
        # Update rightmost element (can only move down)
        dp[n-1] = max(1, dp[n-1] - dungeon[i][n-1])
        
        # Update rest of the row
        for j in range(n - 2, -1, -1):
            min_health_next = min(dp[j], dp[j+1])
            dp[j] = max(1, min_health_next - dungeon[i][j])
    
    return dp[0]


def calculate_minimum_hp_with_path(dungeon):
    """
    FIND OPTIMAL PATH:
    =================
    Return minimum health and the actual optimal path.
    
    Time Complexity: O(m*n) - DP + path reconstruction
    Space Complexity: O(m*n) - DP table + path tracking
    """
    if not dungeon or not dungeon[0]:
        return 1, []
    
    m, n = len(dungeon), len(dungeon[0])
    
    # DP table and path tracking
    dp = [[0] * n for _ in range(m)]
    parent = [[None] * n for _ in range(m)]
    
    # Initialize destination
    dp[m-1][n-1] = max(1, 1 - dungeon[m-1][n-1])
    
    # Fill last row
    for j in range(n - 2, -1, -1):
        dp[m-1][j] = max(1, dp[m-1][j+1] - dungeon[m-1][j])
        parent[m-1][j] = (m-1, j+1)
    
    # Fill last column
    for i in range(m - 2, -1, -1):
        dp[i][n-1] = max(1, dp[i+1][n-1] - dungeon[i][n-1])
        parent[i][n-1] = (i+1, n-1)
    
    # Fill rest of the table
    for i in range(m - 2, -1, -1):
        for j in range(n - 2, -1, -1):
            right_health = dp[i][j+1]
            down_health = dp[i+1][j]
            
            if right_health <= down_health:
                min_health_next = right_health
                parent[i][j] = (i, j+1)
            else:
                min_health_next = down_health
                parent[i][j] = (i+1, j)
            
            dp[i][j] = max(1, min_health_next - dungeon[i][j])
    
    # Reconstruct path
    path = []
    current = (0, 0)
    
    while current is not None:
        path.append(current)
        current = parent[current[0]][current[1]]
    
    return dp[0][0], path


def calculate_minimum_hp_analysis(dungeon):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step DP computation and health tracking.
    
    Time Complexity: O(m*n) - analysis computation
    Space Complexity: O(m*n) - temporary tables
    """
    if not dungeon or not dungeon[0]:
        return 1
    
    m, n = len(dungeon), len(dungeon[0])
    
    print("Dungeon Grid:")
    for i, row in enumerate(dungeon):
        print(f"  Row {i}: {row}")
    
    print(f"\nDP Computation (Bottom-Up, Right-Left):")
    
    # Create DP table for visualization
    dp = [[0] * n for _ in range(m)]
    
    # Initialize destination
    dp[m-1][n-1] = max(1, 1 - dungeon[m-1][n-1])
    print(f"Initialize destination dp[{m-1}][{n-1}] = max(1, 1 - {dungeon[m-1][n-1]}) = {dp[m-1][n-1]}")
    
    # Fill last row
    if n > 1:
        print(f"\nFilling last row (can only move right):")
        for j in range(n - 2, -1, -1):
            dp[m-1][j] = max(1, dp[m-1][j+1] - dungeon[m-1][j])
            print(f"  dp[{m-1}][{j}] = max(1, {dp[m-1][j+1]} - {dungeon[m-1][j]}) = {dp[m-1][j]}")
    
    # Fill last column
    if m > 1:
        print(f"\nFilling last column (can only move down):")
        for i in range(m - 2, -1, -1):
            dp[i][n-1] = max(1, dp[i+1][n-1] - dungeon[i][n-1])
            print(f"  dp[{i}][{n-1}] = max(1, {dp[i+1][n-1]} - {dungeon[i][n-1]}) = {dp[i][n-1]}")
    
    # Fill rest of table
    if m > 1 and n > 1:
        print(f"\nFilling rest of grid:")
        for i in range(m - 2, -1, -1):
            for j in range(n - 2, -1, -1):
                right_health = dp[i][j+1]
                down_health = dp[i+1][j]
                min_health_next = min(right_health, down_health)
                dp[i][j] = max(1, min_health_next - dungeon[i][j])
                
                print(f"  dp[{i}][{j}] = max(1, min({right_health}, {down_health}) - {dungeon[i][j]}) = max(1, {min_health_next - dungeon[i][j]}) = {dp[i][j]}")
    
    print(f"\nFinal DP table (minimum health needed at each position):")
    for i, row in enumerate(dp):
        print(f"  Row {i}: {row}")
    
    # Show optimal path
    min_health, path = calculate_minimum_hp_with_path([row[:] for row in dungeon])
    print(f"\nOptimal path: {path}")
    
    # Simulate the journey
    print(f"\nJourney simulation with initial health {min_health}:")
    health = min_health
    for i, (row, col) in enumerate(path):
        health += dungeon[row][col]
        print(f"  Step {i}: Move to ({row},{col}), room value = {dungeon[row][col]}, health = {health}")
    
    return dp[0][0]


def calculate_minimum_hp_variants():
    """
    TEST PROBLEM VARIANTS:
    =====================
    Test different scenarios and edge cases.
    """
    test_cases = [
        # Basic cases
        ([[-3,5]], 4, "Simple 1x2 case"),
        ([[-3,5],[-1,3]], 3, "Simple 2x2 case"),
        ([[1,-3,3],[0,-2,0],[-3,-3,-3]], 3, "Standard 3x3 case"),
        
        # Edge cases
        ([[1]], 1, "Single positive cell"),
        ([[-5]], 6, "Single negative cell"),
        ([[0]], 1, "Single zero cell"),
        
        # All positive
        ([[1,2,3],[4,5,6]], 1, "All positive values"),
        
        # All negative  
        ([[-1,-2],[-3,-4]], 10, "All negative values"),
        
        # Mixed patterns
        ([[1,-1,0],[2,-2,1]], 1, "Mixed positive/negative"),
        ([[-5,1,2],[3,-4,1]], 4, "Negative start, mixed path"),
        
        # Larger cases
        ([[2,-3,3],[-5,-10,1],[10,30,-5]], 7, "Complex 3x3 case")
    ]
    
    print("Testing Problem Variants:")
    print("=" * 60)
    
    for i, (dungeon, expected, description) in enumerate(test_cases):
        print(f"\nTest {i+1}: {description}")
        print(f"Dungeon: {dungeon}")
        print(f"Expected: {expected}")
        
        result = calculate_minimum_hp_dp([row[:] for row in dungeon])
        print(f"Result: {result} {'✓' if result == expected else '✗'}")
        
        if len(dungeon) <= 3 and len(dungeon[0]) <= 3:
            min_health, path = calculate_minimum_hp_with_path([row[:] for row in dungeon])
            print(f"Optimal path: {path}")


# Test cases
def test_calculate_minimum_hp():
    """Test all implementations with various inputs"""
    test_cases = [
        ([[-3,5]], 4),
        ([[-3,5],[-1,3]], 3),
        ([[1,-3,3],[0,-2,0],[-3,-3,-3]], 3),
        ([[1]], 1),
        ([[-5]], 6),
        ([[0]], 1),
        ([[1,2],[3,4]], 1),
        ([[-1,-2],[-3,-4]], 10),
        ([[2,-3,3],[-5,-10,1],[10,30,-5]], 7)
    ]
    
    print("Testing Dungeon Game Solutions:")
    print("=" * 70)
    
    for i, (dungeon, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Dungeon: {dungeon}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for large inputs)
        if len(dungeon) <= 3 and len(dungeon[0]) <= 3:
            try:
                brute = calculate_minimum_hp_brute_force([row[:] for row in dungeon])
                print(f"Brute Force:      {brute:>5} {'✓' if brute == expected else '✗'}")
            except:
                print(f"Brute Force:      Timeout")
        
        memo = calculate_minimum_hp_memoization([row[:] for row in dungeon])
        dp_result = calculate_minimum_hp_dp([row[:] for row in dungeon])
        space_opt = calculate_minimum_hp_space_optimized([row[:] for row in dungeon])
        
        print(f"Memoization:      {memo:>5} {'✓' if memo == expected else '✗'}")
        print(f"DP (2D):          {dp_result:>5} {'✓' if dp_result == expected else '✗'}")
        print(f"Space Optimized:  {space_opt:>5} {'✓' if space_opt == expected else '✗'}")
        
        # Show path for small cases
        if len(dungeon) <= 3 and len(dungeon[0]) <= 3:
            min_health, path = calculate_minimum_hp_with_path([row[:] for row in dungeon])
            print(f"Optimal path: {path}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    calculate_minimum_hp_analysis([[1,-3,3],[0,-2,0],[-3,-3,-3]])
    
    # Variants testing
    print(f"\n" + "=" * 70)
    calculate_minimum_hp_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. BACKWARD DP: Must work from destination to start")
    print("2. HEALTH CONSTRAINT: Must maintain health ≥ 1 at all times")
    print("3. GREEDY CHOICE: Choose path requiring minimum initial health")
    print("4. STATE DEFINITION: dp[i][j] = min health needed at position (i,j)")
    print("5. RECURRENCE: max(1, min_health_next - current_room_value)")
    
    print("\n" + "=" * 70)
    print("Algorithm Comparison:")
    print("Brute Force:      Try all paths, track minimum health")
    print("Memoization:      Top-down with caching")
    print("DP (2D):          Bottom-up from destination")
    print("Space Optimized:  Row-by-row processing")
    print("With Path:        DP + path reconstruction")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(2^(m+n)), Space: O(m+n)")
    print("Memoization:      Time: O(m*n),     Space: O(m*n)")
    print("DP (2D):          Time: O(m*n),     Space: O(m*n)")
    print("Space Optimized:  Time: O(m*n),     Space: O(n)")
    print("With Path:        Time: O(m*n),     Space: O(m*n)")


if __name__ == "__main__":
    test_calculate_minimum_hp()


"""
PATTERN RECOGNITION:
==================
Dungeon Game is a unique DP problem requiring REVERSE thinking:
- Standard path problems: optimize from start to end
- Dungeon Game: must work backwards from end to start
- Constraint: health must stay ≥ 1 at ALL times
- Goal: minimize INITIAL health needed

KEY INSIGHT - BACKWARD DP NECESSITY:
===================================
**Why Forward DP Fails**:
- Forward: Know starting health, compute ending health
- Problem: Need minimum starting health (unknown)
- Constraint: Health ≥ 1 throughout journey

**Why Backward DP Works**:
- Backward: Know constraint at end (health ≥ 1)
- Propagate minimum requirements backwards
- At start: get minimum initial health needed

**Mathematical Formulation**:
```
dp[i][j] = minimum health needed when entering cell (i,j)
dp[i][j] = max(1, min(dp[i+1][j], dp[i][j+1]) - dungeon[i][j])
```

ALGORITHM APPROACHES:
====================

1. **Bottom-Up DP (Optimal)**: O(m×n) time, O(m×n) space
   - Start from destination (bottom-right)
   - Propagate minimum health requirements backward
   - Most intuitive for this problem

2. **Top-Down Memoization**: O(m×n) time, O(m×n) space
   - Recursive approach with caching
   - Natural problem decomposition

3. **Space Optimized**: O(m×n) time, O(n) space
   - Process row by row from bottom
   - Reuse single array

4. **Brute Force**: O(2^(m+n)) time
   - Try all paths, track health constraints
   - Exponential complexity

DP STATE DEFINITION:
===================
**State**: dp[i][j] = minimum health required when entering cell (i,j)

**Goal**: Find dp[0][0] (minimum initial health)

**Constraint**: After entering any cell, health ≥ 1

RECURRENCE RELATION:
===================
```python
# Base case (destination)
dp[m-1][n-1] = max(1, 1 - dungeon[m-1][n-1])

# General case
next_min_health = min(dp[i+1][j], dp[i][j+1])
dp[i][j] = max(1, next_min_health - dungeon[i][j])
```

**Logic Explanation**:
1. `next_min_health`: minimum health needed for optimal next step
2. `next_min_health - dungeon[i][j]`: health needed before entering current cell
3. `max(1, ...)`: ensure health never drops below 1

BOUNDARY CONDITIONS:
===================
**Last Row**: Can only move right
```python
for j in range(n-2, -1, -1):
    dp[m-1][j] = max(1, dp[m-1][j+1] - dungeon[m-1][j])
```

**Last Column**: Can only move down
```python
for i in range(m-2, -1, -1):
    dp[i][n-1] = max(1, dp[i+1][n-1] - dungeon[i][n-1])
```

SPACE OPTIMIZATION:
==================
```python
# Use 1D array, process bottom to top
dp = [0] * n

# Initialize last row
dp[n-1] = max(1, 1 - dungeon[m-1][n-1])
for j in range(n-2, -1, -1):
    dp[j] = max(1, dp[j+1] - dungeon[m-1][j])

# Process remaining rows
for i in range(m-2, -1, -1):
    dp[n-1] = max(1, dp[n-1] - dungeon[i][n-1])  # rightmost
    for j in range(n-2, -1, -1):
        dp[j] = max(1, min(dp[j], dp[j+1]) - dungeon[i][j])
```

EDGE CASES:
==========
1. **Single cell**: max(1, 1 - dungeon[0][0])
2. **All positive**: Initial health = 1 (optimal)
3. **All negative**: Sum all negatives + 1
4. **Mixed values**: Complex optimization needed

PATH RECONSTRUCTION:
===================
Track optimal choices during DP:
```python
if dp[i][j+1] <= dp[i+1][j]:
    parent[i][j] = (i, j+1)  # go right
else:
    parent[i][j] = (i+1, j)  # go down
```

MATHEMATICAL PROPERTIES:
=======================
- **Monotonicity**: More negative values → higher initial health
- **Optimality**: Greedy choice at each step leads to global optimum
- **Constraint propagation**: Health requirements propagate backward

APPLICATIONS:
============
1. **Game Design**: RPG character health planning
2. **Resource Management**: Minimum starting resources
3. **Risk Assessment**: Worst-case scenario planning
4. **Optimization**: Backward constraint propagation

VARIANTS TO PRACTICE:
====================
- Minimum Path Sum (64) - optimize cost instead of health
- Maximum Path Sum - maximize instead of minimize
- Unique Paths (62/63) - count paths instead of optimize
- Cherry Pickup (741) - bidirectional path optimization

INTERVIEW TIPS:
==============
1. **Recognize backward nature**: Key insight for approach
2. **Explain health constraint**: Why health ≥ 1 matters
3. **Show recurrence logic**: How to propagate requirements
4. **Handle edge cases**: Single cell, all positive/negative
5. **Demonstrate optimization**: 2D → 1D space reduction
6. **Trace example**: Step-by-step backward computation
7. **Discuss alternatives**: Why forward DP doesn't work
8. **Path reconstruction**: How to find optimal route
9. **Real applications**: Game design, resource planning
10. **Complexity analysis**: Why O(m×n) is necessary

COMMON MISTAKES:
===============
1. **Forward DP attempt**: Trying to work from start to end
2. **Ignoring health constraint**: Allowing health < 1
3. **Wrong base case**: Incorrect destination initialization
4. **Boundary errors**: Mishandling last row/column
5. **Integer overflow**: Not considering large negative sums

This problem beautifully demonstrates how constraints can fundamentally 
change the approach needed, requiring backward instead of forward thinking.
"""
