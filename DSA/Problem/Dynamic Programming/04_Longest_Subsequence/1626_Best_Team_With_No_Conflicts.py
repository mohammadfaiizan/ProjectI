"""
LeetCode 1626: Best Team With No Conflicts
Difficulty: Medium
Category: Longest Subsequence Problems (Weighted LIS variant)

PROBLEM DESCRIPTION:
===================
You are the manager of a basketball team. For the upcoming tournament, you want to choose the team 
with the highest overall score. The score of the team is the sum of scores of all the players in the team.

However, the basketball team is not allowed to have conflicts. A conflict exists if a younger player 
has a strictly higher score than an older player. A team is valid if there are no conflicts.

Given two arrays scores and ages, where each scores[i] and ages[i] represents the score and age of 
the ith player, respectively, return the maximum possible score of a valid team.

Example 1:
Input: scores = [1,3,5,10,15], ages = [1,2,3,4,5]
Output: 34
Explanation: You can choose all the players.

Example 2:
Input: scores = [4,5,6,5], ages = [2,1,2,1]
Output: 16
Explanation: Choose the last 3 players. The maximum score is 4 + 5 + 6 + 5 = 20. However, you cannot 
choose the player with score 5 and age 1 because he conflicts with the older player with score 6 and age 2.

Example 3:
Input: scores = [1,2,3,5], ages = [8,9,10,1]
Output: 6
Explanation: Choose players with ages 8, 9, 10. The maximum score is 1 + 2 + 3 = 6.

Constraints:
- 1 <= scores.length == ages.length <= 1000
- 1 <= scores[i] <= 10^6
- 1 <= ages[i] <= 1000
"""

def best_team_score_brute_force(scores, ages):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible team combinations and check for conflicts.
    
    Time Complexity: O(2^n * n^2) - 2^n subsets, n^2 to check conflicts
    Space Complexity: O(n) - recursion stack
    """
    n = len(scores)
    players = list(zip(ages, scores))
    
    def has_conflict(team):
        """Check if team has any conflicts"""
        team.sort()  # Sort by age
        for i in range(len(team)):
            for j in range(i + 1, len(team)):
                age1, score1 = team[i]
                age2, score2 = team[j]
                # Younger player (age1 < age2) has higher score
                if age1 < age2 and score1 > score2:
                    return True
        return False
    
    def generate_teams(index, current_team):
        if index >= n:
            if not has_conflict(current_team):
                return sum(score for _, score in current_team)
            return 0
        
        # Skip current player
        skip = generate_teams(index + 1, current_team)
        
        # Include current player
        current_team.append(players[index])
        include = generate_teams(index + 1, current_team)
        current_team.pop()
        
        return max(skip, include)
    
    return generate_teams(0, [])


def best_team_score_dp_weighted_lis(scores, ages):
    """
    DP APPROACH - WEIGHTED LIS:
    ===========================
    Sort by age, then by score, and find maximum weight increasing subsequence.
    
    Time Complexity: O(n^2) - nested loops after sorting
    Space Complexity: O(n) - DP array
    """
    n = len(scores)
    
    # Create players list and sort by age, then by score
    players = sorted(zip(ages, scores))
    
    # dp[i] = maximum score of valid team ending with player i
    dp = [0] * n
    
    for i in range(n):
        age_i, score_i = players[i]
        dp[i] = score_i  # At least include current player
        
        # Check all previous players
        for j in range(i):
            age_j, score_j = players[j]
            
            # No conflict if ages are equal or scores are non-decreasing
            if age_j <= age_i and score_j <= score_i:
                dp[i] = max(dp[i], dp[j] + score_i)
    
    return max(dp)


def best_team_score_memoization(scores, ages):
    """
    MEMOIZATION APPROACH:
    ====================
    Use recursive approach with memoization.
    
    Time Complexity: O(n^2) - memoized states
    Space Complexity: O(n^2) - memoization table
    """
    n = len(scores)
    players = sorted(zip(ages, scores))
    memo = {}
    
    def dp(index, last_score):
        if index >= n:
            return 0
        
        if (index, last_score) in memo:
            return memo[(index, last_score)]
        
        age, score = players[index]
        
        # Skip current player
        skip = dp(index + 1, last_score)
        
        # Include current player if no conflict
        include = 0
        if score >= last_score:
            include = score + dp(index + 1, score)
        
        result = max(skip, include)
        memo[(index, last_score)] = result
        return result
    
    return dp(0, 0)


def best_team_score_optimized_sorting(scores, ages):
    """
    OPTIMIZED SORTING STRATEGY:
    ===========================
    Use different sorting strategies for optimization.
    
    Time Complexity: O(n^2) - DP computation
    Space Complexity: O(n) - DP array
    """
    n = len(scores)
    
    # Sort by age first, then by score in ascending order
    # This ensures that for same age, we process lower scores first
    players = sorted(zip(ages, scores), key=lambda x: (x[0], x[1]))
    
    dp = [0] * n
    
    for i in range(n):
        age_i, score_i = players[i]
        dp[i] = score_i
        
        for j in range(i):
            age_j, score_j = players[j]
            
            # Since we sorted by age then score, we only need to check scores
            if score_j <= score_i:
                dp[i] = max(dp[i], dp[j] + score_i)
    
    return max(dp) if dp else 0


def best_team_score_with_team(scores, ages):
    """
    FIND ACTUAL BEST TEAM:
    ======================
    Return both maximum score and the actual team composition.
    
    Time Complexity: O(n^2) - DP + reconstruction
    Space Complexity: O(n) - DP array + parent tracking
    """
    n = len(scores)
    players = sorted((ages[i], scores[i], i) for i in range(n))
    
    dp = [0] * n
    parent = [-1] * n
    
    max_score = 0
    max_index = -1
    
    for i in range(n):
        age_i, score_i, orig_i = players[i]
        dp[i] = score_i
        
        for j in range(i):
            age_j, score_j, orig_j = players[j]
            
            if score_j <= score_i and dp[j] + score_i > dp[i]:
                dp[i] = dp[j] + score_i
                parent[i] = j
        
        if dp[i] > max_score:
            max_score = dp[i]
            max_index = i
    
    # Reconstruct team
    team_indices = []
    current = max_index
    
    while current != -1:
        _, _, orig_index = players[current]
        team_indices.append(orig_index)
        current = parent[current]
    
    team_indices.reverse()
    team = [(ages[i], scores[i]) for i in team_indices]
    
    return max_score, team


def best_team_score_segment_tree(scores, ages):
    """
    SEGMENT TREE APPROACH:
    =====================
    Use segment tree for range maximum queries optimization.
    
    Time Complexity: O(n log n) - sorting + segment tree
    Space Complexity: O(n) - segment tree
    """
    n = len(scores)
    
    # Sort by age, then by score
    players = sorted(zip(ages, scores))
    
    # Coordinate compression for scores
    all_scores = sorted(set(score for _, score in players))
    score_to_idx = {score: i for i, score in enumerate(all_scores)}
    
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
    
    seg_tree = SegmentTree(len(all_scores))
    max_score = 0
    
    for age, score in players:
        score_idx = score_to_idx[score]
        
        # Query maximum score for all scores <= current score
        prev_max = seg_tree.query(1, 0, len(all_scores) - 1, 0, score_idx)
        
        # Current team score
        current_score = prev_max + score
        max_score = max(max_score, current_score)
        
        # Update segment tree
        seg_tree.update(1, 0, len(all_scores) - 1, score_idx, current_score)
    
    return max_score


def best_team_score_analysis(scores, ages):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step DP computation and insights.
    
    Time Complexity: O(n^2) - DP computation
    Space Complexity: O(n) - DP array
    """
    n = len(scores)
    
    print(f"Original data:")
    for i in range(n):
        print(f"  Player {i}: age={ages[i]}, score={scores[i]}")
    
    # Sort players
    players = sorted(zip(ages, scores))
    print(f"\nSorted players (by age, then score):")
    for i, (age, score) in enumerate(players):
        print(f"  Player {i}: age={age}, score={score}")
    
    dp = [0] * n
    
    print(f"\nDP computation:")
    for i in range(n):
        age_i, score_i = players[i]
        dp[i] = score_i
        print(f"\nPlayer {i} (age={age_i}, score={score_i}):")
        print(f"  Initial dp[{i}] = {score_i}")
        
        for j in range(i):
            age_j, score_j = players[j]
            
            if score_j <= score_i:
                new_score = dp[j] + score_i
                if new_score > dp[i]:
                    print(f"  Can add to team ending at {j}: {dp[j]} + {score_i} = {new_score}")
                    dp[i] = new_score
        
        print(f"  Final dp[{i}] = {dp[i]}")
    
    result = max(dp)
    print(f"\nMaximum team score: {result}")
    
    return result


# Test cases
def test_best_team_score():
    """Test all implementations with various inputs"""
    test_cases = [
        ([1,3,5,10,15], [1,2,3,4,5], 34),
        ([4,5,6,5], [2,1,2,1], 16),
        ([1,2,3,5], [8,9,10,1], 6),
        ([1,1,1], [1,2,3], 3),
        ([5,4,3,2,1], [1,2,3,4,5], 5),
        ([9,2,8,8,2], [4,1,3,3,5], 26),
        ([1], [1], 1),
        ([10,5], [1,2], 15),
        ([319776,611683,835240,602289,430007,574,142444,858606,734364,896074], 
         [1,1,1,1,1,1,1,1,1,1], 5431037),
        ([596,277,897,622,500,299,34,536,797,32,264,948,645,537,83,589,770], 
         [18,52,60,79,72,28,81,33,96,15,18,5,17,96,57,72,72], 3287)
    ]
    
    print("Testing Best Team With No Conflicts Solutions:")
    print("=" * 70)
    
    for i, (scores, ages, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: scores = {scores[:5]}{'...' if len(scores) > 5 else ''}")
        print(f"ages = {ages[:5]}{'...' if len(ages) > 5 else ''}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for large inputs)
        if len(scores) <= 8:
            try:
                brute = best_team_score_brute_force(scores.copy(), ages.copy())
                print(f"Brute Force:      {brute:>7} {'✓' if brute == expected else '✗'}")
            except:
                print(f"Brute Force:      Timeout")
        
        dp_lis = best_team_score_dp_weighted_lis(scores.copy(), ages.copy())
        memo = best_team_score_memoization(scores.copy(), ages.copy())
        optimized = best_team_score_optimized_sorting(scores.copy(), ages.copy())
        
        print(f"DP (Weighted LIS):{dp_lis:>7} {'✓' if dp_lis == expected else '✗'}")
        print(f"Memoization:      {memo:>7} {'✓' if memo == expected else '✗'}")
        print(f"Optimized:        {optimized:>7} {'✓' if optimized == expected else '✗'}")
        
        if len(scores) <= 15:
            seg_tree = best_team_score_segment_tree(scores.copy(), ages.copy())
            print(f"Segment Tree:     {seg_tree:>7} {'✓' if seg_tree == expected else '✗'}")
        
        # Show actual team for small cases
        if len(scores) <= 8:
            max_score, team = best_team_score_with_team(scores.copy(), ages.copy())
            print(f"Best team: {team}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    best_team_score_analysis([4,5,6,5], [2,1,2,1])
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. SORTING STRATEGY: Sort by age first, then by score")
    print("2. WEIGHTED LIS: Find maximum weight increasing subsequence of scores")
    print("3. CONFLICT RULE: Younger player can't have higher score than older")
    print("4. DP STATE: dp[i] = max score of valid team ending with player i")
    
    print("\n" + "=" * 70)
    print("Algorithm Comparison:")
    print("Brute Force:      Try all team combinations")
    print("DP (Weighted LIS): Sort + weighted LIS on scores")
    print("Memoization:      Recursive with caching")
    print("Optimized:        Improved sorting strategy")
    print("Segment Tree:     Advanced range query optimization")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(2^n * n²), Space: O(n)")
    print("DP (Weighted LIS): Time: O(n²),      Space: O(n)")
    print("Memoization:      Time: O(n²),      Space: O(n²)")
    print("Optimized:        Time: O(n²),      Space: O(n)")
    print("Segment Tree:     Time: O(n log n), Space: O(n)")


if __name__ == "__main__":
    test_best_team_score()


"""
PATTERN RECOGNITION:
==================
This is a Weighted LIS problem with constraints:
- Each player has two attributes: age and score
- Constraint: No younger player can have higher score than older player
- Goal: Maximize total score (weight) of valid team
- Classic weighted LIS after proper sorting

KEY INSIGHT - SORTING STRATEGY:
==============================
**Critical sorting approach**:
1. Sort by age ascending (primary key)
2. Sort by score ascending (secondary key)

**Why this works**:
- After sorting, we only need to ensure scores are non-decreasing
- Age constraint is automatically satisfied by sorting
- Transforms 2D constraint into 1D LIS problem

MATHEMATICAL FORMULATION:
========================
**Conflict condition**: 
Player i conflicts with player j if age[i] < age[j] AND score[i] > score[j]

**After sorting by (age, score)**:
- Age constraint: automatically satisfied
- Score constraint: score[i] ≤ score[j] for valid extension

ALGORITHM APPROACHES:
====================

1. **Weighted LIS (Optimal)**: O(n²)
   - Sort by (age, score)
   - Apply weighted LIS on scores
   - dp[i] = max score ending at player i

2. **Memoization**: O(n²)
   - Recursive with (index, last_score) state
   - Natural top-down formulation

3. **Segment Tree**: O(n log n)
   - Advanced optimization using range queries
   - Coordinate compression + segment tree

4. **Brute Force**: O(2^n × n²)
   - Try all team combinations
   - Check conflicts for each team

STATE DEFINITION:
================
dp[i] = maximum total score of valid team ending with player i

RECURRENCE RELATION:
===================
```
dp[i] = max(score[i], max(dp[j] + score[i])) 
        for all j < i where score[j] ≤ score[i]
```

Base case: dp[i] = score[i] (team with only player i)

SORTING CORRECTNESS:
===================
**Why sort by (age, score)**:
1. Age ordering ensures no age conflicts
2. Score ordering within same age prevents internal conflicts
3. Allows simple 1D DP on scores

**Example**:
Before: [(age=2,score=4), (age=1,score=5), (age=2,score=6), (age=1,score=5)]
After:  [(age=1,score=5), (age=1,score=5), (age=2,score=4), (age=2,score=6)]

OPTIMIZATION TECHNIQUES:
=======================
1. **Segment Tree**: O(n log n) using coordinate compression
2. **Binary Search**: For LIS-style optimization (complex due to weights)
3. **Early Termination**: Skip impossible states
4. **Space Optimization**: Rolling DP (not beneficial here)

APPLICATIONS:
============
1. **Team Selection**: Sports team optimization
2. **Resource Allocation**: Age/skill constraints
3. **Hiring Problems**: Experience/salary constraints
4. **Scheduling**: Age-priority systems

VARIANTS TO PRACTICE:
====================
- Russian Doll Envelopes (354) - 2D LIS without weights
- Largest Divisible Subset (368) - LIS with divisibility
- Maximum Height by Stacking Cuboids (1691) - 3D version
- Weighted Job Scheduling - interval version

EDGE CASES:
==========
1. **Same ages**: All valid combinations possible
2. **All conflicts**: Return single highest score
3. **No conflicts**: Return sum of all scores
4. **Single player**: Return that player's score

INTERVIEW TIPS:
==============
1. **Recognize as weighted LIS**: Key insight
2. **Explain sorting strategy**: Why (age, score) ordering
3. **Show constraint transformation**: 2D → 1D problem
4. **Trace DP computation**: Step-by-step example
5. **Handle reconstruction**: Build actual team
6. **Discuss optimizations**: Segment tree approach
7. **Edge cases**: Same ages, all conflicts
8. **Mathematical proof**: Why sorting preserves optimality
9. **Complexity analysis**: Why O(n²) is necessary for basic DP
10. **Real applications**: Team sports, hiring, resource allocation
"""
