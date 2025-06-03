"""
Dynamic Programming - Interview Problems
This module implements popular DP interview problems including decode ways,
arithmetic slices, dice rolls, frog jump, and ninja training problems.
"""

from typing import List, Dict, Tuple, Optional
import time
from functools import lru_cache

# ==================== DECODE WAYS PROBLEMS ====================

class DecodeWays:
    """
    Decode Ways Problems
    
    Problems involving decoding strings with various constraints.
    """
    
    def num_decodings(self, s: str) -> int:
        """
        Decode Ways I - Count ways to decode string
        
        LeetCode 91 - Decode Ways
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            s: String of digits
        
        Returns:
            Number of ways to decode the string
        """
        if not s or s[0] == '0':
            return 0
        
        n = len(s)
        prev2 = 1  # dp[i-2]
        prev1 = 1  # dp[i-1]
        
        for i in range(1, n):
            current = 0
            
            # Single digit decode
            if s[i] != '0':
                current += prev1
            
            # Two digit decode
            two_digit = int(s[i-1:i+1])
            if 10 <= two_digit <= 26:
                current += prev2
            
            prev2 = prev1
            prev1 = current
        
        return prev1
    
    def num_decodings_with_star(self, s: str) -> int:
        """
        Decode Ways II - String with '*' wildcard
        
        LeetCode 639 - Decode Ways II
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            s: String of digits and '*'
        
        Returns:
            Number of ways to decode with wildcards
        """
        MOD = 10**9 + 7
        
        def ways_single(char: str) -> int:
            """Ways to decode single character"""
            if char == '*':
                return 9  # 1-9
            elif char == '0':
                return 0
            else:
                return 1
        
        def ways_double(first: str, second: str) -> int:
            """Ways to decode two characters"""
            if first == '*' and second == '*':
                return 15  # 11-19, 21-26
            elif first == '*':
                if second <= '6':
                    return 2  # 1X, 2X
                else:
                    return 1  # 1X only
            elif second == '*':
                if first == '1':
                    return 9  # 11-19
                elif first == '2':
                    return 6  # 21-26
                else:
                    return 0
            else:
                two_digit = int(first + second)
                return 1 if 10 <= two_digit <= 26 else 0
        
        n = len(s)
        if n == 0:
            return 0
        
        prev2 = 1
        prev1 = ways_single(s[0])
        
        for i in range(1, n):
            current = (ways_single(s[i]) * prev1) % MOD
            current = (current + ways_double(s[i-1], s[i]) * prev2) % MOD
            
            prev2 = prev1
            prev1 = current
        
        return prev1
    
    def num_decodings_with_reconstruction(self, s: str) -> List[str]:
        """
        Find all possible decoded strings
        
        Args:
            s: String of digits
        
        Returns:
            List of all possible decodings
        """
        if not s:
            return []
        
        result = []
        
        def backtrack(index: int, current: List[str]):
            if index == len(s):
                result.append(''.join(chr(ord('A') + int(c) - 1) for c in current))
                return
            
            # Single digit
            if s[index] != '0':
                current.append(s[index])
                backtrack(index + 1, current)
                current.pop()
            
            # Two digits
            if index + 1 < len(s):
                two_digit = int(s[index:index+2])
                if 10 <= two_digit <= 26:
                    current.append(str(two_digit))
                    backtrack(index + 2, current)
                    current.pop()
        
        backtrack(0, [])
        return result
    
    def num_decodings_with_cost(self, s: str, costs: List[int]) -> int:
        """
        Minimum cost to decode string
        
        Args:
            s: String of digits
            costs: Cost array where costs[i] is cost of decoding to letter i+1
        
        Returns:
            Minimum cost to decode string
        """
        if not s or s[0] == '0':
            return float('inf')
        
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        dp[1] = costs[int(s[0]) - 1]
        
        for i in range(2, n + 1):
            # Single digit
            if s[i-1] != '0':
                digit = int(s[i-1])
                dp[i] = min(dp[i], dp[i-1] + costs[digit - 1])
            
            # Two digits
            two_digit = int(s[i-2:i])
            if 10 <= two_digit <= 26:
                dp[i] = min(dp[i], dp[i-2] + costs[two_digit - 1])
        
        return dp[n] if dp[n] != float('inf') else -1

# ==================== ARITHMETIC SLICES ====================

class ArithmeticSlices:
    """
    Arithmetic Slices Problems
    
    Problems involving arithmetic progressions in arrays.
    """
    
    def number_of_arithmetic_slices(self, nums: List[int]) -> int:
        """
        Count arithmetic slices in array
        
        LeetCode 413 - Arithmetic Slices
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            nums: Array of integers
        
        Returns:
            Number of arithmetic slices
        """
        if len(nums) < 3:
            return 0
        
        count = 0
        current = 0
        
        for i in range(2, len(nums)):
            if nums[i] - nums[i-1] == nums[i-1] - nums[i-2]:
                current += 1
                count += current
            else:
                current = 0
        
        return count
    
    def number_of_arithmetic_slices_ii(self, nums: List[int]) -> int:
        """
        Count arithmetic subsequences (not necessarily contiguous)
        
        LeetCode 446 - Arithmetic Slices II - Subsequence
        
        Time Complexity: O(n²)
        Space Complexity: O(n²)
        
        Args:
            nums: Array of integers
        
        Returns:
            Number of arithmetic subsequences
        """
        n = len(nums)
        if n < 3:
            return 0
        
        # dp[i][d] = number of arithmetic subsequences ending at i with difference d
        dp = [defaultdict(int) for _ in range(n)]
        result = 0
        
        for i in range(1, n):
            for j in range(i):
                diff = nums[i] - nums[j]
                
                # Add subsequences ending at j with difference diff
                dp[i][diff] += dp[j][diff] + 1
                
                # Count valid subsequences (length >= 3)
                result += dp[j][diff]
        
        return result
    
    def longest_arithmetic_slice(self, nums: List[int]) -> int:
        """
        Find length of longest arithmetic slice
        
        Args:
            nums: Array of integers
        
        Returns:
            Length of longest arithmetic slice
        """
        if len(nums) < 2:
            return len(nums)
        
        max_length = 2
        current_length = 2
        
        for i in range(2, len(nums)):
            if nums[i] - nums[i-1] == nums[i-1] - nums[i-2]:
                current_length += 1
                max_length = max(max_length, current_length)
            else:
                current_length = 2
        
        return max_length
    
    def arithmetic_slices_with_sum(self, nums: List[int], target_sum: int) -> int:
        """
        Count arithmetic slices with specific sum
        
        Args:
            nums: Array of integers
            target_sum: Target sum for slices
        
        Returns:
            Number of arithmetic slices with target sum
        """
        n = len(nums)
        count = 0
        
        for i in range(n):
            for j in range(i + 2, n + 1):  # At least 3 elements
                subarray = nums[i:j]
                if len(subarray) >= 3 and self._is_arithmetic(subarray):
                    if sum(subarray) == target_sum:
                        count += 1
        
        return count
    
    def _is_arithmetic(self, arr: List[int]) -> bool:
        """Check if array is arithmetic progression"""
        if len(arr) < 2:
            return True
        
        diff = arr[1] - arr[0]
        for i in range(2, len(arr)):
            if arr[i] - arr[i-1] != diff:
                return False
        
        return True

# ==================== DICE ROLL PROBLEMS ====================

class DiceRollProblems:
    """
    Dice Roll DP Problems
    
    Problems involving dice rolls and target sums.
    """
    
    def num_rolls_to_target(self, n: int, k: int, target: int) -> int:
        """
        Number of ways to roll dice to get target sum
        
        LeetCode 1155 - Number of Dice Rolls With Target Sum
        
        Time Complexity: O(n * k * target)
        Space Complexity: O(target)
        
        Args:
            n: Number of dice
            k: Number of faces per die
            target: Target sum
        
        Returns:
            Number of ways to achieve target sum
        """
        MOD = 10**9 + 7
        
        # dp[i] = ways to get sum i
        dp = [0] * (target + 1)
        dp[0] = 1
        
        for dice in range(n):
            new_dp = [0] * (target + 1)
            
            for current_sum in range(target + 1):
                if dp[current_sum] > 0:
                    for face in range(1, k + 1):
                        if current_sum + face <= target:
                            new_dp[current_sum + face] = (
                                new_dp[current_sum + face] + dp[current_sum]
                            ) % MOD
            
            dp = new_dp
        
        return dp[target]
    
    def num_rolls_to_target_2d(self, n: int, k: int, target: int) -> int:
        """
        2D DP approach for dice rolls
        
        Args:
            n: Number of dice
            k: Number of faces per die
            target: Target sum
        
        Returns:
            Number of ways to achieve target sum
        """
        MOD = 10**9 + 7
        
        # dp[i][j] = ways to get sum j using i dice
        dp = [[0] * (target + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        
        for i in range(1, n + 1):
            for j in range(i, min(i * k, target) + 1):
                for face in range(1, min(k, j) + 1):
                    dp[i][j] = (dp[i][j] + dp[i-1][j-face]) % MOD
        
        return dp[n][target]
    
    def probability_of_target(self, n: int, k: int, target: int) -> float:
        """
        Probability of getting target sum with n dice
        
        Args:
            n: Number of dice
            k: Number of faces per die
            target: Target sum
        
        Returns:
            Probability of getting target sum
        """
        ways = self.num_rolls_to_target(n, k, target)
        total_outcomes = k ** n
        
        return ways / total_outcomes if total_outcomes > 0 else 0.0
    
    def dice_roll_simulation(self, n: int, rollMax: List[int]) -> int:
        """
        Dice roll simulation with consecutive roll limits
        
        LeetCode 1223 - Dice Roll Simulation
        
        Args:
            n: Number of rolls
            rollMax: Maximum consecutive rolls for each face
        
        Returns:
            Number of valid sequences
        """
        MOD = 10**9 + 7
        k = len(rollMax)
        
        # dp[i][j][c] = ways after i rolls, last face j, consecutive count c
        memo = {}
        
        def dp(rolls_left: int, last_face: int, consecutive: int) -> int:
            if rolls_left == 0:
                return 1
            
            if (rolls_left, last_face, consecutive) in memo:
                return memo[(rolls_left, last_face, consecutive)]
            
            result = 0
            
            for face in range(k):
                if face == last_face:
                    if consecutive < rollMax[face]:
                        result = (result + dp(rolls_left - 1, face, consecutive + 1)) % MOD
                else:
                    result = (result + dp(rolls_left - 1, face, 1)) % MOD
            
            memo[(rolls_left, last_face, consecutive)] = result
            return result
        
        return dp(n, -1, 0)

# ==================== FROG JUMP PROBLEMS ====================

class FrogJumpProblems:
    """
    Frog Jump DP Problems
    
    Various frog jumping problems with different constraints.
    """
    
    def min_cost_climbing_stairs(self, cost: List[int]) -> int:
        """
        Minimum cost climbing stairs (frog with cost)
        
        LeetCode 746 - Min Cost Climbing Stairs
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            cost: Cost array for each step
        
        Returns:
            Minimum cost to reach the top
        """
        n = len(cost)
        
        prev2 = 0
        prev1 = 0
        
        for i in range(2, n + 1):
            current = min(prev1 + cost[i-1], prev2 + cost[i-2])
            prev2 = prev1
            prev1 = current
        
        return prev1
    
    def frog_jump_k_steps(self, stones: List[int], k: int) -> int:
        """
        Frog can jump 1 to k steps, minimize energy
        
        Args:
            stones: Array representing stone heights
            k: Maximum jump distance
        
        Returns:
            Minimum energy to reach last stone
        """
        n = len(stones)
        if n <= 1:
            return 0
        
        dp = [float('inf')] * n
        dp[0] = 0
        
        for i in range(1, n):
            for j in range(max(0, i - k), i):
                energy = abs(stones[i] - stones[j])
                dp[i] = min(dp[i], dp[j] + energy)
        
        return dp[n - 1]
    
    def frog_jump_with_positions(self, stones: List[int]) -> bool:
        """
        Frog jump across river (variable jump sizes)
        
        LeetCode 403 - Frog Jump
        
        Time Complexity: O(n²)
        Space Complexity: O(n²)
        
        Args:
            stones: Array of stone positions
        
        Returns:
            True if frog can cross the river
        """
        if not stones or stones[1] != 1:
            return False
        
        stone_set = set(stones)
        memo = {}
        
        def can_jump(position: int, k: int) -> bool:
            if position == stones[-1]:
                return True
            
            if (position, k) in memo:
                return memo[(position, k)]
            
            result = False
            
            for next_k in [k - 1, k, k + 1]:
                if next_k > 0:
                    next_position = position + next_k
                    if next_position in stone_set:
                        if can_jump(next_position, next_k):
                            result = True
                            break
            
            memo[(position, k)] = result
            return result
        
        return can_jump(1, 1)
    
    def count_frog_jump_ways(self, n: int, k: int) -> int:
        """
        Count ways for frog to jump n steps with max k step size
        
        Args:
            n: Number of steps
            k: Maximum step size
        
        Returns:
            Number of ways to reach step n
        """
        if n <= 0:
            return 0 if n < 0 else 1
        
        dp = [0] * (n + 1)
        dp[0] = 1
        
        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                dp[i] += dp[i - j]
        
        return dp[n]
    
    def frog_jump_with_obstacles(self, stones: List[int], obstacles: List[int]) -> int:
        """
        Frog jump with obstacles (can't land on obstacle stones)
        
        Args:
            stones: Available stone positions
            obstacles: Obstacle positions (can't land here)
        
        Returns:
            Minimum jumps to reach last stone, -1 if impossible
        """
        obstacle_set = set(obstacles)
        valid_stones = [stone for stone in stones if stone not in obstacle_set]
        
        if not valid_stones or len(valid_stones) < 2:
            return -1 if len(valid_stones) < 2 else 0
        
        n = len(valid_stones)
        dp = [float('inf')] * n
        dp[0] = 0
        
        for i in range(1, n):
            for j in range(i):
                jump_distance = valid_stones[i] - valid_stones[j]
                if jump_distance <= 2:  # Frog can jump 1 or 2 units
                    dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n - 1] if dp[n - 1] != float('inf') else -1

# ==================== NINJA TRAINING PROBLEMS ====================

class NinjaTrainingProblems:
    """
    Ninja Training DP Problems
    
    Problems from Striver's DP series and similar training scenarios.
    """
    
    def ninja_training(self, points: List[List[int]]) -> int:
        """
        Ninja Training - Maximum points without same activity on consecutive days
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            points: 2D array where points[i][j] is points for activity j on day i
        
        Returns:
            Maximum points ninja can earn
        """
        if not points:
            return 0
        
        n = len(points)
        activities = len(points[0])
        
        # dp[i] = max points on day i for activity i
        prev = points[0][:]
        
        for day in range(1, n):
            current = [0] * activities
            
            for activity in range(activities):
                for prev_activity in range(activities):
                    if activity != prev_activity:
                        current[activity] = max(current[activity], 
                                               prev[prev_activity] + points[day][activity])
            
            prev = current
        
        return max(prev)
    
    def ninja_training_optimized(self, points: List[List[int]]) -> int:
        """
        Space-optimized version for 3 activities
        
        Args:
            points: 2D array with 3 activities per day
        
        Returns:
            Maximum points for ninja training
        """
        if not points:
            return 0
        
        # For 3 activities, we can track explicitly
        prev0, prev1, prev2 = points[0][0], points[0][1], points[0][2]
        
        for day in range(1, len(points)):
            curr0 = points[day][0] + max(prev1, prev2)
            curr1 = points[day][1] + max(prev0, prev2)
            curr2 = points[day][2] + max(prev0, prev1)
            
            prev0, prev1, prev2 = curr0, curr1, curr2
        
        return max(prev0, prev1, prev2)
    
    def house_robber_variant(self, houses: List[int]) -> int:
        """
        House Robber - Can't rob adjacent houses
        
        LeetCode 198 - House Robber
        
        Args:
            houses: Array of money in each house
        
        Returns:
            Maximum money that can be robbed
        """
        if not houses:
            return 0
        if len(houses) == 1:
            return houses[0]
        
        prev2 = houses[0]
        prev1 = max(houses[0], houses[1])
        
        for i in range(2, len(houses)):
            current = max(prev1, prev2 + houses[i])
            prev2 = prev1
            prev1 = current
        
        return prev1
    
    def ninja_and_friends(self, grid: List[List[int]]) -> int:
        """
        Two ninjas collecting maximum cherries/points
        
        Args:
            grid: 2D grid with points/cherries
        
        Returns:
            Maximum points both ninjas can collect
        """
        if not grid or not grid[0]:
            return 0
        
        rows, cols = len(grid), len(grid[0])
        
        # dp[r][c1][c2] = max points when ninja1 at (r,c1) and ninja2 at (r,c2)
        memo = {}
        
        def dp(row: int, col1: int, col2: int) -> int:
            # Base cases
            if (row >= rows or col1 < 0 or col1 >= cols or 
                col2 < 0 or col2 >= cols):
                return float('-inf')
            
            if row == rows - 1:
                if col1 == col2:
                    return grid[row][col1]
                else:
                    return grid[row][col1] + grid[row][col2]
            
            if (row, col1, col2) in memo:
                return memo[(row, col1, col2)]
            
            # Current points
            points = grid[row][col1]
            if col1 != col2:
                points += grid[row][col2]
            
            # Try all combinations of moves
            max_next = float('-inf')
            for dc1 in [-1, 0, 1]:
                for dc2 in [-1, 0, 1]:
                    next_points = dp(row + 1, col1 + dc1, col2 + dc2)
                    max_next = max(max_next, next_points)
            
            result = points + max_next
            memo[(row, col1, col2)] = result
            return result
        
        return dp(0, 0, cols - 1)
    
    def maximum_path_sum_triangle(self, triangle: List[List[int]]) -> int:
        """
        Maximum path sum in triangle (ninja climbing mountain)
        
        LeetCode 120 - Triangle
        
        Args:
            triangle: Triangular array of numbers
        
        Returns:
            Maximum path sum from top to bottom
        """
        if not triangle:
            return 0
        
        # Bottom-up approach
        dp = triangle[-1][:]
        
        for row in range(len(triangle) - 2, -1, -1):
            for col in range(len(triangle[row])):
                dp[col] = triangle[row][col] + min(dp[col], dp[col + 1])
        
        return dp[0]

# ==================== PERFORMANCE ANALYSIS ====================

def performance_comparison():
    """Compare performance of different interview DP problems"""
    print("=== Interview DP Performance Analysis ===\n")
    
    import random
    from collections import defaultdict
    
    # Test decode ways
    decode = DecodeWays()
    test_strings = ["12", "226", "06", "111111111111"]
    
    print("Decode Ways Performance:")
    for s in test_strings:
        start_time = time.time()
        result = decode.num_decodings(s)
        time_taken = time.time() - start_time
        print(f"  String '{s}': {result} ways ({time_taken:.6f}s)")
    
    # Test arithmetic slices
    arithmetic = ArithmeticSlices()
    test_arrays = [[1, 2, 3, 4], [1, 3, 5, 7, 9], [1, 2, 3, 4, 7, 8, 9]]
    
    print("\nArithmetic Slices Performance:")
    for arr in test_arrays:
        start_time = time.time()
        result = arithmetic.number_of_arithmetic_slices(arr)
        time_taken = time.time() - start_time
        print(f"  Array {arr}: {result} slices ({time_taken:.6f}s)")
    
    # Test dice rolls
    dice = DiceRollProblems()
    test_cases = [(2, 6, 7), (3, 6, 10), (4, 6, 15)]
    
    print("\nDice Roll Performance:")
    for n, k, target in test_cases:
        start_time = time.time()
        result = dice.num_rolls_to_target(n, k, target)
        time_taken = time.time() - start_time
        print(f"  {n} dice, {k} faces, target {target}: {result} ways ({time_taken:.6f}s)")

# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Interview DP Problems Demo ===\n")
    
    # Decode Ways
    print("1. Decode Ways Problems:")
    decode = DecodeWays()
    
    test_string = "226"
    ways = decode.num_decodings(test_string)
    print(f"  Ways to decode '{test_string}': {ways}")
    
    star_string = "1*"
    star_ways = decode.num_decodings_with_star(star_string)
    print(f"  Ways to decode '{star_string}' with wildcards: {star_ways}")
    
    all_decodings = decode.num_decodings_with_reconstruction("123")
    print(f"  All decodings of '123': {all_decodings}")
    
    costs = [1] * 26  # Equal cost for all letters
    min_cost = decode.num_decodings_with_cost("12", costs)
    print(f"  Minimum cost to decode '12': {min_cost}")
    print()
    
    # Arithmetic Slices
    print("2. Arithmetic Slices Problems:")
    arithmetic = ArithmeticSlices()
    
    nums = [1, 2, 3, 4]
    slices = arithmetic.number_of_arithmetic_slices(nums)
    print(f"  Arithmetic slices in {nums}: {slices}")
    
    # Import defaultdict for arithmetic slices II
    from collections import defaultdict
    subsequences = arithmetic.number_of_arithmetic_slices_ii(nums)
    print(f"  Arithmetic subsequences in {nums}: {subsequences}")
    
    longest = arithmetic.longest_arithmetic_slice([1, 3, 5, 7, 9, 11])
    print(f"  Longest arithmetic slice: {longest}")
    
    with_sum = arithmetic.arithmetic_slices_with_sum([2, 4, 6, 8], 20)
    print(f"  Arithmetic slices with sum 20: {with_sum}")
    print()
    
    # Dice Roll Problems
    print("3. Dice Roll Problems:")
    dice = DiceRollProblems()
    
    ways_2d = dice.num_rolls_to_target(2, 6, 7)
    ways_1d = dice.num_rolls_to_target_2d(2, 6, 7)
    print(f"  Ways to get sum 7 with 2 dice (6 faces): {ways_2d}")
    print(f"  Same result with 2D DP: {ways_1d}")
    
    probability = dice.probability_of_target(2, 6, 7)
    print(f"  Probability of sum 7: {probability:.4f}")
    
    rollMax = [1, 1, 2, 2, 2, 3]
    simulation = dice.dice_roll_simulation(2, rollMax)
    print(f"  Valid sequences with roll limits: {simulation}")
    print()
    
    # Frog Jump Problems
    print("4. Frog Jump Problems:")
    frog = FrogJumpProblems()
    
    cost = [10, 15, 20]
    min_cost = frog.min_cost_climbing_stairs(cost)
    print(f"  Min cost climbing stairs {cost}: {min_cost}")
    
    stones_height = [10, 30, 40, 20]
    k = 2
    min_energy = frog.frog_jump_k_steps(stones_height, k)
    print(f"  Min energy for frog jump (k={k}): {min_energy}")
    
    river_stones = [0, 1, 3, 5, 6, 8, 12, 17]
    can_cross = frog.frog_jump_with_positions(river_stones)
    print(f"  Can frog cross river: {can_cross}")
    
    jump_ways = frog.count_frog_jump_ways(4, 2)
    print(f"  Ways to jump 4 steps (max 2): {jump_ways}")
    print()
    
    # Ninja Training Problems
    print("5. Ninja Training Problems:")
    ninja = NinjaTrainingProblems()
    
    training_points = [
        [10, 40, 70],
        [20, 50, 80],
        [30, 60, 90]
    ]
    max_points = ninja.ninja_training(training_points)
    max_points_opt = ninja.ninja_training_optimized(training_points)
    print(f"  Ninja training max points: {max_points}")
    print(f"  Optimized version: {max_points_opt}")
    
    houses = [2, 7, 9, 3, 1]
    max_rob = ninja.house_robber_variant(houses)
    print(f"  House robber max money: {max_rob}")
    
    cherry_grid = [
        [3, 1, 1],
        [2, 5, 1],
        [1, 5, 5],
        [2, 1, 1]
    ]
    max_cherries = ninja.ninja_and_friends(cherry_grid)
    print(f"  Two ninjas max cherries: {max_cherries}")
    
    triangle = [
        [2],
        [3, 4],
        [6, 5, 7],
        [4, 1, 8, 3]
    ]
    max_path = ninja.maximum_path_sum_triangle(triangle)
    print(f"  Triangle max path sum: {max_path}")
    print()
    
    # Performance comparison
    performance_comparison()
    
    # Pattern Recognition Guide
    print("\n=== Interview DP Pattern Recognition ===")
    print("Common Interview DP Categories:")
    print("  1. String Manipulation: Decode ways, edit distance")
    print("  2. Array Processing: Arithmetic slices, subarray problems")
    print("  3. Counting Problems: Dice rolls, ways to reach target")
    print("  4. Path Problems: Frog jump, minimum cost paths")
    print("  5. Game Theory: Optimal strategy, competitive scenarios")
    
    print("\nDP State Design Patterns:")
    print("  1. Linear DP: dp[i] depends on dp[i-1], dp[i-2], etc.")
    print("  2. 2D DP: dp[i][j] for two-dimensional state space")
    print("  3. State Compression: Reduce space from O(n) to O(1)")
    print("  4. Memoization: Top-down approach with function caching")
    
    print("\nOptimization Techniques:")
    print("  1. Space Optimization: Rolling arrays, state compression")
    print("  2. Mathematical Insights: Closed-form solutions when possible")
    print("  3. Early Termination: Pruning impossible states")
    print("  4. Preprocessing: Compute auxiliary information")
    
    print("\nInterview Tips:")
    print("  1. Start with brute force, then optimize")
    print("  2. Identify overlapping subproblems")
    print("  3. Define state clearly and transitions")
    print("  4. Consider both top-down and bottom-up")
    print("  5. Analyze time and space complexity")
    print("  6. Test with edge cases")
    
    print("\nCommon Mistakes:")
    print("  1. Incorrect base case handling")
    print("  2. Off-by-one errors in indexing")
    print("  3. Not considering all possible transitions")
    print("  4. Missing edge cases (empty input, single element)")
    print("  5. Incorrect state definition")
    
    print("\n=== Demo Complete ===") 