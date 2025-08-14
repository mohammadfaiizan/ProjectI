"""
Recursion Interview Problems
============================

Topics: Common interview questions, problem-solving patterns, optimization tricks
Companies: Google, Amazon, Microsoft, Facebook, Apple, Netflix, Uber, LinkedIn
Difficulty: Easy to Hard
Time Complexity: Varies by problem and optimization
Space Complexity: O(h) to O(n) depending on recursion depth
"""

from typing import List, Dict, Optional, Tuple, Set
import bisect

class RecursionInterviewProblems:
    
    def __init__(self):
        """Initialize with tracking and memoization support"""
        self.call_count = 0
        self.memo = {}
    
    # ==========================================
    # 1. CLASSIC INTERVIEW PROBLEMS
    # ==========================================
    
    def climbing_stairs(self, n: int) -> int:
        """
        Climbing Stairs: How many ways to climb n stairs (1 or 2 steps at a time)
        
        Company: Amazon, Facebook, Google
        Difficulty: Easy
        
        Recurrence: f(n) = f(n-1) + f(n-2)
        Time: O(n) with memoization, Space: O(n)
        """
        if n in self.memo:
            return self.memo[n]
        
        if n <= 2:
            return n
        
        result = self.climbing_stairs(n - 1) + self.climbing_stairs(n - 2)
        self.memo[n] = result
        return result
    
    def unique_paths(self, m: int, n: int, row: int = 0, col: int = 0) -> int:
        """
        Unique Paths: Count paths from top-left to bottom-right in m×n grid
        
        Company: Microsoft, Amazon
        Difficulty: Medium
        
        Time: O(m*n), Space: O(m*n)
        """
        # Base cases
        if row >= m or col >= n:
            return 0
        if row == m - 1 and col == n - 1:
            return 1
        
        # Check memo
        if (row, col) in self.memo:
            return self.memo[(row, col)]
        
        # Move right or down
        paths = self.unique_paths(m, n, row, col + 1) + self.unique_paths(m, n, row + 1, col)
        self.memo[(row, col)] = paths
        return paths
    
    def coin_change_ways(self, amount: int, coins: List[int], index: int = 0) -> int:
        """
        Coin Change: Count ways to make amount using given coins
        
        Company: Google, Facebook
        Difficulty: Medium
        
        Time: O(amount * len(coins)), Space: O(amount * len(coins))
        """
        # Base cases
        if amount == 0:
            return 1  # One way: use no coins
        if amount < 0 or index >= len(coins):
            return 0  # No way
        
        # Check memo
        if (amount, index) in self.memo:
            return self.memo[(amount, index)]
        
        # Include current coin or exclude it
        include = self.coin_change_ways(amount - coins[index], coins, index)  # Can reuse same coin
        exclude = self.coin_change_ways(amount, coins, index + 1)
        
        result = include + exclude
        self.memo[(amount, index)] = result
        return result
    
    def house_robber(self, houses: List[int], index: int = 0) -> int:
        """
        House Robber: Rob houses to maximize money, can't rob adjacent houses
        
        Company: Amazon, Microsoft
        Difficulty: Medium
        
        Time: O(n), Space: O(n)
        """
        # Base cases
        if index >= len(houses):
            return 0
        
        # Check memo
        if index in self.memo:
            return self.memo[index]
        
        # Rob current house or skip it
        rob_current = houses[index] + self.house_robber(houses, index + 2)
        skip_current = self.house_robber(houses, index + 1)
        
        result = max(rob_current, skip_current)
        self.memo[index] = result
        return result
    
    def decode_ways(self, s: str, index: int = 0) -> int:
        """
        Decode Ways: Count ways to decode a numeric string to letters
        
        Company: Facebook, Microsoft
        Difficulty: Medium
        
        A=1, B=2, ..., Z=26
        Time: O(n), Space: O(n)
        """
        # Base cases
        if index >= len(s):
            return 1  # Empty string has one way
        if s[index] == '0':
            return 0  # Can't decode leading zero
        
        # Check memo
        if index in self.memo:
            return self.memo[index]
        
        ways = 0
        
        # Decode single digit (1-9)
        ways += self.decode_ways(s, index + 1)
        
        # Decode two digits (10-26)
        if index + 1 < len(s):
            two_digit = int(s[index:index + 2])
            if 10 <= two_digit <= 26:
                ways += self.decode_ways(s, index + 2)
        
        self.memo[index] = ways
        return ways
    
    # ==========================================
    # 2. ARRAY AND STRING PROBLEMS
    # ==========================================
    
    def longest_increasing_subsequence(self, nums: List[int], index: int = 0, prev: int = float('-inf')) -> int:
        """
        Longest Increasing Subsequence using recursion
        
        Company: Google, Amazon
        Difficulty: Medium
        
        Time: O(2^n) without memo, Space: O(n^2) with memo
        """
        # Base case
        if index >= len(nums):
            return 0
        
        # Check memo (using string key for complex state)
        key = f"{index}_{prev}"
        if key in self.memo:
            return self.memo[key]
        
        # Don't include current element
        exclude = self.longest_increasing_subsequence(nums, index + 1, prev)
        
        # Include current element if it's greater than previous
        include = 0
        if nums[index] > prev:
            include = 1 + self.longest_increasing_subsequence(nums, index + 1, nums[index])
        
        result = max(include, exclude)
        self.memo[key] = result
        return result
    
    def word_break_all_ways(self, s: str, word_dict: List[str], start: int = 0) -> List[str]:
        """
        Word Break II: Return all possible sentences
        
        Company: Google, Amazon
        Difficulty: Hard
        
        Time: O(2^n) worst case, Space: O(2^n)
        """
        # Base case
        if start >= len(s):
            return [""]
        
        # Check memo
        if start in self.memo:
            return self.memo[start]
        
        result = []
        
        # Try all words that match current position
        for word in word_dict:
            if s[start:].startswith(word):
                # Recursively get sentences for remaining string
                rest_sentences = self.word_break_all_ways(s, word_dict, start + len(word))
                
                for sentence in rest_sentences:
                    if sentence:
                        result.append(word + " " + sentence)
                    else:
                        result.append(word)
        
        self.memo[start] = result
        return result
    
    def palindrome_partitioning(self, s: str, start: int = 0, current: List[str] = None) -> List[List[str]]:
        """
        Palindrome Partitioning: Split string into palindromic substrings
        
        Company: Amazon, Microsoft
        Difficulty: Medium
        
        Time: O(2^n), Space: O(n)
        """
        if current is None:
            current = []
        
        # Base case: reached end of string
        if start >= len(s):
            return [current[:]]  # Return copy of current partition
        
        result = []
        
        # Try all possible partitions starting from current position
        for end in range(start, len(s)):
            substring = s[start:end + 1]
            
            # If current substring is palindrome
            if self.is_palindrome_recursive(substring):
                current.append(substring)
                result.extend(self.palindrome_partitioning(s, end + 1, current))
                current.pop()  # Backtrack
        
        return result
    
    def is_palindrome_recursive(self, s: str, left: int = 0, right: int = None) -> bool:
        """Check if string is palindrome recursively"""
        if right is None:
            right = len(s) - 1
        
        if left >= right:
            return True
        
        if s[left] != s[right]:
            return False
        
        return self.is_palindrome_recursive(s, left + 1, right - 1)
    
    # ==========================================
    # 3. TREE AND GRAPH PROBLEMS
    # ==========================================
    
    def binary_tree_paths(self, root, current_path: List[str] = None) -> List[str]:
        """
        Binary Tree Paths: Find all root-to-leaf paths
        
        Company: Google, Facebook
        Difficulty: Easy
        
        Time: O(n), Space: O(n)
        """
        if current_path is None:
            current_path = []
        
        if not root:
            return []
        
        # Add current node to path
        current_path.append(str(root.val))
        
        # If leaf node, create path string
        if not root.left and not root.right:
            path_string = "->".join(current_path)
            current_path.pop()  # Backtrack
            return [path_string]
        
        # Recursively get paths from left and right subtrees
        all_paths = []
        if root.left:
            all_paths.extend(self.binary_tree_paths(root.left, current_path))
        if root.right:
            all_paths.extend(self.binary_tree_paths(root.right, current_path))
        
        current_path.pop()  # Backtrack
        return all_paths
    
    def path_sum_all_paths(self, root, target_sum: int, current_path: List[int] = None) -> List[List[int]]:
        """
        Path Sum II: Find all root-to-leaf paths with given sum
        
        Company: Amazon, Microsoft
        Difficulty: Medium
        
        Time: O(n), Space: O(n)
        """
        if current_path is None:
            current_path = []
        
        if not root:
            return []
        
        # Add current node to path
        current_path.append(root.val)
        target_sum -= root.val
        
        result = []
        
        # If leaf node and sum matches
        if not root.left and not root.right and target_sum == 0:
            result.append(current_path[:])  # Make a copy
        
        # Recursively search in left and right subtrees
        if root.left:
            result.extend(self.path_sum_all_paths(root.left, target_sum, current_path))
        if root.right:
            result.extend(self.path_sum_all_paths(root.right, target_sum, current_path))
        
        current_path.pop()  # Backtrack
        return result
    
    def count_univalue_subtrees(self, root) -> int:
        """
        Count Univalue Subtrees: Count subtrees where all nodes have same value
        
        Company: Google
        Difficulty: Medium
        
        Time: O(n), Space: O(h)
        """
        self.univalue_count = 0
        
        def is_univalue(node) -> bool:
            if not node:
                return True
            
            left_univalue = is_univalue(node.left)
            right_univalue = is_univalue(node.right)
            
            # Check if current subtree is univalue
            is_current_univalue = True
            
            if node.left and node.left.val != node.val:
                is_current_univalue = False
            if node.right and node.right.val != node.val:
                is_current_univalue = False
            
            if left_univalue and right_univalue and is_current_univalue:
                self.univalue_count += 1
                return True
            
            return False
        
        is_univalue(root)
        return self.univalue_count
    
    # ==========================================
    # 4. OPTIMIZATION AND GAME THEORY
    # ==========================================
    
    def predict_winner(self, nums: List[int], start: int = 0, end: int = None) -> bool:
        """
        Predict the Winner: Two players pick from array ends, can player 1 win?
        
        Company: Google, Amazon
        Difficulty: Medium
        
        Time: O(n^2), Space: O(n^2)
        """
        if end is None:
            end = len(nums) - 1
        
        # Base case: only one element
        if start == end:
            return nums[start] >= 0  # Player 1 gets this, player 2 gets nothing
        
        # Check memo
        if (start, end) in self.memo:
            return self.memo[(start, end)]
        
        # Player 1 picks from start or end
        # Player 1 wins if either choice leads to their victory
        pick_start = nums[start] - (nums[start + 1] if not self.predict_winner(nums, start + 1, end) else 0)
        pick_end = nums[end] - (nums[end - 1] if not self.predict_winner(nums, start, end - 1) else 0)
        
        result = max(pick_start, pick_end) >= 0
        self.memo[(start, end)] = result
        return result
    
    def can_partition(self, nums: List[int], index: int = 0, current_sum: int = 0, target: int = None) -> bool:
        """
        Partition Equal Subset Sum: Can array be partitioned into two equal sum subsets?
        
        Company: Amazon, Facebook
        Difficulty: Medium
        
        Time: O(n * sum), Space: O(n * sum)
        """
        if target is None:
            total = sum(nums)
            if total % 2 != 0:
                return False
            target = total // 2
        
        # Base cases
        if current_sum == target:
            return True
        if index >= len(nums) or current_sum > target:
            return False
        
        # Check memo
        if (index, current_sum) in self.memo:
            return self.memo[(index, current_sum)]
        
        # Include current number or exclude it
        include = self.can_partition(nums, index + 1, current_sum + nums[index], target)
        exclude = self.can_partition(nums, index + 1, current_sum, target)
        
        result = include or exclude
        self.memo[(index, current_sum)] = result
        return result
    
    def target_sum(self, nums: List[int], target: int, index: int = 0) -> int:
        """
        Target Sum: Count ways to assign +/- to numbers to reach target
        
        Company: Facebook, Google
        Difficulty: Medium
        
        Time: O(n * sum), Space: O(n * sum)
        """
        # Base case
        if index >= len(nums):
            return 1 if target == 0 else 0
        
        # Check memo
        if (index, target) in self.memo:
            return self.memo[(index, target)]
        
        # Add current number or subtract it
        add = self.target_sum(nums, target - nums[index], index + 1)
        subtract = self.target_sum(nums, target + nums[index], index + 1)
        
        result = add + subtract
        self.memo[(index, target)] = result
        return result
    
    # ==========================================
    # 5. INTERVIEW TIPS AND PATTERNS
    # ==========================================
    
    def interview_approach_guide(self) -> None:
        """
        Guide for approaching recursion problems in interviews
        """
        print("=== RECURSION INTERVIEW APPROACH ===")
        
        print("\n1. PROBLEM ANALYSIS:")
        print("   • Identify base cases (when recursion stops)")
        print("   • Look for subproblem structure")
        print("   • Check if same subproblems appear multiple times")
        print("   • Consider state space (what parameters change)")
        
        print("\n2. SOLUTION STEPS:")
        print("   • Start with brute force recursive solution")
        print("   • Add memoization if overlapping subproblems exist")
        print("   • Consider space optimization (bottom-up DP)")
        print("   • Analyze time and space complexity")
        
        print("\n3. COMMON PATTERNS:")
        print("   • Choice problems: include/exclude current element")
        print("   • Path problems: explore all possible paths")
        print("   • Partition problems: split into valid groups")
        print("   • Game theory: optimal play for both players")
        
        print("\n4. OPTIMIZATION TECHNIQUES:")
        print("   • Memoization: O(states) space, O(states * transitions) time")
        print("   • Tail recursion: can be converted to iteration")
        print("   • Rolling arrays: reduce space complexity")
        print("   • Early pruning: avoid impossible branches")
        
        print("\n5. INTERVIEW TIPS:")
        print("   • Always start with examples and trace through")
        print("   • Draw recursion tree for small inputs")
        print("   • Identify overlapping subproblems")
        print("   • Discuss optimization opportunities")
        print("   • Consider edge cases and boundary conditions")

# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_interview_problems():
    """Demonstrate common recursion interview problems"""
    print("=== RECURSION INTERVIEW PROBLEMS DEMONSTRATION ===\n")
    
    problems = RecursionInterviewProblems()
    
    # 1. Classic Problems
    print("=== CLASSIC INTERVIEW PROBLEMS ===")
    
    # Climbing Stairs
    print("1. Climbing Stairs (n=5):")
    problems.memo.clear()
    ways = problems.climbing_stairs(5)
    print(f"   Ways to climb 5 stairs: {ways}")
    
    # Unique Paths
    print("\n2. Unique Paths (3x3 grid):")
    problems.memo.clear()
    paths = problems.unique_paths(3, 3)
    print(f"   Unique paths in 3x3 grid: {paths}")
    
    # Coin Change Ways
    print("\n3. Coin Change Ways (amount=4, coins=[1,2,3]):")
    problems.memo.clear()
    ways = problems.coin_change_ways(4, [1, 2, 3])
    print(f"   Ways to make amount 4: {ways}")
    
    # House Robber
    print("\n4. House Robber ([2,7,9,3,1]):")
    problems.memo.clear()
    max_money = problems.house_robber([2, 7, 9, 3, 1])
    print(f"   Maximum money that can be robbed: {max_money}")
    
    # Decode Ways
    print("\n5. Decode Ways ('226'):")
    problems.memo.clear()
    ways = problems.decode_ways('226')
    print(f"   Ways to decode '226': {ways}")
    print()
    
    # 2. Array and String Problems
    print("=== ARRAY AND STRING PROBLEMS ===")
    
    # Longest Increasing Subsequence
    print("1. Longest Increasing Subsequence ([10,9,2,5,3,7,101,18]):")
    problems.memo.clear()
    lis_length = problems.longest_increasing_subsequence([10, 9, 2, 5, 3, 7, 101, 18])
    print(f"   LIS length: {lis_length}")
    
    # Word Break All Ways
    print("\n2. Word Break All Ways ('catsanddog', ['cat','cats','and','sand','dog']):")
    problems.memo.clear()
    sentences = problems.word_break_all_ways('catsanddog', ['cat', 'cats', 'and', 'sand', 'dog'])
    print(f"   Possible sentences: {sentences}")
    
    # Palindrome Partitioning
    print("\n3. Palindrome Partitioning ('aab'):")
    partitions = problems.palindrome_partitioning('aab')
    print(f"   Palindromic partitions: {partitions}")
    print()
    
    # 3. Optimization Problems
    print("=== OPTIMIZATION PROBLEMS ===")
    
    # Partition Equal Subset Sum
    print("1. Partition Equal Subset Sum ([1,5,11,5]):")
    problems.memo.clear()
    can_partition = problems.can_partition([1, 5, 11, 5])
    print(f"   Can partition into equal subsets: {can_partition}")
    
    # Target Sum
    print("\n2. Target Sum ([1,1,1,1,1], target=3):")
    problems.memo.clear()
    ways = problems.target_sum([1, 1, 1, 1, 1], 3)
    print(f"   Ways to reach target 3: {ways}")
    
    # Predict Winner
    print("\n3. Predict Winner ([1,5,2]):")
    problems.memo.clear()
    player1_wins = problems.predict_winner([1, 5, 2])
    print(f"   Player 1 can win: {player1_wins}")
    print()
    
    # Interview approach guide
    problems.interview_approach_guide()

if __name__ == "__main__":
    demonstrate_interview_problems()
    
    print("\n=== INTERVIEW SUCCESS STRATEGY ===")
    print("🎯 Before coding:")
    print("   1. Understand the problem thoroughly")
    print("   2. Ask clarifying questions")
    print("   3. Work through examples manually")
    print("   4. Identify the recursive pattern")
    print("   5. Discuss time/space complexity")
    
    print("\n💡 During coding:")
    print("   1. Start with base cases")
    print("   2. Write recursive relation clearly")
    print("   3. Add memoization if needed")
    print("   4. Test with example inputs")
    print("   5. Optimize if time permits")
    
    print("\n🔍 Common mistakes to avoid:")
    print("   • Forgetting base cases")
    print("   • Not handling edge cases")
    print("   • Inefficient state representation")
    print("   • Stack overflow due to deep recursion")
    print("   • Not considering memoization optimization")
    
    print("\n📚 Study recommendations:")
    print("   • Practice 2-3 problems daily")
    print("   • Focus on different problem types")
    print("   • Understand the underlying patterns")
    print("   • Time yourself on problem solving")
    print("   • Review and optimize your solutions")
