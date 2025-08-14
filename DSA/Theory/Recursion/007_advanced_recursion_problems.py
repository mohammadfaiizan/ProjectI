"""
Advanced Recursion Problems
===========================

Topics: Complex recursive algorithms, mathematical recursion, string problems
Companies: Google, Amazon, Microsoft, Facebook, Apple, Netflix, Uber
Difficulty: Hard
Time Complexity: Varies (O(n) to O(2^n))
Space Complexity: O(h) to O(n) for recursion stack
"""

from typing import List, Dict, Tuple, Optional, Set
import math

class AdvancedRecursionProblems:
    
    def __init__(self):
        """Initialize with tracking capabilities"""
        self.call_count = 0
        self.memo = {}
    
    # ==========================================
    # 1. MATHEMATICAL RECURSION PROBLEMS
    # ==========================================
    
    def josephus_problem(self, n: int, k: int) -> int:
        """
        Josephus Problem: n people in circle, eliminate every kth person
        
        Find the position of the survivor (0-indexed)
        
        Recurrence: J(n,k) = (J(n-1,k) + k) % n
        Base case: J(1,k) = 0
        
        Time: O(n), Space: O(n)
        """
        print(f"Josephus({n}, {k})")
        
        # Base case: only one person left
        if n == 1:
            return 0
        
        # Recursive case
        result = (self.josephus_problem(n - 1, k) + k) % n
        print(f"J({n},{k}) = (J({n-1},{k}) + {k}) % {n} = {result}")
        return result
    
    def catalan_number(self, n: int) -> int:
        """
        Calculate nth Catalan number using recursion
        
        Catalan numbers count various combinatorial structures
        C(n) = sum(C(i) * C(n-1-i)) for i from 0 to n-1
        C(0) = 1
        
        Time: O(4^n), Space: O(n) - exponential without memoization
        """
        if n <= 1:
            return 1
        
        if n in self.memo:
            return self.memo[n]
        
        result = 0
        for i in range(n):
            result += self.catalan_number(i) * self.catalan_number(n - 1 - i)
        
        self.memo[n] = result
        return result
    
    def tower_of_hanoi(self, n: int, source: str, destination: str, auxiliary: str) -> List[str]:
        """
        Solve Tower of Hanoi puzzle
        
        Move n disks from source to destination using auxiliary peg
        
        Time: O(2^n), Space: O(n)
        """
        moves = []
        
        def hanoi_recursive(disks: int, src: str, dest: str, aux: str) -> None:
            if disks == 1:
                move = f"Move disk 1 from {src} to {dest}"
                moves.append(move)
                print(f"  {move}")
                return
            
            # Move n-1 disks from source to auxiliary
            hanoi_recursive(disks - 1, src, aux, dest)
            
            # Move largest disk from source to destination
            move = f"Move disk {disks} from {src} to {dest}"
            moves.append(move)
            print(f"  {move}")
            
            # Move n-1 disks from auxiliary to destination
            hanoi_recursive(disks - 1, aux, dest, src)
        
        print(f"Solving Tower of Hanoi with {n} disks:")
        hanoi_recursive(n, source, destination, auxiliary)
        return moves
    
    def ackermann_function(self, m: int, n: int) -> int:
        """
        Calculate Ackermann function - grows extremely quickly
        
        A(m,n) = n+1 if m=0
               = A(m-1,1) if m>0 and n=0  
               = A(m-1,A(m,n-1)) if m>0 and n>0
        
        Time: Extremely large, Space: O(depth)
        """
        self.call_count += 1
        
        if m == 0:
            return n + 1
        elif n == 0:
            return self.ackermann_function(m - 1, 1)
        else:
            return self.ackermann_function(m - 1, self.ackermann_function(m, n - 1))
    
    # ==========================================
    # 2. STRING RECURSION PROBLEMS
    # ==========================================
    
    def edit_distance(self, word1: str, word2: str, i: int = None, j: int = None) -> int:
        """
        Calculate minimum edit distance between two strings
        
        Operations: insert, delete, replace
        
        Time: O(3^max(m,n)) without memoization, O(m*n) with memoization
        Space: O(max(m,n))
        """
        if i is None:
            i = len(word1)
        if j is None:
            j = len(word2)
        
        # Base cases
        if i == 0:
            return j  # Insert all characters of word2
        if j == 0:
            return i  # Delete all characters of word1
        
        # Check memo
        if (i, j) in self.memo:
            return self.memo[(i, j)]
        
        # If characters match, no operation needed
        if word1[i - 1] == word2[j - 1]:
            result = self.edit_distance(word1, word2, i - 1, j - 1)
        else:
            # Try all three operations and take minimum
            insert_op = self.edit_distance(word1, word2, i, j - 1) + 1
            delete_op = self.edit_distance(word1, word2, i - 1, j) + 1
            replace_op = self.edit_distance(word1, word2, i - 1, j - 1) + 1
            
            result = min(insert_op, delete_op, replace_op)
        
        self.memo[(i, j)] = result
        return result
    
    def longest_palindromic_subsequence(self, s: str, i: int = 0, j: int = None) -> int:
        """
        Find length of longest palindromic subsequence
        
        Time: O(2^n) without memoization, O(n^2) with memoization
        Space: O(n^2)
        """
        if j is None:
            j = len(s) - 1
        
        # Base cases
        if i > j:
            return 0
        if i == j:
            return 1
        
        # Check memo
        if (i, j) in self.memo:
            return self.memo[(i, j)]
        
        # If characters match, include both and recurse on inner substring
        if s[i] == s[j]:
            result = 2 + self.longest_palindromic_subsequence(s, i + 1, j - 1)
        else:
            # Take maximum of excluding either first or last character
            result = max(
                self.longest_palindromic_subsequence(s, i + 1, j),
                self.longest_palindromic_subsequence(s, i, j - 1)
            )
        
        self.memo[(i, j)] = result
        return result
    
    def count_palindromic_substrings(self, s: str) -> int:
        """
        Count all palindromic substrings in a string
        
        Time: O(n^3), Space: O(n)
        """
        def is_palindrome(string: str, start: int, end: int) -> bool:
            """Check if substring is palindrome recursively"""
            if start >= end:
                return True
            
            if string[start] != string[end]:
                return False
            
            return is_palindrome(string, start + 1, end - 1)
        
        count = 0
        n = len(s)
        
        # Check all possible substrings
        for i in range(n):
            for j in range(i, n):
                if is_palindrome(s, i, j):
                    count += 1
                    print(f"Found palindrome: '{s[i:j+1]}'")
        
        return count
    
    def generate_parentheses(self, n: int) -> List[str]:
        """
        Generate all valid combinations of n pairs of parentheses
        
        Time: O(4^n / sqrt(n)) - Catalan number
        Space: O(4^n / sqrt(n))
        """
        result = []
        
        def backtrack(current: str, open_count: int, close_count: int) -> None:
            print(f"Building: '{current}', open={open_count}, close={close_count}")
            
            # Base case: used all n pairs
            if len(current) == 2 * n:
                result.append(current)
                print(f"  ✓ Complete: '{current}'")
                return
            
            # Add opening parenthesis if we haven't used all n
            if open_count < n:
                backtrack(current + "(", open_count + 1, close_count)
            
            # Add closing parenthesis if it doesn't exceed opening ones
            if close_count < open_count:
                backtrack(current + ")", open_count, close_count + 1)
        
        backtrack("", 0, 0)
        return result
    
    # ==========================================
    # 3. ARRAY AND LIST RECURSION PROBLEMS
    # ==========================================
    
    def merge_k_sorted_lists(self, lists: List[List[int]]) -> List[int]:
        """
        Merge k sorted lists using divide and conquer
        
        Time: O(n log k) where n is total number of elements
        Space: O(log k)
        """
        if not lists:
            return []
        
        def merge_two_lists(list1: List[int], list2: List[int]) -> List[int]:
            """Merge two sorted lists"""
            if not list1:
                return list2
            if not list2:
                return list1
            
            if list1[0] <= list2[0]:
                return [list1[0]] + merge_two_lists(list1[1:], list2)
            else:
                return [list2[0]] + merge_two_lists(list1, list2[1:])
        
        def merge_k_helper(lists: List[List[int]]) -> List[int]:
            """Recursively merge k lists using divide and conquer"""
            if len(lists) == 1:
                return lists[0]
            
            mid = len(lists) // 2
            left = merge_k_helper(lists[:mid])
            right = merge_k_helper(lists[mid:])
            
            return merge_two_lists(left, right)
        
        return merge_k_helper(lists)
    
    def quick_select(self, arr: List[int], k: int, left: int = 0, right: int = None) -> int:
        """
        Find kth smallest element using quickselect algorithm
        
        Time: O(n) average, O(n^2) worst case
        Space: O(log n) average
        """
        if right is None:
            right = len(arr) - 1
        
        print(f"QuickSelect: arr[{left}:{right+1}] = {arr[left:right+1]}, k={k}")
        
        if left == right:
            return arr[left]
        
        # Partition the array
        pivot_index = self.partition_for_quickselect(arr, left, right)
        
        print(f"  Pivot at index {pivot_index}, value {arr[pivot_index]}")
        
        # If pivot is the kth element
        if pivot_index == k:
            return arr[pivot_index]
        elif pivot_index > k:
            # kth element is in left partition
            return self.quick_select(arr, k, left, pivot_index - 1)
        else:
            # kth element is in right partition
            return self.quick_select(arr, k, pivot_index + 1, right)
    
    def partition_for_quickselect(self, arr: List[int], left: int, right: int) -> int:
        """Partition array for quickselect"""
        pivot = arr[right]
        i = left - 1
        
        for j in range(left, right):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        
        arr[i + 1], arr[right] = arr[right], arr[i + 1]
        return i + 1
    
    def count_inversions(self, arr: List[int], temp_arr: List[int], left: int = 0, right: int = None) -> int:
        """
        Count number of inversions in array using divide and conquer
        
        Inversion: pair (i,j) where i < j but arr[i] > arr[j]
        
        Time: O(n log n), Space: O(n)
        """
        if right is None:
            right = len(arr) - 1
            temp_arr.extend([0] * len(arr))
        
        inv_count = 0
        
        if left < right:
            mid = (left + right) // 2
            
            # Count inversions in left and right halves
            inv_count += self.count_inversions(arr, temp_arr, left, mid)
            inv_count += self.count_inversions(arr, temp_arr, mid + 1, right)
            
            # Count inversions between left and right halves
            inv_count += self.merge_and_count(arr, temp_arr, left, mid, right)
        
        return inv_count
    
    def merge_and_count(self, arr: List[int], temp_arr: List[int], left: int, mid: int, right: int) -> int:
        """Merge two sorted halves and count inversions"""
        i, j, k = left, mid + 1, left
        inv_count = 0
        
        # Merge the two halves
        while i <= mid and j <= right:
            if arr[i] <= arr[j]:
                temp_arr[k] = arr[i]
                i += 1
            else:
                temp_arr[k] = arr[j]
                # All elements from i to mid are greater than arr[j]
                inv_count += (mid - i + 1)
                j += 1
            k += 1
        
        # Copy remaining elements
        while i <= mid:
            temp_arr[k] = arr[i]
            i += 1
            k += 1
        
        while j <= right:
            temp_arr[k] = arr[j]
            j += 1
            k += 1
        
        # Copy back the merged elements
        for i in range(left, right + 1):
            arr[i] = temp_arr[i]
        
        return inv_count
    
    # ==========================================
    # 4. GRAPH RECURSION PROBLEMS
    # ==========================================
    
    def count_paths_in_grid(self, m: int, n: int, obstacles: Set[Tuple[int, int]] = None) -> int:
        """
        Count number of paths from top-left to bottom-right in m×n grid
        
        Can only move right or down, avoid obstacles
        
        Time: O(2^(m+n)) without memoization, O(m*n) with memoization
        Space: O(m+n)
        """
        if obstacles is None:
            obstacles = set()
        
        def count_paths_helper(row: int, col: int) -> int:
            # Base cases
            if row >= m or col >= n or (row, col) in obstacles:
                return 0
            
            if row == m - 1 and col == n - 1:  # Reached destination
                return 1
            
            # Check memo
            if (row, col) in self.memo:
                return self.memo[(row, col)]
            
            # Count paths going right and down
            paths = (count_paths_helper(row, col + 1) + 
                    count_paths_helper(row + 1, col))
            
            self.memo[(row, col)] = paths
            return paths
        
        return count_paths_helper(0, 0)
    
    def word_break(self, s: str, word_dict: List[str], start: int = 0) -> bool:
        """
        Check if string can be segmented into dictionary words
        
        Time: O(2^n) without memoization, O(n^2) with memoization
        Space: O(n)
        """
        # Base case: reached end of string
        if start >= len(s):
            return True
        
        # Check memo
        if start in self.memo:
            return self.memo[start]
        
        # Try all possible words starting at current position
        for word in word_dict:
            if s[start:].startswith(word):
                if self.word_break(s, word_dict, start + len(word)):
                    self.memo[start] = True
                    return True
        
        self.memo[start] = False
        return False

# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_advanced_recursion():
    """Demonstrate all advanced recursion problems"""
    print("=== ADVANCED RECURSION PROBLEMS DEMONSTRATION ===\n")
    
    problems = AdvancedRecursionProblems()
    
    # 1. Mathematical Problems
    print("=== MATHEMATICAL RECURSION PROBLEMS ===")
    
    # Josephus Problem
    print("1. Josephus Problem (n=5, k=3):")
    survivor = problems.josephus_problem(5, 3)
    print(f"   Survivor position: {survivor}")
    print()
    
    # Catalan Numbers
    print("2. Catalan Numbers:")
    for i in range(6):
        problems.memo.clear()
        catalan = problems.catalan_number(i)
        print(f"   C({i}) = {catalan}")
    print()
    
    # Tower of Hanoi
    print("3. Tower of Hanoi (n=3):")
    moves = problems.tower_of_hanoi(3, "A", "C", "B")
    print(f"   Total moves: {len(moves)}")
    print()
    
    # Ackermann Function (use small values!)
    print("4. Ackermann Function:")
    for m in range(3):
        for n in range(3):
            problems.call_count = 0
            result = problems.ackermann_function(m, n)
            print(f"   A({m},{n}) = {result} ({problems.call_count} calls)")
    print()
    
    # 2. String Problems
    print("=== STRING RECURSION PROBLEMS ===")
    
    # Edit Distance
    print("1. Edit Distance:")
    problems.memo.clear()
    word1, word2 = "intention", "execution"
    distance = problems.edit_distance(word1, word2)
    print(f"   Edit distance between '{word1}' and '{word2}': {distance}")
    print()
    
    # Longest Palindromic Subsequence
    print("2. Longest Palindromic Subsequence:")
    problems.memo.clear()
    s = "bbbab"
    lps_length = problems.longest_palindromic_subsequence(s)
    print(f"   LPS length in '{s}': {lps_length}")
    print()
    
    # Count Palindromic Substrings
    print("3. Count Palindromic Substrings:")
    s = "aaba"
    count = problems.count_palindromic_substrings(s)
    print(f"   Palindromic substrings in '{s}': {count}")
    print()
    
    # Generate Parentheses
    print("4. Generate Parentheses (n=3):")
    parentheses = problems.generate_parentheses(3)
    print(f"   Valid combinations: {parentheses}")
    print()
    
    # 3. Array Problems
    print("=== ARRAY RECURSION PROBLEMS ===")
    
    # Merge K Sorted Lists
    print("1. Merge K Sorted Lists:")
    lists = [[1, 4, 5], [1, 3, 4], [2, 6]]
    merged = problems.merge_k_sorted_lists(lists)
    print(f"   Input: {lists}")
    print(f"   Merged: {merged}")
    print()
    
    # Quick Select
    print("2. Quick Select (find 3rd smallest):")
    arr = [3, 2, 1, 5, 6, 4]
    k = 2  # 0-indexed, so 3rd smallest
    arr_copy = arr[:]
    kth_element = problems.quick_select(arr_copy, k)
    print(f"   Array: {arr}")
    print(f"   3rd smallest element: {kth_element}")
    print()
    
    # Count Inversions
    print("3. Count Inversions:")
    arr = [8, 4, 2, 1]
    temp_arr = []
    arr_copy = arr[:]
    inversions = problems.count_inversions(arr_copy, temp_arr)
    print(f"   Array: {arr}")
    print(f"   Number of inversions: {inversions}")
    print()
    
    # 4. Graph Problems
    print("=== GRAPH RECURSION PROBLEMS ===")
    
    # Count Paths in Grid
    print("1. Count Paths in Grid (3x3):")
    problems.memo.clear()
    paths = problems.count_paths_in_grid(3, 3)
    print(f"   Paths from (0,0) to (2,2): {paths}")
    print()
    
    # Word Break
    print("2. Word Break:")
    problems.memo.clear()
    s = "leetcode"
    word_dict = ["leet", "code"]
    can_break = problems.word_break(s, word_dict)
    print(f"   Can '{s}' be segmented using {word_dict}: {can_break}")

if __name__ == "__main__":
    demonstrate_advanced_recursion()
    
    print("\n=== ADVANCED RECURSION INSIGHTS ===")
    print("🧠 Problem-solving strategies:")
    print("   1. Identify the recursive substructure")
    print("   2. Define clear base cases")
    print("   3. Use memoization for overlapping subproblems")
    print("   4. Consider divide-and-conquer for large problems")
    print("   5. Optimize space when possible")
    
    print("\n🎯 Common patterns:")
    print("   • Mathematical recurrences (Fibonacci, Catalan)")
    print("   • String processing (edit distance, palindromes)")
    print("   • Array partitioning (quickselect, merge operations)")
    print("   • Combinatorial generation (parentheses, subsets)")
    print("   • Graph traversal and path counting")
    
    print("\n⚡ Performance considerations:")
    print("   • Exponential time complexity without optimization")
    print("   • Memoization can reduce to polynomial time")
    print("   • Space complexity limited by recursion depth")
    print("   • Consider iterative alternatives for deep recursion")
    
    print("\n🔧 Debugging tips:")
    print("   • Trace through small examples manually")
    print("   • Add print statements to track recursive calls")
    print("   • Verify base cases handle edge conditions")
    print("   • Test with different input sizes and patterns")
