"""
10. Regular Expression Matching - Multiple Approaches
Difficulty: Hard

Given an input string s and a pattern p, implement regular expression matching 
with support for '.' and '*' where:
- '.' Matches any single character.
- '*' Matches zero or more of the preceding element.

The matching should cover the entire input string (not partial).

LeetCode Problem: https://leetcode.com/problems/regular-expression-matching/

Example:
Input: s = "aa", p = "a*"
Output: true
Explanation: '*' means zero or more of the preceding element 'a'. Therefore, by repeating 'a' once, it becomes "aa".
"""

from typing import Dict, Tuple, Optional

class TrieNode:
    """Trie node for regex pattern storage"""
    def __init__(self):
        self.children = {}
        self.is_pattern_end = False
        self.pattern = ""

class Solution:
    
    def isMatch1(self, s: str, p: str) -> bool:
        """
        Approach 1: Recursive Solution with Memoization
        
        Classic recursive approach with caching.
        
        Time: O(|s| * |p|) with memoization
        Space: O(|s| * |p|) for memoization table
        """
        memo = {}
        
        def dp(i: int, j: int) -> bool:
            if (i, j) in memo:
                return memo[(i, j)]
            
            # Base cases
            if j >= len(p):
                result = i >= len(s)
            elif i >= len(s):
                # Check if remaining pattern can match empty string
                result = all(k + 1 < len(p) and p[k + 1] == '*' 
                           for k in range(j, len(p), 2))
            else:
                # Character matching
                first_match = (p[j] == '.' or p[j] == s[i])
                
                # Look ahead for '*'
                if j + 1 < len(p) and p[j + 1] == '*':
                    # Two choices: use '*' (match 0 times) or don't use it
                    result = (dp(i, j + 2) or  # Match 0 times
                             (first_match and dp(i + 1, j)))  # Match 1+ times
                else:
                    # Regular character match
                    result = first_match and dp(i + 1, j + 1)
            
            memo[(i, j)] = result
            return result
        
        return dp(0, 0)
    
    def isMatch2(self, s: str, p: str) -> bool:
        """
        Approach 2: Bottom-up Dynamic Programming
        
        Iterative DP approach building table from bottom up.
        
        Time: O(|s| * |p|)
        Space: O(|s| * |p|)
        """
        m, n = len(s), len(p)
        
        # dp[i][j] = does s[0:i] match p[0:j]
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        
        # Empty pattern matches empty string
        dp[0][0] = True
        
        # Handle patterns like a*, a*b*, a*b*c* that can match empty string
        for j in range(2, n + 1):
            if p[j - 1] == '*':
                dp[0][j] = dp[0][j - 2]
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    # '*' can match 0 times or 1+ times
                    dp[i][j] = dp[i][j - 2]  # Match 0 times
                    
                    if p[j - 2] == '.' or p[j - 2] == s[i - 1]:
                        dp[i][j] = dp[i][j] or dp[i - 1][j]  # Match 1+ times
                else:
                    # Regular character or '.'
                    if p[j - 1] == '.' or p[j - 1] == s[i - 1]:
                        dp[i][j] = dp[i - 1][j - 1]
        
        return dp[m][n]
    
    def isMatch3(self, s: str, p: str) -> bool:
        """
        Approach 3: Space-Optimized DP
        
        Optimize space usage by using only two rows.
        
        Time: O(|s| * |p|)
        Space: O(|p|)
        """
        m, n = len(s), len(p)
        
        # Use two arrays for current and previous row
        prev = [False] * (n + 1)
        curr = [False] * (n + 1)
        
        # Base case
        prev[0] = True
        
        # Initialize first row
        for j in range(2, n + 1):
            if p[j - 1] == '*':
                prev[j] = prev[j - 2]
        
        # Fill DP table row by row
        for i in range(1, m + 1):
            curr[0] = False  # Empty pattern can't match non-empty string
            
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    curr[j] = curr[j - 2]  # Match 0 times
                    
                    if p[j - 2] == '.' or p[j - 2] == s[i - 1]:
                        curr[j] = curr[j] or prev[j]  # Match 1+ times
                else:
                    if p[j - 1] == '.' or p[j - 1] == s[i - 1]:
                        curr[j] = prev[j - 1]
                    else:
                        curr[j] = False
            
            # Swap arrays
            prev, curr = curr, prev
        
        return prev[n]
    
    def isMatch4(self, s: str, p: str) -> bool:
        """
        Approach 4: Finite State Automaton
        
        Build NFA (Non-deterministic Finite Automaton) from pattern.
        
        Time: O(|p|^2 + |s| * |p|)
        Space: O(|p|^2)
        """
        # Build NFA from pattern
        states = self._build_nfa(p)
        
        # Simulate NFA on input string
        current_states = {0}  # Start state
        
        for char in s:
            current_states = self._epsilon_closure(current_states, states)
            next_states = set()
            
            for state in current_states:
                if state < len(states) and states[state]:
                    transition_char, next_state = states[state]
                    if transition_char == char or transition_char == '.':
                        next_states.add(next_state)
            
            current_states = next_states
        
        # Check if any final state is reachable
        current_states = self._epsilon_closure(current_states, states)
        return len(p) in current_states
    
    def _build_nfa(self, pattern: str) -> Dict[int, Optional[Tuple[str, int]]]:
        """Build NFA from regex pattern"""
        states = {}
        state_count = 0
        
        for i, char in enumerate(pattern):
            if i + 1 < len(pattern) and pattern[i + 1] == '*':
                # Character followed by '*'
                states[state_count] = (char, state_count)  # Self loop
                states[state_count + 1] = None  # Epsilon transition
                state_count += 2
            else:
                # Regular character
                states[state_count] = (char, state_count + 1)
                state_count += 1
        
        return states
    
    def _epsilon_closure(self, states: set, nfa: Dict[int, Optional[Tuple[str, int]]]) -> set:
        """Compute epsilon closure of states"""
        closure = set(states)
        stack = list(states)
        
        while stack:
            state = stack.pop()
            if state in nfa and nfa[state] is None:
                # Epsilon transition
                next_state = state + 1
                if next_state not in closure:
                    closure.add(next_state)
                    stack.append(next_state)
        
        return closure
    
    def isMatch5(self, s: str, p: str) -> bool:
        """
        Approach 5: Trie-based Pattern Compilation
        
        Compile pattern into trie for structured matching.
        
        Time: O(|p| + |s| * |p|)
        Space: O(|p|)
        """
        # Build pattern trie
        root = TrieNode()
        self._compile_pattern_to_trie(p, root, 0)
        
        # Match string against trie
        return self._match_with_trie(s, 0, root)
    
    def _compile_pattern_to_trie(self, pattern: str, node: TrieNode, index: int) -> None:
        """Compile regex pattern into trie structure"""
        if index >= len(pattern):
            node.is_pattern_end = True
            return
        
        char = pattern[index]
        
        if index + 1 < len(pattern) and pattern[index + 1] == '*':
            # Character followed by '*'
            if char not in node.children:
                node.children[char] = TrieNode()
            
            # Zero occurrences
            self._compile_pattern_to_trie(pattern, node, index + 2)
            
            # One or more occurrences
            self._compile_pattern_to_trie(pattern, node.children[char], index)
            
            # Continue after '*'
            self._compile_pattern_to_trie(pattern, node, index + 2)
        else:
            # Regular character or '.'
            if char not in node.children:
                node.children[char] = TrieNode()
            
            self._compile_pattern_to_trie(pattern, node.children[char], index + 1)
    
    def _match_with_trie(self, s: str, index: int, node: TrieNode) -> bool:
        """Match string with compiled trie"""
        if index >= len(s):
            return node.is_pattern_end
        
        char = s[index]
        
        # Try exact character match
        if char in node.children:
            if self._match_with_trie(s, index + 1, node.children[char]):
                return True
        
        # Try '.' wildcard
        if '.' in node.children:
            if self._match_with_trie(s, index + 1, node.children['.']):
                return True
        
        return False
    
    def isMatch6(self, s: str, p: str) -> bool:
        """
        Approach 6: Iterative Simulation with Stack
        
        Non-recursive simulation using explicit stack.
        
        Time: O(|s| * |p|)
        Space: O(|s| * |p|) for stack
        """
        # Stack contains (string_index, pattern_index) pairs
        stack = [(0, 0)]
        visited = set()
        
        while stack:
            si, pi = stack.pop()
            
            if (si, pi) in visited:
                continue
            visited.add((si, pi))
            
            # Base cases
            if pi >= len(p):
                if si >= len(s):
                    return True
                continue
            
            if si >= len(s):
                # Check if remaining pattern can match empty string
                if all(k + 1 < len(p) and p[k + 1] == '*' 
                       for k in range(pi, len(p), 2)):
                    return True
                continue
            
            # Character matching
            first_match = (p[pi] == '.' or p[pi] == s[si])
            
            # Handle '*' operator
            if pi + 1 < len(p) and p[pi + 1] == '*':
                # Two choices
                stack.append((si, pi + 2))  # Match 0 times
                if first_match:
                    stack.append((si + 1, pi))  # Match 1+ times
            else:
                # Regular character
                if first_match:
                    stack.append((si + 1, pi + 1))
        
        return False


def test_basic_cases():
    """Test basic regex matching functionality"""
    print("=== Testing Basic Cases ===")
    
    solution = Solution()
    
    test_cases = [
        # LeetCode examples
        ("aa", "a", False),
        ("aa", "a*", True),
        ("ab", ".*", True),
        ("aab", "c*a*b", True),
        ("mississippi", "mis*is*p*.", False),
        
        # Basic patterns
        ("", "", True),
        ("", "a*", True),
        ("", ".*", True),
        ("a", "", False),
        ("a", "a", True),
        ("a", ".", True),
        
        # Star patterns
        ("aaa", "a*", True),
        ("aaa", "aa*", True),
        ("", "a*b*", True),
        ("ab", "a*b", True),
        ("aab", "a*b", True),
        
        # Complex patterns
        ("abcd", "a.*d", True),
        ("abcd", "a.c*d", True),
        ("abcd", "a.*c*d", True),
        ("abc", "a*b*c*", True),
    ]
    
    approaches = [
        ("Recursive+Memo", solution.isMatch1),
        ("Bottom-up DP", solution.isMatch2),
        ("Space-optimized", solution.isMatch3),
        ("Finite Automaton", solution.isMatch4),
        ("Trie-based", solution.isMatch5),
        ("Iterative Stack", solution.isMatch6),
    ]
    
    for i, (s, p, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: s='{s}', p='{p}'")
        print(f"Expected: {expected}")
        
        for name, method in approaches:
            try:
                result = method(s, p)
                status = "✓" if result == expected else "✗"
                print(f"  {name:15}: {result} {status}")
            except Exception as e:
                print(f"  {name:15}: Error - {e}")


def demonstrate_dp_table():
    """Demonstrate DP table construction"""
    print("\n=== DP Table Construction Demo ===")
    
    s = "aab"
    p = "c*a*b"
    
    print(f"String: '{s}'")
    print(f"Pattern: '{p}'")
    
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    
    # Base case
    dp[0][0] = True
    print(f"\nStep 1: Initialize dp[0][0] = True (empty matches empty)")
    
    # Initialize first row
    print(f"\nStep 2: Initialize first row (empty string vs pattern):")
    for j in range(2, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 2]
            print(f"  dp[0][{j}] = {dp[0][j]} (pattern '{p[:j]}' vs empty)")
    
    # Fill table
    print(f"\nStep 3: Fill DP table:")
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                dp[i][j] = dp[i][j - 2]  # Match 0 times
                
                if p[j - 2] == '.' or p[j - 2] == s[i - 1]:
                    dp[i][j] = dp[i][j] or dp[i - 1][j]  # Match 1+ times
                
                print(f"  dp[{i}][{j}] = {dp[i][j]} ('{s[:i]}' vs '{p[:j]}')")
            else:
                if p[j - 1] == '.' or p[j - 1] == s[i - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                
                print(f"  dp[{i}][{j}] = {dp[i][j]} ('{s[:i]}' vs '{p[:j]}')")
    
    # Show final table
    print(f"\nFinal DP Table:")
    print("     ", end="")
    for j in range(n + 1):
        if j == 0:
            print("ε", end="  ")
        else:
            print(f"{p[j-1]}", end="  ")
    print()
    
    for i in range(m + 1):
        if i == 0:
            print("ε   ", end="")
        else:
            print(f"{s[i-1]}   ", end="")
        
        for j in range(n + 1):
            print("T" if dp[i][j] else "F", end="  ")
        print()
    
    print(f"\nResult: dp[{m}][{n}] = {dp[m][n]}")


def demonstrate_pattern_analysis():
    """Demonstrate pattern analysis and matching process"""
    print("\n=== Pattern Analysis Demo ===")
    
    patterns = ["a*", ".*", "a*b", "a.*b", "a*b*c*"]
    test_string = "abc"
    
    print(f"Test string: '{test_string}'")
    print(f"Analyzing different patterns:")
    
    solution = Solution()
    
    for pattern in patterns:
        print(f"\nPattern: '{pattern}'")
        
        # Analyze pattern structure
        components = []
        i = 0
        while i < len(pattern):
            if i + 1 < len(pattern) and pattern[i + 1] == '*':
                components.append(f"{pattern[i]}* (zero or more '{pattern[i]}')")
                i += 2
            else:
                if pattern[i] == '.':
                    components.append(". (any character)")
                else:
                    components.append(f"'{pattern[i]}' (exact match)")
                i += 1
        
        print(f"  Components: {' + '.join(components)}")
        
        # Test matching
        result = solution.isMatch1(test_string, pattern)
        print(f"  Matches '{test_string}': {result}")
        
        # Show why/why not
        if result:
            print(f"  ✓ Pattern can generate '{test_string}'")
        else:
            print(f"  ✗ Pattern cannot generate '{test_string}'")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solution = Solution()
    
    edge_cases = [
        # Empty cases
        ("", "", True, "Both empty"),
        ("", "a", False, "Empty string, non-empty pattern"),
        ("a", "", False, "Non-empty string, empty pattern"),
        ("", "a*", True, "Empty string, star pattern"),
        ("", ".*", True, "Empty string, dot-star"),
        
        # Single characters
        ("a", "a", True, "Single character match"),
        ("a", "b", False, "Single character mismatch"),
        ("a", ".", True, "Single dot"),
        
        # Star patterns
        ("", "a*b*c*", True, "Multiple stars matching empty"),
        ("abc", "a*b*c*", True, "Multiple stars matching string"),
        ("aabbcc", "a*b*c*", True, "Multiple occurrences"),
        
        # Complex cases
        ("abcdef", "a.*f", True, "Dot-star in middle"),
        ("abcdef", "a.*g", False, "Dot-star with wrong end"),
        ("aaaa", "a*a", True, "Star followed by same character"),
        ("aaaa", ".*a", True, "Dot-star with specific end"),
        
        # Tricky patterns
        ("aaa", "ab*a*c*a", True, "Complex pattern with stars"),
        ("ab", ".*c", False, "Dot-star with required character"),
    ]
    
    for s, p, expected, description in edge_cases:
        print(f"\n{description}:")
        print(f"  s='{s}', p='{p}'")
        
        try:
            result = solution.isMatch1(s, p)
            status = "✓" if result == expected else "✗"
            print(f"  Result: {result} {status}")
        except Exception as e:
            print(f"  Error: {e}")


def benchmark_approaches():
    """Benchmark different approaches"""
    print("\n=== Benchmarking Approaches ===")
    
    import time
    import random
    import string
    
    solution = Solution()
    
    # Generate test cases
    def generate_string(length: int) -> str:
        return ''.join(random.choices(string.ascii_lowercase[:3], k=length))
    
    def generate_pattern(length: int) -> str:
        pattern = ""
        i = 0
        while i < length:
            char = random.choice(string.ascii_lowercase[:3] + ['.'])
            pattern += char
            
            # 30% chance to add '*'
            if random.random() < 0.3 and i + 1 < length:
                pattern += '*'
                i += 1
            i += 1
        
        return pattern
    
    test_scenarios = [
        ("Short", [(generate_string(5), generate_pattern(5)) for _ in range(20)]),
        ("Medium", [(generate_string(10), generate_pattern(8)) for _ in range(20)]),
        ("Long", [(generate_string(15), generate_pattern(12)) for _ in range(10)]),
    ]
    
    approaches = [
        ("Recursive+Memo", solution.isMatch1),
        ("Bottom-up DP", solution.isMatch2),
        ("Space-optimized", solution.isMatch3),
        ("Iterative Stack", solution.isMatch6),
    ]
    
    for scenario_name, test_cases in test_scenarios:
        print(f"\n--- {scenario_name} Dataset ---")
        print(f"Test cases: {len(test_cases)}")
        
        for approach_name, method in approaches:
            start_time = time.time()
            
            results = []
            for s, p in test_cases:
                try:
                    result = method(s, p)
                    results.append(result)
                except:
                    results.append(False)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / len(test_cases)
            
            true_count = sum(results)
            print(f"  {approach_name:15}: {avg_time*1000:.2f}ms/case ({true_count}/{len(test_cases)} matches)")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    solution = Solution()
    
    # Application 1: File pattern matching
    print("1. File Pattern Matching:")
    
    filenames = ["config.txt", "data.csv", "image.jpg", "script.py", "readme.md"]
    patterns = [".*\\.txt", ".*\\.py", "c.*", ".*a.*"]
    
    print(f"   Files: {filenames}")
    print(f"   Patterns: {patterns}")
    
    for pattern in patterns:
        print(f"\n   Pattern '{pattern}' matches:")
        for filename in filenames:
            if solution.isMatch1(filename, pattern):
                print(f"     ✓ {filename}")
    
    # Application 2: Input validation
    print(f"\n2. Input Validation:")
    
    inputs = ["abc123", "hello", "123", "a1b2c3", ""]
    validation_patterns = [
        (".*[0-9].*", "Contains digits"),
        ("[a-z]*", "Only lowercase letters"),
        (".*[a-z].*[0-9].*", "Has letters and digits"),
    ]
    
    print(f"   Inputs: {inputs}")
    
    for pattern, description in validation_patterns:
        print(f"\n   Validation: {description} ('{pattern}')")
        for input_str in inputs:
            valid = solution.isMatch1(input_str, pattern)
            print(f"     '{input_str}': {'Valid' if valid else 'Invalid'}")
    
    # Application 3: Log parsing
    print(f"\n3. Log Pattern Matching:")
    
    log_entries = [
        "ERROR 2024-01-01",
        "INFO system startup",
        "WARNING memory low", 
        "ERROR database fail",
        "DEBUG trace info"
    ]
    
    log_patterns = [
        ("ERROR.*", "Error messages"),
        (".*[0-9].*", "Contains timestamps"),
        (".*system.*", "System-related"),
    ]
    
    print(f"   Log entries: {len(log_entries)}")
    
    for pattern, description in log_patterns:
        print(f"\n   Filter: {description} ('{pattern}')")
        matches = [entry for entry in log_entries 
                  if solution.isMatch1(entry, pattern)]
        for match in matches:
            print(f"     {match}")


def analyze_complexity():
    """Analyze time and space complexity"""
    print("\n=== Complexity Analysis ===")
    
    complexity_analysis = [
        ("Recursive + Memoization",
         "Time: O(|s| * |p|) - each subproblem solved once",
         "Space: O(|s| * |p|) - memoization table + recursion stack"),
        
        ("Bottom-up DP",
         "Time: O(|s| * |p|) - fill entire DP table",
         "Space: O(|s| * |p|) - DP table storage"),
        
        ("Space-optimized DP",
         "Time: O(|s| * |p|) - same computation",
         "Space: O(|p|) - only two rows needed"),
        
        ("Finite State Automaton",
         "Time: O(|p|^2 + |s| * |p|) - build NFA + simulate",
         "Space: O(|p|^2) - NFA state transitions"),
        
        ("Iterative with Stack",
         "Time: O(|s| * |p|) - explicit state exploration",
         "Space: O(|s| * |p|) - stack + visited set"),
    ]
    
    print("Approach Analysis:")
    for approach, time_complexity, space_complexity in complexity_analysis:
        print(f"\n{approach}:")
        print(f"  {time_complexity}")
        print(f"  {space_complexity}")
    
    print(f"\nWhere:")
    print(f"  |s| = length of input string")
    print(f"  |p| = length of pattern")
    
    print(f"\nRecommendations:")
    print(f"  • Use Space-optimized DP for memory-constrained environments")
    print(f"  • Use Bottom-up DP for clarity and debugging")
    print(f"  • Use Recursive+Memo for easier implementation")
    print(f"  • Use Finite Automaton for multiple string matching")


if __name__ == "__main__":
    test_basic_cases()
    demonstrate_dp_table()
    demonstrate_pattern_analysis()
    test_edge_cases()
    benchmark_approaches()
    demonstrate_real_world_applications()
    analyze_complexity()

"""
10. Regular Expression Matching demonstrates comprehensive regex implementation:

1. Recursive + Memoization - Classic top-down approach with caching
2. Bottom-up DP - Iterative table-filling dynamic programming
3. Space-optimized DP - Memory-efficient two-row approach
4. Finite State Automaton - NFA construction and simulation
5. Trie-based - Pattern compilation into trie structure
6. Iterative with Stack - Non-recursive explicit state exploration

Each approach offers different trade-offs between implementation complexity,
time efficiency, and space usage for regular expression matching problems.
"""
