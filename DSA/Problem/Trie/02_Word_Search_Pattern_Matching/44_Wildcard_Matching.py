"""
44. Wildcard Matching - Multiple Approaches
Difficulty: Hard

Given an input string (s) and a pattern (p), implement wildcard pattern matching 
with support for '?' and '*' where:
- '?' Matches any single character.
- '*' Matches any sequence of characters (including the empty sequence).

The matching should cover the entire input string (not partial).

LeetCode Problem: https://leetcode.com/problems/wildcard-matching/

Example:
Input: s = "aa", p = "a"
Output: false
Explanation: "a" does not match the entire string "aa".
"""

from typing import List, Dict, Tuple

class Solution:
    
    def isMatch1(self, s: str, p: str) -> bool:
        """
        Approach 1: Dynamic Programming (Bottom-up)
        
        Classic DP approach building table from bottom up.
        
        Time: O(|s| * |p|)
        Space: O(|s| * |p|)
        """
        m, n = len(s), len(p)
        
        # dp[i][j] = does s[0:i] match p[0:j]
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        
        # Empty pattern matches empty string
        dp[0][0] = True
        
        # Handle patterns with leading '*'
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                dp[0][j] = dp[0][j - 1]
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    # '*' can match empty sequence or any character
                    dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
                elif p[j - 1] == '?' or p[j - 1] == s[i - 1]:
                    # '?' matches any character, or exact match
                    dp[i][j] = dp[i - 1][j - 1]
        
        return dp[m][n]
    
    def isMatch2(self, s: str, p: str) -> bool:
        """
        Approach 2: Space-Optimized DP
        
        Optimize space usage by using only two arrays.
        
        Time: O(|s| * |p|)
        Space: O(|p|)
        """
        m, n = len(s), len(p)
        
        # Use two arrays for current and previous row
        prev = [False] * (n + 1)
        curr = [False] * (n + 1)
        
        # Base case
        prev[0] = True
        
        # Initialize first row for patterns with '*'
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                prev[j] = prev[j - 1]
        
        # Fill DP table row by row
        for i in range(1, m + 1):
            curr[0] = False  # Empty pattern can't match non-empty string
            
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    curr[j] = curr[j - 1] or prev[j]
                elif p[j - 1] == '?' or p[j - 1] == s[i - 1]:
                    curr[j] = prev[j - 1]
                else:
                    curr[j] = False
            
            # Swap arrays
            prev, curr = curr, prev
        
        return prev[n]
    
    def isMatch3(self, s: str, p: str) -> bool:
        """
        Approach 3: Recursive with Memoization
        
        Top-down recursive approach with caching.
        
        Time: O(|s| * |p|)
        Space: O(|s| * |p|) for memoization
        """
        memo = {}
        
        def dp(i: int, j: int) -> bool:
            if (i, j) in memo:
                return memo[(i, j)]
            
            # Base cases
            if j >= len(p):
                result = i >= len(s)
            elif i >= len(s):
                # Check if remaining pattern is all '*'
                result = all(c == '*' for c in p[j:])
            else:
                if p[j] == '*':
                    # '*' can match empty or any sequence
                    result = dp(i, j + 1) or dp(i + 1, j)
                elif p[j] == '?' or p[j] == s[i]:
                    # Character match
                    result = dp(i + 1, j + 1)
                else:
                    # No match
                    result = False
            
            memo[(i, j)] = result
            return result
        
        return dp(0, 0)
    
    def isMatch4(self, s: str, p: str) -> bool:
        """
        Approach 4: Two Pointers with Backtracking
        
        Greedy approach with backtracking when needed.
        
        Time: O(|s| * |p|) worst case, O(|s| + |p|) average
        Space: O(1)
        """
        s_idx = p_idx = 0
        star_idx = star_match = -1
        
        while s_idx < len(s):
            # Characters match or pattern has '?'
            if p_idx < len(p) and (p[p_idx] == s[s_idx] or p[p_idx] == '?'):
                s_idx += 1
                p_idx += 1
            # Pattern has '*'
            elif p_idx < len(p) and p[p_idx] == '*':
                star_idx = p_idx  # Remember '*' position
                star_match = s_idx  # Remember current string position
                p_idx += 1
            # Backtrack if we have seen '*'
            elif star_idx != -1:
                p_idx = star_idx + 1
                star_match += 1
                s_idx = star_match
            else:
                return False
        
        # Skip remaining '*' in pattern
        while p_idx < len(p) and p[p_idx] == '*':
            p_idx += 1
        
        return p_idx == len(p)
    
    def isMatch5(self, s: str, p: str) -> bool:
        """
        Approach 5: Finite State Automaton
        
        Build automaton from pattern and simulate on string.
        
        Time: O(|p|^2 + |s| * |p|)
        Space: O(|p|^2)
        """
        # Simplify pattern by removing consecutive '*'
        simplified_pattern = []
        for char in p:
            if not simplified_pattern or char != '*' or simplified_pattern[-1] != '*':
                simplified_pattern.append(char)
        
        p = ''.join(simplified_pattern)
        
        # Build state transition table
        states = self._build_automaton(p)
        
        # Simulate automaton
        current_states = {0}
        
        for char in s:
            next_states = set()
            
            for state in current_states:
                for next_state, conditions in states.get(state, []):
                    if self._matches_condition(char, conditions):
                        next_states.add(next_state)
            
            current_states = next_states
            if not current_states:
                return False
        
        # Check if any accepting state is reached
        return len(p) in current_states
    
    def _build_automaton(self, pattern: str) -> Dict[int, List[Tuple[int, str]]]:
        """Build finite automaton from pattern"""
        states = {}
        
        for i, char in enumerate(pattern):
            if char == '*':
                # Self-loop and forward transition
                states[i] = [(i, 'any'), (i + 1, 'epsilon')]
            else:
                # Regular transition
                states[i] = [(i + 1, char)]
        
        return states
    
    def _matches_condition(self, char: str, condition: str) -> bool:
        """Check if character matches transition condition"""
        if condition == 'any' or condition == '?':
            return True
        elif condition == 'epsilon':
            return False  # Epsilon transitions handled separately
        else:
            return char == condition
    
    def isMatch6(self, s: str, p: str) -> bool:
        """
        Approach 6: Segment-based Matching
        
        Split pattern by '*' and match segments.
        
        Time: O(|s| * number_of_segments)
        Space: O(number_of_segments)
        """
        # Split pattern by '*'
        segments = []
        current_segment = ""
        
        for char in p:
            if char == '*':
                if current_segment:
                    segments.append(current_segment)
                    current_segment = ""
                # Mark that we have '*' between segments
                if not segments or segments[-1] != '*':
                    segments.append('*')
            else:
                current_segment += char
        
        if current_segment:
            segments.append(current_segment)
        
        return self._match_segments(s, segments)
    
    def _match_segments(self, s: str, segments: List[str]) -> bool:
        """Match string against pattern segments"""
        if not segments:
            return not s
        
        if segments[0] == '*':
            # First segment is '*'
            if len(segments) == 1:
                return True  # Only '*' matches everything
            
            # Try matching remaining segments at each position
            for i in range(len(s) + 1):
                if self._match_segments(s[i:], segments[1:]):
                    return True
            return False
        else:
            # First segment is a pattern
            segment = segments[0]
            
            # Must match from beginning
            if len(segment) > len(s):
                return False
            
            # Check if segment matches prefix
            for i in range(len(segment)):
                if segment[i] != '?' and segment[i] != s[i]:
                    return False
            
            # Recursively match remaining
            return self._match_segments(s[len(segment):], segments[1:])
    
    def isMatch7(self, s: str, p: str) -> bool:
        """
        Approach 7: Iterative with Pattern Preprocessing
        
        Preprocess pattern and use iterative matching.
        
        Time: O(|s| + |p|) preprocessing + O(|s| * |p|) matching
        Space: O(|p|) for preprocessing
        """
        # Preprocess pattern: remove consecutive '*'
        processed_pattern = []
        i = 0
        
        while i < len(p):
            if p[i] == '*':
                processed_pattern.append('*')
                # Skip consecutive '*'
                while i < len(p) and p[i] == '*':
                    i += 1
            else:
                processed_pattern.append(p[i])
                i += 1
        
        return self._iterative_match(s, ''.join(processed_pattern))
    
    def _iterative_match(self, s: str, p: str) -> bool:
        """Iterative matching with preprocessed pattern"""
        m, n = len(s), len(p)
        
        # Use rolling arrays for space efficiency
        prev = [False] * (n + 1)
        curr = [False] * (n + 1)
        
        prev[0] = True
        
        # Initialize for patterns starting with '*'
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                prev[j] = prev[j - 1]
        
        for i in range(1, m + 1):
            curr[0] = False
            
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    curr[j] = curr[j - 1] or prev[j]
                elif p[j - 1] == '?' or p[j - 1] == s[i - 1]:
                    curr[j] = prev[j - 1]
                else:
                    curr[j] = False
            
            prev, curr = curr, prev
        
        return prev[n]


def test_basic_cases():
    """Test basic wildcard matching functionality"""
    print("=== Testing Basic Cases ===")
    
    solution = Solution()
    
    test_cases = [
        # LeetCode examples
        ("aa", "a", False),
        ("aa", "*", True),
        ("cb", "?a", False),
        ("adceb", "*a*b*", True),
        ("acdcb", "a*c?b", False),
        
        # Basic patterns
        ("", "", True),
        ("", "*", True),
        ("a", "", False),
        ("a", "a", True),
        ("a", "?", True),
        ("a", "*", True),
        
        # Multiple wildcards
        ("abc", "a?c", True),
        ("abc", "a*c", True),
        ("abc", "*", True),
        ("abc", "?*", True),
        ("abc", "*?", True),
        
        # Complex patterns
        ("abcdef", "a*f", True),
        ("abcdef", "a*g", False),
        ("abcdef", "*d*", True),
        ("abcdef", "ab*ef", True),
        ("abcdef", "ab*cd*ef", True),
        
        # Edge cases
        ("", "?", False),
        ("a", "??", False),
        ("aa", "?", False),
        ("aa", "??", True),
    ]
    
    approaches = [
        ("Bottom-up DP", solution.isMatch1),
        ("Space-optimized", solution.isMatch2),
        ("Recursive+Memo", solution.isMatch3),
        ("Two Pointers", solution.isMatch4),
        ("Finite Automaton", solution.isMatch5),
        ("Segment-based", solution.isMatch6),
        ("Iterative", solution.isMatch7),
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
    
    s = "adceb"
    p = "*a*b*"
    
    print(f"String: '{s}'")
    print(f"Pattern: '{p}'")
    
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    
    # Base case
    dp[0][0] = True
    print(f"\nStep 1: dp[0][0] = True (empty matches empty)")
    
    # Initialize first row
    print(f"\nStep 2: Handle patterns with leading '*':")
    for j in range(1, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 1]
            print(f"  dp[0][{j}] = {dp[0][j]} (pattern '{p[:j]}' vs empty)")
    
    # Fill table
    print(f"\nStep 3: Fill DP table:")
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
                print(f"  dp[{i}][{j}] = {dp[i][j]} ('{s[:i]}' vs '{p[:j]}') - star")
            elif p[j - 1] == '?' or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                print(f"  dp[{i}][{j}] = {dp[i][j]} ('{s[:i]}' vs '{p[:j]}') - match")
            else:
                dp[i][j] = False
                print(f"  dp[{i}][{j}] = {dp[i][j]} ('{s[:i]}' vs '{p[:j]}') - no match")
    
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


def demonstrate_two_pointers():
    """Demonstrate two pointers approach"""
    print("\n=== Two Pointers Approach Demo ===")
    
    s = "adceb"
    p = "*a*b*"
    
    print(f"String: '{s}'")
    print(f"Pattern: '{p}'")
    
    s_idx = p_idx = 0
    star_idx = star_match = -1
    steps = []
    
    while s_idx < len(s):
        if p_idx < len(p) and (p[p_idx] == s[s_idx] or p[p_idx] == '?'):
            steps.append(f"Match '{s[s_idx]}' with '{p[p_idx]}' at s[{s_idx}], p[{p_idx}]")
            s_idx += 1
            p_idx += 1
        elif p_idx < len(p) and p[p_idx] == '*':
            steps.append(f"Found '*' at p[{p_idx}], remember position")
            star_idx = p_idx
            star_match = s_idx
            p_idx += 1
        elif star_idx != -1:
            steps.append(f"Backtrack: '*' at p[{star_idx}] matches '{s[star_match]}'")
            p_idx = star_idx + 1
            star_match += 1
            s_idx = star_match
        else:
            steps.append(f"No match for '{s[s_idx]}' and no '*' to backtrack")
            break
    
    # Handle remaining '*' in pattern
    while p_idx < len(p) and p[p_idx] == '*':
        steps.append(f"Skip remaining '*' at p[{p_idx}]")
        p_idx += 1
    
    print(f"\nMatching steps:")
    for i, step in enumerate(steps):
        print(f"  {i+1}. {step}")
    
    result = p_idx == len(p)
    print(f"\nResult: {result} (reached end of pattern: {p_idx == len(p)})")


def benchmark_approaches():
    """Benchmark different approaches"""
    print("\n=== Benchmarking Approaches ===")
    
    import time
    import random
    import string
    
    solution = Solution()
    
    # Generate test data
    def generate_string(length: int) -> str:
        return ''.join(random.choices(string.ascii_lowercase[:3], k=length))
    
    def generate_pattern(length: int) -> str:
        pattern = ""
        for _ in range(length):
            choice = random.random()
            if choice < 0.6:  # 60% normal characters
                pattern += random.choice(string.ascii_lowercase[:3])
            elif choice < 0.8:  # 20% '?'
                pattern += '?'
            else:  # 20% '*'
                pattern += '*'
        return pattern
    
    test_scenarios = [
        ("Short", [(generate_string(8), generate_pattern(6)) for _ in range(30)]),
        ("Medium", [(generate_string(15), generate_pattern(10)) for _ in range(20)]),
        ("Long", [(generate_string(25), generate_pattern(15)) for _ in range(10)]),
    ]
    
    approaches = [
        ("Bottom-up DP", solution.isMatch1),
        ("Space-optimized", solution.isMatch2),
        ("Two Pointers", solution.isMatch4),
        ("Iterative", solution.isMatch7),
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


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solution = Solution()
    
    edge_cases = [
        # Empty cases
        ("", "", True, "Both empty"),
        ("", "*", True, "Empty string, star pattern"),
        ("", "?", False, "Empty string, question mark"),
        ("a", "", False, "Non-empty string, empty pattern"),
        
        # Single characters
        ("a", "a", True, "Exact single character"),
        ("a", "?", True, "Question mark single character"),
        ("a", "*", True, "Star single character"),
        ("a", "b", False, "Different single character"),
        
        # Multiple stars
        ("", "***", True, "Multiple stars with empty string"),
        ("abc", "***", True, "Multiple stars with string"),
        ("abc", "*a*b*c*", True, "Stars between characters"),
        
        # Complex patterns
        ("abcdef", "*", True, "Single star matches everything"),
        ("abcdef", "?*", True, "Question mark then star"),
        ("abcdef", "*?", True, "Star then question mark"),
        ("abcdef", "??????", True, "All question marks"),
        ("abcdef", "?????", False, "Too few question marks"),
        ("abcdef", "???????", False, "Too many question marks"),
        
        # Patterns with no stars or question marks
        ("hello", "hello", True, "Exact match"),
        ("hello", "world", False, "Complete mismatch"),
        
        # Boundary conditions
        ("a", "a*", True, "Character followed by star"),
        ("a", "*a", True, "Star followed by character"),
        ("ab", "a*b", True, "Star in middle"),
        ("aaa", "a*a", True, "Star with repeated character"),
    ]
    
    for s, p, expected, description in edge_cases:
        print(f"\n{description}:")
        print(f"  s='{s}', p='{p}'")
        
        try:
            result = solution.isMatch4(s, p)  # Using two pointers approach
            status = "✓" if result == expected else "✗"
            print(f"  Result: {result} {status}")
        except Exception as e:
            print(f"  Error: {e}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    solution = Solution()
    
    # Application 1: File globbing
    print("1. File System Globbing:")
    
    filenames = [
        "config.txt", "config.ini", "data.csv", "image.jpg", 
        "script.py", "readme.md", "test.py", "backup.bak"
    ]
    
    glob_patterns = [
        ("*.txt", "Text files"),
        ("*.py", "Python files"),
        ("config.*", "Config files"),
        ("*.*", "All files with extension"),
        ("????.*", "4-character names"),
    ]
    
    print(f"   Files: {filenames}")
    
    for pattern, description in glob_patterns:
        print(f"\n   Pattern '{pattern}' ({description}):")
        matches = [f for f in filenames if solution.isMatch4(f, pattern)]
        for match in matches:
            print(f"     ✓ {match}")
    
    # Application 2: Log filtering
    print(f"\n2. Log File Filtering:")
    
    log_entries = [
        "2024-01-01 ERROR database connection failed",
        "2024-01-01 INFO system startup complete",
        "2024-01-01 WARNING memory usage high",
        "2024-01-02 ERROR network timeout",
        "2024-01-02 DEBUG user login attempt"
    ]
    
    log_filters = [
        ("*ERROR*", "Error messages"),
        ("2024-01-01*", "Messages from Jan 1st"),
        ("*memory*", "Memory-related messages"),
        ("*DEBUG*", "Debug messages"),
    ]
    
    print(f"   Log entries: {len(log_entries)}")
    
    for pattern, description in log_filters:
        print(f"\n   Filter '{pattern}' ({description}):")
        matches = [entry for entry in log_entries 
                  if solution.isMatch4(entry, pattern)]
        for match in matches:
            print(f"     {match}")
    
    # Application 3: URL routing
    print(f"\n3. URL Route Matching:")
    
    urls = [
        "/api/users/123",
        "/api/users/456/profile", 
        "/api/posts/789",
        "/admin/dashboard",
        "/public/images/logo.png"
    ]
    
    route_patterns = [
        ("/api/users/*", "User API routes"),
        ("/api/*", "All API routes"),
        ("*/profile", "Profile pages"),
        ("/admin/*", "Admin routes"),
        ("*.png", "PNG images"),
    ]
    
    print(f"   URLs: {urls}")
    
    for pattern, description in route_patterns:
        print(f"\n   Route '{pattern}' ({description}):")
        matches = [url for url in urls if solution.isMatch4(url, pattern)]
        for match in matches:
            print(f"     {match}")


def analyze_complexity():
    """Analyze time and space complexity"""
    print("\n=== Complexity Analysis ===")
    
    complexity_analysis = [
        ("Bottom-up DP",
         "Time: O(|s| * |p|) - fill entire DP table",
         "Space: O(|s| * |p|) - full DP table"),
        
        ("Space-optimized DP",
         "Time: O(|s| * |p|) - same computation",
         "Space: O(|p|) - only two rows needed"),
        
        ("Recursive + Memoization",
         "Time: O(|s| * |p|) - each subproblem solved once",
         "Space: O(|s| * |p|) - memoization + recursion"),
        
        ("Two Pointers",
         "Time: O(|s| * |p|) worst case, O(|s| + |p|) average",
         "Space: O(1) - constant extra space"),
        
        ("Finite State Automaton",
         "Time: O(|p|^2 + |s| * |p|) - build automaton + simulate",
         "Space: O(|p|^2) - automaton transitions"),
        
        ("Segment-based",
         "Time: O(|s| * number_of_segments) - match each segment",
         "Space: O(number_of_segments) - segment storage"),
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
    print(f"  • Use Two Pointers for best average performance and minimal space")
    print(f"  • Use Space-optimized DP for guaranteed O(|s|*|p|) time")
    print(f"  • Use Bottom-up DP for debugging and clarity")
    print(f"  • Use Segment-based for patterns with many stars")


if __name__ == "__main__":
    test_basic_cases()
    demonstrate_dp_table()
    demonstrate_two_pointers()
    benchmark_approaches()
    test_edge_cases()
    demonstrate_real_world_applications()
    analyze_complexity()

"""
44. Wildcard Matching demonstrates comprehensive wildcard pattern matching:

1. Bottom-up DP - Classic dynamic programming table approach
2. Space-optimized DP - Memory-efficient two-row approach  
3. Recursive + Memoization - Top-down approach with caching
4. Two Pointers - Greedy approach with backtracking (most efficient)
5. Finite State Automaton - Automaton construction and simulation
6. Segment-based - Split pattern by '*' and match segments
7. Iterative - Preprocessed pattern with iterative matching

Each approach offers different trade-offs between time complexity,
space usage, and implementation complexity for wildcard matching.
"""
