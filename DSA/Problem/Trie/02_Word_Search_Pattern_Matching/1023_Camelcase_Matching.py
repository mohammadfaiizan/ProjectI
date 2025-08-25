"""
1023. Camelcase Matching - Multiple Approaches
Difficulty: Medium

Given an array of strings queries and a string pattern, return a boolean array answer 
where answer[i] is true if queries[i] matches pattern, and false otherwise.

A query word matches pattern if you can insert lowercase letters to pattern to make 
it equal to query. You may insert each character at any position and you may not 
insert any character.

LeetCode Problem: https://leetcode.com/problems/camelcase-matching/

Example:
Input: queries = ["FooBar","FooBarTest","FootBall","FrameBuffer","ForceFeedBack"], pattern = "FB"
Output: [true,false,false,true,false]
"""

from typing import List

class TrieNode:
    """Trie node for camelcase pattern matching"""
    def __init__(self):
        self.children = {}
        self.is_pattern_end = False
        self.patterns = []  # Store patterns that end at this node

class Solution:
    
    def camelMatch1(self, queries: List[str], pattern: str) -> List[bool]:
        """
        Approach 1: Two Pointers Matching
        
        Use two pointers to match pattern with each query.
        
        Time: O(n * (m + p)) where n=queries, m=query length, p=pattern length
        Space: O(1)
        """
        def matches_pattern(query: str, pattern: str) -> bool:
            """Check if query matches pattern using two pointers"""
            i = j = 0  # i for query, j for pattern
            
            while i < len(query) and j < len(pattern):
                if query[i] == pattern[j]:
                    # Characters match, advance both pointers
                    i += 1
                    j += 1
                elif query[i].islower():
                    # Query has lowercase, pattern doesn't - skip it
                    i += 1
                else:
                    # Query has uppercase that doesn't match pattern
                    return False
            
            # Check remaining characters in query
            while i < len(query):
                if query[i].isupper():
                    return False  # Extra uppercase in query
                i += 1
            
            # All pattern characters should be matched
            return j == len(pattern)
        
        return [matches_pattern(query, pattern) for query in queries]
    
    def camelMatch2(self, queries: List[str], pattern: str) -> List[bool]:
        """
        Approach 2: Regular Expression Approach
        
        Convert pattern to regex and match against queries.
        
        Time: O(n * m) with regex optimization
        Space: O(p) for regex pattern
        """
        import re
        
        # Build regex pattern
        # Insert [a-z]* between each character in pattern
        regex_parts = []
        for i, char in enumerate(pattern):
            if i > 0:
                regex_parts.append('[a-z]*')  # Allow lowercase letters
            regex_parts.append(re.escape(char))
        
        # Add optional lowercase at beginning and end
        regex_pattern = '^[a-z]*' + ''.join(regex_parts) + '[a-z]*$'
        
        compiled_regex = re.compile(regex_pattern)
        
        return [bool(compiled_regex.match(query)) for query in queries]
    
    def camelMatch3(self, queries: List[str], pattern: str) -> List[bool]:
        """
        Approach 3: Dynamic Programming
        
        Use DP to check if query can match pattern.
        
        Time: O(n * m * p) where n=queries, m=query length, p=pattern length
        Space: O(m * p) for DP table
        """
        def dp_match(query: str, pattern: str) -> bool:
            """Check match using dynamic programming"""
            m, p = len(query), len(pattern)
            
            # dp[i][j] = can query[0:i] match pattern[0:j]
            dp = [[False] * (p + 1) for _ in range(m + 1)]
            
            # Empty pattern matches empty query
            dp[0][0] = True
            
            # Fill first column: empty pattern can match lowercase prefix
            for i in range(1, m + 1):
                if query[i-1].islower():
                    dp[i][0] = dp[i-1][0]
            
            # Fill DP table
            for i in range(1, m + 1):
                for j in range(1, p + 1):
                    if query[i-1] == pattern[j-1]:
                        # Characters match
                        dp[i][j] = dp[i-1][j-1]
                    elif query[i-1].islower():
                        # Query has extra lowercase - can skip
                        dp[i][j] = dp[i-1][j]
                    # else: query has uppercase that doesn't match - stays False
            
            return dp[m][p]
        
        return [dp_match(query, pattern) for query in queries]
    
    def camelMatch4(self, queries: List[str], pattern: str) -> List[bool]:
        """
        Approach 4: Trie-based Pattern Matching
        
        Build trie for pattern and match queries against it.
        
        Time: O(p + n * m) where p=pattern length
        Space: O(p) for trie
        """
        # Build trie for pattern
        root = TrieNode()
        
        # Insert pattern into trie with special handling
        node = root
        for char in pattern:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_pattern_end = True
        
        def match_with_trie(query: str) -> bool:
            """Match query against trie pattern"""
            def dfs(query_idx: int, trie_node: TrieNode, pattern_idx: int) -> bool:
                # Successfully matched entire pattern
                if pattern_idx == len(pattern):
                    # Check remaining query characters are all lowercase
                    while query_idx < len(query):
                        if query[query_idx].isupper():
                            return False
                        query_idx += 1
                    return True
                
                # Reached end of query but not pattern
                if query_idx >= len(query):
                    return False
                
                query_char = query[query_idx]
                pattern_char = pattern[pattern_idx]
                
                if query_char == pattern_char:
                    # Characters match - advance both
                    return dfs(query_idx + 1, trie_node, pattern_idx + 1)
                elif query_char.islower():
                    # Skip lowercase in query
                    return dfs(query_idx + 1, trie_node, pattern_idx)
                else:
                    # Uppercase mismatch
                    return False
            
            return dfs(0, root, 0)
        
        return [match_with_trie(query) for query in queries]
    
    def camelMatch5(self, queries: List[str], pattern: str) -> List[bool]:
        """
        Approach 5: State Machine Approach
        
        Model pattern matching as finite state machine.
        
        Time: O(n * m)
        Space: O(p) for state machine
        """
        def state_machine_match(query: str, pattern: str) -> bool:
            """Use state machine for pattern matching"""
            state = 0  # Current position in pattern
            
            for char in query:
                if state < len(pattern) and char == pattern[state]:
                    # Advance state on pattern match
                    state += 1
                elif char.islower():
                    # Lowercase characters are always allowed
                    continue
                else:
                    # Uppercase character that doesn't match pattern
                    return False
            
            # Must have matched entire pattern
            return state == len(pattern)
        
        return [state_machine_match(query, pattern) for query in queries]
    
    def camelMatch6(self, queries: List[str], pattern: str) -> List[bool]:
        """
        Approach 6: Optimized Two Pointers with Early Termination
        
        Enhanced two pointers with optimizations.
        
        Time: O(n * (m + p)) with early termination
        Space: O(1)
        """
        def optimized_match(query: str, pattern: str) -> bool:
            """Optimized matching with early termination"""
            # Quick checks
            if not pattern:
                return not any(c.isupper() for c in query)
            
            # Count uppercase letters
            query_upper = sum(1 for c in query if c.isupper())
            pattern_upper = sum(1 for c in pattern if c.isupper())
            
            # Query must have exactly the same number of uppercase letters
            if query_upper != pattern_upper:
                return False
            
            # Two pointers matching
            i = j = 0
            
            while i < len(query) and j < len(pattern):
                if query[i] == pattern[j]:
                    i += 1
                    j += 1
                elif query[i].islower():
                    i += 1
                else:
                    return False
            
            # Check remaining query characters
            while i < len(query):
                if query[i].isupper():
                    return False
                i += 1
            
            return j == len(pattern)
        
        return [optimized_match(query, pattern) for query in queries]


def test_basic_cases():
    """Test basic functionality"""
    print("=== Testing Basic Cases ===")
    
    solution = Solution()
    
    test_cases = [
        # LeetCode example
        (["FooBar","FooBarTest","FootBall","FrameBuffer","ForceFeedBack"], "FB",
         [True, False, False, True, False]),
        
        # Simple cases
        (["CompetitiveProgramming"], "CP", [True]),
        (["CompetitiveProgramming"], "CPP", [False]),
        (["ForceFeedBack"], "FB", [True]),
        (["ForceFeedBack"], "FeedBack", [False]),  # Missing uppercase
        
        # Edge cases
        ([""], "", [True]),
        (["a"], "", [True]),
        (["A"], "", [False]),  # Extra uppercase
        (["Ab"], "A", [True]),
        (["aB"], "B", [True]),
        
        # Complex patterns
        (["ILoveYou"], "ILY", [True]),
        (["ILoveYou"], "ILoveYou", [True]),
        (["iLoveYou"], "ILY", [False]),  # Wrong case
    ]
    
    approaches = [
        ("Two Pointers", solution.camelMatch1),
        ("Regex", solution.camelMatch2),
        ("Dynamic Programming", solution.camelMatch3),
        ("Trie-based", solution.camelMatch4),
        ("State Machine", solution.camelMatch5),
        ("Optimized 2P", solution.camelMatch6),
    ]
    
    for i, (queries, pattern, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: pattern='{pattern}', queries={queries}")
        print(f"Expected: {expected}")
        
        for name, method in approaches:
            try:
                result = method(queries[:], pattern)
                status = "✓" if result == expected else "✗"
                print(f"  {name:15}: {result} {status}")
            except Exception as e:
                print(f"  {name:15}: Error - {e}")


def demonstrate_matching_process():
    """Demonstrate the matching process step by step"""
    print("\n=== Matching Process Demo ===")
    
    query = "FooBar"
    pattern = "FB"
    
    print(f"Query: '{query}'")
    print(f"Pattern: '{pattern}'")
    print(f"Goal: Check if pattern can match query by inserting lowercase letters")
    
    # Two pointers approach
    print(f"\nTwo Pointers Approach:")
    
    i = j = 0  # i for query, j for pattern
    steps = []
    
    while i < len(query) and j < len(pattern):
        query_char = query[i]
        pattern_char = pattern[j]
        
        if query_char == pattern_char:
            steps.append(f"  Step {len(steps)+1}: Match '{query_char}' at query[{i}] with pattern[{j}]")
            i += 1
            j += 1
        elif query_char.islower():
            steps.append(f"  Step {len(steps)+1}: Skip lowercase '{query_char}' at query[{i}]")
            i += 1
        else:
            steps.append(f"  Step {len(steps)+1}: Uppercase mismatch '{query_char}' != '{pattern_char}' - FAIL")
            break
    
    # Check remaining characters
    while i < len(query):
        char = query[i]
        if char.isupper():
            steps.append(f"  Step {len(steps)+1}: Extra uppercase '{char}' at end - FAIL")
            break
        else:
            steps.append(f"  Step {len(steps)+1}: Skip remaining lowercase '{char}'")
        i += 1
    
    for step in steps:
        print(step)
    
    success = j == len(pattern) and i == len(query)
    print(f"\nResult: {'MATCH' if success else 'NO MATCH'}")
    print(f"Pattern fully matched: {j == len(pattern)}")
    print(f"Query fully processed: {i == len(query)}")


def demonstrate_pattern_analysis():
    """Demonstrate pattern analysis techniques"""
    print("\n=== Pattern Analysis Demo ===")
    
    queries = ["FooBar", "FooBarTest", "FootBall", "FrameBuffer"]
    pattern = "FB"
    
    print(f"Pattern: '{pattern}'")
    print(f"Queries: {queries}")
    
    # Analyze each query
    for query in queries:
        print(f"\nAnalyzing '{query}':")
        
        # Count uppercase letters
        query_upper = [c for c in query if c.isupper()]
        pattern_upper = [c for c in pattern if c.isupper()]
        
        print(f"  Query uppercase: {query_upper}")
        print(f"  Pattern uppercase: {pattern_upper}")
        
        # Check uppercase count
        if len(query_upper) != len(pattern_upper):
            print(f"  ✗ Different uppercase count: {len(query_upper)} != {len(pattern_upper)}")
            continue
        
        # Check if uppercase letters match in order
        uppercase_match = True
        for i, (q_char, p_char) in enumerate(zip(query_upper, pattern_upper)):
            if q_char != p_char:
                print(f"  ✗ Uppercase mismatch at position {i}: '{q_char}' != '{p_char}'")
                uppercase_match = False
                break
        
        if uppercase_match:
            print(f"  ✓ Uppercase letters match in order")
            
            # Show how lowercase can be inserted
            result = []
            j = 0  # pattern index
            
            for char in query:
                if j < len(pattern) and char == pattern[j]:
                    result.append(f"[{char}]")  # Pattern character
                    j += 1
                else:
                    result.append(char)  # Inserted lowercase
            
            print(f"  Pattern insertion: {''.join(result)}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solution = Solution()
    
    edge_cases = [
        # Empty cases
        ([], "", "Empty queries"),
        ([""], "", "Empty query and pattern"),
        (["a"], "", "Empty pattern with lowercase"),
        (["A"], "", "Empty pattern with uppercase"),
        
        # Single character cases
        (["a"], "a", "Single lowercase match"),
        (["A"], "A", "Single uppercase match"),
        (["a"], "A", "Case mismatch"),
        (["A"], "a", "Case mismatch reverse"),
        
        # Only lowercase
        (["hello"], "", "Only lowercase query"),
        (["hello"], "h", "Lowercase query with pattern"),
        
        # Only uppercase
        (["HELLO"], "HELLO", "All uppercase match"),
        (["HELLO"], "HEL", "Partial uppercase match"),
        
        # Mixed complex cases
        (["aAbBcC"], "ABC", "Alternating case"),
        (["AaBbCc"], "ABC", "Pattern with lowercase between"),
        (["AbC"], "AC", "Missing middle uppercase"),
    ]
    
    for queries, pattern, description in edge_cases:
        print(f"\n{description}:")
        print(f"  Queries: {queries}")
        print(f"  Pattern: '{pattern}'")
        
        try:
            result = solution.camelMatch1(queries, pattern)
            print(f"  Result: {result}")
        except Exception as e:
            print(f"  Error: {e}")


def benchmark_approaches():
    """Benchmark different approaches"""
    print("\n=== Benchmarking Approaches ===")
    
    import time
    import random
    import string
    
    solution = Solution()
    
    # Generate test data
    def generate_camelcase_word(length: int) -> str:
        """Generate a camelcase word"""
        word = ""
        for i in range(length):
            if i == 0 or random.random() < 0.3:  # 30% chance for uppercase
                word += random.choice(string.ascii_uppercase)
            else:
                word += random.choice(string.ascii_lowercase)
        return word
    
    def generate_pattern(word: str) -> str:
        """Generate a pattern from a word"""
        pattern = ""
        for char in word:
            if char.isupper() and random.random() < 0.7:  # 70% chance to include uppercase
                pattern += char
        return pattern
    
    test_scenarios = [
        ("Small", [generate_camelcase_word(8) for _ in range(20)]),
        ("Medium", [generate_camelcase_word(12) for _ in range(100)]),
        ("Large", [generate_camelcase_word(16) for _ in range(500)]),
    ]
    
    approaches = [
        ("Two Pointers", solution.camelMatch1),
        ("Regex", solution.camelMatch2),
        ("State Machine", solution.camelMatch5),
        ("Optimized 2P", solution.camelMatch6),
    ]
    
    for scenario_name, queries in test_scenarios:
        # Generate pattern from first query
        pattern = generate_pattern(queries[0]) if queries else ""
        
        print(f"\n--- {scenario_name} Dataset ---")
        print(f"Queries: {len(queries)}, Pattern: '{pattern}', Avg query length: {sum(len(q) for q in queries)/len(queries):.1f}")
        
        for approach_name, method in approaches:
            start_time = time.time()
            
            # Run multiple times for better measurement
            for _ in range(5):
                result = method(queries[:], pattern)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 5
            
            true_count = sum(result) if queries else 0
            print(f"  {approach_name:15}: {avg_time*1000:.2f}ms ({true_count}/{len(queries)} matches)")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    solution = Solution()
    
    # Application 1: Class name validation
    print("1. Class Name Validation:")
    
    class_names = [
        "UserAuthenticationService",
        "DatabaseConnectionManager", 
        "FileSystemUtility",
        "NetworkRequestHandler",
        "ConfigurationParser"
    ]
    
    abbreviation = "UAS"  # UserAuthenticationService
    
    print(f"   Class names: {class_names}")
    print(f"   Looking for abbreviation: '{abbreviation}'")
    
    matches = solution.camelMatch1(class_names, abbreviation)
    
    for name, match in zip(class_names, matches):
        print(f"     '{name}': {'✓' if match else '✗'}")
    
    # Application 2: API endpoint matching
    print(f"\n2. API Endpoint Matching:")
    
    api_endpoints = [
        "GetUserProfile",
        "PostUserData", 
        "DeleteUserAccount",
        "GetProductInfo",
        "UpdateProductPrice"
    ]
    
    api_pattern = "GUP"  # GetUserProfile
    
    print(f"   API endpoints: {api_endpoints}")
    print(f"   Pattern: '{api_pattern}'")
    
    api_matches = solution.camelMatch1(api_endpoints, api_pattern)
    
    for endpoint, match in zip(api_endpoints, api_matches):
        print(f"     '{endpoint}': {'matches' if match else 'no match'}")
    
    # Application 3: Variable name suggestions
    print(f"\n3. Variable Name Suggestions:")
    
    variable_names = [
        "firstName",
        "lastName",
        "fullName",
        "emailAddress",
        "phoneNumber"
    ]
    
    var_pattern = "fN"  # firstName, fullName
    
    print(f"   Variable names: {variable_names}")
    print(f"   Typing pattern: '{var_pattern}'")
    
    var_matches = solution.camelMatch1(variable_names, var_pattern)
    
    suggestions = [name for name, match in zip(variable_names, var_matches) if match]
    print(f"   Suggestions: {suggestions}")


def analyze_complexity():
    """Analyze time and space complexity"""
    print("\n=== Complexity Analysis ===")
    
    complexity_analysis = [
        ("Two Pointers",
         "Time: O(n * (m + p)) - scan each query once",
         "Space: O(1) - constant extra space"),
        
        ("Regular Expression",
         "Time: O(n * m) - regex matching per query", 
         "Space: O(p) - compiled regex pattern"),
        
        ("Dynamic Programming",
         "Time: O(n * m * p) - DP table for each query",
         "Space: O(m * p) - DP table storage"),
        
        ("State Machine",
         "Time: O(n * m) - linear scan per query",
         "Space: O(1) - constant state storage"),
        
        ("Optimized Two Pointers",
         "Time: O(n * (m + p)) - with early termination",
         "Space: O(1) - constant extra space"),
    ]
    
    print("Approach Analysis:")
    for approach, time_complexity, space_complexity in complexity_analysis:
        print(f"\n{approach}:")
        print(f"  {time_complexity}")
        print(f"  {space_complexity}")
    
    print(f"\nWhere:")
    print(f"  n = number of queries")
    print(f"  m = average query length")  
    print(f"  p = pattern length")
    
    print(f"\nRecommendations:")
    print(f"  • Use Two Pointers for simplicity and good performance")
    print(f"  • Use Regex for complex pattern requirements")
    print(f"  • Use State Machine for streaming/online scenarios")
    print(f"  • Use Optimized approach for performance-critical applications")


if __name__ == "__main__":
    test_basic_cases()
    demonstrate_matching_process()
    demonstrate_pattern_analysis()
    test_edge_cases()
    benchmark_approaches()
    demonstrate_real_world_applications()
    analyze_complexity()

"""
1023. Camelcase Matching demonstrates multiple pattern matching approaches:

1. Two Pointers - Simple scan matching pattern characters in order
2. Regular Expression - Convert pattern to regex with lowercase insertions
3. Dynamic Programming - DP table to track matching possibilities
4. Trie-based - Build pattern trie for structured matching
5. State Machine - Model pattern matching as finite state automaton
6. Optimized Two Pointers - Enhanced with early termination optimizations

Each approach shows different strategies for handling camelcase pattern
matching with varying complexity and optimization techniques.
"""
