"""
String Regex and Wildcard Matching
==================================

Topics: Regular expressions, wildcard patterns, string matching
Companies: Google, Facebook, Microsoft, Amazon
Difficulty: Hard
"""

from typing import List, Dict

class StringRegexWildcard:
    
    # ==========================================
    # 1. REGULAR EXPRESSION MATCHING
    # ==========================================
    
    def is_match_regex(self, s: str, p: str) -> bool:
        """LC 10: Regular Expression Matching - Time: O(m*n), Space: O(m*n)"""
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        
        dp[0][0] = True
        
        # Handle patterns like a*b*c*
        for j in range(2, n + 1):
            if p[j-1] == '*':
                dp[0][j] = dp[0][j-2]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j-1] == s[i-1] or p[j-1] == '.':
                    dp[i][j] = dp[i-1][j-1]
                elif p[j-1] == '*':
                    dp[i][j] = dp[i][j-2]  # zero occurrences
                    if p[j-2] == s[i-1] or p[j-2] == '.':
                        dp[i][j] = dp[i][j] or dp[i-1][j]
        
        return dp[m][n]
    
    def is_match_regex_recursive(self, s: str, p: str) -> bool:
        """Recursive approach with memoization"""
        memo = {}
        
        def dp(i: int, j: int) -> bool:
            if (i, j) in memo:
                return memo[(i, j)]
            
            if j == len(p):
                return i == len(s)
            
            first_match = i < len(s) and (p[j] == s[i] or p[j] == '.')
            
            if j + 1 < len(p) and p[j + 1] == '*':
                result = dp(i, j + 2) or (first_match and dp(i + 1, j))
            else:
                result = first_match and dp(i + 1, j + 1)
            
            memo[(i, j)] = result
            return result
        
        return dp(0, 0)
    
    # ==========================================
    # 2. WILDCARD PATTERN MATCHING
    # ==========================================
    
    def is_match_wildcard(self, s: str, p: str) -> bool:
        """LC 44: Wildcard Pattern Matching - Time: O(m*n), Space: O(m*n)"""
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        
        dp[0][0] = True
        
        # Handle patterns starting with *
        for j in range(1, n + 1):
            if p[j-1] == '*':
                dp[0][j] = dp[0][j-1]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j-1] == '*':
                    dp[i][j] = dp[i-1][j] or dp[i][j-1]
                elif p[j-1] == '?' or s[i-1] == p[j-1]:
                    dp[i][j] = dp[i-1][j-1]
        
        return dp[m][n]
    
    def is_match_wildcard_optimized(self, s: str, p: str) -> bool:
        """Space optimized wildcard matching - Space: O(n)"""
        m, n = len(s), len(p)
        dp = [False] * (n + 1)
        dp[0] = True
        
        # Handle patterns starting with *
        for j in range(1, n + 1):
            if p[j-1] == '*':
                dp[j] = dp[j-1]
        
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = False
            
            for j in range(1, n + 1):
                temp = dp[j]
                if p[j-1] == '*':
                    dp[j] = dp[j] or dp[j-1]
                elif p[j-1] == '?' or s[i-1] == p[j-1]:
                    dp[j] = prev
                else:
                    dp[j] = False
                prev = temp
        
        return dp[n]
    
    # ==========================================
    # 3. ADVANCED PATTERN MATCHING
    # ==========================================
    
    def match_multiple_patterns(self, s: str, patterns: List[str]) -> List[bool]:
        """Check if string matches any of multiple patterns"""
        results = []
        for pattern in patterns:
            if '*' in pattern or '?' in pattern:
                results.append(self.is_match_wildcard(s, pattern))
            else:
                results.append(s == pattern)
        return results
    
    def find_pattern_positions(self, s: str, pattern: str) -> List[int]:
        """Find all positions where pattern matches in string"""
        positions = []
        
        for i in range(len(s) - len(pattern) + 1):
            substring = s[i:i + len(pattern)]
            if self.is_match_wildcard(substring, pattern):
                positions.append(i)
        
        return positions
    
    def longest_matching_prefix(self, s: str, pattern: str) -> str:
        """Find longest prefix of s that matches pattern"""
        for i in range(len(s), 0, -1):
            if self.is_match_wildcard(s[:i], pattern):
                return s[:i]
        return ""
    
    # ==========================================
    # 4. GLOB PATTERN MATCHING
    # ==========================================
    
    def match_glob_pattern(self, s: str, pattern: str) -> bool:
        """Match glob patterns with *, ?, [], {}, etc."""
        def match_char_class(char: str, char_class: str) -> bool:
            """Match character against character class [a-z], [abc], etc."""
            if char_class.startswith('[') and char_class.endswith(']'):
                char_set = char_class[1:-1]
                
                # Handle negation [!abc] or [^abc]
                if char_set.startswith('!') or char_set.startswith('^'):
                    char_set = char_set[1:]
                    return char not in char_set
                
                # Handle ranges [a-z]
                if '-' in char_set and len(char_set) == 3:
                    start, end = char_set[0], char_set[2]
                    return start <= char <= end
                
                return char in char_set
            
            return char == char_class
        
        def match_brace_expansion(s: str, pattern: str) -> bool:
            """Handle brace expansion {a,b,c}"""
            if '{' not in pattern:
                return self.is_match_wildcard(s, pattern)
            
            start = pattern.find('{')
            end = pattern.find('}', start)
            
            if start == -1 or end == -1:
                return self.is_match_wildcard(s, pattern)
            
            prefix = pattern[:start]
            suffix = pattern[end + 1:]
            options = pattern[start + 1:end].split(',')
            
            for option in options:
                new_pattern = prefix + option + suffix
                if match_brace_expansion(s, new_pattern):
                    return True
            
            return False
        
        return match_brace_expansion(s, pattern)
    
    # ==========================================
    # 5. REGEX COMPILATION AND OPTIMIZATION
    # ==========================================
    
    def compile_regex(self, pattern: str) -> Dict:
        """Compile regex pattern for faster matching"""
        compiled = {
            'pattern': pattern,
            'states': [],
            'transitions': {},
            'accepting_states': set()
        }
        
        # Simplified NFA construction
        state_count = 0
        
        for i, char in enumerate(pattern):
            if char == '.':
                compiled['states'].append(('any', state_count))
            elif char == '*':
                compiled['states'].append(('star', state_count))
            else:
                compiled['states'].append(('char', state_count, char))
            state_count += 1
        
        compiled['accepting_states'].add(state_count - 1)
        return compiled
    
    def match_compiled_regex(self, s: str, compiled_regex: Dict) -> bool:
        """Match string against compiled regex"""
        # Simplified matching logic
        current_states = {0}
        
        for char in s:
            next_states = set()
            
            for state in current_states:
                if state < len(compiled_regex['states']):
                    state_info = compiled_regex['states'][state]
                    
                    if state_info[0] == 'char' and len(state_info) > 2:
                        if char == state_info[2]:
                            next_states.add(state + 1)
                    elif state_info[0] == 'any':
                        next_states.add(state + 1)
                    elif state_info[0] == 'star':
                        next_states.add(state)  # Stay in same state
                        next_states.add(state + 1)  # Move to next state
            
            current_states = next_states
        
        return bool(current_states & compiled_regex['accepting_states'])
    
    # ==========================================
    # 6. PATTERN VALIDATION
    # ==========================================
    
    def is_valid_regex(self, pattern: str) -> bool:
        """Check if regex pattern is valid"""
        stack = []
        i = 0
        
        while i < len(pattern):
            char = pattern[i]
            
            if char == '(':
                stack.append('(')
            elif char == ')':
                if not stack or stack[-1] != '(':
                    return False
                stack.pop()
            elif char == '[':
                stack.append('[')
            elif char == ']':
                if not stack or stack[-1] != '[':
                    return False
                stack.pop()
            elif char == '\\':
                if i + 1 >= len(pattern):
                    return False
                i += 1  # Skip escaped character
            elif char == '*' or char == '+' or char == '?':
                if i == 0:
                    return False
                prev_char = pattern[i - 1]
                if prev_char in '*+?':
                    return False
            
            i += 1
        
        return len(stack) == 0
    
    def escape_regex_chars(self, s: str) -> str:
        """Escape special regex characters in string"""
        special_chars = r'\.^$*+?{}[]|()'
        escaped = []
        
        for char in s:
            if char in special_chars:
                escaped.append('\\' + char)
            else:
                escaped.append(char)
        
        return ''.join(escaped)

# Test Examples
def run_examples():
    srw = StringRegexWildcard()
    
    print("=== REGEX AND WILDCARD EXAMPLES ===\n")
    
    # Regular expression matching
    print("1. REGULAR EXPRESSION MATCHING:")
    print(f"Regex match 'aa' with 'a*': {srw.is_match_regex('aa', 'a*')}")
    print(f"Regex match 'ab' with '.*': {srw.is_match_regex('ab', '.*')}")
    
    # Wildcard matching
    print(f"Wildcard match 'adceb' with '*a*b*': {srw.is_match_wildcard('adceb', '*a*b*')}")
    print(f"Wildcard match 'acdcb' with 'a*c?b': {srw.is_match_wildcard('acdcb', 'a*c?b')}")
    
    # Advanced patterns
    print("\n2. ADVANCED PATTERNS:")
    patterns = ['a*b', 'a?c', 'hello']
    results = srw.match_multiple_patterns('abc', patterns)
    print(f"Multiple pattern matches for 'abc': {dict(zip(patterns, results))}")
    
    positions = srw.find_pattern_positions('abcabc', 'a?c')
    print(f"Pattern positions of 'a?c' in 'abcabc': {positions}")
    
    # Glob patterns
    print(f"Glob match 'test.txt' with '*.txt': {srw.match_glob_pattern('test.txt', '*.txt')}")
    
    # Pattern validation
    print("\n3. PATTERN VALIDATION:")
    print(f"Is valid regex 'a*b+': {srw.is_valid_regex('a*b+')}")
    print(f"Is valid regex '*abc': {srw.is_valid_regex('*abc')}")
    
    escaped = srw.escape_regex_chars("hello.world")
    print(f"Escaped regex: '{escaped}'")

if __name__ == "__main__":
    run_examples() 