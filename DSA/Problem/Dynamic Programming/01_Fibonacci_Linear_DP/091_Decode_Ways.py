"""
LeetCode 91: Decode Ways
Difficulty: Medium
Category: Fibonacci & Linear DP

PROBLEM DESCRIPTION:
===================
A message containing letters from A-Z can be encoded into numbers using the following mapping:

'A' -> "1"
'B' -> "2"
...
'Z' -> "26"

To decode an encoded message, all the digits must be grouped and then mapped back into letters 
using the reverse of the mapping above (there may be multiple ways). For example, "11106" can 
be mapped into:

"AAJF" with the grouping (1 1 10 6)
"KJF" with the grouping (11 10 6)

Note that the grouping (1 11 06) is invalid because "06" cannot be mapped into 'F' since "6" is different from "06".

Given a string s containing only digits, return the number of ways to decode it.

Example 1:
Input: s = "12"
Output: 2
Explanation: "12" could be decoded as "AB" (1 2) or "L" (12).

Example 2:
Input: s = "226"
Output: 3
Explanation: "226" could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).

Example 3:
Input: s = "06"
Output: 0
Explanation: "06" cannot be mapped to "F" because of the leading zero ("6" is different from "06").

Constraints:
- 1 <= s.length <= 100
- s contains only digits and may contain leading zeros.
"""

def num_decodings_bruteforce(s):
    """
    BRUTE FORCE APPROACH - RECURSION:
    ================================
    Try all possible ways to decode the string.
    At each position, try taking 1 digit or 2 digits if valid.
    
    Time Complexity: O(2^n) - exponential due to overlapping subproblems
    Space Complexity: O(n) - recursion stack depth
    """
    def decode_from(index):
        # Base case: reached end of string
        if index == len(s):
            return 1
        
        # Invalid: leading zero
        if s[index] == '0':
            return 0
        
        # Try single digit (always valid if not '0')
        ways = decode_from(index + 1)
        
        # Try two digits if valid
        if index + 1 < len(s):
            two_digit = int(s[index:index + 2])
            if 10 <= two_digit <= 26:
                ways += decode_from(index + 2)
        
        return ways
    
    return decode_from(0)


def num_decodings_memoization(s):
    """
    DYNAMIC PROGRAMMING - TOP DOWN (MEMOIZATION):
    ============================================
    Use memoization to avoid recalculating same subproblems.
    
    Time Complexity: O(n) - each subproblem calculated once
    Space Complexity: O(n) - memoization table + recursion stack
    """
    memo = {}
    
    def decode_from(index):
        if index == len(s):
            return 1
        
        if s[index] == '0':
            return 0
        
        if index in memo:
            return memo[index]
        
        # Single digit
        ways = decode_from(index + 1)
        
        # Two digits
        if index + 1 < len(s):
            two_digit = int(s[index:index + 2])
            if 10 <= two_digit <= 26:
                ways += decode_from(index + 2)
        
        memo[index] = ways
        return ways
    
    return decode_from(0)


def num_decodings_tabulation(s):
    """
    DYNAMIC PROGRAMMING - BOTTOM UP (TABULATION):
    =============================================
    Build solution from bottom up using DP array.
    dp[i] = number of ways to decode s[i:]
    
    Time Complexity: O(n) - single pass through string
    Space Complexity: O(n) - DP array
    """
    if not s or s[0] == '0':
        return 0
    
    n = len(s)
    dp = [0] * (n + 1)
    dp[n] = 1  # Base case: empty string has 1 way
    
    # Fill DP array from right to left
    for i in range(n - 1, -1, -1):
        if s[i] == '0':
            dp[i] = 0
        else:
            # Single digit
            dp[i] = dp[i + 1]
            
            # Two digits
            if i + 1 < n:
                two_digit = int(s[i:i + 2])
                if 10 <= two_digit <= 26:
                    dp[i] += dp[i + 2]
    
    return dp[0]


def num_decodings_space_optimized(s):
    """
    SPACE OPTIMIZED DYNAMIC PROGRAMMING:
    ===================================
    Since we only need next two values, use variables instead of array.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if not s or s[0] == '0':
        return 0
    
    n = len(s)
    next2 = 1  # dp[i+2]
    next1 = 1  # dp[i+1]
    
    for i in range(n - 1, -1, -1):
        current = 0
        
        if s[i] != '0':
            # Single digit
            current = next1
            
            # Two digits
            if i + 1 < n:
                two_digit = int(s[i:i + 2])
                if 10 <= two_digit <= 26:
                    current += next2
        
        # Update for next iteration
        next2 = next1
        next1 = current
    
    return next1


def num_decodings_forward_dp(s):
    """
    FORWARD DP APPROACH:
    ===================
    Build solution forward: dp[i] = ways to decode s[0:i]
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(n) - DP array
    """
    if not s or s[0] == '0':
        return 0
    
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1  # Empty string
    dp[1] = 1  # First character (already checked not '0')
    
    for i in range(2, n + 1):
        # Single digit (current character)
        if s[i - 1] != '0':
            dp[i] += dp[i - 1]
        
        # Two digits (previous + current character)
        two_digit = int(s[i - 2:i])
        if 10 <= two_digit <= 26:
            dp[i] += dp[i - 2]
    
    return dp[n]


def num_decodings_forward_optimized(s):
    """
    FORWARD DP SPACE OPTIMIZED:
    ==========================
    Forward approach with O(1) space.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if not s or s[0] == '0':
        return 0
    
    n = len(s)
    prev2 = 1  # dp[i-2]
    prev1 = 1  # dp[i-1]
    
    for i in range(2, n + 1):
        current = 0
        
        # Single digit
        if s[i - 1] != '0':
            current += prev1
        
        # Two digits
        two_digit = int(s[i - 2:i])
        if 10 <= two_digit <= 26:
            current += prev2
        
        # Update for next iteration
        prev2 = prev1
        prev1 = current
    
    return prev1


def num_decodings_with_decoding_list(s):
    """
    FIND ALL POSSIBLE DECODINGS:
    ===========================
    Return count and actual list of all possible decodings.
    
    Time Complexity: O(2^n) - generate all possible decodings
    Space Complexity: O(2^n) - store all decodings
    """
    if not s or s[0] == '0':
        return 0, []
    
    all_decodings = []
    
    def decode_from(index, current_decoding):
        if index == len(s):
            all_decodings.append(current_decoding)
            return
        
        if s[index] == '0':
            return
        
        # Single digit
        digit = int(s[index])
        letter = chr(ord('A') + digit - 1)
        decode_from(index + 1, current_decoding + letter)
        
        # Two digits
        if index + 1 < len(s):
            two_digit = int(s[index:index + 2])
            if 10 <= two_digit <= 26:
                letter = chr(ord('A') + two_digit - 1)
                decode_from(index + 2, current_decoding + letter)
    
    decode_from(0, "")
    return len(all_decodings), all_decodings


def num_decodings_optimized_conditions(s):
    """
    OPTIMIZED WITH CONDITION CHECKING:
    =================================
    Optimized version with better condition handling.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if not s or s[0] == '0':
        return 0
    
    n = len(s)
    if n == 1:
        return 1
    
    prev2 = 1  # Ways to decode up to i-2
    prev1 = 1  # Ways to decode up to i-1
    
    for i in range(1, n):
        current = 0
        
        # Check single digit (current character)
        if s[i] != '0':
            current += prev1
        
        # Check two digits (previous + current)
        prev_digit = int(s[i - 1])
        curr_digit = int(s[i])
        
        if prev_digit == 1 or (prev_digit == 2 and curr_digit <= 6):
            current += prev2
        
        prev2 = prev1
        prev1 = current
    
    return prev1


def num_decodings_edge_case_handler(s):
    """
    COMPREHENSIVE EDGE CASE HANDLING:
    ================================
    Handle all possible edge cases explicitly.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    if not s:
        return 0
    
    n = len(s)
    
    # Edge cases
    if s[0] == '0':
        return 0
    
    if n == 1:
        return 1
    
    # Check for consecutive zeros (impossible to decode)
    for i in range(1, n):
        if s[i] == '0' and s[i-1] not in '12':
            return 0
    
    prev2 = 1
    prev1 = 1
    
    for i in range(1, n):
        current = 0
        
        # Single digit (if not zero)
        if s[i] != '0':
            current += prev1
        
        # Two digits (if valid range 10-26)
        two_digit_num = int(s[i-1:i+1])
        if 10 <= two_digit_num <= 26:
            current += prev2
        
        prev2 = prev1
        prev1 = current
    
    return prev1


# Test cases
def test_num_decodings():
    """Test all implementations with various inputs"""
    test_cases = [
        ("12", 2),
        ("226", 3),
        ("06", 0),
        ("0", 0),
        ("10", 1),
        ("27", 1),
        ("11106", 2),
        ("111111", 21),
        ("1201234", 3),
        ("12120", 3),
        ("1111111111", 89),
        ("100", 0),
        ("101", 1)
    ]
    
    print("Testing Decode Ways Solutions:")
    print("=" * 70)
    
    for i, (s, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: s = '{s}'")
        print(f"Expected: {expected}")
        
        # Test approaches (skip brute force for long strings)
        if len(s) <= 10:
            brute = num_decodings_bruteforce(s)
            print(f"Brute Force:      {brute:>3} {'✓' if brute == expected else '✗'}")
        
        memo = num_decodings_memoization(s)
        tab = num_decodings_tabulation(s)
        space_opt = num_decodings_space_optimized(s)
        forward = num_decodings_forward_dp(s)
        forward_opt = num_decodings_forward_optimized(s)
        opt_cond = num_decodings_optimized_conditions(s)
        edge_handler = num_decodings_edge_case_handler(s)
        
        print(f"Memoization:      {memo:>3} {'✓' if memo == expected else '✗'}")
        print(f"Tabulation:       {tab:>3} {'✓' if tab == expected else '✗'}")
        print(f"Space Optimized:  {space_opt:>3} {'✓' if space_opt == expected else '✗'}")
        print(f"Forward DP:       {forward:>3} {'✓' if forward == expected else '✗'}")
        print(f"Forward Opt:      {forward_opt:>3} {'✓' if forward_opt == expected else '✗'}")
        print(f"Opt Conditions:   {opt_cond:>3} {'✓' if opt_cond == expected else '✗'}")
        print(f"Edge Handler:     {edge_handler:>3} {'✓' if edge_handler == expected else '✗'}")
        
        # Show all decodings for small strings
        if expected > 0 and expected <= 10 and len(s) <= 6:
            count, decodings = num_decodings_with_decoding_list(s)
            print(f"All decodings: {decodings}")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(2^n),   Space: O(n)")
    print("Memoization:      Time: O(n),     Space: O(n)")
    print("Tabulation:       Time: O(n),     Space: O(n)")
    print("Space Optimized:  Time: O(n),     Space: O(1)")
    print("Forward DP:       Time: O(n),     Space: O(n)")
    print("Forward Opt:      Time: O(n),     Space: O(1)")
    print("Opt Conditions:   Time: O(n),     Space: O(1)")
    print("Edge Handler:     Time: O(n),     Space: O(1)")


if __name__ == "__main__":
    test_num_decodings()


"""
PATTERN RECOGNITION:
==================
This is a Fibonacci-like DP problem with constraints:
- At each position, can take 1 or 2 digits
- 1 digit: must not be '0'
- 2 digits: must be in range [10, 26]
- Count number of valid ways to decode

KEY INSIGHTS:
============
1. Similar to climbing stairs with conditions
2. At position i, ways depend on:
   - Taking 1 digit: dp[i-1] if s[i] != '0'
   - Taking 2 digits: dp[i-2] if s[i-1:i+1] in [10,26]
3. Critical edge cases: leading zeros, invalid ranges

STATE DEFINITION:
================
dp[i] = number of ways to decode string s[0:i] or s[i:] (depending on direction)

RECURRENCE RELATION:
===================
Forward: dp[i] = (dp[i-1] if valid_single) + (dp[i-2] if valid_double)
Backward: dp[i] = (dp[i+1] if valid_single) + (dp[i+2] if valid_double)

Base cases:
- Forward: dp[0] = 1, dp[1] = 1 if s[0] != '0'
- Backward: dp[n] = 1, dp[n-1] = 1 if s[n-1] != '0'

EDGE CASES TO HANDLE:
====================
1. String starts with '0' → return 0
2. '0' not preceded by '1' or '2' → return 0
3. Two-digit numbers > 26 → only single digit valid
4. Empty string → return 0 or 1 (depending on definition)

OPTIMIZATION TECHNIQUES:
=======================
1. Space optimization: O(n) → O(1) using only prev2, prev1
2. Condition optimization: direct digit comparison instead of string conversion
3. Early termination: detect impossible cases upfront

VARIANTS TO PRACTICE:
====================
- Decode Ways II (639) - with '*' wildcard characters
- Integer Break (343) - similar constraint optimization
- Unique Paths (62) - similar Fibonacci pattern

INTERVIEW TIPS:
==============
1. Identify as Fibonacci-like problem with constraints
2. Clarify valid ranges (1-26 for A-Z)
3. Handle edge cases carefully (zeros, invalid ranges)
4. Show both forward and backward DP approaches
5. Optimize space from O(n) to O(1)
6. Trace through examples to verify logic
7. Discuss string parsing vs character manipulation
"""
