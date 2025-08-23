"""
1542. Find Longest Awesome Substring - Multiple Approaches
Difficulty: Hard

You are given a string s. An awesome substring is a non-empty substring of s such that we can make any number of swaps in order to make it a palindrome.

Return the length of the maximum length awesome substring of s.

Note: This problem is included in the Tree Algorithms category because it demonstrates
advanced string processing techniques that are often used in tree-based string algorithms
and suffix tree constructions.
"""

from typing import Dict, List

class FindLongestAwesome:
    """Multiple approaches to find longest awesome substring"""
    
    def longestAwesome_bitmask_prefix(self, s: str) -> int:
        """
        Approach 1: Bitmask with Prefix State Tracking
        
        Use bitmask to track character parity and find longest valid substring.
        
        Time: O(N), Space: O(2^10) = O(1024)
        """
        n = len(s)
        if n == 0:
            return 0
        
        # Bitmask to track parity of each digit (0-9)
        mask = 0
        # Map from mask to first occurrence index
        first_occurrence = {0: -1}  # Empty prefix has mask 0
        max_length = 1  # At least one character is always awesome
        
        for i in range(n):
            digit = int(s[i])
            # Flip bit for current digit
            mask ^= (1 << digit)
            
            # Case 1: Even length palindrome (all digits have even count)
            if mask in first_occurrence:
                max_length = max(max_length, i - first_occurrence[mask])
            else:
                first_occurrence[mask] = i
            
            # Case 2: Odd length palindrome (exactly one digit has odd count)
            for d in range(10):
                target_mask = mask ^ (1 << d)
                if target_mask in first_occurrence:
                    max_length = max(max_length, i - first_occurrence[target_mask])
        
        return max_length
    
    def longestAwesome_optimized_bitmask(self, s: str) -> int:
        """
        Approach 2: Optimized Bitmask with Early Termination
        
        Optimize bitmask approach with better pruning.
        
        Time: O(N), Space: O(2^10)
        """
        n = len(s)
        if n == 0:
            return 0
        
        # Use array instead of dict for better performance
        first_seen = [-2] * (1 << 10)  # -2 means not seen
        first_seen[0] = -1  # Empty prefix
        
        mask = 0
        max_len = 1
        
        for i in range(n):
            digit = int(s[i])
            mask ^= (1 << digit)
            
            # Check for even-length palindrome
            if first_seen[mask] != -2:
                max_len = max(max_len, i - first_seen[mask])
            else:
                first_seen[mask] = i
            
            # Check for odd-length palindrome
            for d in range(10):
                odd_mask = mask ^ (1 << d)
                if first_seen[odd_mask] != -2:
                    max_len = max(max_len, i - first_seen[odd_mask])
        
        return max_len
    
    def longestAwesome_sliding_window_optimized(self, s: str) -> int:
        """
        Approach 3: Sliding Window with Bitmask Optimization
        
        Use sliding window concept with bitmask state tracking.
        
        Time: O(N), Space: O(2^10)
        """
        n = len(s)
        if n == 0:
            return 0
        
        # Track first occurrence of each bitmask state
        state_first = {}
        state_first[0] = -1
        
        current_state = 0
        result = 1
        
        for i in range(n):
            digit = int(s[i])
            current_state ^= (1 << digit)
            
            # Check if we've seen this exact state before (even palindrome)
            if current_state in state_first:
                result = max(result, i - state_first[current_state])
            else:
                state_first[current_state] = i
            
            # Check all possible odd palindromes (one digit with odd count)
            for bit in range(10):
                target_state = current_state ^ (1 << bit)
                if target_state in state_first:
                    result = max(result, i - state_first[target_state])
        
        return result
    
    def longestAwesome_dp_approach(self, s: str) -> int:
        """
        Approach 4: Dynamic Programming Approach
        
        Use DP to track valid substrings with character count analysis.
        
        Time: O(N²), Space: O(N)
        """
        n = len(s)
        if n == 0:
            return 0
        
        max_length = 1
        
        # For each starting position
        for start in range(n):
            count = [0] * 10
            
            # Extend substring from start
            for end in range(start, n):
                count[int(s[end])] += 1
                
                # Check if current substring can form palindrome
                odd_count = sum(1 for c in count if c % 2 == 1)
                
                if odd_count <= 1:  # Can form palindrome
                    max_length = max(max_length, end - start + 1)
        
        return max_length
    
    def longestAwesome_hash_based(self, s: str) -> int:
        """
        Approach 5: Hash-based State Tracking
        
        Use hash-based approach for state management.
        
        Time: O(N), Space: O(min(N, 2^10))
        """
        n = len(s)
        if n == 0:
            return 0
        
        # Use dictionary for sparse representation
        seen_states = {0: -1}
        current_mask = 0
        max_awesome = 1
        
        for i in range(n):
            digit = int(s[i])
            current_mask ^= (1 << digit)
            
            # Check for even-length awesome substring
            if current_mask in seen_states:
                length = i - seen_states[current_mask]
                max_awesome = max(max_awesome, length)
            else:
                seen_states[current_mask] = i
            
            # Check for odd-length awesome substring
            for d in range(10):
                target_mask = current_mask ^ (1 << d)
                if target_mask in seen_states:
                    length = i - seen_states[target_mask]
                    max_awesome = max(max_awesome, length)
        
        return max_awesome
    
    def longestAwesome_trie_based(self, s: str) -> int:
        """
        Approach 6: Trie-based Approach for State Management
        
        Use trie-like structure to manage bitmask states efficiently.
        
        Time: O(N), Space: O(2^10)
        """
        class TrieNode:
            def __init__(self):
                self.children = {}
                self.first_index = -1
        
        n = len(s)
        if n == 0:
            return 0
        
        root = TrieNode()
        root.first_index = -1
        
        mask = 0
        max_len = 1
        
        # Insert initial state
        current = root
        for bit in range(10):
            if mask & (1 << bit):
                if 1 not in current.children:
                    current.children[1] = TrieNode()
                current = current.children[1]
            else:
                if 0 not in current.children:
                    current.children[0] = TrieNode()
                current = current.children[0]
        current.first_index = -1
        
        for i in range(n):
            digit = int(s[i])
            mask ^= (1 << digit)
            
            # Search for current mask (even palindrome)
            current = root
            found = True
            for bit in range(10):
                bit_val = 1 if (mask & (1 << bit)) else 0
                if bit_val not in current.children:
                    found = False
                    break
                current = current.children[bit_val]
            
            if found and current.first_index != -1:
                max_len = max(max_len, i - current.first_index)
            
            # Search for masks differing by one bit (odd palindrome)
            for d in range(10):
                target_mask = mask ^ (1 << d)
                current = root
                found = True
                
                for bit in range(10):
                    bit_val = 1 if (target_mask & (1 << bit)) else 0
                    if bit_val not in current.children:
                        found = False
                        break
                    current = current.children[bit_val]
                
                if found and current.first_index != -1:
                    max_len = max(max_len, i - current.first_index)
            
            # Insert current mask if not seen
            current = root
            for bit in range(10):
                bit_val = 1 if (mask & (1 << bit)) else 0
                if bit_val not in current.children:
                    current.children[bit_val] = TrieNode()
                current = current.children[bit_val]
            
            if current.first_index == -1:
                current.first_index = i
        
        return max_len
    
    def longestAwesome_mathematical_approach(self, s: str) -> int:
        """
        Approach 7: Mathematical Approach with Bit Manipulation
        
        Use mathematical properties of palindromes and bit manipulation.
        
        Time: O(N), Space: O(2^10)
        """
        n = len(s)
        if n == 0:
            return 0
        
        # Precompute all possible single-bit differences
        single_bit_masks = [1 << i for i in range(10)]
        
        # Track first occurrence of each state
        first_occurrence = [n] * (1 << 10)
        first_occurrence[0] = -1
        
        state = 0
        max_length = 1
        
        for i in range(n):
            digit = int(s[i])
            state ^= (1 << digit)
            
            # Update first occurrence if this is the first time seeing this state
            if first_occurrence[state] == n:
                first_occurrence[state] = i
            
            # Check current state (even palindrome)
            if first_occurrence[state] < i:
                max_length = max(max_length, i - first_occurrence[state])
            
            # Check all single-bit differences (odd palindrome)
            for mask in single_bit_masks:
                target_state = state ^ mask
                if first_occurrence[target_state] < n:
                    max_length = max(max_length, i - first_occurrence[target_state])
        
        return max_length

def test_longest_awesome():
    """Test longest awesome substring algorithms"""
    solver = FindLongestAwesome()
    
    test_cases = [
        ("3242415", 5, "Example 1"),
        ("12345678", 1, "No awesome substring > 1"),
        ("213123", 6, "Entire string is awesome"),
        ("00", 2, "Simple case"),
        ("1234567890", 1, "All different digits"),
    ]
    
    algorithms = [
        ("Bitmask Prefix", solver.longestAwesome_bitmask_prefix),
        ("Optimized Bitmask", solver.longestAwesome_optimized_bitmask),
        ("Sliding Window", solver.longestAwesome_sliding_window_optimized),
        ("DP Approach", solver.longestAwesome_dp_approach),
        ("Hash-based", solver.longestAwesome_hash_based),
        ("Mathematical", solver.longestAwesome_mathematical_approach),
    ]
    
    print("=== Testing Find Longest Awesome Substring ===")
    
    for s, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"String: '{s}'")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(s)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:18} | {status} | Length: {result}")
            except Exception as e:
                print(f"{alg_name:18} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_longest_awesome()

"""
Find Longest Awesome Substring demonstrates advanced string
processing with bitmask techniques, palindrome analysis,
and state-based dynamic programming approaches.
"""
