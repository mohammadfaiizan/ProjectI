"""
String Basics and Operations - Professional Interview Level
===========================================================

This module covers fundamental string operations with advanced examples
commonly asked in technical interviews at top-tier companies.

Topics Covered:
- Advanced string manipulation
- Character frequency analysis
- String validation problems
- ASCII and Unicode operations
- Memory-efficient string operations

Author: Interview Prep DSA Module
Difficulty: Easy to Medium
Companies: Google, Microsoft, Amazon, Facebook, Apple
"""

from collections import Counter, defaultdict
import string
import re
from typing import List, Dict, Set, Tuple

class StringBasicsAndOperations:
    """Professional-level string operations for technical interviews"""
    
    def __init__(self):
        self.test_cases = []
    
    # ==========================================
    # 1. ADVANCED STRING REVERSAL TECHNIQUES
    # ==========================================
    
    def reverse_string_inplace(self, s: List[str]) -> None:
        """
        LC 344: Reverse String In-Place
        Time: O(n), Space: O(1)
        Companies: Google, Microsoft
        """
        left, right = 0, len(s) - 1
        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
    
    def reverse_words_in_string(self, s: str) -> str:
        """
        LC 151: Reverse Words in a String
        Time: O(n), Space: O(n)
        Companies: Microsoft, Amazon
        """
        # Remove extra spaces and split
        words = s.strip().split()
        # Reverse the list of words
        return ' '.join(reversed(words))
    
    def reverse_words_in_string_optimal(self, s: str) -> str:
        """
        Optimal solution without built-in functions
        Time: O(n), Space: O(n)
        """
        # Convert to list for manipulation
        chars = list(s)
        n = len(chars)
        
        # Remove extra spaces
        write = 0
        for read in range(n):
            if chars[read] != ' ':
                if write != 0:
                    chars[write] = ' '
                    write += 1
                while read < n and chars[read] != ' ':
                    chars[write] = chars[read]
                    write += 1
                    read += 1
        
        chars = chars[:write]  # Trim array
        
        # Reverse entire string
        self._reverse_range(chars, 0, len(chars) - 1)
        
        # Reverse each word
        start = 0
        for end in range(len(chars) + 1):
            if end == len(chars) or chars[end] == ' ':
                self._reverse_range(chars, start, end - 1)
                start = end + 1
        
        return ''.join(chars)
    
    def _reverse_range(self, chars: List[str], left: int, right: int) -> None:
        """Helper function to reverse characters in range"""
        while left < right:
            chars[left], chars[right] = chars[right], chars[left]
            left += 1
            right -= 1
    
    # ==========================================
    # 2. CHARACTER FREQUENCY AND ANALYSIS
    # ==========================================
    
    def first_unique_character(self, s: str) -> int:
        """
        LC 387: First Unique Character in String
        Time: O(n), Space: O(1) - limited alphabet
        Companies: Amazon, Microsoft
        """
        freq = Counter(s)
        
        for i, char in enumerate(s):
            if freq[char] == 1:
                return i
        return -1
    
    def character_replacement(self, s: str, k: int) -> int:
        """
        LC 424: Longest Repeating Character Replacement
        Time: O(n), Space: O(1)
        Companies: Google, Facebook
        """
        char_count = {}
        max_count = 0
        max_length = 0
        left = 0
        
        for right in range(len(s)):
            char_count[s[right]] = char_count.get(s[right], 0) + 1
            max_count = max(max_count, char_count[s[right]])
            
            # If window size - max_count > k, shrink window
            if right - left + 1 - max_count > k:
                char_count[s[left]] -= 1
                left += 1
            
            max_length = max(max_length, right - left + 1)
        
        return max_length
    
    def custom_sort_string(self, order: str, s: str) -> str:
        """
        LC 791: Custom Sort String
        Time: O(n), Space: O(1)
        Companies: Facebook, Google
        """
        count = Counter(s)
        result = []
        
        # Add characters in order
        for char in order:
            if char in count:
                result.extend([char] * count[char])
                del count[char]
        
        # Add remaining characters
        for char, freq in count.items():
            result.extend([char] * freq)
        
        return ''.join(result)
    
    # ==========================================
    # 3. STRING VALIDATION PROBLEMS
    # ==========================================
    
    def is_valid_parentheses(self, s: str) -> bool:
        """
        LC 20: Valid Parentheses
        Time: O(n), Space: O(n)
        Companies: Google, Microsoft, Amazon
        """
        stack = []
        mapping = {')': '(', '}': '{', ']': '['}
        
        for char in s:
            if char in mapping:
                if not stack or stack.pop() != mapping[char]:
                    return False
            else:
                stack.append(char)
        
        return not stack
    
    def is_valid_palindrome(self, s: str) -> bool:
        """
        LC 125: Valid Palindrome
        Time: O(n), Space: O(1)
        Companies: Microsoft, Facebook
        """
        left, right = 0, len(s) - 1
        
        while left < right:
            # Skip non-alphanumeric characters
            while left < right and not s[left].isalnum():
                left += 1
            while left < right and not s[right].isalnum():
                right -= 1
            
            if s[left].lower() != s[right].lower():
                return False
            
            left += 1
            right -= 1
        
        return True
    
    def is_valid_number(self, s: str) -> bool:
        """
        LC 65: Valid Number
        Time: O(n), Space: O(1)
        Companies: LinkedIn, Facebook
        """
        s = s.strip()
        if not s:
            return False
        
        num_seen = False
        dot_seen = False
        e_seen = False
        num_after_e = True
        
        for i, char in enumerate(s):
            if char.isdigit():
                num_seen = True
                num_after_e = True
            elif char == '.':
                if dot_seen or e_seen:
                    return False
                dot_seen = True
            elif char.lower() == 'e':
                if e_seen or not num_seen:
                    return False
                e_seen = True
                num_after_e = False
            elif char in '+-':
                if i != 0 and s[i-1].lower() != 'e':
                    return False
            else:
                return False
        
        return num_seen and num_after_e
    
    # ==========================================
    # 4. ASCII AND UNICODE OPERATIONS
    # ==========================================
    
    def to_lower_case(self, s: str) -> str:
        """
        LC 709: To Lower Case (without built-in)
        Time: O(n), Space: O(n)
        Companies: Google
        """
        result = []
        
        for char in s:
            ascii_val = ord(char)
            if 65 <= ascii_val <= 90:  # A-Z
                result.append(chr(ascii_val + 32))
            else:
                result.append(char)
        
        return ''.join(result)
    
    def unique_morse_code_words(self, words: List[str]) -> int:
        """
        LC 804: Unique Morse Code Words
        Time: O(n), Space: O(n)
        Companies: Google
        """
        morse_code = [".-","-...","-.-.","-..",".","..-.","--.",
                     "....","..",".---","-.-",".-..","--","-.",
                     "---",".--.","--.-",".-.","...","-","..-",
                     "...-",".--","-..-","-.--","--.."]
        
        transformations = set()
        
        for word in words:
            transformation = ''
            for char in word:
                transformation += morse_code[ord(char) - ord('a')]
            transformations.add(transformation)
        
        return len(transformations)
    
    def is_alien_sorted(self, words: List[str], order: str) -> bool:
        """
        LC 953: Verifying Alien Dictionary
        Time: O(n*m), Space: O(1)
        Companies: Facebook, Google
        """
        # Create order mapping
        order_map = {char: i for i, char in enumerate(order)}
        
        def is_smaller_or_equal(word1: str, word2: str) -> bool:
            i = 0
            while i < len(word1) and i < len(word2):
                if order_map[word1[i]] < order_map[word2[i]]:
                    return True
                elif order_map[word1[i]] > order_map[word2[i]]:
                    return False
                i += 1
            
            return len(word1) <= len(word2)
        
        for i in range(len(words) - 1):
            if not is_smaller_or_equal(words[i], words[i + 1]):
                return False
        
        return True
    
    # ==========================================
    # 5. MEMORY-EFFICIENT STRING OPERATIONS
    # ==========================================
    
    def string_compression(self, chars: List[str]) -> int:
        """
        LC 443: String Compression
        Time: O(n), Space: O(1)
        Companies: Microsoft, Apple
        """
        write = 0
        i = 0
        
        while i < len(chars):
            char = chars[i]
            count = 0
            
            # Count consecutive characters
            while i < len(chars) and chars[i] == char:
                count += 1
                i += 1
            
            # Write character
            chars[write] = char
            write += 1
            
            # Write count if > 1
            if count > 1:
                for digit in str(count):
                    chars[write] = digit
                    write += 1
        
        return write
    
    def remove_duplicates(self, s: str) -> str:
        """
        LC 1047: Remove All Adjacent Duplicates
        Time: O(n), Space: O(n)
        Companies: Facebook, Amazon
        """
        stack = []
        
        for char in s:
            if stack and stack[-1] == char:
                stack.pop()
            else:
                stack.append(char)
        
        return ''.join(stack)
    
    def remove_k_duplicates(self, s: str, k: int) -> str:
        """
        LC 1209: Remove All Adjacent Duplicates in String II
        Time: O(n), Space: O(n)
        Companies: Facebook
        """
        stack = []  # [(char, count)]
        
        for char in s:
            if stack and stack[-1][0] == char:
                stack[-1] = (char, stack[-1][1] + 1)
                if stack[-1][1] == k:
                    stack.pop()
            else:
                stack.append((char, 1))
        
        result = []
        for char, count in stack:
            result.extend([char] * count)
        
        return ''.join(result)
        
    # ==========================================
    # 6. ADVANCED STRING MANIPULATION
    # ==========================================
    
    def multiply_strings(self, num1: str, num2: str) -> str:
        """
        LC 43: Multiply Strings
        Time: O(m*n), Space: O(m+n)
        Companies: Facebook, Google
        """
        if num1 == "0" or num2 == "0":
            return "0"
        
        m, n = len(num1), len(num2)
        result = [0] * (m + n)
        
        # Reverse both numbers
        num1, num2 = num1[::-1], num2[::-1]
        
        for i in range(m):
            for j in range(n):
                digit1, digit2 = int(num1[i]), int(num2[j])
                result[i + j] += digit1 * digit2
                
                # Handle carry
                result[i + j + 1] += result[i + j] // 10
                result[i + j] %= 10
        
        # Remove leading zeros and reverse
        result = result[::-1]
        start = 0
        while start < len(result) and result[start] == 0:
            start += 1
        
        return ''.join(map(str, result[start:]))
    
    def add_strings(self, num1: str, num2: str) -> str:
        """
        LC 415: Add Strings
        Time: O(max(m,n)), Space: O(max(m,n))
        Companies: Google, Airbnb
        """
        result = []
        carry = 0
        i, j = len(num1) - 1, len(num2) - 1
        
        while i >= 0 or j >= 0 or carry:
            digit1 = int(num1[i]) if i >= 0 else 0
            digit2 = int(num2[j]) if j >= 0 else 0
            
            total = digit1 + digit2 + carry
            result.append(str(total % 10))
            carry = total // 10
            
            i -= 1
            j -= 1
        
        return ''.join(reversed(result))

def run_examples():
    """Run comprehensive examples for all string operations"""
    sb = StringBasicsAndOperations()
    
    print("=== STRING BASICS AND OPERATIONS EXAMPLES ===\n")
    
    # 1. String Reversal Examples
    print("1. STRING REVERSAL:")
    s1 = ["h", "e", "l", "l", "o"]
    sb.reverse_string_inplace(s1)
    print(f"Reverse in-place: {s1}")
    
    print(f"Reverse words: '{sb.reverse_words_in_string('  hello   world  ')}'")
    print(f"Reverse words optimal: '{sb.reverse_words_in_string_optimal('  hello   world  ')}'")
    
    # 2. Character Frequency Examples
    print("\n2. CHARACTER FREQUENCY:")
    print(f"First unique character in 'leetcode': {sb.first_unique_character('leetcode')}")
    print(f"Longest repeating replacement in 'ABAB' with k=2: {sb.character_replacement('ABAB', 2)}")
    print(f"Custom sort 'cba' with 'cbafg': '{sb.custom_sort_string('cba', 'cbafg')}'")
    
    # 3. String Validation Examples
    print("\n3. STRING VALIDATION:")
    print(f"Valid parentheses '()[]{{}}': {sb.is_valid_parentheses('()[]{}')}")
    print(f"Valid palindrome 'A man a plan a canal Panama': {sb.is_valid_palindrome('A man, a plan, a canal: Panama')}")
    print(f"Valid number '0': {sb.is_valid_number('0')}")
    print(f"Valid number '3.14e-9': {sb.is_valid_number('3.14e-9')}")
    
    # 4. ASCII Operations Examples
    print("\n4. ASCII OPERATIONS:")
    print(f"To lowercase 'Hello': '{sb.to_lower_case('Hello')}'")
    print(f"Unique morse words ['gin','zen','gig','msg']: {sb.unique_morse_code_words(['gin','zen','gig','msg'])}")
    print(f"Alien sorted ['hello','leetcode'] with 'hlabcdefgijkmnopqrstuvwxyz': {sb.is_alien_sorted(['hello','leetcode'], 'hlabcdefgijkmnopqrstuvwxyz')}")
    
    # 5. Memory-Efficient Operations Examples
    print("\n5. MEMORY-EFFICIENT OPERATIONS:")
    chars = ["a","a","b","b","c","c","c"]
    length = sb.string_compression(chars)
    print(f"String compression: {chars[:length]} (length: {length})")
    
    print(f"Remove duplicates 'abbaca': '{sb.remove_duplicates('abbaca')}'")
    print(f"Remove k=3 duplicates 'abccccded': '{sb.remove_k_duplicates('abccccded', 3)}'")
    
    # 6. Advanced Manipulation Examples
    print("\n6. ADVANCED MANIPULATION:")
    print(f"Multiply strings '123' × '456': '{sb.multiply_strings('123', '456')}'")
    print(f"Add strings '123' + '456': '{sb.add_strings('123', '456')}'")

if __name__ == "__main__":
    run_examples() 