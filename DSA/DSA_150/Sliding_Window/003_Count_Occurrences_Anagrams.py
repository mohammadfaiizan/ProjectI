"""
Problem: Count Occurrences of Anagrams
URL: https://www.geeksforgeeks.org/problems/count-occurences-of-anagrams5839/1

Problem Statement:
Given a word and a text, return the count of the occurrences of anagrams of the word in the text.

Sample Input/Output:
Input: text = "forxxorfxdofr", word = "for"
Output: 3
Explanation: Anagrams of "for" are "for", "orf", "ofr". All appear in text.

Input: text = "aabaabaa", word = "aaba"
Output: 4
Explanation: Anagrams of "aaba" appear 4 times in text.
"""

from typing import List
from collections import Counter, defaultdict

class Solution:
    def Search_Anagrams_Brute_Force(self, text: str, word: str) -> int:
        """
        Brute Force Approach - Check all substrings
        Time Complexity: O(n * m * log m) where n=len(text), m=len(word)
        Space Complexity: O(m)
        """
        def Are_Anagrams(s1: str, s2: str) -> bool:
            return sorted(s1) == sorted(s2)
        
        count = 0
        word_len = len(word)
        
        for i in range(len(text) - word_len + 1):
            substring = text[i:i + word_len]
            if Are_Anagrams(substring, word):
                count += 1
        
        return count
    
    def Search_Anagrams_Counter_Comparison(self, text: str, word: str) -> int:
        """
        Counter Comparison Approach - Use Counter for each window
        Time Complexity: O(n * m)
        Space Complexity: O(m)
        """
        word_counter = Counter(word)
        count = 0
        word_len = len(word)
        
        for i in range(len(text) - word_len + 1):
            substring = text[i:i + word_len]
            if Counter(substring) == word_counter:
                count += 1
        
        return count
    
    def Search_Anagrams_Sliding_Window_Optimal(self, text: str, word: str) -> int:
        """
        Sliding Window Approach - Optimal solution
        Time Complexity: O(n + m)
        Space Complexity: O(m)
        """
        if len(word) > len(text):
            return 0
        
        word_count = Counter(word)
        window_count = Counter()
        count = 0
        word_len = len(word)
        
        for i in range(len(text)):
            window_count[text[i]] += 1
            
            if i >= word_len:
                left_char = text[i - word_len]
                window_count[left_char] -= 1
                if window_count[left_char] == 0:
                    del window_count[left_char]
            
            if i >= word_len - 1:
                if window_count == word_count:
                    count += 1
        
        return count
    
    def Search_Anagrams_Character_Frequency(self, text: str, word: str) -> int:
        """
        Character Frequency Approach - Track frequency differences
        Time Complexity: O(n)
        Space Complexity: O(1) - assuming only lowercase letters
        """
        if len(word) > len(text):
            return 0
        
        word_freq = [0] * 26
        window_freq = [0] * 26
        
        for char in word:
            word_freq[ord(char) - ord('a')] += 1
        
        count = 0
        word_len = len(word)
        
        for i in range(len(text)):
            window_freq[ord(text[i]) - ord('a')] += 1
            
            if i >= word_len:
                window_freq[ord(text[i - word_len]) - ord('a')] -= 1
            
            if i >= word_len - 1:
                if word_freq == window_freq:
                    count += 1
        
        return count
    
    def Search_Anagrams_HashMap_Sliding(self, text: str, word: str) -> int:
        """
        HashMap Sliding Window - Using dictionary for frequency
        Time Complexity: O(n)
        Space Complexity: O(m)
        """
        if len(word) > len(text):
            return 0
        
        word_map = defaultdict(int)
        for char in word:
            word_map[char] += 1
        
        window_map = defaultdict(int)
        count = 0
        matches = 0
        word_len = len(word)
        
        for i in range(len(text)):
            right_char = text[i]
            window_map[right_char] += 1
            
            if window_map[right_char] == word_map[right_char]:
                matches += 1
            elif window_map[right_char] == word_map[right_char] + 1:
                matches -= 1
            
            if i >= word_len:
                left_char = text[i - word_len]
                if window_map[left_char] == word_map[left_char]:
                    matches -= 1
                elif window_map[left_char] == word_map[left_char] + 1:
                    matches += 1
                window_map[left_char] -= 1
            
            if i >= word_len - 1 and matches == len(word_map):
                count += 1
        
        return count

def Test_Search_Anagrams():
    solution = Solution()
    
    test_cases = [
        ("forxxorfxdofr", "for", 3),
        ("aabaabaa", "aaba", 4),
        ("abab", "ab", 2),
        ("cbaebabacd", "abc", 2),
        ("baa", "aa", 1)
    ]
    
    for text, word, expected in test_cases:
        result1 = solution.Search_Anagrams_Brute_Force(text, word)
        result2 = solution.Search_Anagrams_Counter_Comparison(text, word)
        result3 = solution.Search_Anagrams_Sliding_Window_Optimal(text, word)
        result4 = solution.Search_Anagrams_Character_Frequency(text, word)
        result5 = solution.Search_Anagrams_HashMap_Sliding(text, word)
        
        print(f"Text: '{text}', Word: '{word}'")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Counter Comparison: {result2}")
        print(f"Sliding Window Optimal: {result3}")
        print(f"Character Frequency: {result4}")
        print(f"HashMap Sliding: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Search_Anagrams()
