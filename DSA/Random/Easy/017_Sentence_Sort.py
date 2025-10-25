"""
Problem: Sentence Sort
URL: https://leetcode.com/problems/sorting-the-sentence/

Problem Statement:
A sentence is a list of words that are separated by a single space with no leading or 
trailing spaces. Each word consists of lowercase and uppercase English letters.

A sentence can be shuffled by appending the 1-indexed word position to each word then 
rearranging the words in the sentence.

Given a shuffled sentence s containing no more than 9 words, reconstruct and return the 
original sentence.

Sample Input/Output:
Input: s = "is2 sentence4 This1 a3"
Output: "This is a sentence"

Input: s = "Myself2 Me1 I4 and3"
Output: "Me Myself and I"

Input: s = "practice1"
Output: "practice"
"""

from typing import List

class Solution:
    def Sort_Sentence_Brute_Force(self, s: str) -> str:
        """
        Brute Force Approach - Manual sorting
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        words = s.split()
        n = len(words)
        result = [''] * n
        
        for word in words:
            position = int(word[-1]) - 1
            actual_word = word[:-1]
            result[position] = actual_word
        
        return ' '.join(result)
    
    def Sort_Sentence_Dictionary(self, s: str) -> str:
        """
        Dictionary Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        words = s.split()
        word_dict = {}
        
        for word in words:
            position = int(word[-1])
            actual_word = word[:-1]
            word_dict[position] = actual_word
        
        result = []
        for i in range(1, len(words) + 1):
            result.append(word_dict[i])
        
        return ' '.join(result)
    
    def Sort_Sentence_Sorted(self, s: str) -> str:
        """
        Sorted with Key Approach - Optimal solution
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        words = s.split()
        sorted_words = sorted(words, key=lambda x: int(x[-1]))
        
        return ' '.join(word[:-1] for word in sorted_words)
    
    def Sort_Sentence_List_Comprehension(self, s: str) -> str:
        """
        List Comprehension with Fixed Array
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        words = s.split()
        result = [''] * len(words)
        
        for word in words:
            result[int(word[-1]) - 1] = word[:-1]
        
        return ' '.join(result)
    
    def Sort_Sentence_Tuple_Sort(self, s: str) -> str:
        """
        Tuple Sort Approach
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        words = s.split()
        word_tuples = [(int(word[-1]), word[:-1]) for word in words]
        word_tuples.sort()
        
        return ' '.join(word for _, word in word_tuples)

def Test_Sort_Sentence():
    solution = Solution()
    
    test_cases = [
        ("is2 sentence4 This1 a3", "This is a sentence"),
        ("Myself2 Me1 I4 and3", "Me Myself and I"),
        ("practice1", "practice"),
        ("Hello1 World2", "Hello World"),
        ("3is 1This 2a 4test", "This a is test")
    ]
    
    for s, expected in test_cases:
        result1 = solution.Sort_Sentence_Brute_Force(s)
        result2 = solution.Sort_Sentence_Dictionary(s)
        result3 = solution.Sort_Sentence_Sorted(s)
        result4 = solution.Sort_Sentence_List_Comprehension(s)
        result5 = solution.Sort_Sentence_Tuple_Sort(s)
        
        print(f"Input: '{s}'")
        print(f"Expected: '{expected}'")
        print(f"Brute Force: '{result1}'")
        print(f"Dictionary: '{result2}'")
        print(f"Sorted: '{result3}'")
        print(f"List Comprehension: '{result4}'")
        print(f"Tuple Sort: '{result5}'")
        print("-" * 50)

if __name__ == "__main__":
    Test_Sort_Sentence()

