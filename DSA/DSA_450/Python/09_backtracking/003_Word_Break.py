"""
Problem: Word Break
URL: https://practice.geeksforgeeks.org/problems/word-break-part-23249/1

Problem Statement:
Given string s and dictionary, find all possible sentences by breaking s into dictionary words.

Sample Input/Output:
Input: s="catsanddog", dict=["cat","cats","and","sand","dog"]
Output: ["cats and dog","cat sand dog"]
Explanation: Two ways to break the string into valid words
"""


class Solution:
    def Word_Break_Backtracking(self, s, dict_list):
        """
        Backtracking with substring check
        Time Complexity: O(2^n * n)
        Space Complexity: O(n)
        """
        result = []
        word_set = set(dict_list)
        current_sentence = ""
        
        def backtrack(start):
            if start == len(s):
                result.append(current_sentence[1:])
                return
            
            for end in range(start + 1, len(s) + 1):
                word = s[start:end]
                if word in word_set:
                    old_sentence = current_sentence
                    current_sentence += " " + word
                    backtrack(end)
                    current_sentence = old_sentence
        
        backtrack(0)
        return result


def Test_Word_Break():
    solution = Solution()
    
    s = "catsanddog"
    dict_list = ["cat", "cats", "and", "sand", "dog"]
    sentences = solution.Word_Break_Backtracking(s, dict_list)
    
    print("Possible sentences:")
    for sentence in sentences:
        print(sentence)


if __name__ == "__main__":
    Test_Word_Break()
