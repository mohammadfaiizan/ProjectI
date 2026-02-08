"""
Problem: Print All Anagrams Together
URL: https://practice.geeksforgeeks.org/problems/print-anagrams-together/1

Problem Statement:
Given a sequence of words, group all anagrams together.

Sample Input/Output:
Input: ["cat","dog","tac","god","act","ogd"]
Output: groups [cat,tac,act],[dog,god,ogd]
"""


class Solution:
    def Anagrams_Sorted_Key(self, words):
        """
        Anagrams_Sorted_Key (sort each word as key, group using map, O(N * K log K))
        Time Complexity: O(N * K log K) where N is number of words, K is average length
        Space Complexity: O(N * K)
        """
        groups = {}
        
        for word in words:
            key = ''.join(sorted(word))
            if key not in groups:
                groups[key] = []
            groups[key].append(word)
        
        return list(groups.values())
    
    def Anagrams_Count_Key(self, words):
        """
        Anagrams_Count_Key (use character frequency as key, O(N * K))
        Time Complexity: O(N * K) where N is number of words, K is average length
        Space Complexity: O(N * K)
        """
        groups = {}
        
        for word in words:
            count = [0] * 26
            for c in word:
                count[ord(c) - ord('a')] += 1
            
            key = ""
            for i in range(26):
                if count[i] > 0:
                    key += chr(ord('a') + i) + str(count[i])
            
            if key not in groups:
                groups[key] = []
            groups[key].append(word)
        
        return list(groups.values())
    
    def Anagrams_Trie(self, words):
        """
        Anagrams_Trie (group by sorted characters using trie-like structure)
        Time Complexity: O(N * K log K)
        Space Complexity: O(N * K)
        """
        groups = {}
        
        for word in words:
            key = ''.join(sorted(word))
            if key not in groups:
                groups[key] = []
            groups[key].append(word)
        
        return list(groups.values())


def Test_Anagrams_Together():
    solution = Solution()
    
    print("=== Test Case 1 ===")
    words1 = ["cat", "dog", "tac", "god", "act", "ogd"]
    print(f"Input: {' '.join(words1)}")
    
    result1 = solution.Anagrams_Sorted_Key(words1)
    print("Output (Sorted Key):")
    for group in result1:
        print(f"[{','.join(group)}]", end=" ")
    print()
    
    result1b = solution.Anagrams_Count_Key(words1)
    print("Output (Count Key):")
    for group in result1b:
        print(f"[{','.join(group)}]", end=" ")
    print()
    
    print("\n=== Test Case 2 ===")
    words2 = ["eat", "tea", "tan", "ate", "nat", "bat"]
    print(f"Input: {' '.join(words2)}")
    
    result2 = solution.Anagrams_Sorted_Key(words2)
    print("Output:")
    for group in result2:
        print(f"[{','.join(group)}]", end=" ")
    print()
    
    print("\n=== Test Case 3 ===")
    words3 = ["listen", "silent", "enlist", "hello", "world"]
    print(f"Input: {' '.join(words3)}")
    
    result3 = solution.Anagrams_Sorted_Key(words3)
    print("Output:")
    for group in result3:
        print(f"[{','.join(group)}]", end=" ")
    print()
    
    print("\n=== Test Case 4 (Single group) ===")
    words4 = ["abc", "bca", "cab"]
    print(f"Input: {' '.join(words4)}")
    
    result4 = solution.Anagrams_Sorted_Key(words4)
    print("Output:")
    for group in result4:
        print(f"[{','.join(group)}]", end=" ")
    print()
    
    print("\n=== Test Case 5 (No anagrams) ===")
    words5 = ["apple", "banana", "cherry"]
    print(f"Input: {' '.join(words5)}")
    
    result5 = solution.Anagrams_Sorted_Key(words5)
    print("Output:")
    for group in result5:
        print(f"[{','.join(group)}]", end=" ")
    print()


if __name__ == "__main__":
    Test_Anagrams_Together()
