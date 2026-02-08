"""
Problem: Alien Dictionary (Find Order of Characters)
URL: https://practice.geeksforgeeks.org/problems/alien-dictionary/1

Problem Statement:
Given a sorted dictionary of words in an alien language, find the order of characters in that language. The words are sorted lexicographically according to the alien language rules.

Sample Input/Output:
Input: words=["baa","abcd","abca","cab","cad"], k=4
Output: b d a c
"""

from collections import deque


class Solution:
    def Alien_Dict_Topological(self, words, k):
        """
        Build DAG from adjacent word comparisons + topological sort
        Time Complexity: O(N*|S|+K) where N=number of words, |S|=avg length, K=alphabet size
        Space Complexity: O(K)
        """
        adj = [[] for _ in range(k)]
        inDegree = [-1] * k
        
        for word in words:
            for c in word:
                if inDegree[ord(c) - ord('a')] == -1:
                    inDegree[ord(c) - ord('a')] = 0
        
        for i in range(len(words) - 1):
            word1 = words[i]
            word2 = words[i + 1]
            
            length = min(len(word1), len(word2))
            found = False
            
            for j in range(length):
                if word1[j] != word2[j]:
                    u = ord(word1[j]) - ord('a')
                    v = ord(word2[j]) - ord('a')
                    adj[u].append(v)
                    inDegree[v] += 1
                    found = True
                    break
            
            if not found and len(word1) > len(word2):
                return ""
        
        q = deque()
        for i in range(k):
            if inDegree[i] == 0:
                q.append(i)
        
        result = []
        while q:
            u = q.popleft()
            result.append(chr(ord('a') + u))
            
            for v in adj[u]:
                inDegree[v] -= 1
                if inDegree[v] == 0:
                    q.append(v)
        
        for i in range(k):
            if inDegree[i] != -1 and inDegree[i] > 0:
                return ""
        
        return ''.join(result)


def Test_Alien_Dict():
    solution = Solution()
    
    print("Test Case 1: Standard alien dictionary")
    words1 = ["baa", "abcd", "abca", "cab", "cad"]
    k1 = 4
    result1 = solution.Alien_Dict_Topological(words1, k1)
    print(f"Order: {' '.join(result1)}")
    
    print("\nTest Case 2: Simple order")
    words2 = ["caa", "aaa", "aab"]
    k2 = 3
    result2 = solution.Alien_Dict_Topological(words2, k2)
    print(f"Order: {' '.join(result2)}")
    
    print("\nTest Case 3: Single character")
    words3 = ["a"]
    k3 = 1
    result3 = solution.Alien_Dict_Topological(words3, k3)
    print(f"Order: {' '.join(result3)}")


if __name__ == "__main__":
    Test_Alien_Dict()
