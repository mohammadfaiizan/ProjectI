"""
Problem: Word Ladder
URL: https://leetcode.com/problems/word-ladder/

Problem Statement:
A transformation sequence from word beginWord to word endWord using a dictionary wordList is a sequence of words 
beginWord -> s1 -> s2 -> ... -> sk such that:
- Every adjacent pair of words differs by exactly one letter.
- Every si for 1 <= i <= k is in wordList. Note that beginWord does not need to be in wordList.
- sk == endWord
Given two words, beginWord and endWord, and a dictionary wordList, return the length of the shortest transformation sequence from beginWord to endWord, or 0 if no such sequence exists.

Sample Input/Output:
Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
Output: 5
Explanation: One shortest transformation sequence is "hit" -> "hot" -> "dot" -> "dog" -> "cog", which is 5 words long.

Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]
Output: 0
Explanation: The endWord "cog" is not in wordList, therefore there is no valid transformation sequence.
"""

from typing import List
from collections import deque, defaultdict

class Solution:
    def Ladder_Length_BFS_Basic(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        """
        BFS Basic - Standard BFS with character substitution
        Time Complexity: O(M² * N)
        Space Complexity: O(M * N)
        """
        if endWord not in wordList:
            return 0
        
        wordSet = set(wordList)
        queue = deque([(beginWord, 1)])
        visited = {beginWord}
        
        while queue:
            word, length = queue.popleft()
            
            if word == endWord:
                return length
            
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c != word[i]:
                        new_word = word[:i] + c + word[i+1:]
                        if new_word in wordSet and new_word not in visited:
                            visited.add(new_word)
                            queue.append((new_word, length + 1))
        
        return 0
    
    def Ladder_Length_Bidirectional_BFS(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        """
        Bidirectional BFS - Search from both ends
        Time Complexity: O(M² * N)
        Space Complexity: O(M * N)
        """
        if endWord not in wordList:
            return 0
        
        wordSet = set(wordList)
        
        begin_set = {beginWord}
        end_set = {endWord}
        visited = set()
        length = 1
        
        while begin_set and end_set:
            if len(begin_set) > len(end_set):
                begin_set, end_set = end_set, begin_set
            
            temp_set = set()
            
            for word in begin_set:
                for i in range(len(word)):
                    for c in 'abcdefghijklmnopqrstuvwxyz':
                        if c != word[i]:
                            new_word = word[:i] + c + word[i+1:]
                            
                            if new_word in end_set:
                                return length + 1
                            
                            if new_word in wordSet and new_word not in visited:
                                visited.add(new_word)
                                temp_set.add(new_word)
            
            begin_set = temp_set
            length += 1
        
        return 0
    
    def Ladder_Length_Preprocess_Graph(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        """
        Preprocess Graph - Build adjacency list first
        Time Complexity: O(M² * N)
        Space Complexity: O(M² * N)
        """
        if endWord not in wordList:
            return 0
        
        wordList.append(beginWord)
        graph = defaultdict(list)
        
        for i in range(len(wordList)):
            for j in range(i + 1, len(wordList)):
                word1, word2 = wordList[i], wordList[j]
                diff_count = sum(c1 != c2 for c1, c2 in zip(word1, word2))
                if diff_count == 1:
                    graph[word1].append(word2)
                    graph[word2].append(word1)
        
        queue = deque([(beginWord, 1)])
        visited = {beginWord}
        
        while queue:
            word, length = queue.popleft()
            
            if word == endWord:
                return length
            
            for neighbor in graph[word]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, length + 1))
        
        return 0
    
    def Ladder_Length_Pattern_Dict(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        """
        Pattern Dictionary - Group words by pattern
        Time Complexity: O(M² * N)
        Space Complexity: O(M² * N)
        """
        if endWord not in wordList:
            return 0
        
        wordList.append(beginWord)
        pattern_dict = defaultdict(list)
        
        for word in wordList:
            for i in range(len(word)):
                pattern = word[:i] + '*' + word[i+1:]
                pattern_dict[pattern].append(word)
        
        queue = deque([(beginWord, 1)])
        visited = {beginWord}
        
        while queue:
            word, length = queue.popleft()
            
            if word == endWord:
                return length
            
            for i in range(len(word)):
                pattern = word[:i] + '*' + word[i+1:]
                for neighbor in pattern_dict[pattern]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, length + 1))
        
        return 0
    
    def Ladder_Length_DFS_Memoization(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        """
        DFS with Memoization - Recursive approach with caching
        Time Complexity: O(M² * N)
        Space Complexity: O(M * N)
        """
        if endWord not in wordList:
            return 0
        
        wordSet = set(wordList)
        memo = {}
        
        def Is_One_Diff(word1: str, word2: str) -> bool:
            return sum(c1 != c2 for c1, c2 in zip(word1, word2)) == 1
        
        def DFS(current: str, target: str, visited: frozenset) -> int:
            if current == target:
                return 1
            
            if (current, target, visited) in memo:
                return memo[(current, target, visited)]
            
            min_length = float('inf')
            
            for word in wordSet:
                if word not in visited and Is_One_Diff(current, word):
                    new_visited = visited | {word}
                    length = DFS(word, target, new_visited)
                    if length != float('inf'):
                        min_length = min(min_length, length + 1)
            
            memo[(current, target, visited)] = min_length
            return min_length
        
        result = DFS(beginWord, endWord, frozenset([beginWord]))
        return result if result != float('inf') else 0

def Test_Ladder_Length():
    solution = Solution()
    
    test_cases = [
        ("hit", "cog", ["hot","dot","dog","lot","log","cog"], 5),
        ("hit", "cog", ["hot","dot","dog","lot","log"], 0),
        ("a", "c", ["a","b","c"], 2),
        ("hot", "dog", ["hot","dog"], 0),
        ("hot", "dog", ["hot","hog","dog"], 3)
    ]
    
    methods = [
        ("BFS Basic", solution.Ladder_Length_BFS_Basic),
        ("Bidirectional BFS", solution.Ladder_Length_Bidirectional_BFS),
        ("Preprocess Graph", solution.Ladder_Length_Preprocess_Graph),
        ("Pattern Dictionary", solution.Ladder_Length_Pattern_Dict)
    ]
    
    for beginWord, endWord, wordList, expected in test_cases:
        print(f"Begin: {beginWord}, End: {endWord}")
        print(f"WordList: {wordList}")
        print(f"Expected: {expected}")
        
        for method_name, method in methods:
            result = method(beginWord, endWord, wordList.copy())
            print(f"{method_name}: {result}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Ladder_Length()
