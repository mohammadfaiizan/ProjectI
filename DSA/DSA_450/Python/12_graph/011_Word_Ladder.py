"""
Problem: Word Ladder
URL: https://leetcode.com/problems/word-ladder/

Problem Statement:
Find shortest transformation sequence length from beginWord to endWord, changing one letter at a time with each word in wordList.

Sample Input/Output:
Input: begin="hit", end="cog", wordList=["hot","dot","dog","lot","log","cog"]
Output: 5
"""

from collections import deque


class Solution:
    def Word_Ladder_BFS(self, beginWord, endWord, wordList):
        """
        BFS - Try All 26 Chars at Each Position
        Time Complexity: O(M^2 * N) where M=word length, N=list size
        Space Complexity: O(N)
        """
        wordSet = set(wordList)
        
        if endWord not in wordSet:
            return 0
        
        q = deque()
        q.append((beginWord, 1))
        wordSet.discard(beginWord)
        
        while q:
            word, level = q.popleft()
            
            if word == endWord:
                return level
            
            word_list = list(word)
            for i in range(len(word_list)):
                original = word_list[i]
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c == original:
                        continue
                    
                    word_list[i] = c
                    newWord = ''.join(word_list)
                    if newWord in wordSet:
                        q.append((newWord, level + 1))
                        wordSet.discard(newWord)
                word_list[i] = original
        
        return 0

    def Word_Ladder_Bidirectional_BFS(self, beginWord, endWord, wordList):
        """
        Bidirectional BFS
        Time Complexity: O(M^2 * N)
        Space Complexity: O(N)
        """
        wordSet = set(wordList)
        
        if endWord not in wordSet:
            return 0
        
        beginSet = {beginWord}
        endSet = {endWord}
        wordSet.discard(beginWord)
        wordSet.discard(endWord)
        
        level = 1
        
        while beginSet and endSet:
            if len(beginSet) > len(endSet):
                beginSet, endSet = endSet, beginSet
            
            nextSet = set()
            
            for word in beginSet:
                word_list = list(word)
                for i in range(len(word_list)):
                    original = word_list[i]
                    for c in 'abcdefghijklmnopqrstuvwxyz':
                        if c == original:
                            continue
                        
                        word_list[i] = c
                        newWord = ''.join(word_list)
                        
                        if newWord in endSet:
                            return level + 1
                        
                        if newWord in wordSet:
                            nextSet.add(newWord)
                            wordSet.discard(newWord)
                    word_list[i] = original
            
            beginSet = nextSet
            level += 1
        
        return 0


def Test_Word_Ladder():
    solution = Solution()
    
    print("Test: Word Ladder")
    beginWord = "hit"
    endWord = "cog"
    wordList = ["hot", "dot", "dog", "lot", "log", "cog"]
    
    result1 = solution.Word_Ladder_BFS(beginWord, endWord, wordList.copy())
    print(f"Shortest sequence length (BFS): {result1}")
    
    wordList2 = ["hot", "dot", "dog", "lot", "log", "cog"]
    result2 = solution.Word_Ladder_Bidirectional_BFS(beginWord, endWord, wordList2)
    print(f"Shortest sequence length (Bidirectional BFS): {result2}")
    
    print("\nTest 2: No valid transformation")
    beginWord2 = "hit"
    endWord2 = "cog"
    wordList3 = ["hot", "dot", "dog", "lot", "log"]
    
    result3 = solution.Word_Ladder_BFS(beginWord2, endWord2, wordList3)
    print(f"Shortest sequence length: {result3}")


if __name__ == "__main__":
    Test_Word_Ladder()
