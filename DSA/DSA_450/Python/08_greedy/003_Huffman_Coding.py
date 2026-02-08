"""
Problem: Huffman Coding
URL: https://practice.geeksforgeeks.org/problems/huffman-encoding3345/1

Problem Statement:
Given a string S with distinct character frequencies, build a Huffman tree and generate Huffman codes for each character. Return the codes sorted by character.

Sample Input/Output:
Input: S = "abcdef", freq[] = {5, 9, 12, 13, 16, 45}
Output: {"0", "100", "101", "1100", "1101", "111"}
Explanation: Huffman tree built from frequencies, codes assigned based on path from root.
"""

import heapq


class Node:
    def __init__(self, data, freq):
        self.data = data
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq


class Solution:
    def Huffman_Codes_MinHeap(self, S, f, n):
        """
        Build Huffman tree using min-heap, traverse to generate codes
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        pq = []
        
        for i in range(n):
            heapq.heappush(pq, Node(S[i], f[i]))
        
        while len(pq) > 1:
            left = heapq.heappop(pq)
            right = heapq.heappop(pq)
            
            merged = Node('$', left.freq + right.freq)
            merged.left = left
            merged.right = right
            heapq.heappush(pq, merged)
        
        codes = [''] * n
        code = ""
        self.Generate_Codes(pq[0], code, codes, S)
        
        return codes
    
    def Generate_Codes(self, root, code, codes, S):
        if root is None:
            return
        
        if root.data != '$':
            idx = S.find(root.data)
            codes[idx] = code
            return
        
        self.Generate_Codes(root.left, code + "0", codes, S)
        self.Generate_Codes(root.right, code + "1", codes, S)


def Test_Huffman_Coding():
    solution = Solution()
    S = "abcdef"
    f = [5, 9, 12, 13, 16, 45]
    codes = solution.Huffman_Codes_MinHeap(S, f, len(S))
    print("Huffman codes:")
    for i in range(len(codes)):
        print(f"{S[i]}: {codes[i]}")


if __name__ == "__main__":
    Test_Huffman_Coding()
