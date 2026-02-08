"""
Problem: Maximum XOR of Two Numbers in an Array
URL: https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/

Problem Statement:
Given an integer array nums, return the maximum result of nums[i] XOR nums[j],
where 0 <= i <= j < n.

Sample Input/Output:
Input: nums = [3, 10, 5, 25, 2, 8]
Output: 28
Explanation: The maximum XOR is 5 XOR 25 = 28.

Input: nums = [14, 70, 53, 83, 49, 91, 36, 80, 92, 51, 66, 70]
Output: 127
"""


class TrieNode:
    def __init__(self):
        self.children = {}


class BinaryTrie:
    def __init__(self):
        self.root = TrieNode()
    
    def Insert(self, num):
        curr = self.root
        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            if bit not in curr.children:
                curr.children[bit] = TrieNode()
            curr = curr.children[bit]
    
    def Get_Max_XOR(self, num):
        curr = self.root
        max_xor = 0
        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            opp_bit = 1 - bit
            if opp_bit in curr.children:
                max_xor |= (1 << i)
                curr = curr.children[opp_bit]
            else:
                curr = curr.children[bit]
        return max_xor


class Solution:
    def Max_XOR_Binary_Trie(self, nums):
        """
        Binary Trie - Insert all, then greedily pick opposite bits
        Time Complexity: O(32 * n)
        Space Complexity: O(32 * n)
        """
        trie = BinaryTrie()
        for num in nums:
            trie.Insert(num)
        result = 0
        for num in nums:
            result = max(result, trie.Get_Max_XOR(num))
        return result
    
    def Max_XOR_Brute(self, nums):
        """
        Brute Force - Check all pairs
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        result = 0
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                result = max(result, nums[i] ^ nums[j])
        return result


def Test_Maximum_XOR():
    solution = Solution()
    
    nums1 = [3, 10, 5, 25, 2, 8]
    print(f"Array: [3,10,5,25,2,8]")
    print(f"Binary Trie: {solution.Max_XOR_Binary_Trie(nums1)}")
    print(f"Brute Force: {solution.Max_XOR_Brute(nums1)}")
    print('-' * 50)
    
    nums2 = [14, 70, 53, 83, 49, 91, 36, 80, 92, 51, 66, 70]
    print(f"Array: [14,70,53,83,49,91,36,80,92,51,66,70]")
    print(f"Binary Trie: {solution.Max_XOR_Binary_Trie(nums2)}")
    print(f"Brute Force: {solution.Max_XOR_Brute(nums2)}")
    print('-' * 50)
    
    nums3 = [0]
    print(f"Array: [0]")
    print(f"Binary Trie: {solution.Max_XOR_Binary_Trie(nums3)}")
    print(f"Brute Force: {solution.Max_XOR_Brute(nums3)}")
    print('-' * 50)
    
    nums4 = [1, 2, 3, 4, 5]
    print(f"Array: [1,2,3,4,5]")
    print(f"Binary Trie: {solution.Max_XOR_Binary_Trie(nums4)}")
    print(f"Brute Force: {solution.Max_XOR_Brute(nums4)}")


if __name__ == "__main__":
    Test_Maximum_XOR()
