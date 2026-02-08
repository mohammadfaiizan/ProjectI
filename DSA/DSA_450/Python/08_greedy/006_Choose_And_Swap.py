"""
Problem: Choose and Swap
URL: https://practice.geeksforgeeks.org/problems/choose-and-swap0531/1

Problem Statement:
Given a string S of lowercase alphabets, choose two characters and swap all occurrences of first character with second and vice versa. Find the lexicographically smallest string possible.

Sample Input/Output:
Input: S = "ccad"
Output: "aacd"
Explanation: Swap 'c' with 'a' to get "aacd" which is lexicographically smallest.
"""


class Solution:
    def Choose_And_Swap_First_Occurrence(self, A):
        """
        Track first occurrence of each character, find first char that can be swapped
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        first_occurrence = [-1] * 26
        
        for i in range(len(A)):
            if first_occurrence[ord(A[i]) - ord('a')] == -1:
                first_occurrence[ord(A[i]) - ord('a')] = i
        
        swap_char1 = None
        swap_char2 = None
        
        for i in range(len(A)):
            for j in range(ord(A[i]) - ord('a')):
                if first_occurrence[j] != -1 and first_occurrence[j] > i:
                    swap_char1 = A[i]
                    swap_char2 = chr(ord('a') + j)
                    break
            if swap_char1 is not None:
                break
        
        if swap_char1 is None:
            return A
        
        result = list(A)
        for i in range(len(result)):
            if result[i] == swap_char1:
                result[i] = swap_char2
            elif result[i] == swap_char2:
                result[i] = swap_char1
        
        return ''.join(result)


def Test_Choose_And_Swap():
    solution = Solution()
    S = "ccad"
    print(f"Original: {S}")
    print(f"Result: {solution.Choose_And_Swap_First_Occurrence(S)}")


if __name__ == "__main__":
    Test_Choose_And_Swap()
