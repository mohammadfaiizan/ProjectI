"""
Problem: Rearrange Array Alternating Positive and Negative
URL: https://www.geeksforgeeks.org/rearrange-array-alternating-positive-negative-items-o1-extra-space/

Problem Statement:
Given an array of positive and negative numbers, arrange them in an alternate fashion
such that every positive number is followed by a negative and vice-versa. Order of elements
should be maintained. Extra positive or negative elements are placed at the end.

Sample Input/Output:
Input: arr = [1, 2, 3, -4, -1, 4]
Output: [-4, 1, -1, 2, 3, 4]

Input: arr = [-5, -2, 5, 2, 4, 7, 1, 8, 0, -8]
Output: [-5, 5, -2, 2, -8, 4, 7, 1, 8, 0]
"""


class Solution:
    def Rearrange_Alternating_Right_Rotate_Optimal(self, arr):
        """
        Right Rotate Based - Track out-of-place elements and rotate
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        n = len(arr)
        outofplace = -1
        for index in range(n):
            if outofplace >= 0:
                if (arr[index] >= 0 and arr[outofplace] < 0) or \
                   (arr[index] < 0 and arr[outofplace] >= 0):
                    self.Right_Rotate(arr, outofplace, index)
                    if index - outofplace >= 2:
                        outofplace += 2
                    else:
                        outofplace = -1
            if outofplace == -1:
                if (arr[index] >= 0 and index % 2 == 0) or \
                   (arr[index] < 0 and index % 2 == 1):
                    outofplace = index

    def Rearrange_Alternating_Extra_Space(self, arr):
        """
        Extra Space - Separate positives and negatives then interleave
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        pos = []
        neg = []
        for x in arr:
            if x >= 0:
                pos.append(x)
            else:
                neg.append(x)
        i = 0
        p = 0
        n_idx = 0
        place_neg = True
        while p < len(pos) and n_idx < len(neg):
            if place_neg:
                arr[i] = neg[n_idx]
                n_idx += 1
            else:
                arr[i] = pos[p]
                p += 1
            i += 1
            place_neg = not place_neg
        while p < len(pos):
            arr[i] = pos[p]
            i += 1
            p += 1
        while n_idx < len(neg):
            arr[i] = neg[n_idx]
            i += 1
            n_idx += 1

    def Right_Rotate(self, arr, outofplace, cur):
        temp = arr[cur]
        for i in range(cur, outofplace, -1):
            arr[i] = arr[i - 1]
        arr[outofplace] = temp


def Test_Rearrange_Alternating():
    solution = Solution()

    test_cases = [
        [1, 2, 3, -4, -1, 4],
        [-5, -2, 5, 2, 4, 7, 1, 8, 0, -8],
        [-1, -2, -3, 1, 2, 3]
    ]

    for arr in test_cases:
        print("Original:", arr)

        arr1 = arr.copy()
        arr2 = arr.copy()

        solution.Rearrange_Alternating_Right_Rotate_Optimal(arr1)
        print("Right Rotate:", arr1)

        solution.Rearrange_Alternating_Extra_Space(arr2)
        print("Extra Space:", arr2)

        print("-" * 50)


if __name__ == "__main__":
    Test_Rearrange_Alternating()
