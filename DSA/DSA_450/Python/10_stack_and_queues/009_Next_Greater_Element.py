"""
Problem: Next Greater Element
URL: https://practice.geeksforgeeks.org/problems/next-larger-element-1587115620/1

Problem Statement:
For each element in the array, find the next greater element to its right. If no greater element exists, return -1.

Sample Input/Output:
Input: [4,5,2,25]
Output: [5,25,25,-1]
Input: [13,7,6,12]
Output: [-1,12,12,-1]
"""


class Solution:
    def Next_Greater_BruteForce(self, arr):
        """
        Find next greater element using brute force.
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        n = len(arr)
        result = [-1] * n
        for i in range(n):
            for j in range(i + 1, n):
                if arr[j] > arr[i]:
                    result[i] = arr[j]
                    break
        return result

    def Next_Greater_Stack(self, arr):
        """
        Find next greater element using stack.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(arr)
        result = [-1] * n
        st = []
        for i in range(n - 1, -1, -1):
            while st and st[-1] <= arr[i]:
                st.pop()
            if st:
                result[i] = st[-1]
            st.append(arr[i])
        return result

    def Next_Greater_Stack_LeftToRight(self, arr):
        """
        Find next greater element using stack left to right.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(arr)
        result = [-1] * n
        st = []
        for i in range(n):
            while st and arr[st[-1]] < arr[i]:
                result[st[-1]] = arr[i]
                st.pop()
            st.append(i)
        return result


def Test_Next_Greater_Element():
    solution = Solution()
    print("Next Greater Element Tests:")
    
    arr1 = [4, 5, 2, 25]
    result1 = solution.Next_Greater_Stack(arr1)
    print(f"Input: [4,5,2,25]")
    print(f"Output: {result1}")
    
    arr2 = [13, 7, 6, 12]
    result2 = solution.Next_Greater_Stack(arr2)
    print(f"\nInput: [13,7,6,12]")
    print(f"Output: {result2}")
    
    arr3 = [1, 3, 2, 4]
    result3 = solution.Next_Greater_Stack(arr3)
    print(f"\nInput: [1,3,2,4]")
    print(f"Output: {result3}")


if __name__ == "__main__":
    Test_Next_Greater_Element()
