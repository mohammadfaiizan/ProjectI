"""
Problem: Majority Element
URL: https://practice.geeksforgeeks.org/problems/majority-element-1587115620/1

Problem Statement:
Given an array A of N elements. Find the majority element in the array. A majority element in an array A of size N is an element that appears more than N/2 times in the array.

Sample Input/Output:
Input: N = 3, A[] = {1,2,3}
Output: -1

Input: N = 5, A[] = {3,1,3,3,2}
Output: 3
"""


class Solution:
    def Majority_Element_HashMap(self, a, size):
        """
        Using hashmap to count occurrences
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        count = {}
        for i in range(size):
            count[a[i]] = count.get(a[i], 0) + 1
            if count[a[i]] > size // 2:
                return a[i]
        return -1

    def Majority_Element_Sorting(self, a, size):
        """
        Sort array and check middle element
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        a_sorted = sorted(a)
        candidate = a_sorted[size // 2]
        count = 0
        for i in range(size):
            if a_sorted[i] == candidate:
                count += 1
        return candidate if count > size // 2 else -1

    def Majority_Element_Moore_Voting(self, a, size):
        """
        Moore's Voting Algorithm
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        candidate = -1
        votes = 0
        
        for i in range(size):
            if votes == 0:
                candidate = a[i]
                votes = 1
            else:
                if a[i] == candidate:
                    votes += 1
                else:
                    votes -= 1
        
        count = 0
        for i in range(size):
            if a[i] == candidate:
                count += 1
        
        return candidate if count > size // 2 else -1


def Test_Majority_Element():
    sol = Solution()
    tests = [
        [1, 2, 3],
        [3, 1, 3, 3, 2],
        [1, 1, 1, 2, 2],
        [1],
        [1, 2, 2, 2, 3]
    ]

    for arr in tests:
        size = len(arr)
        print("Array:", end=" ")
        for num in arr:
            print(num, end=" ")
        print()
        
        res1 = sol.Majority_Element_HashMap(arr[:], size)
        print(f"HashMap: {res1}")
        
        res2 = sol.Majority_Element_Sorting(arr[:], size)
        print(f"Sorting: {res2}")
        
        res3 = sol.Majority_Element_Moore_Voting(arr[:], size)
        print(f"Moore's Voting: {res3}")
        
        print("-" * 50)


if __name__ == "__main__":
    Test_Majority_Element()
