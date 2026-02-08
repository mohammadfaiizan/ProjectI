"""
Problem: Next Smaller Element
URL: https://www.geeksforgeeks.org/next-smaller-element/

Problem Statement:
Given an array, print the Next Smaller Element (NSE) for every element. The Next smaller Element for an element x is the first smaller element on the right side of x in array. Elements for which no smaller element exist, consider next smaller element as -1.

Sample Input/Output:
Input: arr[] = [4,8,5,2,25]
Output: [2,5,2,-1,-1]
Explanation: Next smaller element for 4 is 2, for 8 is 5, for 5 is 2, for 2 is -1, for 25 is -1.
"""


class Solution:
    def Next_Smaller_Element_Brute_Force(self, arr):
        """
        Find next smaller element using brute force.
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        n = len(arr)
        result = [-1] * n
        
        for i in range(n):
            for j in range(i + 1, n):
                if arr[j] < arr[i]:
                    result[i] = arr[j]
                    break
        
        return result
    
    def Next_Smaller_Element_Stack_Right_To_Left(self, arr):
        """
        Find next smaller element using stack right-to-left.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(arr)
        result = [-1] * n
        st = []
        
        for i in range(n - 1, -1, -1):
            while st and st[-1] >= arr[i]:
                st.pop()
            
            if st:
                result[i] = st[-1]
            
            st.append(arr[i])
        
        return result
    
    def Next_Smaller_Element_Stack_Left_To_Right(self, arr):
        """
        Find next smaller element using stack left-to-right with map.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(arr)
        result = [-1] * n
        st = []
        nextSmaller = {}
        
        for i in range(n):
            while st and arr[st[-1]] > arr[i]:
                nextSmaller[st[-1]] = arr[i]
                st.pop()
            st.append(i)
        
        for i in range(n):
            if i in nextSmaller:
                result[i] = nextSmaller[i]
        
        return result


def Test_Next_Smaller_Element():
    solution = Solution()
    
    arr1 = [4, 8, 5, 2, 25]
    result1 = solution.Next_Smaller_Element_Stack_Right_To_Left(arr1)
    print(f"Test 1 - Stack Right to Left: {result1}")
    
    arr2 = [13, 7, 6, 12]
    result2 = solution.Next_Smaller_Element_Stack_Right_To_Left(arr2)
    print(f"Test 2 - Stack Right to Left: {result2}")
    
    arr3 = [11, 13, 21, 3]
    result3 = solution.Next_Smaller_Element_Stack_Right_To_Left(arr3)
    print(f"Test 3 - Stack Right to Left: {result3}")
    
    arr4 = [4, 8, 5, 2, 25]
    result4 = solution.Next_Smaller_Element_Brute_Force(arr4)
    print(f"Test 4 - Brute Force: {result4}")


if __name__ == "__main__":
    Test_Next_Smaller_Element()
