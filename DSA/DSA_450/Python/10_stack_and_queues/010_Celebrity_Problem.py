"""
Problem: The Celebrity Problem
URL: https://practice.geeksforgeeks.org/problems/the-celebrity-problem/1

Problem Statement:
In a party of n people, find the celebrity. A celebrity is someone who is known by everyone but knows nobody.

Sample Input/Output:
Input: Matrix representing who knows whom
Output: Index of celebrity or -1 if none exists
"""


class Solution:
    def Find_Celebrity_Stack(self, M, n):
        """
        Find celebrity using stack approach.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        st = []
        for i in range(n):
            st.append(i)
        
        while len(st) > 1:
            a = st.pop()
            b = st.pop()
            
            if M[a][b] == 1:
                st.append(b)
            else:
                st.append(a)
        
        candidate = st[0]
        
        for i in range(n):
            if i != candidate:
                if M[candidate][i] == 1 or M[i][candidate] == 0:
                    return -1
        
        return candidate

    def Find_Celebrity_TwoPointer(self, M, n):
        """
        Find celebrity using two pointer approach.
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        left = 0
        right = n - 1
        
        while left < right:
            if M[left][right] == 1:
                left += 1
            else:
                right -= 1
        
        candidate = left
        
        for i in range(n):
            if i != candidate:
                if M[candidate][i] == 1 or M[i][candidate] == 0:
                    return -1
        
        return candidate

    def Find_Celebrity_BruteForce(self, M, n):
        """
        Find celebrity using brute force approach.
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        for i in range(n):
            isCelebrity = True
            for j in range(n):
                if i != j:
                    if M[i][j] == 1 or M[j][i] == 0:
                        isCelebrity = False
                        break
            if isCelebrity:
                return i
        return -1


def Test_Celebrity_Problem():
    solution = Solution()
    print("Celebrity Problem Tests:")
    
    M1 = [
        [0, 1, 0],
        [0, 0, 0],
        [0, 1, 0]
    ]
    print(f"\nTest Case 1 (Celebrity exists at index 1):")
    print(f"Stack approach: {solution.Find_Celebrity_Stack(M1, 3)}")
    print(f"Two-pointer approach: {solution.Find_Celebrity_TwoPointer(M1, 3)}")
    print(f"Brute force approach: {solution.Find_Celebrity_BruteForce(M1, 3)}")
    
    M2 = [
        [0, 1],
        [1, 0]
    ]
    print(f"\nTest Case 2 (No celebrity):")
    print(f"Stack approach: {solution.Find_Celebrity_Stack(M2, 2)}")
    print(f"Two-pointer approach: {solution.Find_Celebrity_TwoPointer(M2, 2)}")
    print(f"Brute force approach: {solution.Find_Celebrity_BruteForce(M2, 2)}")
    
    M3 = [
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0]
    ]
    print(f"\nTest Case 3 (Celebrity exists at index 2):")
    print(f"Stack approach: {solution.Find_Celebrity_Stack(M3, 4)}")
    print(f"Two-pointer approach: {solution.Find_Celebrity_TwoPointer(M3, 4)}")
    print(f"Brute force approach: {solution.Find_Celebrity_BruteForce(M3, 4)}")


if __name__ == "__main__":
    Test_Celebrity_Problem()
