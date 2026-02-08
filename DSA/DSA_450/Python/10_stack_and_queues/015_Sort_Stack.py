"""
Problem: Sort a Stack using Recursion
URL: https://practice.geeksforgeeks.org/problems/sort-a-stack/1

Problem Statement:
Sort a stack in ascending order (top is largest) using recursion or a temporary stack.

Sample Input/Output:
Input: stack [34,3,31,98,92,23]
Output: sorted stack
"""


class Solution:
    def Sorted_Insert(self, st, x):
        """
        Insert element in sorted position.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not st or st[-1] <= x:
            st.append(x)
            return
        top = st.pop()
        self.Sorted_Insert(st, x)
        st.append(top)

    def Sort_Stack_Recursion(self, st):
        """
        Sort stack using recursion.
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        """
        if not st:
            return
        top = st.pop()
        self.Sort_Stack_Recursion(st)
        self.Sorted_Insert(st, top)

    def Sort_Stack_Iterative(self, st):
        """
        Sort stack using iterative approach with temp stack.
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        """
        temp = []
        while st:
            x = st.pop()
            while temp and temp[-1] > x:
                st.append(temp.pop())
            temp.append(x)
        while temp:
            st.append(temp.pop())


def Test_Sort_Stack():
    solution = Solution()
    
    print("=== Recursion Approach ===")
    st1 = [34, 3, 31, 98, 92, 23]
    print(f"Before sort: {st1}")
    solution.Sort_Stack_Recursion(st1)
    print(f"After sort: {st1}")
    
    print("\n=== Iterative Approach ===")
    st2 = [34, 3, 31, 98, 92, 23]
    print(f"Before sort: {st2}")
    solution.Sort_Stack_Iterative(st2)
    print(f"After sort: {st2}")


if __name__ == "__main__":
    Test_Sort_Stack()
