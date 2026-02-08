"""
Problem: Insert Element at the Bottom of a Stack
URL: https://stackoverflow.com/questions/45130465/inserting-at-the-end-of-stack

Problem Statement:
Insert an element at the bottom of a stack without using any other data structure.

Sample Input/Output:
Input: stack [1,2,3,4] insert 0 at bottom
Output: [0,1,2,3,4]
"""


class Solution:
    def Insert_At_Bottom_Recursion(self, st, x):
        """
        Insert element at bottom using recursion.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not st:
            st.append(x)
            return
        top = st.pop()
        self.Insert_At_Bottom_Recursion(st, x)
        st.append(top)

    def Insert_At_Bottom_Temp_Stack(self, st, x):
        """
        Insert element at bottom using temporary stack.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        temp = []
        while st:
            temp.append(st.pop())
        st.append(x)
        while temp:
            st.append(temp.pop())


def Test_Insert_At_Bottom():
    solution = Solution()
    
    print("=== Recursion Approach ===")
    st1 = [1, 2, 3, 4]
    print(f"Before: {st1}")
    solution.Insert_At_Bottom_Recursion(st1, 0)
    print(f"After inserting 0: {st1}")
    
    print("\n=== Temp Stack Approach ===")
    st2 = [1, 2, 3, 4]
    print(f"Before: {st2}")
    solution.Insert_At_Bottom_Temp_Stack(st2, 0)
    print(f"After inserting 0: {st2}")


if __name__ == "__main__":
    Test_Insert_At_Bottom()
