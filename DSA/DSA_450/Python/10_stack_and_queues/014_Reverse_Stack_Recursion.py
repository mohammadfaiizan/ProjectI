"""
Problem: Reverse a Stack using Recursion
URL: https://www.geeksforgeeks.org/reverse-a-stack-using-recursion/

Problem Statement:
Reverse a stack using recursion only (no extra data structure). Uses insert_at_bottom helper.

Sample Input/Output:
Input: stack [1,2,3,4,5]
Output: [5,4,3,2,1]
"""


class Solution:
    def Insert_At_Bottom(self, st, x):
        """
        Insert element at bottom of stack.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not st:
            st.append(x)
            return
        top = st.pop()
        self.Insert_At_Bottom(st, x)
        st.append(top)

    def Reverse_Stack_Recursion(self, st):
        """
        Reverse stack using recursion.
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        """
        if not st:
            return
        top = st.pop()
        self.Reverse_Stack_Recursion(st)
        self.Insert_At_Bottom(st, top)


def Test_Reverse_Stack():
    solution = Solution()
    
    st = [1, 2, 3, 4, 5]
    print(f"Before reverse: {st}")
    solution.Reverse_Stack_Recursion(st)
    print(f"After reverse: {st}")
    
    st2 = [10, 20, 30]
    print(f"\nBefore reverse: {st2}")
    solution.Reverse_Stack_Recursion(st2)
    print(f"After reverse: {st2}")


if __name__ == "__main__":
    Test_Reverse_Stack()
