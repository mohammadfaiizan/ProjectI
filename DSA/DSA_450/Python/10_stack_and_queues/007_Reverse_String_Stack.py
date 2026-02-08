"""
Problem: Reverse a String using Stack
URL: https://practice.geeksforgeeks.org/problems/reverse-a-string-using-stack/1

Problem Statement:
Reverse a string using stack data structure. Push all characters to stack then pop them back.

Sample Input/Output:
Input: "hello"
Output: "olleh"
Input: "abc"
Output: "cba"
"""


class Solution:
    def Reverse_Stack(self, s):
        """
        Reverse string using stack.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        st = []
        for c in s:
            st.append(c)
        result = ""
        while st:
            result += st.pop()
        return result

    def Reverse_TwoPointer(self, s):
        """
        Reverse string using two pointers.
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        s_list = list(s)
        left = 0
        right = len(s_list) - 1
        while left < right:
            s_list[left], s_list[right] = s_list[right], s_list[left]
            left += 1
            right -= 1
        return "".join(s_list)


def Test_Reverse_String_Stack():
    solution = Solution()
    print("Reverse String Stack Tests:")
    
    print(f'"hello" -> "{solution.Reverse_Stack("hello")}"')
    print(f'"abc" -> "{solution.Reverse_Stack("abc")}"')
    print(f'"" -> "{solution.Reverse_Stack("")}"')
    print(f'"a" -> "{solution.Reverse_Stack("a")}"')
    print(f'"racecar" -> "{solution.Reverse_Stack("racecar")}"')
    
    print("\nTwo Pointer Comparison:")
    print(f'"hello" -> "{solution.Reverse_TwoPointer("hello")}"')
    print(f'"abc" -> "{solution.Reverse_TwoPointer("abc")}"')


if __name__ == "__main__":
    Test_Reverse_String_Stack()
