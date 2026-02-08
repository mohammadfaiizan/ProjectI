"""
Problem: Check for Balanced Parentheses
URL: https://practice.geeksforgeeks.org/problems/parenthesis-checker2744/1

Problem Statement:
Given a string of brackets, check if it is balanced. Handle '(', ')', '{', '}', '[', ']'.

Sample Input/Output:
Input: "()"
Output: true
Input: "([{}])"
Output: true
Input: "(]"
Output: false
"""


class Solution:
    def Is_Balanced_Stack(self, s):
        """
        Check if parentheses are balanced using stack.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        st = []
        for c in s:
            if c == '(' or c == '{' or c == '[':
                st.append(c)
            else:
                if not st:
                    return False
                top = st.pop()
                if (c == ')' and top != '(') or \
                   (c == '}' and top != '{') or \
                   (c == ']' and top != '['):
                    return False
        return len(st) == 0


def Test_Balanced_Parentheses():
    solution = Solution()
    print("Balanced Parentheses Tests:")
    
    print(f'"()": {solution.Is_Balanced_Stack("()")}')
    print(f'"([{{}}])": {solution.Is_Balanced_Stack("([{}])")}')
    print(f'"(]": {solution.Is_Balanced_Stack("(]")}')
    print(f'"": {solution.Is_Balanced_Stack("")}')
    print(f'"(((": {solution.Is_Balanced_Stack("(((")}')
    print(f'"()[]{{}}": {solution.Is_Balanced_Stack("()[]{}")}')
    print(f'"([)]": {solution.Is_Balanced_Stack("([)]")}')
    print(f'"({{[]}})": {solution.Is_Balanced_Stack("({[]})")}')


if __name__ == "__main__":
    Test_Balanced_Parentheses()
