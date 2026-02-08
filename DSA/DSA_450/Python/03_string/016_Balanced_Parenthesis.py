"""
Problem: Balanced Parenthesis (Parenthesis Checker)
URL: https://practice.geeksforgeeks.org/problems/parenthesis-checker2744/1

Problem Statement:
Given an expression string x, examine whether the pairs and the orders of
{, }, (, ), [, ] are correct.

Sample Input/Output:
Input: "{([])}"
Output: true

Input: "[(])"
Output: false
"""


class Solution:
    def Balanced_Parenthesis_Stack(self, x):
        """
        Using stack to match opening and closing brackets
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        stack = []
        for c in x:
            if c in '({[':
                stack.append(c)
            else:
                if not stack:
                    return False
                if c == ')' and stack[-1] == '(':
                    stack.pop()
                elif c == '}' and stack[-1] == '{':
                    stack.pop()
                elif c == ']' and stack[-1] == '[':
                    stack.pop()
                else:
                    return False
        return len(stack) == 0

    def Balanced_Parenthesis_Map(self, x):
        """
        Using map for bracket matching
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        stack = []
        mp = {')': '(', '}': '{', ']': '['}

        for c in x:
            if c not in mp:
                stack.append(c)
            else:
                if not stack or stack[-1] != mp[c]:
                    return False
                stack.pop()

        return len(stack) == 0

    def Balanced_Parenthesis_Counter(self, x):
        """
        Counter approach - works only for single type of brackets
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        count = 0
        for c in x:
            if c == '(':
                count += 1
            elif c == ')':
                count -= 1
            if count < 0:
                return False
        return count == 0


def Test_Balanced_Parenthesis():
    sol = Solution()
    tests = ["{([])}", "[(])", "()", "((()))", "{[()]}", "{{[[(())]]}}", "(]", ""]

    for x in tests:
        print(f'Input: "{x}"')
        print(f"Stack: {sol.Balanced_Parenthesis_Stack(x)}")
        print(f"Map: {sol.Balanced_Parenthesis_Map(x)}")
        print('-' * 50)


if __name__ == "__main__":
    Test_Balanced_Parenthesis()
