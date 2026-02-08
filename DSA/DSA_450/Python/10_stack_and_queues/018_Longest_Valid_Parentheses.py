"""
Problem: Length of the Longest Valid Parentheses Substring
URL: https://practice.geeksforgeeks.org/problems/valid-substring0624/1

Problem Statement:
Find length of the longest valid (well-formed) parentheses substring.

Sample Input/Output:
Input: "(()"
Output: 2
Input: ")()())"
Output: 4
Input: "((()()()()(((())"
Output: 8
"""


class Solution:
    def Longest_Valid_Parentheses_Stack(self, s):
        """
        Find longest valid parentheses using stack.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        st = [-1]
        maxLen = 0
        for i in range(len(s)):
            if s[i] == '(':
                st.append(i)
            else:
                st.pop()
                if not st:
                    st.append(i)
                else:
                    maxLen = max(maxLen, i - st[-1])
        return maxLen

    def Longest_Valid_Parentheses_Two_Pass(self, s):
        """
        Find longest valid parentheses using two pass approach.
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        left = right = maxLen = 0
        for i in range(len(s)):
            if s[i] == '(':
                left += 1
            else:
                right += 1
            if left == right:
                maxLen = max(maxLen, 2 * right)
            elif right > left:
                left = right = 0
        
        left = right = 0
        for i in range(len(s) - 1, -1, -1):
            if s[i] == '(':
                left += 1
            else:
                right += 1
            if left == right:
                maxLen = max(maxLen, 2 * left)
            elif left > right:
                left = right = 0
        return maxLen


def Test_Longest_Valid_Parentheses():
    solution = Solution()
    
    print("=== Stack Approach ===")
    print(f'Input: "(()"')
    print(f"Output: {solution.Longest_Valid_Parentheses_Stack('(()')}")
    
    print(f'\nInput: ")()())"')
    print(f"Output: {solution.Longest_Valid_Parentheses_Stack(')()())')}")
    
    print(f'\nInput: "((()()()()(((())"')
    print(f"Output: {solution.Longest_Valid_Parentheses_Stack('((()()()()(((())')}")
    
    print(f'\nInput: ""')
    print(f"Output: {solution.Longest_Valid_Parentheses_Stack('')}")
    
    print(f'\nInput: "()(()"')
    print(f"Output: {solution.Longest_Valid_Parentheses_Stack('()(()')}")
    
    print("\n=== Two-Pass Approach ===")
    print(f'Input: "(()"')
    print(f"Output: {solution.Longest_Valid_Parentheses_Two_Pass('(()')}")
    
    print(f'\nInput: ")()())"')
    print(f"Output: {solution.Longest_Valid_Parentheses_Two_Pass(')()())')}")
    
    print(f'\nInput: "((()()()()(((())"')
    print(f"Output: {solution.Longest_Valid_Parentheses_Two_Pass('((()()()()(((())')}")


if __name__ == "__main__":
    Test_Longest_Valid_Parentheses()
