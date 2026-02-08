"""
Problem: Expression Contains Redundant Brackets or Not
URL: https://www.geeksforgeeks.org/expression-contains-redundant-bracket-not/

Problem Statement:
Check if expression contains redundant brackets (brackets without operator).

Sample Input/Output:
Input: "((a+b))"
Output: true (redundant)
Input: "(a+b*(c-d))"
Output: false
"""


class Solution:
    def Has_Redundant_Brackets_Stack(self, s):
        """
        Check for redundant brackets using stack.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        st = []
        for c in s:
            if c == '(' or c == '+' or c == '-' or c == '*' or c == '/':
                st.append(c)
            elif c == ')':
                hasOperator = False
                while st and st[-1] != '(':
                    if st[-1] == '+' or st[-1] == '-' or st[-1] == '*' or st[-1] == '/':
                        hasOperator = True
                    st.pop()
                if st:
                    st.pop()
                if not hasOperator:
                    return True
        return False


def Test_Redundant_Brackets():
    solution = Solution()
    
    print("=== Redundant Brackets Check ===")
    print(f'Input: "((a+b))"')
    print(f"Output: {'true (redundant)' if solution.Has_Redundant_Brackets_Stack('((a+b))') else 'false'}")
    
    print(f'\nInput: "(a+b*(c-d))"')
    print(f"Output: {'true (redundant)' if solution.Has_Redundant_Brackets_Stack('(a+b*(c-d))') else 'false'}")
    
    print(f'\nInput: "(a+b)"')
    print(f"Output: {'true (redundant)' if solution.Has_Redundant_Brackets_Stack('(a+b)') else 'false'}")
    
    print(f'\nInput: "((a+b)+c)"')
    print(f"Output: {'true (redundant)' if solution.Has_Redundant_Brackets_Stack('((a+b)+c)') else 'false'}")
    
    print(f'\nInput: "(a+(b)/c)"')
    print(f"Output: {'true (redundant)' if solution.Has_Redundant_Brackets_Stack('(a+(b)/c)') else 'false'}")
    
    print(f'\nInput: "(a+b*(c-d))"')
    print(f"Output: {'true (redundant)' if solution.Has_Redundant_Brackets_Stack('(a+b*(c-d))') else 'false'}")
    
    print(f'\nInput: "((a))"')
    print(f"Output: {'true (redundant)' if solution.Has_Redundant_Brackets_Stack('((a))') else 'false'}")


if __name__ == "__main__":
    Test_Redundant_Brackets()
