"""
Problem: Evaluation of Postfix and Prefix Expressions
URL: https://practice.geeksforgeeks.org/problems/evaluation-of-postfix-expression1735/1

Problem Statement:
Evaluate postfix and prefix expressions given as strings with single-digit operands and +,-,*,/,^ operators.

Sample Input/Output:
Input: "231*+9-"
Output: -4
Input: "+9*26"
Output: 21
"""


class Solution:
    def Evaluate_Postfix_Stack(self, postfix):
        """
        Evaluate postfix expression using stack.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        st = []
        for c in postfix:
            if c.isdigit():
                st.append(int(c))
            else:
                op2 = st.pop()
                op1 = st.pop()
                result = 0
                if c == '+':
                    result = op1 + op2
                elif c == '-':
                    result = op1 - op2
                elif c == '*':
                    result = op1 * op2
                elif c == '/':
                    result = op1 // op2
                elif c == '^':
                    result = op1 ** op2
                st.append(result)
        return st[0]

    def Evaluate_Prefix_Stack(self, prefix):
        """
        Evaluate prefix expression using stack.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        st = []
        prefix = prefix[::-1]
        for c in prefix:
            if c.isdigit():
                st.append(int(c))
            else:
                op1 = st.pop()
                op2 = st.pop()
                result = 0
                if c == '+':
                    result = op1 + op2
                elif c == '-':
                    result = op1 - op2
                elif c == '*':
                    result = op1 * op2
                elif c == '/':
                    result = op1 // op2
                elif c == '^':
                    result = op1 ** op2
                st.append(result)
        return st[0]


def Test_Expression_Evaluation():
    solution = Solution()
    
    print("=== Postfix Evaluation ===")
    print(f"231*+9- -> {solution.Evaluate_Postfix_Stack('231*+9-')}")
    print(f"123+* -> {solution.Evaluate_Postfix_Stack('123+*')}")
    print(f"23*4+ -> {solution.Evaluate_Postfix_Stack('23*4+')}")
    print(f"52^3+ -> {solution.Evaluate_Postfix_Stack('52^3+')}")
    
    print("\n=== Prefix Evaluation ===")
    print(f"+9*26 -> {solution.Evaluate_Prefix_Stack('+9*26')}")
    print(f"*+123 -> {solution.Evaluate_Prefix_Stack('*+123')}")
    print(f"+*234 -> {solution.Evaluate_Prefix_Stack('+*234')}")
    print(f"+^523 -> {solution.Evaluate_Prefix_Stack('+^523')}")


if __name__ == "__main__":
    Test_Expression_Evaluation()
