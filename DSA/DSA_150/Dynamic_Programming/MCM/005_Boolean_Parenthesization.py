"""
Problem: Boolean Parenthesization
URL: https://practice.geeksforgeeks.org/problems/boolean-parenthesization5610/1

Problem Statement:
Given a boolean expression with following symbols:
Symbols: 'T' ---> true, 'F' ---> false
Operators: '&' ---> boolean AND, '|' ---> boolean OR, '^' ---> boolean XOR
Count the number of ways we can parenthesize the expression so that the value of expression evaluates to true.

Sample Input/Output:
Input: S = "T|T&F^T"
Output: 4
Explanation: The expression evaluates to true in 4 ways: ((T|T)&(F^T)), ((T|(T&F))^T), (T|((T&F)^T)), (T|(T&(F^T)))

Input: S = "T^F|F"
Output: 2
Explanation: The expression evaluates to true in 2 ways: ((T^F)|F), (T^(F|F))
"""

from typing import List, Tuple

class Solution:
    def Count_Ways_Brute_Force(self, expression: str) -> int:
        """
        Brute Force - Try all possible parenthesizations
        Time Complexity: O(4^n)
        Space Complexity: O(n)
        """
        def Evaluate(expr: str) -> bool:
            if len(expr) == 1:
                return expr == 'T'
            
            result = expr[0] == 'T'
            
            for i in range(1, len(expr), 2):
                operator = expr[i]
                operand = expr[i + 1] == 'T'
                
                if operator == '&':
                    result = result and operand
                elif operator == '|':
                    result = result or operand
                elif operator == '^':
                    result = result != operand
            
            return result
        
        def Generate_All_Parentheses(expr: str) -> List[str]:
            if len(expr) <= 1:
                return [expr]
            
            all_expressions = []
            
            for i in range(1, len(expr), 2):
                left_expressions = Generate_All_Parentheses(expr[:i])
                right_expressions = Generate_All_Parentheses(expr[i + 1:])
                operator = expr[i]
                
                for left in left_expressions:
                    for right in right_expressions:
                        all_expressions.append(f"({left}{operator}{right})")
            
            return all_expressions
        
        all_parentheses = Generate_All_Parentheses(expression)
        count = 0
        
        for expr in all_parentheses:
            if Evaluate(expr.replace('(', '').replace(')', '')):
                count += 1
        
        return count
    
    def Count_Ways_Recursive(self, expression: str) -> int:
        """
        Recursive - MCM pattern counting true evaluations
        Time Complexity: O(4^n)
        Space Complexity: O(n)
        """
        def Count_True_Ways(expr: str, i: int, j: int, is_true: bool) -> int:
            if i == j:
                if is_true:
                    return 1 if expr[i] == 'T' else 0
                else:
                    return 1 if expr[i] == 'F' else 0
            
            count = 0
            
            for k in range(i + 1, j, 2):
                left_true = Count_True_Ways(expr, i, k - 1, True)
                left_false = Count_True_Ways(expr, i, k - 1, False)
                right_true = Count_True_Ways(expr, k + 1, j, True)
                right_false = Count_True_Ways(expr, k + 1, j, False)
                
                operator = expr[k]
                
                if operator == '&':
                    if is_true:
                        count += left_true * right_true
                    else:
                        count += left_true * right_false + left_false * right_true + left_false * right_false
                elif operator == '|':
                    if is_true:
                        count += left_true * right_true + left_true * right_false + left_false * right_true
                    else:
                        count += left_false * right_false
                elif operator == '^':
                    if is_true:
                        count += left_true * right_false + left_false * right_true
                    else:
                        count += left_true * right_true + left_false * right_false
            
            return count
        
        return Count_True_Ways(expression, 0, len(expression) - 1, True)
    
    def Count_Ways_Memoized(self, expression: str) -> int:
        """
        Memoized - Top-down DP with memoization
        Time Complexity: O(n³)
        Space Complexity: O(n³)
        """
        memo = {}
        
        def Count_Ways_Memo(i: int, j: int, is_true: bool) -> int:
            if i == j:
                if is_true:
                    return 1 if expression[i] == 'T' else 0
                else:
                    return 1 if expression[i] == 'F' else 0
            
            if (i, j, is_true) in memo:
                return memo[(i, j, is_true)]
            
            count = 0
            
            for k in range(i + 1, j, 2):
                left_true = Count_Ways_Memo(i, k - 1, True)
                left_false = Count_Ways_Memo(i, k - 1, False)
                right_true = Count_Ways_Memo(k + 1, j, True)
                right_false = Count_Ways_Memo(k + 1, j, False)
                
                operator = expression[k]
                
                if operator == '&':
                    if is_true:
                        count += left_true * right_true
                    else:
                        count += left_true * right_false + left_false * right_true + left_false * right_false
                elif operator == '|':
                    if is_true:
                        count += left_true * right_true + left_true * right_false + left_false * right_true
                    else:
                        count += left_false * right_false
                elif operator == '^':
                    if is_true:
                        count += left_true * right_false + left_false * right_true
                    else:
                        count += left_true * right_true + left_false * right_false
            
            memo[(i, j, is_true)] = count
            return count
        
        return Count_Ways_Memo(0, len(expression) - 1, True)
    
    def Count_Ways_Tabulation_Optimal(self, expression: str) -> int:
        """
        Tabulation Optimal - Bottom-up DP
        Time Complexity: O(n³)
        Space Complexity: O(n²)
        """
        n = len(expression)
        
        dp_true = [[0] * n for _ in range(n)]
        dp_false = [[0] * n for _ in range(n)]
        
        for i in range(0, n, 2):
            dp_true[i][i] = 1 if expression[i] == 'T' else 0
            dp_false[i][i] = 1 if expression[i] == 'F' else 0
        
        for length in range(3, n + 1, 2):
            for i in range(0, n - length + 1, 2):
                j = i + length - 1
                
                for k in range(i + 1, j, 2):
                    left_true = dp_true[i][k - 1]
                    left_false = dp_false[i][k - 1]
                    right_true = dp_true[k + 1][j]
                    right_false = dp_false[k + 1][j]
                    
                    operator = expression[k]
                    
                    if operator == '&':
                        dp_true[i][j] += left_true * right_true
                        dp_false[i][j] += left_true * right_false + left_false * right_true + left_false * right_false
                    elif operator == '|':
                        dp_true[i][j] += left_true * right_true + left_true * right_false + left_false * right_true
                        dp_false[i][j] += left_false * right_false
                    elif operator == '^':
                        dp_true[i][j] += left_true * right_false + left_false * right_true
                        dp_false[i][j] += left_true * right_true + left_false * right_false
        
        return dp_true[0][n - 1]
    
    def Count_Ways_With_Details(self, expression: str) -> Tuple[int, int]:
        """
        With Details - Return both true and false counts
        Time Complexity: O(n³)
        Space Complexity: O(n²)
        """
        n = len(expression)
        
        dp_true = [[0] * n for _ in range(n)]
        dp_false = [[0] * n for _ in range(n)]
        
        for i in range(0, n, 2):
            dp_true[i][i] = 1 if expression[i] == 'T' else 0
            dp_false[i][i] = 1 if expression[i] == 'F' else 0
        
        for length in range(3, n + 1, 2):
            for i in range(0, n - length + 1, 2):
                j = i + length - 1
                
                for k in range(i + 1, j, 2):
                    left_true = dp_true[i][k - 1]
                    left_false = dp_false[i][k - 1]
                    right_true = dp_true[k + 1][j]
                    right_false = dp_false[k + 1][j]
                    
                    operator = expression[k]
                    
                    if operator == '&':
                        dp_true[i][j] += left_true * right_true
                        dp_false[i][j] += left_true * right_false + left_false * right_true + left_false * right_false
                    elif operator == '|':
                        dp_true[i][j] += left_true * right_true + left_true * right_false + left_false * right_true
                        dp_false[i][j] += left_false * right_false
                    elif operator == '^':
                        dp_true[i][j] += left_true * right_false + left_false * right_true
                        dp_false[i][j] += left_true * right_true + left_false * right_false
        
        return dp_true[0][n - 1], dp_false[0][n - 1]
    
    def Count_Ways_Space_Optimized(self, expression: str) -> int:
        """
        Space Optimized - Optimize space usage
        Time Complexity: O(n³)
        Space Complexity: O(n²)
        """
        n = len(expression)
        
        true_dp = [[0] * n for _ in range(n)]
        false_dp = [[0] * n for _ in range(n)]
        
        for i in range(0, n, 2):
            if expression[i] == 'T':
                true_dp[i][i] = 1
            else:
                false_dp[i][i] = 1
        
        for gap in range(2, n, 2):
            for i in range(0, n - gap, 2):
                j = i + gap
                
                for k in range(i + 1, j, 2):
                    lt = true_dp[i][k - 1]
                    lf = false_dp[i][k - 1]
                    rt = true_dp[k + 1][j]
                    rf = false_dp[k + 1][j]
                    
                    op = expression[k]
                    
                    if op == '&':
                        true_dp[i][j] += lt * rt
                        false_dp[i][j] += lt * rf + lf * rt + lf * rf
                    elif op == '|':
                        true_dp[i][j] += lt * rt + lt * rf + lf * rt
                        false_dp[i][j] += lf * rf
                    elif op == '^':
                        true_dp[i][j] += lt * rf + lf * rt
                        false_dp[i][j] += lt * rt + lf * rf
        
        return true_dp[0][n - 1]

def Test_Count_Ways():
    solution = Solution()
    
    test_cases = [
        ("T|T&F^T", 4),
        ("T^F|F", 2),
        ("T|F^T", 3),
        ("T&F|T^F", 2),
        ("T^T^F", 1)
    ]
    
    methods = [
        ("Recursive", solution.Count_Ways_Recursive),
        ("Memoized", solution.Count_Ways_Memoized),
        ("Tabulation Optimal", solution.Count_Ways_Tabulation_Optimal),
        ("Space Optimized", solution.Count_Ways_Space_Optimized)
    ]
    
    for expression, expected in test_cases:
        print(f"Expression: '{expression}'")
        print(f"Expected: {expected}")
        
        if len(expression) <= 7:
            result_bf = solution.Count_Ways_Brute_Force(expression)
            print(f"Brute Force: {result_bf}")
        
        for method_name, method in methods:
            try:
                result = method(expression)
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        true_count, false_count = solution.Count_Ways_With_Details(expression)
        print(f"With Details: True={true_count}, False={false_count}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Count_Ways()
