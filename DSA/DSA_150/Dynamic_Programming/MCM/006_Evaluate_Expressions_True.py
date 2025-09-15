"""
Problem: Evaluate Expression To True
URL: https://www.geeksforgeeks.org/boolean-parenthesization-problem-dp-37/

Problem Statement:
Given a boolean expression with the following symbols:
'T' -> true, 'F' -> false
And following operators filled between symbols:
'&' -> boolean AND, '|' -> boolean OR, '^' -> boolean XOR
Count the number of ways we can parenthesize the expression so that the value of expression evaluates to true.
Note: The answer can be large, so return it modulo 1003.

Sample Input/Output:
Input: S = "T|T&F^T"
Output: 4
Explanation: The given expression can be parenthesized in 4 ways to evaluate to true.

Input: S = "T^F|F"
Output: 2
Explanation: The given expression can be parenthesized in 2 ways to evaluate to true.
"""

from typing import List, Tuple

class Solution:
    MOD = 1003
    
    def Evaluate_Expression_Brute_Force(self, expression: str) -> int:
        """
        Brute Force - Generate all parenthesizations and count true ones
        Time Complexity: O(4^n)
        Space Complexity: O(n)
        """
        def Evaluate_Boolean(expr: str) -> bool:
            if len(expr) == 1:
                return expr == 'T'
            
            stack = []
            
            i = 0
            while i < len(expr):
                if expr[i] in 'TF':
                    stack.append(expr[i] == 'T')
                elif expr[i] in '&|^':
                    if len(stack) >= 2:
                        right = stack.pop()
                        left = stack.pop()
                        
                        if expr[i] == '&':
                            result = left and right
                        elif expr[i] == '|':
                            result = left or right
                        elif expr[i] == '^':
                            result = left != right
                        
                        stack.append(result)
                i += 1
            
            return stack[0] if stack else False
        
        def Generate_All_Ways(expr: str) -> List[str]:
            if len(expr) == 1:
                return [expr]
            
            ways = []
            
            for i in range(1, len(expr), 2):
                left_ways = Generate_All_Ways(expr[:i])
                right_ways = Generate_All_Ways(expr[i + 1:])
                op = expr[i]
                
                for left in left_ways:
                    for right in right_ways:
                        ways.append(f"{left}{op}{right}")
            
            return ways
        
        all_ways = Generate_All_Ways(expression)
        true_count = 0
        
        for way in all_ways:
            if Evaluate_Boolean(way):
                true_count += 1
        
        return true_count % self.MOD
    
    def Evaluate_Expression_Recursive(self, expression: str) -> int:
        """
        Recursive - MCM pattern with boolean evaluation
        Time Complexity: O(4^n)
        Space Complexity: O(n)
        """
        def Count_Ways(i: int, j: int, target: bool) -> int:
            if i == j:
                if target:
                    return 1 if expression[i] == 'T' else 0
                else:
                    return 1 if expression[i] == 'F' else 0
            
            ways = 0
            
            for k in range(i + 1, j, 2):
                left_true = Count_Ways(i, k - 1, True)
                left_false = Count_Ways(i, k - 1, False)
                right_true = Count_Ways(k + 1, j, True)
                right_false = Count_Ways(k + 1, j, False)
                
                total_ways = (left_true + left_false) * (right_true + right_false)
                
                operator = expression[k]
                true_ways = 0
                
                if operator == '&':
                    true_ways = left_true * right_true
                elif operator == '|':
                    true_ways = total_ways - left_false * right_false
                elif operator == '^':
                    true_ways = left_true * right_false + left_false * right_true
                
                if target:
                    ways = (ways + true_ways) % self.MOD
                else:
                    ways = (ways + total_ways - true_ways) % self.MOD
            
            return ways
        
        return Count_Ways(0, len(expression) - 1, True)
    
    def Evaluate_Expression_Memoized(self, expression: str) -> int:
        """
        Memoized - Top-down DP with memoization
        Time Complexity: O(n³)
        Space Complexity: O(n³)
        """
        memo = {}
        
        def Count_Ways_Memo(i: int, j: int, target: bool) -> int:
            if i == j:
                if target:
                    return 1 if expression[i] == 'T' else 0
                else:
                    return 1 if expression[i] == 'F' else 0
            
            if (i, j, target) in memo:
                return memo[(i, j, target)]
            
            ways = 0
            
            for k in range(i + 1, j, 2):
                left_true = Count_Ways_Memo(i, k - 1, True)
                left_false = Count_Ways_Memo(i, k - 1, False)
                right_true = Count_Ways_Memo(k + 1, j, True)
                right_false = Count_Ways_Memo(k + 1, j, False)
                
                operator = expression[k]
                
                if operator == '&':
                    if target:
                        ways = (ways + left_true * right_true) % self.MOD
                    else:
                        ways = (ways + left_true * right_false + left_false * right_true + left_false * right_false) % self.MOD
                elif operator == '|':
                    if target:
                        ways = (ways + left_true * right_true + left_true * right_false + left_false * right_true) % self.MOD
                    else:
                        ways = (ways + left_false * right_false) % self.MOD
                elif operator == '^':
                    if target:
                        ways = (ways + left_true * right_false + left_false * right_true) % self.MOD
                    else:
                        ways = (ways + left_true * right_true + left_false * right_false) % self.MOD
            
            memo[(i, j, target)] = ways
            return ways
        
        return Count_Ways_Memo(0, len(expression) - 1, True)
    
    def Evaluate_Expression_Tabulation_Optimal(self, expression: str) -> int:
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
                        dp_true[i][j] = (dp_true[i][j] + left_true * right_true) % self.MOD
                        dp_false[i][j] = (dp_false[i][j] + left_true * right_false + left_false * right_true + left_false * right_false) % self.MOD
                    elif operator == '|':
                        dp_true[i][j] = (dp_true[i][j] + left_true * right_true + left_true * right_false + left_false * right_true) % self.MOD
                        dp_false[i][j] = (dp_false[i][j] + left_false * right_false) % self.MOD
                    elif operator == '^':
                        dp_true[i][j] = (dp_true[i][j] + left_true * right_false + left_false * right_true) % self.MOD
                        dp_false[i][j] = (dp_false[i][j] + left_true * right_true + left_false * right_false) % self.MOD
        
        return dp_true[0][n - 1]
    
    def Evaluate_Expression_With_Stats(self, expression: str) -> Tuple[int, int, int]:
        """
        With Stats - Return true count, false count, and total ways
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
                        dp_true[i][j] = (dp_true[i][j] + left_true * right_true) % self.MOD
                        dp_false[i][j] = (dp_false[i][j] + left_true * right_false + left_false * right_true + left_false * right_false) % self.MOD
                    elif operator == '|':
                        dp_true[i][j] = (dp_true[i][j] + left_true * right_true + left_true * right_false + left_false * right_true) % self.MOD
                        dp_false[i][j] = (dp_false[i][j] + left_false * right_false) % self.MOD
                    elif operator == '^':
                        dp_true[i][j] = (dp_true[i][j] + left_true * right_false + left_false * right_true) % self.MOD
                        dp_false[i][j] = (dp_false[i][j] + left_true * right_true + left_false * right_false) % self.MOD
        
        true_ways = dp_true[0][n - 1]
        false_ways = dp_false[0][n - 1]
        total_ways = (true_ways + false_ways) % self.MOD
        
        return true_ways, false_ways, total_ways
    
    def Evaluate_Expression_Space_Optimized(self, expression: str) -> int:
        """
        Space Optimized - Reduce space complexity
        Time Complexity: O(n³)
        Space Complexity: O(n²)
        """
        n = len(expression)
        
        T = [[0] * n for _ in range(n)]
        F = [[0] * n for _ in range(n)]
        
        for i in range(0, n, 2):
            T[i][i] = 1 if expression[i] == 'T' else 0
            F[i][i] = 1 if expression[i] == 'F' else 0
        
        for gap in range(2, n, 2):
            for i in range(0, n - gap, 2):
                j = i + gap
                
                for k in range(i + 1, j, 2):
                    tik = T[i][k - 1]
                    fik = F[i][k - 1]
                    tkj = T[k + 1][j]
                    fkj = F[k + 1][j]
                    
                    op = expression[k]
                    
                    if op == '&':
                        T[i][j] = (T[i][j] + tik * tkj) % self.MOD
                        F[i][j] = (F[i][j] + tik * fkj + fik * tkj + fik * fkj) % self.MOD
                    elif op == '|':
                        T[i][j] = (T[i][j] + tik * tkj + tik * fkj + fik * tkj) % self.MOD
                        F[i][j] = (F[i][j] + fik * fkj) % self.MOD
                    elif op == '^':
                        T[i][j] = (T[i][j] + tik * fkj + fik * tkj) % self.MOD
                        F[i][j] = (F[i][j] + tik * tkj + fik * fkj) % self.MOD
        
        return T[0][n - 1]

def Test_Evaluate_Expression():
    solution = Solution()
    
    test_cases = [
        ("T|T&F^T", 4),
        ("T^F|F", 2),
        ("T|F^T", 3),
        ("T&F|T^F", 2),
        ("T^T^F", 1),
        ("F|T^F&T", 3)
    ]
    
    methods = [
        ("Recursive", solution.Evaluate_Expression_Recursive),
        ("Memoized", solution.Evaluate_Expression_Memoized),
        ("Tabulation Optimal", solution.Evaluate_Expression_Tabulation_Optimal),
        ("Space Optimized", solution.Evaluate_Expression_Space_Optimized)
    ]
    
    for expression, expected in test_cases:
        print(f"Expression: '{expression}'")
        print(f"Expected: {expected}")
        
        if len(expression) <= 7:
            result_bf = solution.Evaluate_Expression_Brute_Force(expression)
            print(f"Brute Force: {result_bf}")
        
        for method_name, method in methods:
            try:
                result = method(expression)
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        true_ways, false_ways, total_ways = solution.Evaluate_Expression_With_Stats(expression)
        print(f"With Stats: True={true_ways}, False={false_ways}, Total={total_ways}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Evaluate_Expression()
