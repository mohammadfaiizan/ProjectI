"""
Problem: Factorial of a Large Number
URL: https://practice.geeksforgeeks.org/problems/factorials-of-large-numbers2508/1

Problem Statement:
Given an integer N, find its factorial. The factorial can be very large,
so return the result as a vector of digits.

Sample Input/Output:
Input: N = 5
Output: [1, 2, 0]
Explanation: 5! = 120.

Input: N = 10
Output: [3, 6, 2, 8, 8, 0, 0]
Explanation: 10! = 3628800.
"""


class Solution:
    def Factorial_Array_Multiplication_Optimal(self, n):
        """
        Array Multiplication - Multiply digit by digit with carry
        Time Complexity: O(n * digits)
        Space Complexity: O(digits)
        """
        result = [1]
        size = 1
        for i in range(2, n + 1):
            carry = 0
            for j in range(size):
                prod = result[j] * i + carry
                result[j] = prod % 10
                carry = prod // 10
            while carry:
                result.append(carry % 10)
                carry //= 10
                size += 1
        result.reverse()
        return result

    def Factorial_String_Multiplication(self, n):
        """
        String Based - Use string to handle large number multiplication
        Time Complexity: O(n * digits)
        Space Complexity: O(digits)
        """
        result = "1"
        for i in range(2, n + 1):
            result = self._Multiply_String(result, i)
        digits = [int(c) for c in result]
        return digits

    def _Multiply_String(self, num, x):
        carry = 0
        num_list = list(num)
        for i in range(len(num_list) - 1, -1, -1):
            prod = int(num_list[i]) * x + carry
            num_list[i] = str(prod % 10)
            carry = prod // 10
        prefix = ""
        while carry:
            prefix = str(carry % 10) + prefix
            carry //= 10
        return prefix + "".join(num_list)


def Test_Factorial_Large_Number():
    solution = Solution()

    test_cases = [5, 10, 20, 25]

    for n in test_cases:
        print(f"N={n}")

        r1 = solution.Factorial_Array_Multiplication_Optimal(n)
        print(f"Array: {''.join(map(str, r1))}")

        r2 = solution.Factorial_String_Multiplication(n)
        print(f"String: {''.join(map(str, r2))}")
        print("-" * 50)


if __name__ == "__main__":
    Test_Factorial_Large_Number()
