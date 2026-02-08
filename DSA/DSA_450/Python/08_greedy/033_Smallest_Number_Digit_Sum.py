"""
Problem: Smallest Number With Given Digit Sum
URL: https://practice.geeksforgeeks.org/problems/smallest-number5829/1

Problem Statement:
Find smallest number with M digits and digit sum S.

Sample Input/Output:
Input: M=2, S=9
Output: 18
Explanation: Greedy fill from right with max digits approach.
"""


class Solution:
    def Smallest_Number(self, M, S):
        """
        Greedy fill from right with max digits approach
        Time Complexity: O(m)
        Space Complexity: O(m)
        """
        if S == 0:
            if M == 1:
                return "0"
            return "-1"
        
        if S > 9 * M:
            return "-1"
        
        result = ['0'] * M
        result[0] = '1'
        S -= 1
        
        for i in range(M - 1, -1, -1):
            if S >= 9:
                result[i] = '9'
                S -= 9
            else:
                if i == 0:
                    result[i] = chr(ord('0') + S + 1)
                else:
                    result[i] = chr(ord('0') + S)
                S = 0
        
        return ''.join(result)


def Test_Smallest_Number_Digit_Sum():
    solution = Solution()
    
    print(f"Test 1: {solution.Smallest_Number(2, 9)}")
    print(f"Test 2: {solution.Smallest_Number(3, 20)}")
    print(f"Test 3: {solution.Smallest_Number(1, 9)}")
    print(f"Test 4: {solution.Smallest_Number(2, 0)}")


if __name__ == "__main__":
    Test_Smallest_Number_Digit_Sum()
