"""
Problem: Die Hard
URL: https://www.spoj.com/problems/DIEHARD/

Problem Statement:
A character has health H and armor A. Each time unit, must visit one of 3 places: Air (+3H,+2A), Water(-5H,-10A), Fire(-20H,+5A). Cannot visit same place consecutively. Maximize time units survived.

Sample Input/Output:
Input: H=20, A=8
Output: 5
Explanation: Greedy alternating approach maximizes survival time.
"""


class Solution:
    def Max_Survival_Time_Greedy(self, H, A):
        """
        Greedy alternating approach
        Time Complexity: O(max(H,A))
        Space Complexity: O(1)
        """
        time = 0
        in_air = False
        
        while H > 0 and A > 0:
            if not in_air:
                H += 3
                A += 2
                in_air = True
            else:
                if H > 5 and A > 10:
                    H -= 5
                    A -= 10
                elif H > 20:
                    H -= 20
                    A += 5
                else:
                    break
                in_air = False
            time += 1
        
        return time
    
    def Max_Survival_Time_DP(self, H, A):
        """
        DP memoization approach
        Time Complexity: O(H*A)
        Space Complexity: O(H*A)
        """
        memo = {}
        return self.Max_Survival_Time_DP_Helper(H, A, -1, memo)
    
    def Max_Survival_Time_DP_Helper(self, H, A, last, memo):
        if H <= 0 or A <= 0:
            return 0
        
        key = (H, A)
        if key in memo:
            return memo[key]
        
        max_time = 0
        
        if last != 0:
            max_time = max(max_time, 1 + self.Max_Survival_Time_DP_Helper(H + 3, A + 2, 0, memo))
        if last != 1 and H > 5 and A > 10:
            max_time = max(max_time, 1 + self.Max_Survival_Time_DP_Helper(H - 5, A - 10, 1, memo))
        if last != 2 and H > 20:
            max_time = max(max_time, 1 + self.Max_Survival_Time_DP_Helper(H - 20, A + 5, 2, memo))
        
        memo[key] = max_time
        return max_time


def Test_Die_Hard():
    solution = Solution()
    
    print(f"Test 1 (Greedy): {solution.Max_Survival_Time_Greedy(20, 8)}")
    print(f"Test 1 (DP): {solution.Max_Survival_Time_DP(20, 8)}")
    
    print(f"Test 2 (Greedy): {solution.Max_Survival_Time_Greedy(10, 5)}")
    print(f"Test 2 (DP): {solution.Max_Survival_Time_DP(10, 5)}")


if __name__ == "__main__":
    Test_Die_Hard()
