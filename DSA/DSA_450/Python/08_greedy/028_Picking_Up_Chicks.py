"""
Problem: Picking Up Chicks
URL: https://www.spoj.com/problems/GCJ101BB/

Problem Statement:
N chicks on a road, barn at position B. Each chick has position x[i] and speed v[i]. Time T seconds. Need at least K chicks to reach barn. Find min swaps, or IMPOSSIBLE.

Sample Input/Output:
Input: N=5, K=3, B=10, T=5, positions=[0,2,5,6,7], speeds=[1,1,1,1,4]
Output: 0
Explanation: Greedy right-to-left approach finds minimum swaps needed.
"""


class Solution:
    def Min_Swaps_Or_Impossible(self, N, K, B, T, positions, speeds):
        """
        Greedy right-to-left approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        can_reach = [False] * N
        
        for i in range(N):
            distance = B - positions[i]
            if distance <= T * speeds[i]:
                can_reach[i] = True
        
        reached = 0
        swaps = 0
        slow_behind = 0
        
        for i in range(N - 1, -1, -1):
            if can_reach[i]:
                reached += 1
                swaps += slow_behind
            else:
                slow_behind += 1
            
            if reached >= K:
                return swaps
        
        return -1


def Test_Picking_Up_Chicks():
    solution = Solution()
    
    positions1 = [0, 2, 5, 6, 7]
    speeds1 = [1, 1, 1, 1, 4]
    result1 = solution.Min_Swaps_Or_Impossible(5, 3, 10, 5, positions1, speeds1)
    if result1 == -1:
        print("Test 1: IMPOSSIBLE")
    else:
        print(f"Test 1: {result1}")
    
    positions2 = [0, 1, 2, 3, 4]
    speeds2 = [1, 1, 1, 1, 1]
    result2 = solution.Min_Swaps_Or_Impossible(5, 3, 10, 5, positions2, speeds2)
    if result2 == -1:
        print("Test 2: IMPOSSIBLE")
    else:
        print(f"Test 2: {result2}")


if __name__ == "__main__":
    Test_Picking_Up_Chicks()
