"""
Problem: Aggressive Cows
URL: https://www.spoj.com/problems/AGGRCOW/

Problem Statement:
Place C cows in N stalls to maximize the minimum distance between any two cows.
Binary search on the answer (minimum distance).

Sample Input:
5 3
1 2 8 4 9

Sample Output:
3
"""


class Solution:
    def Solve_Binary_Search_On_Answer(self, stalls, cows):
        """
        Approach: Binary search on minimum distance. For each distance,
        check if we can place all cows with at least that distance.
        Time Complexity: O(n log n + n log(max_pos)) where n = stalls
        Space Complexity: O(1)
        """
        stalls_sorted = sorted(stalls)
        n = len(stalls_sorted)
        left = 0
        right = stalls_sorted[n - 1] - stalls_sorted[0]
        result = 0
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if self.Can_Place_Cows(stalls_sorted, cows, mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return result
    
    def Can_Place_Cows(self, stalls, cows, min_distance):
        count = 1
        last_pos = stalls[0]
        
        for i in range(1, len(stalls)):
            if stalls[i] - last_pos >= min_distance:
                count += 1
                last_pos = stalls[i]
                if count >= cows:
                    return True
        
        return count >= cows


def Test_Aggressive_Cows():
    sol = Solution()
    
    stalls1 = [1, 2, 8, 4, 9]
    assert sol.Solve_Binary_Search_On_Answer(stalls1, 3) == 3
    
    stalls2 = [1, 2, 4, 8, 9]
    assert sol.Solve_Binary_Search_On_Answer(stalls2, 3) == 3
    
    stalls3 = [1, 5, 10]
    assert sol.Solve_Binary_Search_On_Answer(stalls3, 2) == 9
    
    stalls4 = [1, 2, 3, 4, 5]
    assert sol.Solve_Binary_Search_On_Answer(stalls4, 3) == 2
    
    print("All test cases passed!")


if __name__ == "__main__":
    Test_Aggressive_Cows()
