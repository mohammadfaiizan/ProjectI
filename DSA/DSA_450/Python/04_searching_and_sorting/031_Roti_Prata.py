"""
Problem: Roti Prata
URL: https://www.spoj.com/problems/PRATA/

Problem Statement:
Assign parathas to cooks with different ranks.
ith paratha by cook of rank R takes R*i time.
Minimize total time. Binary search on answer.

Sample Input:
P = 10, L = 4, ranks[] = {1, 2, 3, 4}

Sample Output:
12
"""


class Solution:
    def Count_Parathas(self, ranks, n, time_limit):
        """
        Approach: Count how many parathas can be made in given time
        For each cook with rank R, count parathas: floor((-1 + sqrt(1 + 8*time/R)) / 2)
        Or use iterative approach: count parathas where R*i*(i+1)/2 <= time
        
        Time Complexity: O(n) where n is number of cooks
        Space Complexity: O(1)
        """
        total_parathas = 0
        for i in range(n):
            count = 0
            time_used = 0
            paratha_num = 1
            
            while time_used + ranks[i] * paratha_num <= time_limit:
                time_used += ranks[i] * paratha_num
                count += 1
                paratha_num += 1
            total_parathas += count
        return total_parathas

    def Min_Time_To_Make_Parathas(self, P, ranks, n):
        """
        Approach: Binary search on answer
        Search for minimum time needed to make P parathas
        Low = 0, High = P * (P + 1) * max_rank / 2 (upper bound)
        
        Time Complexity: O(n * log(max_time))
        Space Complexity: O(1)
        """
        low = 0
        max_rank = max(ranks)
        high = P * (P + 1) * max_rank // 2
        result = high
        
        while low <= high:
            mid = low + (high - low) // 2
            parathas_made = self.Count_Parathas(ranks, n, mid)
            
            if parathas_made >= P:
                result = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return result


def Test_Roti_Prata():
    sol = Solution()
    
    ranks1 = [1, 2, 3, 4]
    assert sol.Min_Time_To_Make_Parathas(10, ranks1, 4) == 12
    
    ranks2 = [1]
    assert sol.Min_Time_To_Make_Parathas(8, ranks2, 1) == 36
    
    ranks3 = [1, 1, 1, 1]
    assert sol.Min_Time_To_Make_Parathas(10, ranks3, 4) == 4
    
    assert sol.Count_Parathas(ranks1, 4, 12) >= 10
    assert sol.Count_Parathas(ranks1, 4, 11) < 10
    
    print("All test cases passed!")


if __name__ == "__main__":
    Test_Roti_Prata()
