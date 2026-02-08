"""
Problem: Maximum Trains Stoppage
URL: https://www.geeksforgeeks.org/maximum-trains-stoppage-can-provided/

Problem Statement:
Given N platforms and M trains with arrival time, departure time and platform number, find the maximum number of trains that can be stopped without any conflict.

Sample Input/Output:
Input: n = 3, m = 6, arr[] = {1000, 1100, 1200, 1300, 1400, 1500}, dep[] = {1100, 1200, 1300, 1400, 1500, 1600}, plat[] = {1, 1, 2, 2, 3, 3}
Output: 6
Explanation: All trains can be accommodated on their respective platforms.
"""


class Solution:
    def Max_Trains_Stoppage_Activity_Selection(self, n, m, arr, dep, plat):
        """
        Activity selection per platform - sort trains by departure time per platform
        Time Complexity: O(m log m)
        Space Complexity: O(m)
        """
        trains = [(arr[i], dep[i], plat[i]) for i in range(m)]
        
        trains.sort(key=lambda x: (x[2], x[1]))
        
        count = 0
        last_departure = [-1] * (n + 1)
        
        for train in trains:
            arrival, departure, platform = train
            if arrival >= last_departure[platform]:
                count += 1
                last_departure[platform] = departure
        
        return count


def Test_Maximum_Trains_Stoppage():
    solution = Solution()
    n, m = 3, 6
    arr = [1000, 1100, 1200, 1300, 1400, 1500]
    dep = [1100, 1200, 1300, 1400, 1500, 1600]
    plat = [1, 1, 2, 2, 3, 3]
    print(f"Max trains: {solution.Max_Trains_Stoppage_Activity_Selection(n, m, arr, dep, plat)}")


if __name__ == "__main__":
    Test_Maximum_Trains_Stoppage()
