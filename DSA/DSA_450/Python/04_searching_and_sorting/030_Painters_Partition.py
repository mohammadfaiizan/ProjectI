"""
Problem: Painters Partition Problem
URL: https://practice.geeksforgeeks.org/problems/the-painters-partition-problem1535/1

Problem Statement:
Divide n boards among k painters to minimize maximum time.
Each board takes a certain time to paint. Painters work sequentially.

Sample Input:
arr[] = {10, 20, 30, 40}, k = 2

Sample Output:
60
"""


class Solution:
    def Is_Possible(self, arr, n, k, time_limit):
        """
        Approach: Check if it's possible to paint boards with given time limit
        Assign boards to painters greedily, ensuring no painter exceeds time limit
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        current_sum = 0
        painters_used = 1
        
        for i in range(n):
            if arr[i] > time_limit:
                return False
            if current_sum + arr[i] > time_limit:
                painters_used += 1
                current_sum = arr[i]
                if painters_used > k:
                    return False
            else:
                current_sum += arr[i]
        return True

    def Min_Time(self, arr, n, k):
        """
        Approach: Binary search on answer
        Search for minimum maximum time needed
        Low = maximum element, High = sum of all elements
        
        Time Complexity: O(n log(sum))
        Space Complexity: O(1)
        """
        if n < k:
            return max(arr)
        
        low = max(arr)
        high = sum(arr)
        result = high
        
        while low <= high:
            mid = low + (high - low) // 2
            
            if self.Is_Possible(arr, n, k, mid):
                result = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return result


def Test_Painters_Partition():
    sol = Solution()
    
    arr1 = [10, 20, 30, 40]
    assert sol.Min_Time(arr1, 4, 2) == 60
    
    arr2 = [10, 10, 10, 10]
    assert sol.Min_Time(arr2, 4, 2) == 20
    
    arr3 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert sol.Min_Time(arr3, 9, 3) == 17
    
    arr4 = [5]
    assert sol.Min_Time(arr4, 1, 1) == 5
    
    print("All test cases passed!")


if __name__ == "__main__":
    Test_Painters_Partition()
