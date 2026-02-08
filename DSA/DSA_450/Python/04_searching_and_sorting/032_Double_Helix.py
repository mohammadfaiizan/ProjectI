"""
Problem: Double Helix / ANARC05B
URL: https://www.spoj.com/problems/ANARC05B/

Problem Statement:
Two sorted arrays with some common elements.
Find max sum path switching allowed at common points.
Can switch from one array to another only at common elements.

Sample Input:
arr1[] = {2, 3, 7, 10, 12}, arr2[] = {1, 5, 7, 8}

Sample Output:
35
"""


class Solution:
    def Max_Sum_Path(self, arr1, n, arr2, m):
        """
        Approach: Two pointer technique
        Traverse both arrays simultaneously, accumulate sums between common elements
        At common elements, take maximum sum and reset both accumulators
        
        Time Complexity: O(n + m)
        Space Complexity: O(1)
        """
        i = 0
        j = 0
        sum1 = 0
        sum2 = 0
        result = 0
        
        while i < n and j < m:
            if arr1[i] < arr2[j]:
                sum1 += arr1[i]
                i += 1
            elif arr1[i] > arr2[j]:
                sum2 += arr2[j]
                j += 1
            else:
                result += max(sum1, sum2) + arr1[i]
                sum1 = 0
                sum2 = 0
                i += 1
                j += 1
        
        while i < n:
            sum1 += arr1[i]
            i += 1
        
        while j < m:
            sum2 += arr2[j]
            j += 1
        
        result += max(sum1, sum2)
        return result


def Test_Double_Helix():
    sol = Solution()
    
    arr1 = [2, 3, 7, 10, 12]
    arr2 = [1, 5, 7, 8]
    assert sol.Max_Sum_Path(arr1, 5, arr2, 4) == 35
    
    arr3 = [1, 2, 3]
    arr4 = [3, 4, 5]
    assert sol.Max_Sum_Path(arr3, 3, arr4, 3) == 15
    
    arr5 = [1, 2, 3]
    arr6 = [4, 5, 6]
    assert sol.Max_Sum_Path(arr5, 3, arr6, 3) == 21
    
    arr7 = [1]
    arr8 = [1]
    assert sol.Max_Sum_Path(arr7, 1, arr8, 1) == 1
    
    print("All test cases passed!")


if __name__ == "__main__":
    Test_Double_Helix()
