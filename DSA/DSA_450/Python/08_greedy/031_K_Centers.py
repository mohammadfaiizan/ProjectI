"""
Problem: K Centers Problem
URL: https://www.geeksforgeeks.org/k-centers-problem-set-1-greedy-approximate-algorithm/

Problem Statement:
Given N cities with distances, select K centers to minimize max distance from any city to nearest center (2-approximation).

Sample Input/Output:
Input: N=4, K=2, distances matrix
Output: Centers selected and max distance
Explanation: Greedy farthest-first traversal selects optimal centers.
"""


class Solution:
    def Select_K_Centers(self, N, K, distances):
        """
        Greedy farthest-first traversal approach
        Time Complexity: O(n*k)
        Space Complexity: O(n)
        """
        centers = []
        min_dist = [float('inf')] * N
        
        centers.append(0)
        
        for i in range(N):
            min_dist[i] = distances[0][i]
        
        for k in range(1, K):
            farthest_city = -1
            max_dist = 0
            
            for i in range(N):
                if min_dist[i] > max_dist:
                    max_dist = min_dist[i]
                    farthest_city = i
            
            if farthest_city == -1:
                break
            
            centers.append(farthest_city)
            
            for i in range(N):
                min_dist[i] = min(min_dist[i], distances[farthest_city][i])
        
        max_min_dist = max(min_dist)
        
        return (centers, max_min_dist)


def Test_K_Centers():
    solution = Solution()
    
    dist1 = [
        [0, 10, 7, 6],
        [10, 0, 8, 5],
        [7, 8, 0, 12],
        [6, 5, 12, 0]
    ]
    
    result1 = solution.Select_K_Centers(4, 2, dist1)
    print("Test 1 - Centers:", end=" ")
    for c in result1[0]:
        print(c, end=" ")
    print(f", Max Distance: {result1[1]}")
    
    dist2 = [
        [0, 1, 2],
        [1, 0, 3],
        [2, 3, 0]
    ]
    
    result2 = solution.Select_K_Centers(3, 2, dist2)
    print("Test 2 - Centers:", end=" ")
    for c in result2[0]:
        print(c, end=" ")
    print(f", Max Distance: {result2[1]}")


if __name__ == "__main__":
    Test_K_Centers()
