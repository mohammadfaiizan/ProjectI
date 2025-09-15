"""
Problem: K Closest Points to Origin
URL: https://leetcode.com/problems/k-closest-points-to-origin/description/

Problem Statement:
Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane 
and an integer k, return the k closest points to the origin (0, 0).

Sample Input/Output:
Input: points = [[1,3],[-2,2]], k = 1
Output: [[-2,2]]
Explanation: Distance between (1, 3) and origin is sqrt(10), Distance between (-2, 2) and origin is sqrt(8)

Input: points = [[3,3],[5,-1],[-2,4]], k = 2
Output: [[3,3],[-2,4]]
Explanation: The two closest points are [3,3] and [-2,4]
"""

import heapq
import math
from typing import List

class Solution:
    def K_Closest_Brute_Force(self, points: List[List[int]], k: int) -> List[List[int]]:
        """
        Brute Force Approach - Calculate all distances and sort
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        distances = []
        for i, (x, y) in enumerate(points):
            dist = x * x + y * y
            distances.append((dist, i))
        
        distances.sort()
        return [points[distances[i][1]] for i in range(k)]
    
    def K_Closest_Selection_Sort(self, points: List[List[int]], k: int) -> List[List[int]]:
        """
        Selection Sort Approach - Find k closest points
        Time Complexity: O(n * k)
        Space Complexity: O(1)
        """
        n = len(points)
        
        def Get_Distance_Squared(point):
            return point[0] * point[0] + point[1] * point[1]
        
        for i in range(k):
            min_idx = i
            min_dist = Get_Distance_Squared(points[i])
            
            for j in range(i + 1, n):
                curr_dist = Get_Distance_Squared(points[j])
                if curr_dist < min_dist:
                    min_dist = curr_dist
                    min_idx = j
            
            points[i], points[min_idx] = points[min_idx], points[i]
        
        return points[:k]
    
    def K_Closest_Max_Heap_Optimal(self, points: List[List[int]], k: int) -> List[List[int]]:
        """
        Max Heap Approach - Maintain heap of k closest points
        Time Complexity: O(n log k)
        Space Complexity: O(k)
        """
        max_heap = []
        
        for x, y in points:
            dist = x * x + y * y
            if len(max_heap) < k:
                heapq.heappush(max_heap, (-dist, [x, y]))
            elif dist < -max_heap[0][0]:
                heapq.heappop(max_heap)
                heapq.heappush(max_heap, (-dist, [x, y]))
        
        return [point[1] for point in max_heap]
    
    def K_Closest_Min_Heap(self, points: List[List[int]], k: int) -> List[List[int]]:
        """
        Min Heap Approach - All points then extract k smallest
        Time Complexity: O(n log n + k log n)
        Space Complexity: O(n)
        """
        min_heap = []
        for x, y in points:
            dist = x * x + y * y
            heapq.heappush(min_heap, (dist, [x, y]))
        
        result = []
        for _ in range(k):
            dist, point = heapq.heappop(min_heap)
            result.append(point)
        
        return result
    
    def K_Closest_Quick_Select(self, points: List[List[int]], k: int) -> List[List[int]]:
        """
        Quick Select Approach - Partition based algorithm
        Time Complexity: O(n) average, O(n²) worst
        Space Complexity: O(1)
        """
        def Get_Distance_Squared(point):
            return point[0] * point[0] + point[1] * point[1]
        
        def Quick_Select(left, right, k):
            if left >= right:
                return
            
            pivot_idx = Partition(left, right)
            
            if pivot_idx == k:
                return
            elif pivot_idx < k:
                Quick_Select(pivot_idx + 1, right, k)
            else:
                Quick_Select(left, pivot_idx - 1, k)
        
        def Partition(left, right):
            pivot = Get_Distance_Squared(points[right])
            i = left
            
            for j in range(left, right):
                if Get_Distance_Squared(points[j]) <= pivot:
                    points[i], points[j] = points[j], points[i]
                    i += 1
            
            points[i], points[right] = points[right], points[i]
            return i
        
        Quick_Select(0, len(points) - 1, k)
        return points[:k]
    
    def K_Closest_Priority_Queue(self, points: List[List[int]], k: int) -> List[List[int]]:
        """
        Priority Queue using heapq.nsmallest
        Time Complexity: O(n + k log n)
        Space Complexity: O(n)
        """
        def Get_Distance_Squared(point):
            return point[0] * point[0] + point[1] * point[1]
        
        return heapq.nsmallest(k, points, key=Get_Distance_Squared)

def Test_K_Closest_Points():
    solution = Solution()
    
    test_cases = [
        ([[1,3],[-2,2]], 1),
        ([[3,3],[5,-1],[-2,4]], 2),
        ([[0,1],[1,0],[2,0],[1,1],[1,2],[2,1]], 3),
        ([[1,1],[1,1],[1,1]], 2)
    ]
    
    for points, k in test_cases:
        points_copy1 = [p.copy() for p in points]
        points_copy2 = [p.copy() for p in points]
        points_copy3 = [p.copy() for p in points]
        points_copy4 = [p.copy() for p in points]
        points_copy5 = [p.copy() for p in points]
        points_copy6 = [p.copy() for p in points]
        
        result1 = solution.K_Closest_Brute_Force(points_copy1, k)
        result2 = solution.K_Closest_Selection_Sort(points_copy2, k)
        result3 = solution.K_Closest_Max_Heap_Optimal(points_copy3, k)
        result4 = solution.K_Closest_Min_Heap(points_copy4, k)
        result5 = solution.K_Closest_Quick_Select(points_copy5, k)
        result6 = solution.K_Closest_Priority_Queue(points_copy6, k)
        
        print(f"Points: {points}, k: {k}")
        print(f"Brute Force: {result1}")
        print(f"Selection Sort: {result2}")
        print(f"Max Heap Optimal: {result3}")
        print(f"Min Heap: {result4}")
        print(f"Quick Select: {result5}")
        print(f"Priority Queue: {result6}")
        print("-" * 50)

if __name__ == "__main__":
    Test_K_Closest_Points()
