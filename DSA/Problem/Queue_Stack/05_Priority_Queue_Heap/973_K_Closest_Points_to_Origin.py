"""
973. K Closest Points to Origin - Multiple Approaches
Difficulty: Medium

Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane and an integer k, return the k closest points to the origin (0, 0).

The distance between two points on the X-Y plane is the Euclidean distance (i.e., √((x1 - x2)² + (y1 - y2)²)).

You may return the answer in any order. The answer is guaranteed to be unique (except for the order that it is in).
"""

from typing import List
import heapq
import random

class KClosestPointsToOrigin:
    """Multiple approaches to find k closest points to origin"""
    
    def kClosest_max_heap(self, points: List[List[int]], k: int) -> List[List[int]]:
        """
        Approach 1: Max Heap of Size K (Optimal)
        
        Maintain max heap of k closest points.
        
        Time: O(n log k), Space: O(k)
        """
        heap = []  # Max heap: (-distance², point)
        
        for point in points:
            x, y = point
            dist_sq = x * x + y * y  # No need for sqrt for comparison
            
            if len(heap) < k:
                heapq.heappush(heap, (-dist_sq, point))
            elif dist_sq < -heap[0][0]:
                heapq.heappop(heap)
                heapq.heappush(heap, (-dist_sq, point))
        
        return [point for _, point in heap]
    
    def kClosest_min_heap(self, points: List[List[int]], k: int) -> List[List[int]]:
        """
        Approach 2: Min Heap with All Points
        
        Add all points to min heap and extract k smallest.
        
        Time: O(n log n), Space: O(n)
        """
        heap = []
        
        for point in points:
            x, y = point
            dist_sq = x * x + y * y
            heapq.heappush(heap, (dist_sq, point))
        
        result = []
        for _ in range(k):
            _, point = heapq.heappop(heap)
            result.append(point)
        
        return result
    
    def kClosest_sorting(self, points: List[List[int]], k: int) -> List[List[int]]:
        """
        Approach 3: Sorting
        
        Sort all points by distance and return first k.
        
        Time: O(n log n), Space: O(1)
        """
        def distance_squared(point):
            return point[0] * point[0] + point[1] * point[1]
        
        points.sort(key=distance_squared)
        return points[:k]
    
    def kClosest_quickselect(self, points: List[List[int]], k: int) -> List[List[int]]:
        """
        Approach 4: QuickSelect (Optimal average case)
        
        Use quickselect to find k closest points.
        
        Time: O(n) average, O(n²) worst, Space: O(1)
        """
        def distance_squared(point):
            return point[0] * point[0] + point[1] * point[1]
        
        def quickselect(left: int, right: int, k: int):
            if left >= right:
                return
            
            # Random pivot for better average performance
            pivot_idx = random.randint(left, right)
            points[pivot_idx], points[right] = points[right], points[pivot_idx]
            
            # Partition
            pivot_dist = distance_squared(points[right])
            i = left
            
            for j in range(left, right):
                if distance_squared(points[j]) <= pivot_dist:
                    points[i], points[j] = points[j], points[i]
                    i += 1
            
            points[i], points[right] = points[right], points[i]
            
            # Recurse on appropriate side
            if i == k:
                return
            elif i < k:
                quickselect(i + 1, right, k)
            else:
                quickselect(left, i - 1, k)
        
        quickselect(0, len(points) - 1, k)
        return points[:k]
    
    def kClosest_bucket_sort(self, points: List[List[int]], k: int) -> List[List[int]]:
        """
        Approach 5: Bucket Sort (for bounded distances)
        
        Use bucket sort when distances are in a reasonable range.
        
        Time: O(n + max_dist), Space: O(n + max_dist)
        """
        # Calculate max distance squared
        max_dist_sq = max(point[0] * point[0] + point[1] * point[1] for point in points)
        
        # Create buckets
        buckets = [[] for _ in range(max_dist_sq + 1)]
        
        # Distribute points into buckets
        for point in points:
            dist_sq = point[0] * point[0] + point[1] * point[1]
            buckets[dist_sq].append(point)
        
        # Collect k closest points
        result = []
        for dist_sq in range(len(buckets)):
            for point in buckets[dist_sq]:
                if len(result) < k:
                    result.append(point)
                else:
                    return result
        
        return result
    
    def kClosest_divide_conquer(self, points: List[List[int]], k: int) -> List[List[int]]:
        """
        Approach 6: Divide and Conquer
        
        Use divide and conquer approach.
        
        Time: O(n log n), Space: O(log n)
        """
        def distance_squared(point):
            return point[0] * point[0] + point[1] * point[1]
        
        def merge_closest(left_points, right_points, k):
            """Merge two sorted lists and return k closest"""
            result = []
            i = j = 0
            
            while len(result) < k and (i < len(left_points) or j < len(right_points)):
                if i >= len(left_points):
                    result.append(right_points[j])
                    j += 1
                elif j >= len(right_points):
                    result.append(left_points[i])
                    i += 1
                elif distance_squared(left_points[i]) <= distance_squared(right_points[j]):
                    result.append(left_points[i])
                    i += 1
                else:
                    result.append(right_points[j])
                    j += 1
            
            return result
        
        def divide_conquer(points_list, k):
            if len(points_list) <= k:
                return sorted(points_list, key=distance_squared)
            
            mid = len(points_list) // 2
            left_closest = divide_conquer(points_list[:mid], k)
            right_closest = divide_conquer(points_list[mid:], k)
            
            return merge_closest(left_closest, right_closest, k)
        
        return divide_conquer(points, k)


def test_k_closest_points_to_origin():
    """Test k closest points to origin algorithms"""
    solver = KClosestPointsToOrigin()
    
    test_cases = [
        ([[1,3],[-2,2]], 1, [[-2,2]], "Example 1"),
        ([[3,3],[5,-1],[-2,4]], 2, [[3,3],[-2,4]], "Example 2"),
        ([[0,1],[1,0]], 2, [[0,1],[1,0]], "Two points"),
        ([[1,1]], 1, [[1,1]], "Single point"),
        ([[1,0],[2,0],[3,0]], 2, [[1,0],[2,0]], "Points on axis"),
        ([[0,0],[1,1],[2,2]], 2, [[0,0],[1,1]], "Points on diagonal"),
    ]
    
    algorithms = [
        ("Max Heap", solver.kClosest_max_heap),
        ("Min Heap", solver.kClosest_min_heap),
        ("Sorting", solver.kClosest_sorting),
        ("QuickSelect", solver.kClosest_quickselect),
        ("Bucket Sort", solver.kClosest_bucket_sort),
        ("Divide Conquer", solver.kClosest_divide_conquer),
    ]
    
    print("=== Testing K Closest Points to Origin ===")
    
    for points, k, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Points: {points}")
        print(f"k: {k}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func([p[:] for p in points], k)  # Pass copy
                
                # Sort both expected and result by distance for comparison
                def dist_sq(p): return p[0]*p[0] + p[1]*p[1]
                expected_sorted = sorted(expected, key=dist_sq)
                result_sorted = sorted(result, key=dist_sq)
                
                status = "✓" if result_sorted == expected_sorted else "✗"
                print(f"{alg_name:15} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:15} | ERROR: {str(e)[:40]}")


def demonstrate_distance_calculation():
    """Demonstrate distance calculation"""
    print("\n=== Distance Calculation Demonstration ===")
    
    points = [[1,3], [-2,2], [3,3], [5,-1], [-2,4]]
    
    print("Points and their distances from origin (0,0):")
    
    point_distances = []
    for point in points:
        x, y = point
        dist_sq = x * x + y * y
        dist = (dist_sq) ** 0.5
        point_distances.append((point, dist_sq, dist))
        
        print(f"  Point {point}: distance² = {x}² + {y}² = {dist_sq}, distance = {dist:.2f}")
    
    # Sort by distance
    point_distances.sort(key=lambda x: x[1])
    
    print(f"\nPoints sorted by distance:")
    for i, (point, dist_sq, dist) in enumerate(point_distances):
        print(f"  {i+1}. {point} (distance = {dist:.2f})")


def visualize_points_on_plane():
    """Visualize points on coordinate plane"""
    print("\n=== Points Visualization ===")
    
    points = [[1,3], [-2,2], [3,3], [5,-1], [-2,4]]
    k = 2
    
    print(f"Points: {points}")
    print(f"Finding {k} closest to origin (0,0)")
    
    # Calculate distances
    point_distances = []
    for point in points:
        x, y = point
        dist_sq = x * x + y * y
        point_distances.append((point, dist_sq))
    
    # Sort by distance
    point_distances.sort(key=lambda x: x[1])
    
    print(f"\nCoordinate plane visualization:")
    print("     |")
    print("  4  |  *(-2,4)")
    print("  3  |  *(1,3)  *(3,3)")
    print("  2  |  *(-2,2)")
    print("  1  |")
    print("  0--+--+--+--+--")
    print(" -2 -1  0  1  2  3  4  5")
    print(" -1  |           *(5,-1)")
    print("     |")
    
    print(f"\nDistances from origin:")
    for i, (point, dist_sq) in enumerate(point_distances):
        dist = dist_sq ** 0.5
        marker = "✓" if i < k else " "
        print(f"  {marker} {point}: {dist:.2f}")
    
    closest_k = [point for point, _ in point_distances[:k]]
    print(f"\n{k} closest points: {closest_k}")


if __name__ == "__main__":
    test_k_closest_points_to_origin()
    demonstrate_distance_calculation()
    visualize_points_on_plane()

"""
K Closest Points to Origin demonstrates heap applications for geometric
problems, including distance-based selection with multiple optimization
strategies and efficient point selection algorithms.
"""
