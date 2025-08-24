"""
1642. Furthest Building You Can Reach - Multiple Approaches
Difficulty: Medium

You are given an integer array heights representing the heights of buildings, some bricks, and some ladders.

You start your journey from building 0 and move to the next building by possibly using bricks or ladders.

While moving from building i to building i+1 (0-indexed):
- If the current building's height is greater than or equal to the next building's height, you do not need a ladder or bricks.
- If the current building's height is less than the next building's height, you need a ladder or (height[i+1] - height[i]) bricks.

Return the furthest building index (0-indexed) you can reach if you use the given ladders and bricks optimally.
"""

from typing import List
import heapq

class FurthestBuildingYouCanReach:
    """Multiple approaches to find furthest building you can reach"""
    
    def furthestBuilding_greedy_heap(self, heights: List[int], bricks: int, ladders: int) -> int:
        """
        Approach 1: Greedy with Min Heap (Optimal)
        
        Use ladders for largest gaps, bricks for smallest gaps.
        
        Time: O(n log ladders), Space: O(ladders)
        """
        heap = []  # Min heap to store the smallest gaps where we used ladders
        
        for i in range(len(heights) - 1):
            gap = heights[i + 1] - heights[i]
            
            if gap <= 0:
                continue  # No ladder or bricks needed
            
            if len(heap) < ladders:
                # We have ladders available, use one
                heapq.heappush(heap, gap)
            else:
                # No more ladders, decide whether to use bricks or replace a ladder
                if heap and gap > heap[0]:
                    # Replace the smallest ladder usage with bricks
                    smallest_ladder_gap = heapq.heappop(heap)
                    heapq.heappush(heap, gap)
                    bricks -= smallest_ladder_gap
                else:
                    # Use bricks for current gap
                    bricks -= gap
                
                if bricks < 0:
                    return i
        
        return len(heights) - 1
    
    def furthestBuilding_binary_search(self, heights: List[int], bricks: int, ladders: int) -> int:
        """
        Approach 2: Binary Search
        
        Binary search on the answer with greedy verification.
        
        Time: O(n log n), Space: O(n)
        """
        def can_reach(target_idx: int) -> bool:
            """Check if we can reach target_idx with given resources"""
            gaps = []
            
            for i in range(target_idx):
                gap = heights[i + 1] - heights[i]
                if gap > 0:
                    gaps.append(gap)
            
            # Use ladders for largest gaps
            gaps.sort(reverse=True)
            
            # Use ladders for first 'ladders' gaps
            bricks_needed = sum(gaps[ladders:]) if len(gaps) > ladders else 0
            
            return bricks_needed <= bricks
        
        left, right = 0, len(heights) - 1
        
        while left < right:
            mid = (left + right + 1) // 2
            
            if can_reach(mid):
                left = mid
            else:
                right = mid - 1
        
        return left
    
    def furthestBuilding_dp_approach(self, heights: List[int], bricks: int, ladders: int) -> int:
        """
        Approach 3: Dynamic Programming
        
        Use DP to track maximum reachable index with given resources.
        
        Time: O(n * bricks * ladders), Space: O(bricks * ladders)
        """
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def dp(idx: int, remaining_bricks: int, remaining_ladders: int) -> int:
            """Return furthest index reachable from idx with given resources"""
            if idx == len(heights) - 1:
                return idx
            
            gap = heights[idx + 1] - heights[idx]
            
            if gap <= 0:
                # No resources needed
                return dp(idx + 1, remaining_bricks, remaining_ladders)
            
            max_reach = idx  # At least we can stay here
            
            # Try using bricks
            if remaining_bricks >= gap:
                max_reach = max(max_reach, dp(idx + 1, remaining_bricks - gap, remaining_ladders))
            
            # Try using ladder
            if remaining_ladders > 0:
                max_reach = max(max_reach, dp(idx + 1, remaining_bricks, remaining_ladders - 1))
            
            return max_reach
        
        return dp(0, bricks, ladders)
    
    def furthestBuilding_greedy_sorting(self, heights: List[int], bricks: int, ladders: int) -> int:
        """
        Approach 4: Greedy with Sorting
        
        Collect all gaps and use optimal allocation.
        
        Time: O(n log n), Space: O(n)
        """
        gaps = []
        
        # Collect all positive gaps with their indices
        for i in range(len(heights) - 1):
            gap = heights[i + 1] - heights[i]
            if gap > 0:
                gaps.append((gap, i))
        
        # Sort gaps by size (largest first)
        gaps.sort(reverse=True)
        
        # Use ladders for largest gaps
        ladder_used = set()
        for i in range(min(ladders, len(gaps))):
            ladder_used.add(gaps[i][1])
        
        # Use bricks for remaining gaps
        current_bricks = bricks
        
        for i in range(len(heights) - 1):
            gap = heights[i + 1] - heights[i]
            
            if gap <= 0:
                continue
            
            if i not in ladder_used:
                current_bricks -= gap
                if current_bricks < 0:
                    return i
        
        return len(heights) - 1
    
    def furthestBuilding_brute_force(self, heights: List[int], bricks: int, ladders: int) -> int:
        """
        Approach 5: Brute Force with Backtracking
        
        Try all possible combinations of ladder and brick usage.
        
        Time: O(2^n), Space: O(n)
        """
        def backtrack(idx: int, remaining_bricks: int, remaining_ladders: int) -> int:
            """Backtrack to find maximum reachable index"""
            if idx == len(heights) - 1:
                return idx
            
            gap = heights[idx + 1] - heights[idx]
            
            if gap <= 0:
                return backtrack(idx + 1, remaining_bricks, remaining_ladders)
            
            max_reach = idx
            
            # Try using bricks
            if remaining_bricks >= gap:
                max_reach = max(max_reach, backtrack(idx + 1, remaining_bricks - gap, remaining_ladders))
            
            # Try using ladder
            if remaining_ladders > 0:
                max_reach = max(max_reach, backtrack(idx + 1, remaining_bricks, remaining_ladders - 1))
            
            return max_reach
        
        return backtrack(0, bricks, ladders)


def test_furthest_building_you_can_reach():
    """Test furthest building algorithms"""
    solver = FurthestBuildingYouCanReach()
    
    test_cases = [
        ([4,2,7,6,9,14,12], 5, 1, 4, "Example 1"),
        ([4,12,2,7,3,18,20,3,19], 10, 2, 7, "Example 2"),
        ([14,3,19,3], 17, 0, 3, "Example 3"),
        ([1,2,3,4,5], 0, 4, 4, "Only ladders"),
        ([1,2,3,4,5], 10, 0, 4, "Only bricks"),
        ([5,4,3,2,1], 0, 0, 4, "Decreasing heights"),
        ([1,1,1,1], 0, 0, 3, "Same heights"),
        ([1,5,1,5], 3, 1, 3, "Mixed case"),
    ]
    
    algorithms = [
        ("Greedy Heap", solver.furthestBuilding_greedy_heap),
        ("Binary Search", solver.furthestBuilding_binary_search),
        ("DP Approach", solver.furthestBuilding_dp_approach),
        ("Greedy Sorting", solver.furthestBuilding_greedy_sorting),
        # ("Brute Force", solver.furthestBuilding_brute_force),  # Too slow for large inputs
    ]
    
    print("=== Testing Furthest Building You Can Reach ===")
    
    for heights, bricks, ladders, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Heights: {heights}")
        print(f"Bricks: {bricks}, Ladders: {ladders}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(heights[:], bricks, ladders)  # Pass copy
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:15} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:15} | ERROR: {str(e)[:40]}")


def demonstrate_greedy_heap_approach():
    """Demonstrate greedy heap approach step by step"""
    print("\n=== Greedy Heap Approach Step-by-Step Demo ===")
    
    heights = [4, 2, 7, 6, 9, 14, 12]
    bricks = 5
    ladders = 1
    
    print(f"Heights: {heights}")
    print(f"Bricks: {bricks}, Ladders: {ladders}")
    
    print(f"\nStrategy: Use ladders for largest gaps, bricks for smallest gaps")
    
    heap = []
    current_bricks = bricks
    
    for i in range(len(heights) - 1):
        gap = heights[i + 1] - heights[i]
        
        print(f"\nStep {i+1}: Moving from building {i} (height {heights[i]}) to building {i+1} (height {heights[i+1]})")
        
        if gap <= 0:
            print(f"  Gap: {gap} (no resources needed)")
            continue
        
        print(f"  Gap: {gap} (need ladder or {gap} bricks)")
        
        if len(heap) < ladders:
            heapq.heappush(heap, gap)
            print(f"  Used ladder (have {ladders - len(heap)} ladders left)")
            print(f"  Ladder gaps so far: {sorted(heap)}")
        else:
            if heap and gap > heap[0]:
                smallest_ladder_gap = heapq.heappop(heap)
                heapq.heappush(heap, gap)
                current_bricks -= smallest_ladder_gap
                print(f"  Replaced ladder (gap {smallest_ladder_gap}) with bricks")
                print(f"  Used ladder for gap {gap}")
                print(f"  Bricks remaining: {current_bricks}")
            else:
                current_bricks -= gap
                print(f"  Used {gap} bricks")
                print(f"  Bricks remaining: {current_bricks}")
            
            if current_bricks < 0:
                print(f"  Out of bricks! Can't proceed further.")
                print(f"  Furthest building reached: {i}")
                return
        
        print(f"  Current ladder gaps: {sorted(heap)}")
    
    print(f"\nReached the end! Furthest building: {len(heights) - 1}")


if __name__ == "__main__":
    test_furthest_building_you_can_reach()
    demonstrate_greedy_heap_approach()

"""
Furthest Building You Can Reach demonstrates heap applications for
resource optimization problems, including greedy strategies and
multiple approaches for optimal resource allocation.
"""
