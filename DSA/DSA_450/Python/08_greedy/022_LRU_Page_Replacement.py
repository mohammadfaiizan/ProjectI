"""
Problem: LRU Page Replacement
URL: https://practice.geeksforgeeks.org/problems/page-faults-in-lru5603/1

Problem Statement:
Count page faults using LRU page replacement algorithm given page references and capacity.

Sample Input/Output:
Input: pages[] = {5, 0, 1, 3, 2, 4, 1, 0, 5}, capacity = 4
Output: 8
Explanation: Page faults occur when pages are not in memory and capacity is full.
"""


class Solution:
    def LRU_Page_Replacement_HashSet_HashMap(self, pages, capacity):
        """
        HashSet + HashMap index tracking greedy approach
        Time Complexity: O(n * capacity)
        Space Complexity: O(capacity)
        """
        memory = set()
        last_used = {}
        page_faults = 0
        
        for i in range(len(pages)):
            page = pages[i]
            
            if page in memory:
                last_used[page] = i
                continue
            
            page_faults += 1
            
            if len(memory) < capacity:
                memory.add(page)
                last_used[page] = i
            else:
                lru_page = -1
                min_time = float('inf')
                
                for p in memory:
                    if last_used[p] < min_time:
                        min_time = last_used[p]
                        lru_page = p
                
                memory.remove(lru_page)
                memory.add(page)
                last_used[page] = i
        
        return page_faults


def Test_LRU_Page_Replacement():
    solution = Solution()
    
    pages1 = [5, 0, 1, 3, 2, 4, 1, 0, 5]
    print(f"Test 1: {solution.LRU_Page_Replacement_HashSet_HashMap(pages1, 4)}")
    
    pages2 = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2]
    print(f"Test 2: {solution.LRU_Page_Replacement_HashSet_HashMap(pages2, 4)}")
    
    pages3 = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
    print(f"Test 3: {solution.LRU_Page_Replacement_HashSet_HashMap(pages3, 3)}")


if __name__ == "__main__":
    Test_LRU_Page_Replacement()
