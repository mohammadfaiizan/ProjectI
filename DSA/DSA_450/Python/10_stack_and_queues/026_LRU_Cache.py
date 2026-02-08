"""
Problem: LRU Cache Implementation
URL: https://practice.geeksforgeeks.org/problems/lru-cache/1

Problem Statement:
Implement LRU Cache with get and put operations in O(1) time.

Sample Input/Output:
Input: capacity=2, put(1,1), put(2,2), get(1), put(3,3), get(2)
Output: 1, -1
"""

from collections import OrderedDict


class LRUCache:
    def __init__(self, cap):
        """
        Initialize LRU cache with given capacity.
        Time Complexity: O(1)
        Space Complexity: O(capacity)
        """
        self.capacity = cap
        self.cache = OrderedDict()

    def Get(self, key):
        """
        Get value for key and mark as recently used.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if key not in self.cache:
            return -1
        value = self.cache[key]
        del self.cache[key]
        self.cache[key] = value
        return value

    def Put(self, key, value):
        """
        Put key-value pair, evicting least recently used if at capacity.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if key in self.cache:
            del self.cache[key]
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value


class Solution:
    def Test_LRU_Cache(self):
        cache = LRUCache(2)
        
        cache.Put(1, 1)
        cache.Put(2, 2)
        print(f"Get(1): {cache.Get(1)}")
        
        cache.Put(3, 3)
        print(f"Get(2): {cache.Get(2)}")
        print(f"Get(3): {cache.Get(3)}")
        print(f"Get(1): {cache.Get(1)}")
        
        cache.Put(4, 4)
        print(f"Get(1): {cache.Get(1)}")
        print(f"Get(3): {cache.Get(3)}")
        print(f"Get(4): {cache.Get(4)}")


def Test_LRU_Cache():
    solution = Solution()
    solution.Test_LRU_Cache()


if __name__ == "__main__":
    Test_LRU_Cache()
