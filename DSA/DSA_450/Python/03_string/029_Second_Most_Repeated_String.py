"""
Problem: Second Most Repeated String in a Sequence
URL: https://practice.geeksforgeeks.org/problems/second-most-repeated-string-in-a-sequence0534/1

Problem Statement:
Given a sequence of strings, find the second most repeated string in the sequence.

Sample Input/Output:
Input: arr = ["aaa", "bbb", "ccc", "bbb", "aaa", "aaa"]
Output: "bbb"

Input: arr = ["abc", "abc", "xyz", "xyz", "xyz"]
Output: "abc"
"""

import heapq


class Solution:
    def Second_Most_Repeated_Map(self, arr):
        """
        Using map to count frequencies, then find second max
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        mp = {}
        mx = 0
        for s in arr:
            mp[s] = mp.get(s, 0) + 1
            mx = max(mx, mp[s])

        secondMax = 0
        ans = ""
        for s, count in mp.items():
            if count != mx and count > secondMax:
                secondMax = count
                ans = s
        return ans

    def Second_Most_Repeated_Sorting(self, arr):
        """
        Sort, count frequencies, sort by frequency
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        mp = {}
        for s in arr:
            mp[s] = mp.get(s, 0) + 1

        freqs = [(count, s) for s, count in mp.items()]
        freqs.sort(reverse=True)

        if len(freqs) >= 2:
            return freqs[1][1]
        return ""

    def Second_Most_Repeated_Heap(self, arr):
        """
        Using max heap to find top 2
        Time Complexity: O(n + k log k) where k = unique strings
        Space Complexity: O(n)
        """
        mp = {}
        for s in arr:
            mp[s] = mp.get(s, 0) + 1

        pq = [(-count, s) for s, count in mp.items()]
        heapq.heapify(pq)

        if pq:
            heapq.heappop(pq)
        if pq:
            return heapq.heappop(pq)[1]
        return ""


def Test_Second_Most_Repeated():
    sol = Solution()
    tests = [
        ["aaa", "bbb", "ccc", "bbb", "aaa", "aaa"],
        ["abc", "abc", "xyz", "xyz", "xyz"],
        ["one", "two", "three", "one", "two", "one", "two"]
    ]

    for arr in tests:
        print(f"Input: {' '.join(arr)}")
        print(f"Map: {sol.Second_Most_Repeated_Map(arr)}")
        print(f"Sorting: {sol.Second_Most_Repeated_Sorting(arr)}")
        print(f"Heap: {sol.Second_Most_Repeated_Heap(arr)}")
        print('-' * 50)


if __name__ == "__main__":
    Test_Second_Most_Repeated()
