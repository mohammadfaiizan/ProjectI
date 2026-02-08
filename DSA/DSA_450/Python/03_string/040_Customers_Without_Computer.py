"""
Problem: Number of Customers Who Could Not Get a Computer
URL: https://www.geeksforgeeks.org/function-to-find-number-of-customers-who-could-not-get-a-computer/

Problem Statement:
Given N computers in a cafe and a string of uppercase characters representing
customers entering/leaving. First occurrence means entering, second means leaving.
Find the number of customers who could not get a computer.

Sample Input/Output:
Input: N = 2, seq = "ABCBCA"
Output: 1 (C couldn't get a computer)

Input: N = 2, seq = "ABCBCADEED"
Output: 2
"""


class Solution:
    def Disappointed_Array(self, capacity, s):
        """
        Using array to track customer state
        Time Complexity: O(n)
        Space Complexity: O(1) - fixed 26 chars
        """
        cnt = [0] * 26
        occupied = ans = 0

        for i in range(len(s)):
            idx = ord(s[i]) - ord('A')
            if cnt[idx] == 0:
                if occupied < capacity:
                    cnt[idx] = 1
                    occupied += 1
                else:
                    ans += 1
                    cnt[idx] = -1
            elif cnt[idx] == 1:
                cnt[idx] = 0
                occupied -= 1

        return ans

    def Disappointed_Map(self, capacity, s):
        """
        Using dictionary for tracking
        Time Complexity: O(n)
        Space Complexity: O(k) where k = unique customers
        """
        state = {}
        occupied = ans = 0

        for c in s:
            if c not in state or state[c] == 0:
                if occupied < capacity:
                    state[c] = 1
                    occupied += 1
                else:
                    ans += 1
                    state[c] = -1
            elif state[c] == 1:
                state[c] = 0
                occupied -= 1

        return ans


def Test_Customers_Without_Computer():
    sol = Solution()
    tests = [
        (2, "ABCBCA"),
        (2, "ABCBCADEED"),
        (3, "ABCABC"),
        (1, "ABAB"),
        (3, "ABCDABCD")
    ]

    for n, seq in tests:
        print(f"N={n}, Seq: {seq}")
        print(f"Array: {sol.Disappointed_Array(n, seq)}")
        print(f"Map: {sol.Disappointed_Map(n, seq)}")
        print('-' * 50)


if __name__ == "__main__":
    Test_Customers_Without_Computer()
