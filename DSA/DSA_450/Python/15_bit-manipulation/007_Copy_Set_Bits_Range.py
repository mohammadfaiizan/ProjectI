"""
Problem: Copy Set Bits of Y to X in Range [L, R]
URL: https://www.geeksforgeeks.org/copy-set-bits-in-a-range/

Problem Statement:
Given x, y, l, r (1-indexed), copy set bits of y to x in bit positions l to r.

Sample Input/Output:
Input: x=10 (1010), y=13 (1101), l=2, r=3
Output: 14 (1110)
"""


class Solution:
    def Copy_Bits_Mask(self, x, y, l, r):
        """
        Create mask for range [l,r], x = x | (y & mask)
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if l < 1 or r < 1 or l > r:
            return x
        mask = ((1 << (r - l + 1)) - 1) << (l - 1)
        mask = mask & y
        return x | mask

    def Copy_Bits_Loop(self, x, y, l, r):
        """
        Iterate bits l to r, set in x if set in y
        Time Complexity: O(r-l)
        Space Complexity: O(1)
        """
        if l < 1 or r < 1 or l > r:
            return x
        for i in range(l - 1, r):
            if y & (1 << i):
                x |= (1 << i)
        return x


def Test_Copy_Set_Bits_Range():
    solution = Solution()

    print("Testing Copy_Bits_Mask:")
    x1, y1, l1, r1 = 10, 13, 2, 3
    print(f"x={x1} (1010), y={y1} (1101), l={l1}, r={r1} ->",
          solution.Copy_Bits_Mask(x1, y1, l1, r1), "(expected: 14)")

    print("\nTesting Copy_Bits_Loop:")
    x2, y2, l2, r2 = 10, 13, 2, 3
    print(f"x={x2} (1010), y={y2} (1101), l={l2}, r={r2} ->",
          solution.Copy_Bits_Loop(x2, y2, l2, r2), "(expected: 14)")


if __name__ == "__main__":
    Test_Copy_Set_Bits_Range()
