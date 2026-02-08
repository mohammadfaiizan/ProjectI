/*
Problem: Copy Set Bits of Y to X in Range [L, R]
URL: https://www.geeksforgeeks.org/copy-set-bits-in-a-range/

Problem Statement:
Given x, y, l, r (1-indexed), copy set bits of y to x in bit positions l to r.

Sample Input/Output:
Input: x=10 (1010), y=13 (1101), l=2, r=3
Output: 14 (1110)
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Copy_Bits_Mask(int x, int y, int l, int r) {
        /*
        Create mask for range [l,r], x = x | (y & mask)
        Time Complexity: O(1)
        Space Complexity: O(1)
        */
        if (l < 1 || r < 1 || l > r) return x;
        int mask = ((1 << (r - l + 1)) - 1) << (l - 1);
        mask = mask & y;
        return x | mask;
    }

    int Copy_Bits_Loop(int x, int y, int l, int r) {
        /*
        Iterate bits l to r, set in x if set in y
        Time Complexity: O(r-l)
        Space Complexity: O(1)
        */
        if (l < 1 || r < 1 || l > r) return x;
        for (int i = l - 1; i < r; i++) {
            if (y & (1 << i)) {
                x |= (1 << i);
            }
        }
        return x;
    }
};

void Test_Copy_Set_Bits_Range() {
    Solution solution;
    
    cout << "Testing Copy_Bits_Mask:" << endl;
    int x1 = 10, y1 = 13, l1 = 2, r1 = 3;
    cout << "x=" << x1 << " (1010), y=" << y1 << " (1101), l=" << l1 << ", r=" << r1 << " -> " 
         << solution.Copy_Bits_Mask(x1, y1, l1, r1) << " (expected: 14)" << endl;
    
    cout << "\nTesting Copy_Bits_Loop:" << endl;
    int x2 = 10, y2 = 13, l2 = 2, r2 = 3;
    cout << "x=" << x2 << " (1010), y=" << y2 << " (1101), l=" << l2 << ", r=" << r2 << " -> " 
         << solution.Copy_Bits_Loop(x2, y2, l2, r2) << " (expected: 14)" << endl;
}

int main() {
    Test_Copy_Set_Bits_Range();
    return 0;
}
