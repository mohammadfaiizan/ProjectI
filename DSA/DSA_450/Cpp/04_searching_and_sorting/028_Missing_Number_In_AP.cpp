/*
 * Problem: Arithmetic Number / Missing Number in AP
 * URL: https://practice.geeksforgeeks.org/problems/arithmetic-number2815/1
 * 
 * Problem Statement:
 * Given first term A, last term B, and common difference C, check if B exists in the AP.
 * 
 * Sample Input:
 * A = 1, B = 3, C = 2
 * 
 * Sample Output:
 * 1
 */

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    /*
     * Approach: Use mathematical formula to check if B exists in AP
     * Formula: B = A + n*C where n >= 0
     * Rearranging: n = (B - A) / C
     * B exists if (B - A) is divisible by C and n >= 0
     * 
     * Time Complexity: O(1)
     * Space Complexity: O(1)
     */
    int InSequence_Math(long long A, long long B, long long C) {
        if (C == 0) {
            return (A == B) ? 1 : 0;
        }
        long long diff = B - A;
        if ((diff > 0 && C < 0) || (diff < 0 && C > 0)) {
            return 0;
        }
        return (diff % C == 0) ? 1 : 0;
    }

    /*
     * Approach: Iteratively check each term in the AP until we reach or exceed B
     * Start from A and keep adding C until we reach B or exceed it
     * 
     * Time Complexity: O(n) where n is the number of terms
     * Space Complexity: O(1)
     */
    int InSequence_Iterative(long long A, long long B, long long C) {
        if (C == 0) {
            return (A == B) ? 1 : 0;
        }
        long long current = A;
        if (C > 0) {
            while (current < B) {
                current += C;
            }
        } else {
            while (current > B) {
                current += C;
            }
        }
        return (current == B) ? 1 : 0;
    }
};

void Test_Missing_Number_In_AP() {
    Solution sol;
    
    assert(sol.InSequence_Math(1, 3, 2) == 1);
    assert(sol.InSequence_Math(1, 2, 2) == 1);
    assert(sol.InSequence_Math(1, 5, 2) == 1);
    assert(sol.InSequence_Math(1, 4, 2) == 0);
    assert(sol.InSequence_Math(5, 1, -2) == 1);
    assert(sol.InSequence_Math(1, 1, 0) == 1);
    assert(sol.InSequence_Math(1, 2, 0) == 0);
    
    assert(sol.InSequence_Iterative(1, 3, 2) == 1);
    assert(sol.InSequence_Iterative(1, 2, 2) == 1);
    assert(sol.InSequence_Iterative(1, 5, 2) == 1);
    assert(sol.InSequence_Iterative(1, 4, 2) == 0);
    assert(sol.InSequence_Iterative(5, 1, -2) == 1);
    assert(sol.InSequence_Iterative(1, 1, 0) == 1);
    assert(sol.InSequence_Iterative(1, 2, 0) == 0);
    
    cout << "All test cases passed!" << endl;
}

int main() {
    Test_Missing_Number_In_AP();
    return 0;
}
