/*
 * Problem: Smallest Factorial Number with N Trailing Zeros
 * URL: https://practice.geeksforgeeks.org/problems/smallest-factorial-number5929/1
 * 
 * Problem Statement:
 * Find smallest number whose factorial has at least n trailing zeros.
 * Trailing zeros in factorial = count of factor 5 (since 10 = 2*5, and 2s are more abundant)
 * 
 * Sample Input:
 * n = 1
 * 
 * Sample Output:
 * 5
 */

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    /*
     * Approach: Count trailing zeros in factorial of a number
     * Count number of 5s in prime factorization: n/5 + n/25 + n/125 + ...
     * 
     * Time Complexity: O(log n)
     * Space Complexity: O(1)
     */
    long long Count_Trailing_Zeros(long long num) {
        long long count = 0;
        long long divisor = 5;
        while (divisor <= num) {
            count += num / divisor;
            divisor *= 5;
        }
        return count;
    }

    /*
     * Approach: Binary search on answer
     * Search for smallest number whose factorial has at least n trailing zeros
     * Low = 0, High = 5*n (upper bound)
     * 
     * Time Complexity: O(log^2(n))
     * Space Complexity: O(1)
     */
    int Find_Smallest_Number(int n) {
        if (n == 0) return 0;
        
        long long low = 0;
        long long high = 5LL * n;
        long long result = high;
        
        while (low <= high) {
            long long mid = low + (high - low) / 2;
            long long zeros = Count_Trailing_Zeros(mid);
            
            if (zeros >= n) {
                result = mid;
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        
        return result;
    }
};

void Test_Smallest_Number_Trailing_Zeros() {
    Solution sol;
    
    assert(sol.Find_Smallest_Number(1) == 5);
    assert(sol.Find_Smallest_Number(5) == 25);
    assert(sol.Find_Smallest_Number(0) == 0);
    assert(sol.Find_Smallest_Number(6) == 25);
    assert(sol.Find_Smallest_Number(25) == 105);
    
    assert(sol.Count_Trailing_Zeros(5) == 1);
    assert(sol.Count_Trailing_Zeros(10) == 2);
    assert(sol.Count_Trailing_Zeros(25) == 6);
    assert(sol.Count_Trailing_Zeros(100) == 24);
    
    cout << "All test cases passed!" << endl;
}

int main() {
    Test_Smallest_Number_Trailing_Zeros();
    return 0;
}
