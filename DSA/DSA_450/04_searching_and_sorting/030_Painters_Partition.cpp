/*
 * Problem: Painters Partition Problem
 * URL: https://practice.geeksforgeeks.org/problems/the-painters-partition-problem1535/1
 * 
 * Problem Statement:
 * Divide n boards among k painters to minimize maximum time.
 * Each board takes a certain time to paint. Painters work sequentially.
 * 
 * Sample Input:
 * arr[] = {10, 20, 30, 40}, k = 2
 * 
 * Sample Output:
 * 60
 */

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    /*
     * Approach: Check if it's possible to paint boards with given time limit
     * Assign boards to painters greedily, ensuring no painter exceeds time limit
     * 
     * Time Complexity: O(n)
     * Space Complexity: O(1)
     */
    bool Is_Possible(int arr[], int n, int k, long long time_limit) {
        long long current_sum = 0;
        int painters_used = 1;
        
        for (int i = 0; i < n; i++) {
            if (arr[i] > time_limit) {
                return false;
            }
            if (current_sum + arr[i] > time_limit) {
                painters_used++;
                current_sum = arr[i];
                if (painters_used > k) {
                    return false;
                }
            } else {
                current_sum += arr[i];
            }
        }
        return true;
    }

    /*
     * Approach: Binary search on answer
     * Search for minimum maximum time needed
     * Low = maximum element, High = sum of all elements
     * 
     * Time Complexity: O(n log(sum))
     * Space Complexity: O(1)
     */
    long long Min_Time(int arr[], int n, int k) {
        if (n < k) {
            return *max_element(arr, arr + n);
        }
        
        long long low = *max_element(arr, arr + n);
        long long high = accumulate(arr, arr + n, 0LL);
        long long result = high;
        
        while (low <= high) {
            long long mid = low + (high - low) / 2;
            
            if (Is_Possible(arr, n, k, mid)) {
                result = mid;
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        
        return result;
    }
};

void Test_Painters_Partition() {
    Solution sol;
    
    int arr1[] = {10, 20, 30, 40};
    assert(sol.Min_Time(arr1, 4, 2) == 60);
    
    int arr2[] = {10, 10, 10, 10};
    assert(sol.Min_Time(arr2, 4, 2) == 20);
    
    int arr3[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    assert(sol.Min_Time(arr3, 9, 3) == 17);
    
    int arr4[] = {5};
    assert(sol.Min_Time(arr4, 1, 1) == 5);
    
    cout << "All test cases passed!" << endl;
}

int main() {
    Test_Painters_Partition();
    return 0;
}
