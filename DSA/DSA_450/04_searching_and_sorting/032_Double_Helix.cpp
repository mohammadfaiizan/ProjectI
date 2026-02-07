/*
 * Problem: Double Helix / ANARC05B
 * URL: https://www.spoj.com/problems/ANARC05B/
 * 
 * Problem Statement:
 * Two sorted arrays with some common elements.
 * Find max sum path switching allowed at common points.
 * Can switch from one array to another only at common elements.
 * 
 * Sample Input:
 * arr1[] = {2, 3, 7, 10, 12}, arr2[] = {1, 5, 7, 8}
 * 
 * Sample Output:
 * 35
 */

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    /*
     * Approach: Two pointer technique
     * Traverse both arrays simultaneously, accumulate sums between common elements
     * At common elements, take maximum sum and reset both accumulators
     * 
     * Time Complexity: O(n + m)
     * Space Complexity: O(1)
     */
    int Max_Sum_Path(int arr1[], int n, int arr2[], int m) {
        int i = 0, j = 0;
        int sum1 = 0, sum2 = 0;
        int result = 0;
        
        while (i < n && j < m) {
            if (arr1[i] < arr2[j]) {
                sum1 += arr1[i];
                i++;
            } else if (arr1[i] > arr2[j]) {
                sum2 += arr2[j];
                j++;
            } else {
                result += max(sum1, sum2) + arr1[i];
                sum1 = 0;
                sum2 = 0;
                i++;
                j++;
            }
        }
        
        while (i < n) {
            sum1 += arr1[i];
            i++;
        }
        
        while (j < m) {
            sum2 += arr2[j];
            j++;
        }
        
        result += max(sum1, sum2);
        return result;
    }
};

void Test_Double_Helix() {
    Solution sol;
    
    int arr1[] = {2, 3, 7, 10, 12};
    int arr2[] = {1, 5, 7, 8};
    assert(sol.Max_Sum_Path(arr1, 5, arr2, 4) == 35);
    
    int arr3[] = {1, 2, 3};
    int arr4[] = {3, 4, 5};
    assert(sol.Max_Sum_Path(arr3, 3, arr4, 3) == 15);
    
    int arr5[] = {1, 2, 3};
    int arr6[] = {4, 5, 6};
    assert(sol.Max_Sum_Path(arr5, 3, arr6, 3) == 21);
    
    int arr7[] = {1};
    int arr8[] = {1};
    assert(sol.Max_Sum_Path(arr7, 1, arr8, 1) == 1);
    
    cout << "All test cases passed!" << endl;
}

int main() {
    Test_Double_Helix();
    return 0;
}
