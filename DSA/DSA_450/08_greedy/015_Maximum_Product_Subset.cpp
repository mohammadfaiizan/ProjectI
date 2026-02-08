/*
Problem: Maximum Product Subset
URL: https://www.geeksforgeeks.org/maximum-product-subset-array/

Problem Statement:
Find maximum product of a subset of an array (handles negatives and zeros).

Sample Input/Output:
Input: arr[] = {-1, -1, -2, 4, 3}
Output: 24
Explanation: Maximum product is (-1) * (-1) * (-2) * 4 * 3 = 24
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    long long Maximum_Product_Subset_Count_Negatives_Zeros(vector<int>& arr) {
        /*
        Count negatives/zeros greedy approach: Handle negative count and zeros
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int n = arr.size();
        int negative_count = 0;
        int zero_count = 0;
        int max_negative = INT_MIN;
        long long product = 1;
        
        for (int num : arr) {
            if (num == 0) {
                zero_count++;
                continue;
            }
            if (num < 0) {
                negative_count++;
                max_negative = max(max_negative, num);
            }
            product *= num;
        }
        
        if (zero_count == n) {
            return 0;
        }
        
        if (negative_count % 2 == 1) {
            if (negative_count == 1 && zero_count > 0 && negative_count + zero_count == n) {
                return 0;
            }
            product /= max_negative;
        }
        
        return product;
    }
};

void Test_Maximum_Product_Subset() {
    Solution solution;
    
    vector<int> arr1 = {-1, -1, -2, 4, 3};
    cout << "Test 1: " << solution.Maximum_Product_Subset_Count_Negatives_Zeros(arr1) << endl;
    
    vector<int> arr2 = {-1, 0};
    cout << "Test 2: " << solution.Maximum_Product_Subset_Count_Negatives_Zeros(arr2) << endl;
    
    vector<int> arr3 = {0, 0, 0};
    cout << "Test 3: " << solution.Maximum_Product_Subset_Count_Negatives_Zeros(arr3) << endl;
    
    vector<int> arr4 = {-1, -2, -3};
    cout << "Test 4: " << solution.Maximum_Product_Subset_Count_Negatives_Zeros(arr4) << endl;
}

int main() {
    Test_Maximum_Product_Subset();
    return 0;
}
