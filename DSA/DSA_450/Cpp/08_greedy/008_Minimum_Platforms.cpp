/*
Problem: Minimum Platforms
URL: https://practice.geeksforgeeks.org/problems/minimum-platforms-1587115620/1

Problem Statement:
Given arrival and departure times of all trains that reach a railway station, find the minimum number of platforms required for the railway station so that no train waits.

Sample Input/Output:
Input: arr[] = {900, 940, 950, 1100, 1500, 1800}, dep[] = {910, 1200, 1120, 1130, 1900, 2000}
Output: 3
Explanation: Minimum 3 platforms are required to accommodate all trains.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Find_Platform_Sort_Two_Pointer(int arr[], int dep[], int n) {
        /*
        Sort both arrays, use two pointers to track overlapping trains
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        */
        sort(arr, arr + n);
        sort(dep, dep + n);
        
        int platforms = 1;
        int max_platforms = 1;
        int i = 1;
        int j = 0;
        
        while (i < n && j < n) {
            if (arr[i] <= dep[j]) {
                platforms++;
                i++;
            } else {
                platforms--;
                j++;
            }
            max_platforms = max(max_platforms, platforms);
        }
        
        return max_platforms;
    }
};

void Test_Minimum_Platforms() {
    Solution solution;
    int arr[] = {900, 940, 950, 1100, 1500, 1800};
    int dep[] = {910, 1200, 1120, 1130, 1900, 2000};
    int n = 6;
    cout << "Minimum platforms: " << solution.Find_Platform_Sort_Two_Pointer(arr, dep, n) << endl;
}

int main() {
    Test_Minimum_Platforms();
    return 0;
}
