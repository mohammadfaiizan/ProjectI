/*
Problem: Minimum Sum of Two Numbers Formed from Digits of an Array
URL: https://practice.geeksforgeeks.org/problems/minimum-sum4058/1

Problem Statement:
Given an array of digits, form two numbers using all digits such that their sum is minimized.

Sample Input/Output:
Input: [6,8,4,5,2,3]
Output: "604"
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    string Min_Sum_Sort(vector<int>& arr) {
        /*
        Sort Approach
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        */
        sort(arr.begin(), arr.end());
        string num1 = "";
        string num2 = "";
        
        for (int i = 0; i < arr.size(); i++) {
            if (i % 2 == 0) {
                num1 += to_string(arr[i]);
            } else {
                num2 += to_string(arr[i]);
            }
        }
        
        return AddStrings(num1, num2);
    }
    
    string Min_Sum_Min_Heap(vector<int>& arr) {
        /*
        Min Heap Approach
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        */
        priority_queue<int, vector<int>, greater<int>> pq;
        for (int num : arr) {
            pq.push(num);
        }
        
        string num1 = "";
        string num2 = "";
        bool toggle = true;
        
        while (!pq.empty()) {
            int digit = pq.top();
            pq.pop();
            if (toggle) {
                num1 += to_string(digit);
            } else {
                num2 += to_string(digit);
            }
            toggle = !toggle;
        }
        
        return AddStrings(num1, num2);
    }
    
private:
    string AddStrings(string num1, string num2) {
        if (num1.empty()) return num2;
        if (num2.empty()) return num1;
        
        reverse(num1.begin(), num1.end());
        reverse(num2.begin(), num2.end());
        
        string result = "";
        int carry = 0;
        int i = 0, j = 0;
        
        while (i < num1.length() || j < num2.length() || carry) {
            int sum = carry;
            if (i < num1.length()) {
                sum += num1[i] - '0';
                i++;
            }
            if (j < num2.length()) {
                sum += num2[j] - '0';
                j++;
            }
            result += to_string(sum % 10);
            carry = sum / 10;
        }
        
        reverse(result.begin(), result.end());
        
        int start = 0;
        while (start < result.length() && result[start] == '0') {
            start++;
        }
        
        return start == result.length() ? "0" : result.substr(start);
    }
};

void Test_Min_Sum() {
    Solution solution;
    
    vector<int> arr1 = {6, 8, 4, 5, 2, 3};
    cout << "Test 1 Sort: " << solution.Min_Sum_Sort(arr1) << endl;
    vector<int> arr1b = {6, 8, 4, 5, 2, 3};
    cout << "Test 1 Min Heap: " << solution.Min_Sum_Min_Heap(arr1b) << endl;
    
    vector<int> arr2 = {5, 3, 0, 7, 4};
    cout << "Test 2 Sort: " << solution.Min_Sum_Sort(arr2) << endl;
    vector<int> arr2b = {5, 3, 0, 7, 4};
    cout << "Test 2 Min Heap: " << solution.Min_Sum_Min_Heap(arr2b) << endl;
    
    vector<int> arr3 = {1, 2, 3, 4};
    cout << "Test 3 Sort: " << solution.Min_Sum_Sort(arr3) << endl;
    vector<int> arr3b = {1, 2, 3, 4};
    cout << "Test 3 Min Heap: " << solution.Min_Sum_Min_Heap(arr3b) << endl;
}

int main() {
    Test_Min_Sum();
    return 0;
}
