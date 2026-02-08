/*
Problem: Minimum Sum of Squares of Character Counts After Removing K Characters
URL: https://practice.geeksforgeeks.org/problems/game-with-string4100/1

Problem Statement:
Given a string s and an integer k, remove k characters from the string such that the sum of squares of the count of each distinct character remaining in the string is minimized.

Sample Input/Output:
Input: s = "aabcbcbcac", k = 3
Output: 27
Explanation: Remove 3 'c' characters. Remaining: a=3, b=3, c=1. Sum = 3^2 + 3^2 + 1^2 = 19
Actually, optimal: Remove 3 characters to minimize sum of squares.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Min_Sum_Squares_After_Remove_K_Max_Heap(string s, int k) {
        /*
        Max-heap greedy
        Time Complexity: O(n + k log n)
        Space Complexity: O(n)
        */
        unordered_map<char, int> freq;
        for (char c : s) {
            freq[c]++;
        }
        
        priority_queue<int> pq;
        for (auto& pair : freq) {
            pq.push(pair.second);
        }
        
        while (k > 0 && !pq.empty()) {
            int top = pq.top();
            pq.pop();
            top--;
            if (top > 0) {
                pq.push(top);
            }
            k--;
        }
        
        int sum = 0;
        while (!pq.empty()) {
            int val = pq.top();
            pq.pop();
            sum += val * val;
        }
        
        return sum;
    }
};

void Test_Min_Sum_Squares_After_Remove_K() {
    Solution solution;
    
    string s1 = "aabcbcbcac";
    int k1 = 3;
    cout << "Test 1 - Max Heap: " << solution.Min_Sum_Squares_After_Remove_K_Max_Heap(s1, k1) << endl;
    
    string s2 = "abccc";
    int k2 = 1;
    cout << "Test 2 - Max Heap: " << solution.Min_Sum_Squares_After_Remove_K_Max_Heap(s2, k2) << endl;
    
    string s3 = "aaab";
    int k3 = 2;
    cout << "Test 3 - Max Heap: " << solution.Min_Sum_Squares_After_Remove_K_Max_Heap(s3, k3) << endl;
    
    string s4 = "abbccc";
    int k4 = 3;
    cout << "Test 4 - Max Heap: " << solution.Min_Sum_Squares_After_Remove_K_Max_Heap(s4, k4) << endl;
}

int main() {
    Test_Min_Sum_Squares_After_Remove_K();
    return 0;
}
