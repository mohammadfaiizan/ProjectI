/*
Problem: Choose and Swap
URL: https://practice.geeksforgeeks.org/problems/choose-and-swap0531/1

Problem Statement:
Given a string S of lowercase alphabets, choose two characters and swap all occurrences of first character with second and vice versa. Find the lexicographically smallest string possible.

Sample Input/Output:
Input: S = "ccad"
Output: "aacd"
Explanation: Swap 'c' with 'a' to get "aacd" which is lexicographically smallest.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    string Choose_And_Swap_First_Occurrence(string A) {
        /*
        Track first occurrence of each character, find first char that can be swapped
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        vector<int> first_occurrence(26, -1);
        
        for (int i = 0; i < A.length(); i++) {
            if (first_occurrence[A[i] - 'a'] == -1) {
                first_occurrence[A[i] - 'a'] = i;
            }
        }
        
        char swap_char1 = '\0';
        char swap_char2 = '\0';
        
        for (int i = 0; i < A.length(); i++) {
            for (int j = 0; j < A[i] - 'a'; j++) {
                if (first_occurrence[j] != -1 && first_occurrence[j] > i) {
                    swap_char1 = A[i];
                    swap_char2 = 'a' + j;
                    break;
                }
            }
            if (swap_char1 != '\0') break;
        }
        
        if (swap_char1 == '\0') {
            return A;
        }
        
        for (int i = 0; i < A.length(); i++) {
            if (A[i] == swap_char1) {
                A[i] = swap_char2;
            } else if (A[i] == swap_char2) {
                A[i] = swap_char1;
            }
        }
        
        return A;
    }
};

void Test_Choose_And_Swap() {
    Solution solution;
    string S = "ccad";
    cout << "Original: " << S << endl;
    cout << "Result: " << solution.Choose_And_Swap_First_Occurrence(S) << endl;
}

int main() {
    Test_Choose_And_Swap();
    return 0;
}
