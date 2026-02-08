/*
Problem: Stack Permutations
URL: https://www.geeksforgeeks.org/stack-permutations-check-if-an-array-is-stack-permutation-of-other/

Problem Statement:
Given two arrays, check if one is a stack permutation of other (can second be obtained from first using a stack).

Sample Input/Output:
Input: input=[1,2,3], output=[2,1,3]
Output: Yes

Input: input=[1,2,3], output=[3,1,2]
Output: No
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Check_Stack_Permutation_Simulation(vector<int>& input, vector<int>& output) {
        stack<int> st;
        int j = 0;
        for (int i = 0; i < input.size(); i++) {
            st.push(input[i]);
            while (!st.empty() && j < output.size() && st.top() == output[j]) {
                st.pop();
                j++;
            }
        }
        return j == output.size() && st.empty();
    }
};

void Test_Check_Stack_Permutation_Simulation() {
    Solution solution;
    
    vector<int> input1 = {1, 2, 3};
    vector<int> output1 = {2, 1, 3};
    cout << "Input: [1,2,3], Output: [2,1,3] -> " 
         << (solution.Check_Stack_Permutation_Simulation(input1, output1) ? "Yes" : "No") << endl;

    vector<int> input2 = {1, 2, 3};
    vector<int> output2 = {3, 1, 2};
    cout << "Input: [1,2,3], Output: [3,1,2] -> " 
         << (solution.Check_Stack_Permutation_Simulation(input2, output2) ? "Yes" : "No") << endl;

    vector<int> input3 = {1, 2, 3, 4};
    vector<int> output3 = {2, 4, 3, 1};
    cout << "Input: [1,2,3,4], Output: [2,4,3,1] -> " 
         << (solution.Check_Stack_Permutation_Simulation(input3, output3) ? "Yes" : "No") << endl;
}

int main() {
    Test_Check_Stack_Permutation_Simulation();
    return 0;
}
