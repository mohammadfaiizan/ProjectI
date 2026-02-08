/*
Problem: Maximum Sum Possible Equal Sum Three Stacks
URL: https://www.geeksforgeeks.org/find-maximum-sum-possible-equal-sum-three-stacks/

Problem Statement:
Given three stacks, find max possible equal sum by removing top elements.

Sample Input/Output:
Input: stack1=[3,2,1,1,1], stack2=[4,3,2], stack3=[1,1,4,1]
Output: 5
Explanation: Greedy remove from max sum stack approach.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Max_Equal_Sum(vector<int>& stack1, vector<int>& stack2, vector<int>& stack3) {
        /*
        Greedy remove from max sum stack approach
        Time Complexity: O(n1+n2+n3)
        Space Complexity: O(1)
        */
        int sum1 = 0, sum2 = 0, sum3 = 0;
        
        for (int x : stack1) sum1 += x;
        for (int x : stack2) sum2 += x;
        for (int x : stack3) sum3 += x;
        
        int idx1 = 0, idx2 = 0, idx3 = 0;
        
        while (idx1 < stack1.size() && idx2 < stack2.size() && idx3 < stack3.size()) {
            if (sum1 == sum2 && sum2 == sum3) {
                return sum1;
            }
            
            if (sum1 >= sum2 && sum1 >= sum3) {
                sum1 -= stack1[idx1++];
            } else if (sum2 >= sum1 && sum2 >= sum3) {
                sum2 -= stack2[idx2++];
            } else {
                sum3 -= stack3[idx3++];
            }
        }
        
        return 0;
    }
};

void Test_Max_Equal_Sum_Three_Stacks() {
    Solution solution;
    
    vector<int> stack1 = {3, 2, 1, 1, 1};
    vector<int> stack2 = {4, 3, 2};
    vector<int> stack3 = {1, 1, 4, 1};
    cout << "Test 1: " << solution.Max_Equal_Sum(stack1, stack2, stack3) << endl;
    
    vector<int> stack4 = {1, 1, 1, 1};
    vector<int> stack5 = {2, 2};
    vector<int> stack6 = {4};
    cout << "Test 2: " << solution.Max_Equal_Sum(stack4, stack5, stack6) << endl;
    
    vector<int> stack7 = {1, 2, 3};
    vector<int> stack8 = {2, 3, 1};
    vector<int> stack9 = {3, 1, 2};
    cout << "Test 3: " << solution.Max_Equal_Sum(stack7, stack8, stack9) << endl;
}

int main() {
    Test_Max_Equal_Sum_Three_Stacks();
    return 0;
}
