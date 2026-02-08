/*
Problem: Arranging Amplifiers
URL: https://www.spoj.com/problems/ARRANGE/

Problem Statement:
Arrange N amplifiers to maximize a^(b^(c^(...))). Put 1s first, rest in descending order. Special case: swap 2,3 if both present.

Sample Input/Output:
Input: [2,3,1,4]
Output: [1,3,2,4]
Explanation: Sort + special handling for 2 and 3.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> Arrange_Amplifiers(vector<int>& amplifiers) {
        /*
        Sort + special handling approach
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        */
        vector<int> ones;
        vector<int> others;
        
        for (int amp : amplifiers) {
            if (amp == 1) {
                ones.push_back(amp);
            } else {
                others.push_back(amp);
            }
        }
        
        sort(others.begin(), others.end());
        
        vector<int> result;
        result.insert(result.end(), ones.begin(), ones.end());
        
        if (others.size() == 2 && others[0] == 2 && others[1] == 3) {
            result.push_back(3);
            result.push_back(2);
        } else {
            result.insert(result.end(), others.begin(), others.end());
        }
        
        return result;
    }
};

void Test_Arranging_Amplifiers() {
    Solution solution;
    
    vector<int> amps1 = {2, 3, 1, 4};
    vector<int> result1 = solution.Arrange_Amplifiers(amps1);
    cout << "Test 1: ";
    for (int x : result1) cout << x << " ";
    cout << endl;
    
    vector<int> amps2 = {1, 1, 2, 3};
    vector<int> result2 = solution.Arrange_Amplifiers(amps2);
    cout << "Test 2: ";
    for (int x : result2) cout << x << " ";
    cout << endl;
    
    vector<int> amps3 = {4, 5, 6};
    vector<int> result3 = solution.Arrange_Amplifiers(amps3);
    cout << "Test 3: ";
    for (int x : result3) cout << x << " ";
    cout << endl;
}

int main() {
    Test_Arranging_Amplifiers();
    return 0;
}
