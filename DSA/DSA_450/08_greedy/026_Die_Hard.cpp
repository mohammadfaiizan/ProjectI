/*
Problem: Die Hard
URL: https://www.spoj.com/problems/DIEHARD/

Problem Statement:
A character has health H and armor A. Each time unit, must visit one of 3 places: Air (+3H,+2A), Water(-5H,-10A), Fire(-20H,+5A). Cannot visit same place consecutively. Maximize time units survived.

Sample Input/Output:
Input: H=20, A=8
Output: 5
Explanation: Greedy alternating approach maximizes survival time.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Max_Survival_Time_Greedy(int H, int A) {
        /*
        Greedy alternating approach
        Time Complexity: O(max(H,A))
        Space Complexity: O(1)
        */
        int time = 0;
        bool in_air = false;
        
        while (H > 0 && A > 0) {
            if (!in_air) {
                H += 3;
                A += 2;
                in_air = true;
            } else {
                if (H > 5 && A > 10) {
                    H -= 5;
                    A -= 10;
                } else if (H > 20) {
                    H -= 20;
                    A += 5;
                } else {
                    break;
                }
                in_air = false;
            }
            time++;
        }
        
        return time;
    }
    
    int Max_Survival_Time_DP(int H, int A) {
        /*
        DP memoization approach
        Time Complexity: O(H*A)
        Space Complexity: O(H*A)
        */
        map<pair<int, int>, int> memo;
        return Max_Survival_Time_DP_Helper(H, A, -1, memo);
    }
    
private:
    int Max_Survival_Time_DP_Helper(int H, int A, int last, map<pair<int, int>, int>& memo) {
        if (H <= 0 || A <= 0) return 0;
        
        pair<int, int> key = {H, A};
        if (memo.find(key) != memo.end()) return memo[key];
        
        int max_time = 0;
        
        if (last != 0) {
            max_time = max(max_time, 1 + Max_Survival_Time_DP_Helper(H + 3, A + 2, 0, memo));
        }
        if (last != 1 && H > 5 && A > 10) {
            max_time = max(max_time, 1 + Max_Survival_Time_DP_Helper(H - 5, A - 10, 1, memo));
        }
        if (last != 2 && H > 20) {
            max_time = max(max_time, 1 + Max_Survival_Time_DP_Helper(H - 20, A + 5, 2, memo));
        }
        
        memo[key] = max_time;
        return max_time;
    }
};

void Test_Die_Hard() {
    Solution solution;
    
    cout << "Test 1 (Greedy): " << solution.Max_Survival_Time_Greedy(20, 8) << endl;
    cout << "Test 1 (DP): " << solution.Max_Survival_Time_DP(20, 8) << endl;
    
    cout << "Test 2 (Greedy): " << solution.Max_Survival_Time_Greedy(10, 5) << endl;
    cout << "Test 2 (DP): " << solution.Max_Survival_Time_DP(10, 5) << endl;
}

int main() {
    Test_Die_Hard();
    return 0;
}
