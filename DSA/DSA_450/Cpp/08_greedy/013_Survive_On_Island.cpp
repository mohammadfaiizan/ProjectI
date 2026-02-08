/*
Problem: Survive On Island
URL: https://www.geeksforgeeks.org/survival/

Problem Statement:
Given S days to survive, N units of food can be bought per day, M units needed per day, and shop closed on Sundays. Can you survive? If yes, find min buying days.

Sample Input/Output:
Input: S=10, N=16, M=2
Output: 2
Explanation: Need 20 units total. Can buy 16 units on day 1 (Monday), 16 units on day 2 (Tuesday) = 32 units. Shop closed on Sunday (day 7). Can survive with 2 buying days.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Survive_On_Island_Math_Greedy(int S, int N, int M) {
        /*
        Math/greedy approach: Calculate total food needed and check if we can buy enough before Sundays
        Time Complexity: O(1)
        Space Complexity: O(1)
        */
        int total_food_needed = S * M;
        int sundays = S / 7;
        int buying_days_available = S - sundays;
        
        if (buying_days_available * N < total_food_needed) {
            return -1;
        }
        
        return (total_food_needed + N - 1) / N;
    }
};

void Test_Survive_On_Island() {
    Solution solution;
    
    cout << "Test 1: S=10, N=16, M=2" << endl;
    cout << "Result: " << solution.Survive_On_Island_Math_Greedy(10, 16, 2) << endl;
    
    cout << "\nTest 2: S=10, N=20, M=30" << endl;
    cout << "Result: " << solution.Survive_On_Island_Math_Greedy(10, 20, 30) << endl;
    
    cout << "\nTest 3: S=6, N=10, M=2" << endl;
    cout << "Result: " << solution.Survive_On_Island_Math_Greedy(6, 10, 2) << endl;
}

int main() {
    Test_Survive_On_Island();
    return 0;
}
