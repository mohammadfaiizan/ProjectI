/*
Problem: First Circular Tour Visiting All Petrol Pumps
URL: https://practice.geeksforgeeks.org/problems/circular-tour/1

Problem Statement:
Find the first petrol pump from where a circular tour can be completed visiting all pumps.

Sample Input/Output:
Input: petrol=[4,6,7,4], distance=[6,5,3,5]
Output: 1
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Circular_Tour_Brute_Force(vector<int>& petrol, vector<int>& distance) {
        int n = petrol.size();
        for (int start = 0; start < n; start++) {
            int current_petrol = 0;
            int i = start;
            int count = 0;
            while (count < n) {
                current_petrol += petrol[i] - distance[i];
                if (current_petrol < 0) break;
                i = (i + 1) % n;
                count++;
            }
            if (count == n && current_petrol >= 0) {
                return start;
            }
        }
        return -1;
    }

    int Circular_Tour_Deficit_Tracking(vector<int>& petrol, vector<int>& distance) {
        int n = petrol.size();
        int start = 0;
        int deficit = 0;
        int balance = 0;
        
        for (int i = 0; i < n; i++) {
            balance += petrol[i] - distance[i];
            if (balance < 0) {
                deficit += balance;
                start = i + 1;
                balance = 0;
            }
        }
        
        if (deficit + balance >= 0) {
            return start;
        }
        return -1;
    }
};

void Test_Circular_Tour_Brute_Force() {
    Solution solution;
    
    vector<int> petrol1 = {4, 6, 7, 4};
    vector<int> distance1 = {6, 5, 3, 5};
    cout << "Brute Force - Petrol: [4,6,7,4], Distance: [6,5,3,5] -> Start: " 
         << solution.Circular_Tour_Brute_Force(petrol1, distance1) << endl;

    vector<int> petrol2 = {6, 7, 4, 10};
    vector<int> distance2 = {5, 6, 7, 6};
    cout << "Brute Force - Petrol: [6,7,4,10], Distance: [5,6,7,6] -> Start: " 
         << solution.Circular_Tour_Brute_Force(petrol2, distance2) << endl;
}

void Test_Circular_Tour_Deficit_Tracking() {
    Solution solution;
    
    vector<int> petrol1 = {4, 6, 7, 4};
    vector<int> distance1 = {6, 5, 3, 5};
    cout << "Deficit Tracking - Petrol: [4,6,7,4], Distance: [6,5,3,5] -> Start: " 
         << solution.Circular_Tour_Deficit_Tracking(petrol1, distance1) << endl;

    vector<int> petrol2 = {6, 7, 4, 10};
    vector<int> distance2 = {5, 6, 7, 6};
    cout << "Deficit Tracking - Petrol: [6,7,4,10], Distance: [5,6,7,6] -> Start: " 
         << solution.Circular_Tour_Deficit_Tracking(petrol2, distance2) << endl;

    vector<int> petrol3 = {1, 2};
    vector<int> distance3 = {2, 1};
    cout << "Deficit Tracking - Petrol: [1,2], Distance: [2,1] -> Start: " 
         << solution.Circular_Tour_Deficit_Tracking(petrol3, distance3) << endl;
}

int main() {
    Test_Circular_Tour_Brute_Force();
    Test_Circular_Tour_Deficit_Tracking();
    return 0;
}
