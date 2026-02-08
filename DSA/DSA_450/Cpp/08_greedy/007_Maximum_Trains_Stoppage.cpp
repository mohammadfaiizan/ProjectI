/*
Problem: Maximum Trains Stoppage
URL: https://www.geeksforgeeks.org/maximum-trains-stoppage-can-provided/

Problem Statement:
Given N platforms and M trains with arrival time, departure time and platform number, find the maximum number of trains that can be stopped without any conflict.

Sample Input/Output:
Input: n = 3, m = 6, arr[] = {1000, 1100, 1200, 1300, 1400, 1500}, dep[] = {1100, 1200, 1300, 1400, 1500, 1600}, plat[] = {1, 1, 2, 2, 3, 3}
Output: 6
Explanation: All trains can be accommodated on their respective platforms.
*/

#include <bits/stdc++.h>
using namespace std;

struct Train {
    int arrival;
    int departure;
    int platform;
};

class Solution {
public:
    int Max_Trains_Stoppage_Activity_Selection(int n, int m, vector<int>& arr, vector<int>& dep, vector<int>& plat) {
        /*
        Activity selection per platform - sort trains by departure time per platform
        Time Complexity: O(m log m)
        Space Complexity: O(m)
        */
        vector<Train> trains;
        for (int i = 0; i < m; i++) {
            trains.push_back({arr[i], dep[i], plat[i]});
        }
        
        sort(trains.begin(), trains.end(), [](Train a, Train b) {
            if (a.platform != b.platform) {
                return a.platform < b.platform;
            }
            return a.departure < b.departure;
        });
        
        int count = 0;
        vector<int> last_departure(n + 1, -1);
        
        for (auto& train : trains) {
            if (train.arrival >= last_departure[train.platform]) {
                count++;
                last_departure[train.platform] = train.departure;
            }
        }
        
        return count;
    }
};

void Test_Maximum_Trains_Stoppage() {
    Solution solution;
    int n = 3, m = 6;
    vector<int> arr = {1000, 1100, 1200, 1300, 1400, 1500};
    vector<int> dep = {1100, 1200, 1300, 1400, 1500, 1600};
    vector<int> plat = {1, 1, 2, 2, 3, 3};
    cout << "Max trains: " << solution.Max_Trains_Stoppage_Activity_Selection(n, m, arr, dep, plat) << endl;
}

int main() {
    Test_Maximum_Trains_Stoppage();
    return 0;
}
