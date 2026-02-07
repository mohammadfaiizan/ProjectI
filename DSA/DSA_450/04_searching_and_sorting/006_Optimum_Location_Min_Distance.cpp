/*
Problem: Optimum Location of Point to Minimize Total Distance
URL: https://www.geeksforgeeks.org/optimum-location-point-minimize-total-distance/

Problem Statement:
Given a set of points, find the point on a line that minimizes the total distance to all given points. The line is given by ax + by + c = 0.

Sample Input/Output:
Input: points = [(0, 0), (2, 0), (3, 0), (5, 0)], line: y = 0 (a=0, b=1, c=0)
Output: (2.5, 0)

Input: points = [(1, 1), (2, 2), (3, 3)], line: x - y = 0 (a=1, b=-1, c=0)
Output: (2, 2)
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    double Optimum_Location_Ternary_Search(vector<pair<int, int>>& points, double a, double b, double c) {
        /*
        Ternary search to find optimal point on line that minimizes total distance
        Time Complexity: O(n * log(range))
        Space Complexity: O(1)
        */
        auto distance = [&](double x) -> double {
            double y = (a == 0) ? (-c / b) : (-a * x - c) / b;
            double total = 0.0;
            for (auto& p : points) {
                double dx = p.first - x;
                double dy = p.second - y;
                total += sqrt(dx * dx + dy * dy);
            }
            return total;
        };
        
        double left = -1e6, right = 1e6;
        double eps = 1e-6;
        
        while (right - left > eps) {
            double m1 = left + (right - left) / 3.0;
            double m2 = right - (right - left) / 3.0;
            
            if (distance(m1) < distance(m2)) {
                right = m2;
            } else {
                left = m1;
            }
        }
        
        return (left + right) / 2.0;
    }
};

void Test_Optimum_Location_Min_Distance() {
    Solution sol;
    vector<pair<vector<pair<int, int>>, vector<double>>> tests = {
        {{{0, 0}, {2, 0}, {3, 0}, {5, 0}}, {0.0, 1.0, 0.0}},
        {{{1, 1}, {2, 2}, {3, 3}}, {1.0, -1.0, 0.0}},
        {{{0, 1}, {1, 0}, {2, 1}}, {0.0, 1.0, -1.0}}
    };

    for (auto& test : tests) {
        vector<pair<int, int>> points = test.first;
        double a = test.second[0], b = test.second[1], c = test.second[2];
        
        cout << "Points: ";
        for (auto& p : points) {
            cout << "(" << p.first << "," << p.second << ") ";
        }
        cout << endl;
        cout << "Line: " << a << "x + " << b << "y + " << c << " = 0" << endl;
        
        double res = sol.Optimum_Location_Ternary_Search(points, a, b, c);
        double y = (a == 0) ? (-c / b) : (-a * res - c) / b;
        cout << "Optimum Location: (" << res << ", " << y << ")" << endl;
        
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Optimum_Location_Min_Distance();
    return 0;
}
