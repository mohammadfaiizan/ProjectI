/*
Problem: Given N Appointments, Find Conflicting Appointments
URL: https://www.geeksforgeeks.org/given-n-appointments-find-conflicting-appointments/

Problem Statement:
Given N appointments, find all conflicting appointments using interval tree.

Sample Input/Output:
Input: appointments = [(1,5), (3,7), (2,6), (10,15), (5,6), (4,100)]
Output: Conflicts: (1,5) conflicts with (3,7), (2,6)
Explanation: Overlapping intervals conflict with each other.
*/

#include <bits/stdc++.h>
using namespace std;

struct Interval {
    int start;
    int end;
    Interval(int s, int e) : start(s), end(e) {}
};

struct IntervalNode {
    Interval interval;
    int max_end;
    IntervalNode* left;
    IntervalNode* right;
    IntervalNode(Interval i) : interval(i), max_end(i.end), left(NULL), right(NULL) {}
};

void Print_Inorder(IntervalNode* root) {
    if (root == NULL) return;
    Print_Inorder(root->left);
    cout << "[" << root->interval.start << "," << root->interval.end << "] ";
    Print_Inorder(root->right);
}

class Solution {
public:
    bool Do_Overlap(Interval i1, Interval i2) {
        return i1.start < i2.end && i2.start < i1.end;
    }

    IntervalNode* Insert_Interval_Tree(IntervalNode* root, Interval interval, vector<Interval>& conflicts) {
        if (root == NULL) {
            IntervalNode* node = new IntervalNode(interval);
            return node;
        }
        if (Do_Overlap(root->interval, interval)) {
            conflicts.push_back(root->interval);
        }
        if (interval.start < root->interval.start) {
            root->left = Insert_Interval_Tree(root->left, interval, conflicts);
        } else {
            root->right = Insert_Interval_Tree(root->right, interval, conflicts);
        }
        root->max_end = max(root->max_end, interval.end);
        if (root->left) root->max_end = max(root->max_end, root->left->max_end);
        if (root->right) root->max_end = max(root->max_end, root->right->max_end);
        return root;
    }

    vector<vector<Interval>> Find_Conflicts_Interval_Tree(vector<Interval>& appointments) {
        /*
        Interval tree approach
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        */
        vector<vector<Interval>> all_conflicts;
        IntervalNode* root = NULL;
        for (auto& app : appointments) {
            vector<Interval> conflicts;
            root = Insert_Interval_Tree(root, app, conflicts);
            if (!conflicts.empty()) {
                conflicts.push_back(app);
                all_conflicts.push_back(conflicts);
            }
        }
        return all_conflicts;
    }

    vector<vector<Interval>> Find_Conflicts_Brute(vector<Interval>& appointments) {
        /*
        Brute force approach
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        */
        vector<vector<Interval>> all_conflicts;
        int n = appointments.size();
        for (int i = 0; i < n; i++) {
            vector<Interval> conflicts;
            for (int j = 0; j < n; j++) {
                if (i != j && Do_Overlap(appointments[i], appointments[j])) {
                    conflicts.push_back(appointments[j]);
                }
            }
            if (!conflicts.empty()) {
                conflicts.push_back(appointments[i]);
                all_conflicts.push_back(conflicts);
            }
        }
        return all_conflicts;
    }
};

void Test_Conflicting_Appointments() {
    Solution solution;
    vector<Interval> appointments = {
        Interval(1, 5), Interval(3, 7), Interval(2, 6),
        Interval(10, 15), Interval(5, 6), Interval(4, 100)
    };
    vector<vector<Interval>> conflicts1 = solution.Find_Conflicts_Interval_Tree(appointments);
    vector<vector<Interval>> conflicts2 = solution.Find_Conflicts_Brute(appointments);
    cout << "Conflicts (Interval Tree): " << conflicts1.size() << " groups" << endl;
    cout << "Conflicts (Brute Force): " << conflicts2.size() << " groups" << endl;
}

int main() {
    Test_Conflicting_Appointments();
    return 0;
}
