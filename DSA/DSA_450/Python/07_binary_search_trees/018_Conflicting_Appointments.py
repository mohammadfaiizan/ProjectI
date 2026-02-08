"""
Problem: Given N Appointments, Find Conflicting Appointments
URL: https://www.geeksforgeeks.org/given-n-appointments-find-conflicting-appointments/

Problem Statement:
Given N appointments, find all conflicting appointments using interval tree.

Sample Input/Output:
Input: appointments = [(1,5), (3,7), (2,6), (10,15), (5,6), (4,100)]
Output: Conflicts: (1,5) conflicts with (3,7), (2,6)
Explanation: Overlapping intervals conflict with each other.
"""


class Interval:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class IntervalNode:
    def __init__(self, interval):
        self.interval = interval
        self.max_end = interval.end
        self.left = None
        self.right = None


def Print_Inorder(root):
    if root is None:
        return
    Print_Inorder(root.left)
    print(f"[{root.interval.start},{root.interval.end}]", end=" ")
    Print_Inorder(root.right)


class Solution:
    def Do_Overlap(self, i1, i2):
        return i1.start < i2.end and i2.start < i1.end

    def Insert_Interval_Tree(self, root, interval, conflicts):
        if root is None:
            node = IntervalNode(interval)
            return node
        if self.Do_Overlap(root.interval, interval):
            conflicts.append(root.interval)
        if interval.start < root.interval.start:
            root.left = self.Insert_Interval_Tree(root.left, interval, conflicts)
        else:
            root.right = self.Insert_Interval_Tree(root.right, interval, conflicts)
        root.max_end = max(root.max_end, interval.end)
        if root.left:
            root.max_end = max(root.max_end, root.left.max_end)
        if root.right:
            root.max_end = max(root.max_end, root.right.max_end)
        return root

    def Find_Conflicts_Interval_Tree(self, appointments):
        """
        Interval tree approach
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        all_conflicts = []
        root = None
        for app in appointments:
            conflicts = []
            root = self.Insert_Interval_Tree(root, app, conflicts)
            if conflicts:
                conflicts.append(app)
                all_conflicts.append(conflicts)
        return all_conflicts

    def Find_Conflicts_Brute(self, appointments):
        """
        Brute force approach
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        all_conflicts = []
        n = len(appointments)
        for i in range(n):
            conflicts = []
            for j in range(n):
                if i != j and self.Do_Overlap(appointments[i], appointments[j]):
                    conflicts.append(appointments[j])
            if conflicts:
                conflicts.append(appointments[i])
                all_conflicts.append(conflicts)
        return all_conflicts


def Test_Conflicting_Appointments():
    solution = Solution()
    appointments = [
        Interval(1, 5), Interval(3, 7), Interval(2, 6),
        Interval(10, 15), Interval(5, 6), Interval(4, 100)
    ]
    conflicts1 = solution.Find_Conflicts_Interval_Tree(appointments)
    conflicts2 = solution.Find_Conflicts_Brute(appointments)
    print("Conflicts (Interval Tree):", len(conflicts1), "groups")
    print("Conflicts (Brute Force):", len(conflicts2), "groups")


if __name__ == "__main__":
    Test_Conflicting_Appointments()
