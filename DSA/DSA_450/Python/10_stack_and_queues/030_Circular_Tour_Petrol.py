"""
Problem: First Circular Tour Visiting All Petrol Pumps
URL: https://practice.geeksforgeeks.org/problems/circular-tour/1

Problem Statement:
Find the first petrol pump from where a circular tour can be completed visiting all pumps.

Sample Input/Output:
Input: petrol=[4,6,7,4], distance=[6,5,3,5]
Output: 1
"""


class Solution:
    def Circular_Tour_Brute_Force(self, petrol, distance):
        """
        Find starting point using brute force.
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        n = len(petrol)
        for start in range(n):
            current_petrol = 0
            i = start
            count = 0
            while count < n:
                current_petrol += petrol[i] - distance[i]
                if current_petrol < 0:
                    break
                i = (i + 1) % n
                count += 1
            if count == n and current_petrol >= 0:
                return start
        return -1

    def Circular_Tour_Deficit_Tracking(self, petrol, distance):
        """
        Find starting point using deficit tracking.
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(petrol)
        start = 0
        deficit = 0
        balance = 0
        
        for i in range(n):
            balance += petrol[i] - distance[i]
            if balance < 0:
                deficit += balance
                start = i + 1
                balance = 0
        
        if deficit + balance >= 0:
            return start
        return -1


def Test_Circular_Tour_Brute_Force():
    solution = Solution()
    
    petrol1 = [4, 6, 7, 4]
    distance1 = [6, 5, 3, 5]
    result1 = solution.Circular_Tour_Brute_Force(petrol1, distance1)
    print(f"Brute Force - Petrol: [4,6,7,4], Distance: [6,5,3,5] -> Start: {result1}")

    petrol2 = [6, 7, 4, 10]
    distance2 = [5, 6, 7, 6]
    result2 = solution.Circular_Tour_Brute_Force(petrol2, distance2)
    print(f"Brute Force - Petrol: [6,7,4,10], Distance: [5,6,7,6] -> Start: {result2}")


def Test_Circular_Tour_Deficit_Tracking():
    solution = Solution()
    
    petrol1 = [4, 6, 7, 4]
    distance1 = [6, 5, 3, 5]
    result1 = solution.Circular_Tour_Deficit_Tracking(petrol1, distance1)
    print(f"Deficit Tracking - Petrol: [4,6,7,4], Distance: [6,5,3,5] -> Start: {result1}")

    petrol2 = [6, 7, 4, 10]
    distance2 = [5, 6, 7, 6]
    result2 = solution.Circular_Tour_Deficit_Tracking(petrol2, distance2)
    print(f"Deficit Tracking - Petrol: [6,7,4,10], Distance: [5,6,7,6] -> Start: {result2}")

    petrol3 = [1, 2]
    distance3 = [2, 1]
    result3 = solution.Circular_Tour_Deficit_Tracking(petrol3, distance3)
    print(f"Deficit Tracking - Petrol: [1,2], Distance: [2,1] -> Start: {result3}")


if __name__ == "__main__":
    Test_Circular_Tour_Brute_Force()
    Test_Circular_Tour_Deficit_Tracking()
