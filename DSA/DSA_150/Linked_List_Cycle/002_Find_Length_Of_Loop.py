"""
Problem: Find Length of Loop in Linked List
URL: https://www.geeksforgeeks.org/find-length-of-loop-in-linked-list/

Problem Statement:
Given a linked list, find the length of the loop in the linked list. If there is no loop, return 0.

Sample Input/Output:
Input: 1 -> 2 -> 3 -> 4 -> 5 -> 2 (cycle back to node 2)
Output: 4
Explanation: The loop is 2 -> 3 -> 4 -> 5 -> 2, which has length 4.

Input: 1 -> 2 -> 3 -> 4 -> 5 -> NULL
Output: 0
Explanation: There is no loop in the linked list.
"""

from typing import Optional

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def Count_Loop_Length_Hash_Set(self, head: Optional[ListNode]) -> int:
        """
        Hash Set - Store nodes with positions
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        visited = {}
        current = head
        position = 0
        
        while current:
            if current in visited:
                return position - visited[current]
            
            visited[current] = position
            current = current.next
            position += 1
        
        return 0
    
    def Count_Loop_Length_Floyd_Optimal(self, head: Optional[ListNode]) -> int:
        """
        Floyd's Algorithm Optimal - Detect cycle then count length
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head or not head.next:
            return 0
        
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                return self.Count_Loop_From_Meeting_Point(slow)
        
        return 0
    
    def Count_Loop_From_Meeting_Point(self, meeting_point: ListNode) -> int:
        """
        Count Loop From Meeting Point - Count nodes in cycle
        """
        current = meeting_point
        length = 1
        
        while current.next != meeting_point:
            current = current.next
            length += 1
        
        return length
    
    def Count_Loop_Length_Modified_Floyd(self, head: Optional[ListNode]) -> int:
        """
        Modified Floyd - Count while detecting
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head:
            return 0
        
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                length = 0
                current = slow
                
                while True:
                    current = current.next
                    length += 1
                    if current == slow:
                        break
                
                return length
        
        return 0
    
    def Count_Loop_Length_Two_Pointers(self, head: Optional[ListNode]) -> int:
        """
        Two Pointers - Use two pointers after cycle detection
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head:
            return 0
        
        def Has_Cycle_And_Get_Meeting_Point() -> Optional[ListNode]:
            slow = fast = head
            
            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next
                
                if slow == fast:
                    return slow
            
            return None
        
        meeting_point = Has_Cycle_And_Get_Meeting_Point()
        
        if not meeting_point:
            return 0
        
        pointer1 = meeting_point
        pointer2 = meeting_point.next
        length = 1
        
        while pointer1 != pointer2:
            pointer2 = pointer2.next
            length += 1
        
        return length
    
    def Count_Loop_Length_Mark_And_Count(self, head: Optional[ListNode]) -> int:
        """
        Mark and Count - Mark starting point and count
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head:
            return 0
        
        MARKER = "VISITED"
        current = head
        position = 0
        
        while current:
            if hasattr(current, 'marker') and current.marker == MARKER:
                length = 0
                temp = current
                
                while True:
                    temp = temp.next
                    length += 1
                    if temp == current:
                        break
                
                return length
            
            current.marker = MARKER
            current = current.next
            position += 1
        
        return 0
    
    def Count_Loop_Length_Distance_Method(self, head: Optional[ListNode]) -> int:
        """
        Distance Method - Calculate distances to find loop
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head:
            return 0
        
        slow = fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                slow = head
                
                while slow != fast:
                    slow = slow.next
                    fast = fast.next
                
                loop_start = slow
                length = 1
                current = loop_start.next
                
                while current != loop_start:
                    current = current.next
                    length += 1
                
                return length
        
        return 0

def Create_Cyclic_List(values, pos):
    if not values:
        return None
    
    nodes = []
    for val in values:
        nodes.append(ListNode(val))
    
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    
    if pos >= 0 and pos < len(nodes):
        nodes[-1].next = nodes[pos]
    
    return nodes[0]

def Test_Count_Loop_Length():
    solution = Solution()
    
    test_cases = [
        ([1,2,3,4,5], 1, 4),
        ([1,2,3,4,5], 0, 5),
        ([1,2,3,4,5], -1, 0),
        ([1,2], 0, 2),
        ([1], -1, 0),
        ([1,2,3,4,5,6], 2, 4)
    ]
    
    methods = [
        ("Hash Set", solution.Count_Loop_Length_Hash_Set),
        ("Floyd Optimal", solution.Count_Loop_Length_Floyd_Optimal),
        ("Modified Floyd", solution.Count_Loop_Length_Modified_Floyd),
        ("Two Pointers", solution.Count_Loop_Length_Two_Pointers),
        ("Distance Method", solution.Count_Loop_Length_Distance_Method)
    ]
    
    for values, pos, expected in test_cases:
        print(f"Values: {values}, Pos: {pos}")
        print(f"Expected: {expected}")
        
        for method_name, method in methods:
            head = Create_Cyclic_List(values, pos)
            try:
                result = method(head)
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Count_Loop_Length()
