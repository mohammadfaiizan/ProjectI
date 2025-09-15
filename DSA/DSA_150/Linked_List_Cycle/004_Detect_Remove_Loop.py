"""
Problem: Detect and Remove Loop in a Linked List
URL: https://www.geeksforgeeks.org/detect-and-remove-loop-in-a-linked-list/

Problem Statement:
Given a linked list, check if it has a loop. If a loop is present, remove the loop and return the head of the modified list.

Sample Input/Output:
Input: 1 -> 2 -> 3 -> 4 -> 5 -> 2 (cycle back to node 2)
Output: 1 -> 2 -> 3 -> 4 -> 5 -> NULL
Explanation: The loop is removed by breaking the connection from 5 to 2.

Input: 1 -> 2 -> 3 -> 4 -> 5 -> NULL
Output: 1 -> 2 -> 3 -> 4 -> 5 -> NULL
Explanation: No loop present, list remains unchanged.
"""

from typing import Optional

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def Remove_Loop_Hash_Set(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Hash Set - Track visited nodes and remove loop
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not head:
            return head
        
        visited = set()
        current = head
        prev = None
        
        while current:
            if current in visited:
                prev.next = None
                break
            
            visited.add(current)
            prev = current
            current = current.next
        
        return head
    
    def Remove_Loop_Floyd_Optimal(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Floyd's Algorithm Optimal - Detect then remove loop
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head or not head.next:
            return head
        
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                self.Remove_Loop_From_Meeting_Point(head, slow)
                break
        
        return head
    
    def Remove_Loop_From_Meeting_Point(self, head: ListNode, meeting_point: ListNode) -> None:
        """
        Remove Loop From Meeting Point - Find start and remove
        """
        if head == meeting_point:
            while meeting_point.next != head:
                meeting_point = meeting_point.next
            meeting_point.next = None
        else:
            ptr1 = head
            ptr2 = meeting_point
            
            while ptr1.next != ptr2.next:
                ptr1 = ptr1.next
                ptr2 = ptr2.next
            
            ptr2.next = None
    
    def Remove_Loop_Count_Method(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Count Method - Count loop length then remove
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head:
            return head
        
        def Get_Loop_Length() -> int:
            slow = fast = head
            
            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next
                
                if slow == fast:
                    length = 1
                    current = slow
                    
                    while current.next != slow:
                        current = current.next
                        length += 1
                    
                    return length
            
            return 0
        
        loop_length = Get_Loop_Length()
        
        if loop_length == 0:
            return head
        
        ptr1 = head
        ptr2 = head
        
        for _ in range(loop_length):
            ptr2 = ptr2.next
        
        while ptr1.next != ptr2.next:
            ptr1 = ptr1.next
            ptr2 = ptr2.next
        
        ptr2.next = None
        return head
    
    def Remove_Loop_Mark_Nodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Mark Nodes - Mark visited and remove when found
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head:
            return head
        
        MARKER = "VISITED"
        current = head
        prev = None
        
        while current:
            if hasattr(current, 'marker') and current.marker == MARKER:
                prev.next = None
                break
            
            current.marker = MARKER
            prev = current
            current = current.next
        
        return head
    
    def Remove_Loop_Distance_Based(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Distance Based - Calculate distances and remove
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head:
            return head
        
        slow = fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                break
        else:
            return head
        
        slow = head
        
        if slow == fast:
            while fast.next != slow:
                fast = fast.next
        else:
            while slow.next != fast.next:
                slow = slow.next
                fast = fast.next
        
        fast.next = None
        return head
    
    def Remove_Loop_Two_Pointer_Approach(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Two Pointer Approach - Alternative implementation
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head or not head.next:
            return head
        
        def Find_Loop_Start() -> Optional[ListNode]:
            slow = fast = head
            
            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next
                
                if slow == fast:
                    slow = head
                    
                    while slow != fast:
                        slow = slow.next
                        fast = fast.next
                    
                    return slow
            
            return None
        
        loop_start = Find_Loop_Start()
        
        if loop_start:
            current = loop_start
            
            while current.next != loop_start:
                current = current.next
            
            current.next = None
        
        return head

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

def List_To_Array(head, max_length=20):
    result = []
    current = head
    count = 0
    
    while current and count < max_length:
        result.append(current.val)
        current = current.next
        count += 1
    
    if current:
        result.append("...")
    
    return result

def Test_Remove_Loop():
    solution = Solution()
    
    test_cases = [
        ([1,2,3,4,5], 1),
        ([1,2,3,4,5], 0),
        ([1,2,3,4,5], -1),
        ([1,2], 0),
        ([1], -1),
        ([3,2,0,-4], 1)
    ]
    
    methods = [
        ("Hash Set", solution.Remove_Loop_Hash_Set),
        ("Floyd Optimal", solution.Remove_Loop_Floyd_Optimal),
        ("Count Method", solution.Remove_Loop_Count_Method),
        ("Distance Based", solution.Remove_Loop_Distance_Based),
        ("Two Pointer Approach", solution.Remove_Loop_Two_Pointer_Approach)
    ]
    
    for values, pos in test_cases:
        print(f"Values: {values}, Pos: {pos}")
        has_cycle = pos >= 0
        print(f"Has Cycle: {has_cycle}")
        
        for method_name, method in methods:
            head = Create_Cyclic_List(values, pos)
            try:
                result_head = method(head)
                result = List_To_Array(result_head)
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Remove_Loop()
