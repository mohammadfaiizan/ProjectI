"""
Linked List Basics - Fundamental LinkedList Data Structures and Operations
This module implements all types of linked lists with comprehensive operations.
"""

from typing import Optional, Any, List

class ListNode:
    """Basic node class for singly linked list"""
    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val = val
        self.next = next
    
    def __repr__(self):
        return f"ListNode({self.val})"

class DoublyListNode:
    """Node class for doubly linked list"""
    def __init__(self, val: int = 0, prev: Optional['DoublyListNode'] = None, 
                 next: Optional['DoublyListNode'] = None):
        self.val = val
        self.prev = prev
        self.next = next
    
    def __repr__(self):
        return f"DoublyListNode({self.val})"

class SinglyLinkedList:
    """
    Comprehensive Singly Linked List Implementation
    """
    
    def __init__(self):
        """Initialize empty singly linked list"""
        self.head: Optional[ListNode] = None
        self.size = 0
    
    def insert_at_beginning(self, val: int) -> None:
        """
        Insert node at the beginning
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Args:
            val: Value to insert
        """
        new_node = ListNode(val)
        new_node.next = self.head
        self.head = new_node
        self.size += 1
    
    def insert_at_end(self, val: int) -> None:
        """
        Insert node at the end
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            val: Value to insert
        """
        new_node = ListNode(val)
        
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        
        self.size += 1
    
    def insert_at_position(self, pos: int, val: int) -> bool:
        """
        Insert node at specific position (0-indexed)
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            pos: Position to insert at
            val: Value to insert
        
        Returns:
            bool: True if insertion successful
        """
        if pos < 0 or pos > self.size:
            return False
        
        if pos == 0:
            self.insert_at_beginning(val)
            return True
        
        new_node = ListNode(val)
        current = self.head
        
        for _ in range(pos - 1):
            current = current.next
        
        new_node.next = current.next
        current.next = new_node
        self.size += 1
        return True
    
    def delete_by_value(self, val: int) -> bool:
        """
        Delete first occurrence of value
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            val: Value to delete
        
        Returns:
            bool: True if deletion successful
        """
        if not self.head:
            return False
        
        # Delete head
        if self.head.val == val:
            self.head = self.head.next
            self.size -= 1
            return True
        
        # Delete non-head node
        current = self.head
        while current.next:
            if current.next.val == val:
                current.next = current.next.next
                self.size -= 1
                return True
            current = current.next
        
        return False
    
    def delete_by_position(self, pos: int) -> bool:
        """
        Delete node at specific position
        
        Args:
            pos: Position to delete (0-indexed)
        
        Returns:
            bool: True if deletion successful
        """
        if pos < 0 or pos >= self.size or not self.head:
            return False
        
        if pos == 0:
            self.head = self.head.next
            self.size -= 1
            return True
        
        current = self.head
        for _ in range(pos - 1):
            current = current.next
        
        current.next = current.next.next
        self.size -= 1
        return True
    
    def search(self, val: int) -> int:
        """
        Search for value and return position
        
        Args:
            val: Value to search for
        
        Returns:
            int: Position of value (-1 if not found)
        """
        current = self.head
        position = 0
        
        while current:
            if current.val == val:
                return position
            current = current.next
            position += 1
        
        return -1
    
    def get_node_at_position(self, pos: int) -> Optional[ListNode]:
        """
        Get node at specific position
        
        Args:
            pos: Position to get (0-indexed)
        
        Returns:
            Node at position or None
        """
        if pos < 0 or pos >= self.size:
            return None
        
        current = self.head
        for _ in range(pos):
            current = current.next
        
        return current
    
    def display(self) -> List[int]:
        """Display all values in the list"""
        values = []
        current = self.head
        
        while current:
            values.append(current.val)
            current = current.next
        
        return values
    
    def length(self) -> int:
        """Get length of list"""
        return self.size
    
    def is_empty(self) -> bool:
        """Check if list is empty"""
        return self.head is None
    
    def clear(self) -> None:
        """Clear all nodes"""
        self.head = None
        self.size = 0
    
    def reverse(self) -> None:
        """
        Reverse the linked list in place
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        prev = None
        current = self.head
        
        while current:
            next_temp = current.next
            current.next = prev
            prev = current
            current = next_temp
        
        self.head = prev
    
    def get_middle_node(self) -> Optional[ListNode]:
        """
        Get middle node using slow-fast pointer technique
        
        Returns:
            Middle node (for even length, returns second middle)
        """
        if not self.head:
            return None
        
        slow = fast = self.head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        return slow
    
    def has_cycle(self) -> bool:
        """
        Check if list has cycle using Floyd's algorithm
        
        Returns:
            bool: True if cycle exists
        """
        if not self.head:
            return False
        
        slow = fast = self.head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                return True
        
        return False
    
    def print_structure(self) -> str:
        """Print list structure"""
        if not self.head:
            return "Empty List"
        
        values = self.display()
        return " -> ".join(map(str, values)) + " -> NULL"

class DoublyLinkedList:
    """
    Comprehensive Doubly Linked List Implementation
    """
    
    def __init__(self):
        """Initialize empty doubly linked list"""
        self.head: Optional[DoublyListNode] = None
        self.tail: Optional[DoublyListNode] = None
        self.size = 0
    
    def insert_at_beginning(self, val: int) -> None:
        """
        Insert node at beginning
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Args:
            val: Value to insert
        """
        new_node = DoublyListNode(val)
        
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        
        self.size += 1
    
    def insert_at_end(self, val: int) -> None:
        """
        Insert node at end
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Args:
            val: Value to insert
        """
        new_node = DoublyListNode(val)
        
        if not self.tail:
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        
        self.size += 1
    
    def insert_at_position(self, pos: int, val: int) -> bool:
        """
        Insert node at specific position
        
        Args:
            pos: Position to insert at (0-indexed)
            val: Value to insert
        
        Returns:
            bool: True if insertion successful
        """
        if pos < 0 or pos > self.size:
            return False
        
        if pos == 0:
            self.insert_at_beginning(val)
            return True
        
        if pos == self.size:
            self.insert_at_end(val)
            return True
        
        new_node = DoublyListNode(val)
        
        # Optimize: choose direction based on position
        if pos <= self.size // 2:
            # Traverse from head
            current = self.head
            for _ in range(pos):
                current = current.next
        else:
            # Traverse from tail
            current = self.tail
            for _ in range(self.size - pos - 1):
                current = current.prev
        
        # Insert before current
        new_node.next = current
        new_node.prev = current.prev
        current.prev.next = new_node
        current.prev = new_node
        
        self.size += 1
        return True
    
    def delete_by_value(self, val: int) -> bool:
        """
        Delete first occurrence of value
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            val: Value to delete
        
        Returns:
            bool: True if deletion successful
        """
        current = self.head
        
        while current:
            if current.val == val:
                # Update previous node
                if current.prev:
                    current.prev.next = current.next
                else:
                    self.head = current.next
                
                # Update next node
                if current.next:
                    current.next.prev = current.prev
                else:
                    self.tail = current.prev
                
                self.size -= 1
                return True
            
            current = current.next
        
        return False
    
    def delete_by_position(self, pos: int) -> bool:
        """
        Delete node at specific position
        
        Args:
            pos: Position to delete (0-indexed)
        
        Returns:
            bool: True if deletion successful
        """
        if pos < 0 or pos >= self.size:
            return False
        
        # Optimize: choose direction based on position
        if pos <= self.size // 2:
            # Traverse from head
            current = self.head
            for _ in range(pos):
                current = current.next
        else:
            # Traverse from tail
            current = self.tail
            for _ in range(self.size - pos - 1):
                current = current.prev
        
        # Delete current node
        if current.prev:
            current.prev.next = current.next
        else:
            self.head = current.next
        
        if current.next:
            current.next.prev = current.prev
        else:
            self.tail = current.prev
        
        self.size -= 1
        return True
    
    def search(self, val: int) -> int:
        """
        Search for value and return position
        
        Args:
            val: Value to search for
        
        Returns:
            int: Position of value (-1 if not found)
        """
        current = self.head
        position = 0
        
        while current:
            if current.val == val:
                return position
            current = current.next
            position += 1
        
        return -1
    
    def search_optimized(self, val: int) -> int:
        """
        Optimized search using both directions
        
        Args:
            val: Value to search for
        
        Returns:
            int: Position of value (-1 if not found)
        """
        if not self.head:
            return -1
        
        left = self.head
        right = self.tail
        left_pos = 0
        right_pos = self.size - 1
        
        while left and right and left_pos <= right_pos:
            if left.val == val:
                return left_pos
            if right.val == val:
                return right_pos
            
            left = left.next
            right = right.prev
            left_pos += 1
            right_pos -= 1
        
        return -1
    
    def display_forward(self) -> List[int]:
        """Display values forward (head to tail)"""
        values = []
        current = self.head
        
        while current:
            values.append(current.val)
            current = current.next
        
        return values
    
    def display_backward(self) -> List[int]:
        """Display values backward (tail to head)"""
        values = []
        current = self.tail
        
        while current:
            values.append(current.val)
            current = current.prev
        
        return values
    
    def length(self) -> int:
        """Get length of list"""
        return self.size
    
    def is_empty(self) -> bool:
        """Check if list is empty"""
        return self.head is None
    
    def clear(self) -> None:
        """Clear all nodes"""
        self.head = self.tail = None
        self.size = 0
    
    def reverse(self) -> None:
        """
        Reverse the doubly linked list in place
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        current = self.head
        
        while current:
            # Swap next and prev
            current.next, current.prev = current.prev, current.next
            current = current.prev  # Move to what was next
        
        # Swap head and tail
        self.head, self.tail = self.tail, self.head
    
    def print_structure_forward(self) -> str:
        """Print forward structure"""
        if not self.head:
            return "Empty List"
        
        values = self.display_forward()
        return "NULL <-> " + " <-> ".join(map(str, values)) + " <-> NULL"

class CircularLinkedList:
    """
    Circular Singly Linked List Implementation
    """
    
    def __init__(self):
        """Initialize empty circular linked list"""
        self.head: Optional[ListNode] = None
        self.tail: Optional[ListNode] = None
        self.size = 0
    
    def insert_at_beginning(self, val: int) -> None:
        """
        Insert node at the beginning
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Args:
            val: Value to insert
        """
        new_node = ListNode(val)
        
        if not self.head:
            self.head = self.tail = new_node
            new_node.next = new_node
        else:
            new_node.next = self.head
            self.tail.next = new_node
            self.head = new_node
        
        self.size += 1
    
    def insert_at_end(self, val: int) -> None:
        """
        Insert node at the end
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Args:
            val: Value to insert
        """
        new_node = ListNode(val)
        
        if not self.head:
            self.head = self.tail = new_node
            new_node.next = new_node
        else:
            new_node.next = self.head
            self.tail.next = new_node
            self.tail = new_node
        
        self.size += 1
    
    def display(self) -> List[int]:
        """Display all values in the list"""
        if not self.head:
            return []
        
        values = []
        current = self.head
        
        while True:
            values.append(current.val)
            current = current.next
            if current == self.head:
                break
        
        return values
    
    def length(self) -> int:
        """Get length of list"""
        return self.size
    
    def print_structure(self) -> str:
        """Print circular structure"""
        if not self.head:
            return "Empty Circular List"
        
        values = self.display()
        return " -> ".join(map(str, values)) + f" -> {values[0]} (circular)"

class CircularDoublyLinkedList:
    """
    Circular Doubly Linked List Implementation
    """
    
    def __init__(self):
        """Initialize empty circular doubly linked list"""
        self.head: Optional[DoublyListNode] = None
        self.size = 0
    
    def insert_at_beginning(self, val: int) -> None:
        """
        Insert node at beginning
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Args:
            val: Value to insert
        """
        new_node = DoublyListNode(val)
        
        if not self.head:
            self.head = new_node
            new_node.next = new_node.prev = new_node
        else:
            tail = self.head.prev
            
            new_node.next = self.head
            new_node.prev = tail
            self.head.prev = new_node
            tail.next = new_node
            self.head = new_node
        
        self.size += 1
    
    def insert_at_end(self, val: int) -> None:
        """
        Insert node at end
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Args:
            val: Value to insert
        """
        if not self.head:
            self.insert_at_beginning(val)
            return
        
        new_node = DoublyListNode(val)
        tail = self.head.prev
        
        new_node.next = self.head
        new_node.prev = tail
        tail.next = new_node
        self.head.prev = new_node
        
        self.size += 1
    
    def display_forward(self) -> List[int]:
        """Display values forward"""
        if not self.head:
            return []
        
        values = []
        current = self.head
        
        while True:
            values.append(current.val)
            current = current.next
            if current == self.head:
                break
        
        return values
    
    def length(self) -> int:
        """Get length of list"""
        return self.size

class LinkedListComparator:
    """
    Utility class for comparing and analyzing linked lists
    """
    
    @staticmethod
    def are_equal(list1: SinglyLinkedList, list2: SinglyLinkedList) -> bool:
        """
        Check if two singly linked lists are equal
        
        Args:
            list1: First linked list
            list2: Second linked list
        
        Returns:
            bool: True if lists are equal
        """
        if list1.length() != list2.length():
            return False
        
        current1 = list1.head
        current2 = list2.head
        
        while current1 and current2:
            if current1.val != current2.val:
                return False
            current1 = current1.next
            current2 = current2.next
        
        return current1 is None and current2 is None
    
    @staticmethod
    def merge_sorted_lists(list1: SinglyLinkedList, list2: SinglyLinkedList) -> SinglyLinkedList:
        """
        Merge two sorted singly linked lists
        
        Args:
            list1: First sorted linked list
            list2: Second sorted linked list
        
        Returns:
            Merged sorted linked list
        """
        result = SinglyLinkedList()
        
        p1 = list1.head
        p2 = list2.head
        
        while p1 and p2:
            if p1.val <= p2.val:
                result.insert_at_end(p1.val)
                p1 = p1.next
            else:
                result.insert_at_end(p2.val)
                p2 = p2.next
        
        # Add remaining nodes
        while p1:
            result.insert_at_end(p1.val)
            p1 = p1.next
        
        while p2:
            result.insert_at_end(p2.val)
            p2 = p2.next
        
        return result
    
    @staticmethod
    def find_intersection(list1: SinglyLinkedList, list2: SinglyLinkedList) -> Optional[ListNode]:
        """
        Find intersection point of two singly linked lists
        
        Args:
            list1: First linked list
            list2: Second linked list
        
        Returns:
            Intersection node or None
        """
        len1 = list1.length()
        len2 = list2.length()
        
        ptr1 = list1.head
        ptr2 = list2.head
        
        # Align starting points
        if len1 > len2:
            for _ in range(len1 - len2):
                ptr1 = ptr1.next
        else:
            for _ in range(len2 - len1):
                ptr2 = ptr2.next
        
        # Find intersection
        while ptr1 and ptr2:
            if ptr1 == ptr2:
                return ptr1
            ptr1 = ptr1.next
            ptr2 = ptr2.next
        
        return None

class LinkedListUtilities:
    """
    Utility functions for linked list operations
    """
    
    @staticmethod
    def create_from_array(arr: List[int], list_type: str = "singly") -> Any:
        """
        Create linked list from array
        
        Args:
            arr: Array of values
            list_type: Type of list ("singly", "doubly", "circular")
        
        Returns:
            Created linked list
        """
        if list_type == "singly":
            ll = SinglyLinkedList()
        elif list_type == "doubly":
            ll = DoublyLinkedList()
        elif list_type == "circular":
            ll = CircularLinkedList()
        elif list_type == "circular_doubly":
            ll = CircularDoublyLinkedList()
        else:
            raise ValueError("Invalid list type")
        
        for val in arr:
            ll.insert_at_end(val)
        
        return ll
    
    @staticmethod
    def remove_duplicates(ll: SinglyLinkedList) -> None:
        """
        Remove duplicates from singly linked list
        
        Args:
            ll: Singly linked list
        """
        if not ll.head:
            return
        
        seen = set()
        current = ll.head
        prev = None
        
        while current:
            if current.val in seen:
                prev.next = current.next
                ll.size -= 1
            else:
                seen.add(current.val)
                prev = current
            current = current.next
    
    @staticmethod
    def sort_list(ll: SinglyLinkedList) -> None:
        """
        Sort singly linked list using merge sort
        
        Args:
            ll: Singly linked list to sort
        """
        if ll.size <= 1:
            return
        
        ll.head = LinkedListUtilities._merge_sort(ll.head)
    
    @staticmethod
    def _merge_sort(head: ListNode) -> ListNode:
        """Helper method for merge sort"""
        if not head or not head.next:
            return head
        
        # Split the list
        mid = LinkedListUtilities._get_middle(head)
        mid_next = mid.next
        mid.next = None
        
        # Recursively sort both halves
        left = LinkedListUtilities._merge_sort(head)
        right = LinkedListUtilities._merge_sort(mid_next)
        
        # Merge sorted halves
        return LinkedListUtilities._merge(left, right)
    
    @staticmethod
    def _get_middle(head: ListNode) -> ListNode:
        """Get middle node using slow-fast pointer"""
        slow = fast = head
        prev = None
        
        while fast and fast.next:
            prev = slow
            slow = slow.next
            fast = fast.next.next
        
        return prev
    
    @staticmethod
    def _merge(left: ListNode, right: ListNode) -> ListNode:
        """Merge two sorted linked lists"""
        dummy = ListNode(0)
        current = dummy
        
        while left and right:
            if left.val <= right.val:
                current.next = left
                left = left.next
            else:
                current.next = right
                right = right.next
            current = current.next
        
        # Append remaining nodes
        current.next = left or right
        
        return dummy.next

# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Linked List Basics Demo ===\n")
    
    # Example 1: Singly Linked List
    print("1. Singly Linked List Operations:")
    
    sll = SinglyLinkedList()
    
    # Insert operations
    print("Inserting elements:")
    sll.insert_at_beginning(2)
    sll.insert_at_beginning(1)
    sll.insert_at_end(3)
    sll.insert_at_end(4)
    sll.insert_at_position(2, 10)
    
    print(f"List: {sll.display()}")
    print(f"Structure: {sll.print_structure()}")
    print(f"Length: {sll.length()}")
    
    # Search operations
    print(f"Search 10: position {sll.search(10)}")
    print(f"Search 99: position {sll.search(99)}")
    
    # Delete operations
    print(f"Delete value 10: {sll.delete_by_value(10)}")
    print(f"Delete position 0: {sll.delete_by_position(0)}")
    print(f"After deletions: {sll.display()}")
    
    # Other operations
    middle = sll.get_middle_node()
    print(f"Middle node: {middle.val if middle else None}")
    
    # Reverse list
    sll.reverse()
    print(f"After reverse: {sll.display()}")
    print()
    
    # Example 2: Doubly Linked List
    print("2. Doubly Linked List Operations:")
    
    dll = DoublyLinkedList()
    
    # Insert operations
    print("Inserting elements:")
    dll.insert_at_beginning(2)
    dll.insert_at_beginning(1)
    dll.insert_at_end(3)
    dll.insert_at_end(4)
    dll.insert_at_position(2, 10)
    
    print(f"Forward: {dll.display_forward()}")
    print(f"Backward: {dll.display_backward()}")
    print(f"Structure: {dll.print_structure_forward()}")
    print(f"Length: {dll.length()}")
    
    # Search operations
    print(f"Normal search 10: position {dll.search(10)}")
    print(f"Optimized search 10: position {dll.search_optimized(10)}")
    
    # Delete operations
    print(f"Delete value 10: {dll.delete_by_value(10)}")
    print(f"Delete position 2: {dll.delete_by_position(2)}")
    print(f"After deletions: {dll.display_forward()}")
    
    # Reverse list
    dll.reverse()
    print(f"After reverse: {dll.display_forward()}")
    print()
    
    # Example 3: Circular Linked List
    print("3. Circular Linked List Operations:")
    
    cll = CircularLinkedList()
    
    # Insert operations
    cll.insert_at_beginning(3)
    cll.insert_at_beginning(2)
    cll.insert_at_beginning(1)
    cll.insert_at_end(4)
    cll.insert_at_end(5)
    
    print(f"List: {cll.display()}")
    print(f"Structure: {cll.print_structure()}")
    print(f"Length: {cll.length()}")
    print()
    
    # Example 4: Circular Doubly Linked List
    print("4. Circular Doubly Linked List Operations:")
    
    cdll = CircularDoublyLinkedList()
    
    # Insert operations
    cdll.insert_at_beginning(3)
    cdll.insert_at_beginning(2)
    cdll.insert_at_beginning(1)
    cdll.insert_at_end(4)
    cdll.insert_at_end(5)
    
    print(f"Forward: {cdll.display_forward()}")
    print(f"Length: {cdll.length()}")
    print()
    
    # Example 5: Utility Functions
    print("5. Utility Functions:")
    
    # Create from array
    arr = [1, 3, 2, 4, 2, 5]
    sll_from_array = LinkedListUtilities.create_from_array(arr, "singly")
    print(f"Created from array {arr}: {sll_from_array.display()}")
    
    # Remove duplicates
    LinkedListUtilities.remove_duplicates(sll_from_array)
    print(f"After removing duplicates: {sll_from_array.display()}")
    
    # Sort list
    LinkedListUtilities.sort_list(sll_from_array)
    print(f"After sorting: {sll_from_array.display()}")
    
    # Merge sorted lists
    list1 = LinkedListUtilities.create_from_array([1, 3, 5], "singly")
    list2 = LinkedListUtilities.create_from_array([2, 4, 6], "singly")
    merged = LinkedListComparator.merge_sorted_lists(list1, list2)
    print(f"Merged sorted lists: {merged.display()}")
    
    # Compare lists
    list3 = LinkedListUtilities.create_from_array([1, 2, 3, 4, 5, 6], "singly")
    are_equal = LinkedListComparator.are_equal(merged, list3)
    print(f"Are merged and list3 equal: {are_equal}")
    print()
    
    # Example 6: Performance Analysis
    print("6. Performance Analysis:")
    
    # Large list operations
    large_array = list(range(1, 1001))
    
    # Singly linked list
    large_sll = LinkedListUtilities.create_from_array(large_array, "singly")
    print(f"Large singly linked list length: {large_sll.length()}")
    
    # Search performance
    search_val = 500
    pos_sll = large_sll.search(search_val)
    print(f"Search {search_val} in singly linked list: position {pos_sll}")
    
    # Doubly linked list
    large_dll = LinkedListUtilities.create_from_array(large_array, "doubly")
    print(f"Large doubly linked list length: {large_dll.length()}")
    
    # Optimized search
    pos_dll_normal = large_dll.search(search_val)
    pos_dll_optimized = large_dll.search_optimized(search_val)
    print(f"Search {search_val} in doubly linked list (normal): position {pos_dll_normal}")
    print(f"Search {search_val} in doubly linked list (optimized): position {pos_dll_optimized}")
    
    # Insert/delete performance comparison
    print("\nInsert/Delete Performance:")
    
    # Singly linked list - insert at beginning vs end
    test_sll = SinglyLinkedList()
    
    # Insert at beginning (O(1))
    for i in range(10):
        test_sll.insert_at_beginning(i)
    print(f"SLL after 10 inserts at beginning: {test_sll.display()}")
    
    # Insert at end (O(n))
    for i in range(10, 15):
        test_sll.insert_at_end(i)
    print(f"SLL after 5 inserts at end: {test_sll.display()}")
    
    # Doubly linked list - both operations are O(1)
    test_dll = DoublyLinkedList()
    
    for i in range(10):
        test_dll.insert_at_beginning(i)
    print(f"DLL after 10 inserts at beginning: {test_dll.display_forward()}")
    
    for i in range(10, 15):
        test_dll.insert_at_end(i)
    print(f"DLL after 5 inserts at end: {test_dll.display_forward()}")
    
    # Memory efficiency comparison
    print(f"\nMemory usage per node:")
    print(f"  Singly Linked List: 1 pointer (next)")
    print(f"  Doubly Linked List: 2 pointers (prev, next)")
    print(f"  Circular Lists: Same as above + circular connectivity")
    
    print("\n=== Demo Complete ===") 