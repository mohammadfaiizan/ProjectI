"""
Fast & Slow Pointer Techniques - Floyd's Tortoise & Hare Algorithm
This module implements comprehensive two-pointer techniques for linked list problems.
"""

from typing import Optional, List, Set, Tuple

class ListNode:
    """Basic node class for linked list"""
    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val = val
        self.next = next
    
    def __repr__(self):
        return f"ListNode({self.val})"

class FastSlowPointer:
    """
    Implementation of fast and slow pointer techniques
    """
    
    def __init__(self):
        """Initialize fast-slow pointer solver"""
        pass
    
    # ==================== CYCLE DETECTION ====================
    
    def has_cycle(self, head: Optional[ListNode]) -> bool:
        """
        Detect if linked list has a cycle using Floyd's algorithm
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            head: Head of linked list
        
        Returns:
            bool: True if cycle exists
        """
        if not head or not head.next:
            return False
        
        slow = fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                return True
        
        return False
    
    def detect_cycle_with_position(self, head: Optional[ListNode]) -> Tuple[bool, int]:
        """
        Detect cycle and return position where cycle starts (0-indexed)
        
        Args:
            head: Head of linked list
        
        Returns:
            Tuple of (has_cycle, cycle_start_position)
        """
        if not head or not head.next:
            return False, -1
        
        slow = fast = head
        
        # Phase 1: Detect if cycle exists
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                break
        else:
            return False, -1
        
        # Phase 2: Find cycle start position
        slow = head
        position = 0
        
        while slow != fast:
            slow = slow.next
            fast = fast.next
            position += 1
        
        return True, position
    
    def find_cycle_start_node(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Find the node where cycle starts
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            head: Head of linked list
        
        Returns:
            Node where cycle starts or None
        """
        if not head or not head.next:
            return None
        
        slow = fast = head
        
        # Phase 1: Detect cycle
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                break
        else:
            return None
        
        # Phase 2: Find cycle start
        slow = head
        while slow != fast:
            slow = slow.next
            fast = fast.next
        
        return slow
    
    def find_cycle_length(self, head: Optional[ListNode]) -> int:
        """
        Find length of cycle in linked list
        
        Args:
            head: Head of linked list
        
        Returns:
            Length of cycle (0 if no cycle)
        """
        if not head or not head.next:
            return 0
        
        slow = fast = head
        
        # Phase 1: Detect cycle
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                break
        else:
            return 0
        
        # Phase 2: Calculate cycle length
        current = slow
        length = 0
        
        while True:
            current = current.next
            length += 1
            if current == slow:
                break
        
        return length
    
    def has_cycle_with_hash_set(self, head: Optional[ListNode]) -> bool:
        """
        Detect cycle using hash set (alternative approach)
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        
        Args:
            head: Head of linked list
        
        Returns:
            bool: True if cycle exists
        """
        seen = set()
        current = head
        
        while current:
            if current in seen:
                return True
            seen.add(current)
            current = current.next
        
        return False
    
    def remove_cycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Remove cycle from linked list if it exists
        
        Args:
            head: Head of linked list
        
        Returns:
            Head of modified list
        """
        if not head or not head.next:
            return head
        
        slow = fast = head
        
        # Detect cycle
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                break
        else:
            return head  # No cycle
        
        # Find cycle start
        slow = head
        while slow.next != fast.next:
            slow = slow.next
            fast = fast.next
        
        # Remove cycle
        fast.next = None
        
        return head
    
    # ==================== FIND MIDDLE ELEMENT ====================
    
    def find_middle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Find middle node of linked list
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            head: Head of linked list
        
        Returns:
            Middle node (for even length, returns second middle)
        """
        if not head:
            return None
        
        slow = fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        return slow
    
    def find_middle_with_count(self, head: Optional[ListNode]) -> Tuple[Optional[ListNode], int]:
        """
        Find middle node and return total count
        
        Args:
            head: Head of linked list
        
        Returns:
            Tuple of (middle_node, total_count)
        """
        if not head:
            return None, 0
        
        slow = fast = head
        count = 0
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            count += 2
        
        # Adjust count for odd length
        if fast:
            count += 1
        
        return slow, count
    
    def find_middle_previous(self, head: Optional[ListNode]) -> Tuple[Optional[ListNode], Optional[ListNode]]:
        """
        Find middle node and its previous node
        
        Args:
            head: Head of linked list
        
        Returns:
            Tuple of (previous_node, middle_node)
        """
        if not head or not head.next:
            return None, head
        
        prev = None
        slow = fast = head
        
        while fast and fast.next:
            prev = slow
            slow = slow.next
            fast = fast.next.next
        
        return prev, slow
    
    def find_nth_from_middle(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        """
        Find nth node from middle (positive n = forward, negative n = backward)
        
        Args:
            head: Head of linked list
            n: Distance from middle (can be negative)
        
        Returns:
            Node at nth position from middle
        """
        middle = self.find_middle(head)
        if not middle:
            return None
        
        current = middle
        
        # Move forward
        if n > 0:
            for _ in range(n):
                if current and current.next:
                    current = current.next
                else:
                    return None
        
        # Move backward (requires traversal from head)
        elif n < 0:
            # Find position of middle
            middle_pos = 0
            temp = head
            while temp != middle:
                temp = temp.next
                middle_pos += 1
            
            # Calculate target position
            target_pos = middle_pos + n
            if target_pos < 0:
                return None
            
            # Traverse to target position
            current = head
            for _ in range(target_pos):
                current = current.next
        
        return current
    
    # ==================== PALINDROME DETECTION ====================
    
    def is_palindrome_with_reversal(self, head: Optional[ListNode]) -> bool:
        """
        Check if linked list is palindrome by reversing second half
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            head: Head of linked list
        
        Returns:
            bool: True if list is palindrome
        """
        if not head or not head.next:
            return True
        
        # Find middle
        slow = fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        
        # Reverse second half
        second_half = self._reverse_list(slow.next)
        
        # Compare both halves
        first_half = head
        result = True
        
        while second_half:
            if first_half.val != second_half.val:
                result = False
                break
            first_half = first_half.next
            second_half = second_half.next
        
        # Restore original list (optional)
        slow.next = self._reverse_list(slow.next)
        
        return result
    
    def is_palindrome_with_stack(self, head: Optional[ListNode]) -> bool:
        """
        Check palindrome using stack
        
        Time Complexity: O(n)
        Space Complexity: O(n/2)
        
        Args:
            head: Head of linked list
        
        Returns:
            bool: True if list is palindrome
        """
        if not head or not head.next:
            return True
        
        stack = []
        slow = fast = head
        
        # Push first half to stack
        while fast and fast.next:
            stack.append(slow.val)
            slow = slow.next
            fast = fast.next.next
        
        # Skip middle element for odd length
        if fast:
            slow = slow.next
        
        # Compare second half with stack
        while slow:
            if stack.pop() != slow.val:
                return False
            slow = slow.next
        
        return True
    
    def is_palindrome_with_array(self, head: Optional[ListNode]) -> bool:
        """
        Check palindrome by converting to array
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        
        Args:
            head: Head of linked list
        
        Returns:
            bool: True if list is palindrome
        """
        values = []
        current = head
        
        while current:
            values.append(current.val)
            current = current.next
        
        return values == values[::-1]
    
    def get_palindrome_info(self, head: Optional[ListNode]) -> dict:
        """
        Get comprehensive palindrome information
        
        Args:
            head: Head of linked list
        
        Returns:
            Dictionary with palindrome analysis
        """
        if not head:
            return {"is_palindrome": True, "length": 0, "middle_val": None}
        
        # Convert to array for analysis
        values = []
        current = head
        
        while current:
            values.append(current.val)
            current = current.next
        
        length = len(values)
        is_palindrome = values == values[::-1]
        middle_val = values[length // 2] if length > 0 else None
        
        # Find mismatch position if not palindrome
        mismatch_pos = -1
        if not is_palindrome:
            for i in range(length // 2):
                if values[i] != values[length - 1 - i]:
                    mismatch_pos = i
                    break
        
        return {
            "is_palindrome": is_palindrome,
            "length": length,
            "middle_val": middle_val,
            "mismatch_position": mismatch_pos,
            "values": values
        }
    
    # ==================== ADVANCED TWO POINTER TECHNIQUES ====================
    
    def find_nth_from_end(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        """
        Find nth node from end using two pointers
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            head: Head of linked list
            n: Position from end (1-indexed)
        
        Returns:
            Nth node from end
        """
        if not head or n <= 0:
            return None
        
        fast = slow = head
        
        # Move fast pointer n steps ahead
        for _ in range(n):
            if not fast:
                return None
            fast = fast.next
        
        # Move both pointers until fast reaches end
        while fast:
            slow = slow.next
            fast = fast.next
        
        return slow
    
    def find_intersection(self, head1: Optional[ListNode], head2: Optional[ListNode]) -> Optional[ListNode]:
        """
        Find intersection point of two linked lists
        
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        
        Args:
            head1: Head of first linked list
            head2: Head of second linked list
        
        Returns:
            Intersection node or None
        """
        if not head1 or not head2:
            return None
        
        # Get lengths
        len1 = self._get_length(head1)
        len2 = self._get_length(head2)
        
        # Align starting points
        if len1 > len2:
            for _ in range(len1 - len2):
                head1 = head1.next
        else:
            for _ in range(len2 - len1):
                head2 = head2.next
        
        # Find intersection
        while head1 and head2:
            if head1 == head2:
                return head1
            head1 = head1.next
            head2 = head2.next
        
        return None
    
    def find_intersection_elegant(self, head1: Optional[ListNode], head2: Optional[ListNode]) -> Optional[ListNode]:
        """
        Find intersection using elegant two-pointer approach
        
        Args:
            head1: Head of first linked list
            head2: Head of second linked list
        
        Returns:
            Intersection node or None
        """
        if not head1 or not head2:
            return None
        
        ptr1 = head1
        ptr2 = head2
        
        while ptr1 != ptr2:
            ptr1 = head2 if ptr1 is None else ptr1.next
            ptr2 = head1 if ptr2 is None else ptr2.next
        
        return ptr1
    
    def separate_odd_even_positions(self, head: Optional[ListNode]) -> Tuple[Optional[ListNode], Optional[ListNode]]:
        """
        Separate nodes at odd and even positions
        
        Args:
            head: Head of linked list
        
        Returns:
            Tuple of (odd_positioned_head, even_positioned_head)
        """
        if not head:
            return None, None
        
        odd = head
        even = head.next
        even_head = even
        
        while even and even.next:
            odd.next = even.next
            odd = odd.next
            even.next = odd.next
            even = even.next
        
        return head, even_head
    
    def merge_odd_even_positions(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Rearrange list to have all odd positioned nodes first, then even positioned
        
        Args:
            head: Head of linked list
        
        Returns:
            Head of rearranged list
        """
        if not head or not head.next:
            return head
        
        odd_head, even_head = self.separate_odd_even_positions(head)
        
        # Connect odd list to even list
        current = odd_head
        while current.next:
            current = current.next
        current.next = even_head
        
        return odd_head
    
    def find_triplet_with_sum(self, head: Optional[ListNode], target: int) -> List[Tuple[int, int, int]]:
        """
        Find all triplets in sorted linked list that sum to target
        
        Args:
            head: Head of sorted linked list
            target: Target sum
        
        Returns:
            List of triplets (values, not nodes)
        """
        if not head or not head.next or not head.next.next:
            return []
        
        # Convert to array for easier processing
        values = []
        current = head
        while current:
            values.append(current.val)
            current = current.next
        
        triplets = []
        n = len(values)
        
        for i in range(n - 2):
            if i > 0 and values[i] == values[i - 1]:
                continue
            
            left = i + 1
            right = n - 1
            
            while left < right:
                current_sum = values[i] + values[left] + values[right]
                
                if current_sum == target:
                    triplets.append((values[i], values[left], values[right]))
                    
                    # Skip duplicates
                    while left < right and values[left] == values[left + 1]:
                        left += 1
                    while left < right and values[right] == values[right - 1]:
                        right -= 1
                    
                    left += 1
                    right -= 1
                elif current_sum < target:
                    left += 1
                else:
                    right -= 1
        
        return triplets
    
    # ==================== UTILITY METHODS ====================
    
    def _reverse_list(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """Helper method to reverse a linked list"""
        prev = None
        current = head
        
        while current:
            next_temp = current.next
            current.next = prev
            prev = current
            current = next_temp
        
        return prev
    
    def _get_length(self, head: Optional[ListNode]) -> int:
        """Helper method to get length of linked list"""
        length = 0
        current = head
        
        while current:
            length += 1
            current = current.next
        
        return length
    
    def create_list_from_array(self, arr: List[int]) -> Optional[ListNode]:
        """Create linked list from array"""
        if not arr:
            return None
        
        head = ListNode(arr[0])
        current = head
        
        for val in arr[1:]:
            current.next = ListNode(val)
            current = current.next
        
        return head
    
    def list_to_array(self, head: Optional[ListNode]) -> List[int]:
        """Convert linked list to array"""
        result = []
        current = head
        
        while current:
            result.append(current.val)
            current = current.next
        
        return result
    
    def create_cycle(self, head: Optional[ListNode], pos: int) -> Optional[ListNode]:
        """Create cycle in linked list at given position (for testing)"""
        if not head or pos < 0:
            return head
        
        # Find tail and cycle start
        tail = head
        cycle_start = None
        position = 0
        
        while tail.next:
            if position == pos:
                cycle_start = tail
            tail = tail.next
            position += 1
        
        # Create cycle
        if cycle_start:
            tail.next = cycle_start
        
        return head
    
    def print_list_info(self, head: Optional[ListNode], max_nodes: int = 20) -> None:
        """Print list information (handles cycles)"""
        if not head:
            print("Empty list")
            return
        
        visited = set()
        current = head
        values = []
        position = 0
        
        while current and position < max_nodes:
            if current in visited:
                values.append(f"{current.val}(cycle)")
                break
            
            visited.add(current)
            values.append(str(current.val))
            current = current.next
            position += 1
        
        if current and position == max_nodes:
            values.append("...")
        
        print(" -> ".join(values))

# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Fast & Slow Pointer Techniques Demo ===\n")
    
    fsp = FastSlowPointer()
    
    # Example 1: Cycle Detection
    print("1. Cycle Detection:")
    
    # Create list without cycle
    no_cycle_list = fsp.create_list_from_array([1, 2, 3, 4, 5])
    print("List without cycle:")
    fsp.print_list_info(no_cycle_list)
    
    has_cycle = fsp.has_cycle(no_cycle_list)
    print(f"Has cycle: {has_cycle}")
    
    # Create list with cycle
    cycle_list = fsp.create_list_from_array([1, 2, 3, 4, 5])
    cycle_list = fsp.create_cycle(cycle_list, 2)  # Cycle starts at position 2
    print("\nList with cycle (cycle starts at position 2):")
    fsp.print_list_info(cycle_list)
    
    has_cycle, cycle_pos = fsp.detect_cycle_with_position(cycle_list)
    cycle_length = fsp.find_cycle_length(cycle_list)
    cycle_start = fsp.find_cycle_start_node(cycle_list)
    
    print(f"Has cycle: {has_cycle}")
    print(f"Cycle start position: {cycle_pos}")
    print(f"Cycle length: {cycle_length}")
    print(f"Cycle start node value: {cycle_start.val if cycle_start else None}")
    print()
    
    # Example 2: Find Middle Element
    print("2. Find Middle Element:")
    
    test_arrays = [
        [1],
        [1, 2],
        [1, 2, 3],
        [1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6]
    ]
    
    for arr in test_arrays:
        test_list = fsp.create_list_from_array(arr)
        middle = fsp.find_middle(test_list)
        middle_with_count = fsp.find_middle_with_count(test_list)
        prev_middle = fsp.find_middle_previous(test_list)
        
        print(f"List: {arr}")
        print(f"  Middle: {middle.val if middle else None}")
        print(f"  Middle with count: {middle_with_count[0].val if middle_with_count[0] else None}, Count: {middle_with_count[1]}")
        print(f"  Previous of middle: {prev_middle[0].val if prev_middle[0] else None}")
    print()
    
    # Example 3: Palindrome Detection
    print("3. Palindrome Detection:")
    
    palindrome_tests = [
        [1],
        [1, 2, 1],
        [1, 2, 2, 1],
        [1, 2, 3, 2, 1],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 3, 2, 1]
    ]
    
    for arr in palindrome_tests:
        test_list = fsp.create_list_from_array(arr)
        
        is_pal_reversal = fsp.is_palindrome_with_reversal(test_list)
        
        # Recreate list for other methods
        test_list2 = fsp.create_list_from_array(arr)
        is_pal_stack = fsp.is_palindrome_with_stack(test_list2)
        
        test_list3 = fsp.create_list_from_array(arr)
        is_pal_array = fsp.is_palindrome_with_array(test_list3)
        
        test_list4 = fsp.create_list_from_array(arr)
        pal_info = fsp.get_palindrome_info(test_list4)
        
        print(f"List: {arr}")
        print(f"  Palindrome (reversal): {is_pal_reversal}")
        print(f"  Palindrome (stack): {is_pal_stack}")
        print(f"  Palindrome (array): {is_pal_array}")
        print(f"  Detailed info: {pal_info}")
        print()
    
    # Example 4: Nth from End
    print("4. Nth Node from End:")
    
    nth_test_list = fsp.create_list_from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(f"List: {fsp.list_to_array(nth_test_list)}")
    
    for n in [1, 3, 5, 10, 15]:
        nth_node = fsp.find_nth_from_end(nth_test_list, n)
        print(f"  {n}th from end: {nth_node.val if nth_node else 'None'}")
    print()
    
    # Example 5: List Intersection
    print("5. List Intersection:")
    
    # Create intersecting lists
    # List 1: 1 -> 2 -> 3 \
    #                     -> 6 -> 7 -> 8
    # List 2: 4 -> 5 ----/
    
    common = fsp.create_list_from_array([6, 7, 8])
    list1 = fsp.create_list_from_array([1, 2, 3])
    list2 = fsp.create_list_from_array([4, 5])
    
    # Connect to common part
    current1 = list1
    while current1.next:
        current1 = current1.next
    current1.next = common
    
    current2 = list2
    while current2.next:
        current2 = current2.next
    current2.next = common
    
    print("List 1:")
    fsp.print_list_info(list1, 10)
    print("List 2:")
    fsp.print_list_info(list2, 10)
    
    intersection = fsp.find_intersection(list1, list2)
    intersection_elegant = fsp.find_intersection_elegant(list1, list2)
    
    print(f"Intersection node: {intersection.val if intersection else 'None'}")
    print(f"Intersection (elegant): {intersection_elegant.val if intersection_elegant else 'None'}")
    print()
    
    # Example 6: Odd-Even Position Separation
    print("6. Odd-Even Position Separation:")
    
    odd_even_list = fsp.create_list_from_array([1, 2, 3, 4, 5, 6, 7, 8])
    print(f"Original list: {fsp.list_to_array(odd_even_list)}")
    
    odd_head, even_head = fsp.separate_odd_even_positions(odd_even_list)
    print(f"Odd positions: {fsp.list_to_array(odd_head)}")
    print(f"Even positions: {fsp.list_to_array(even_head)}")
    
    # Merge back
    merged_list = fsp.create_list_from_array([1, 2, 3, 4, 5, 6, 7, 8])
    merged_result = fsp.merge_odd_even_positions(merged_list)
    print(f"Merged (odd first, then even): {fsp.list_to_array(merged_result)}")
    print()
    
    # Example 7: Performance Analysis
    print("7. Performance Analysis:")
    
    # Large list operations
    large_array = list(range(1, 10001))
    large_list = fsp.create_list_from_array(large_array)
    
    print(f"Testing on list with {len(large_array)} elements")
    
    # Find middle
    middle = fsp.find_middle(large_list)
    print(f"Middle element: {middle.val if middle else None}")
    
    # Find nth from end
    nth_from_end = fsp.find_nth_from_end(large_list, 100)
    print(f"100th from end: {nth_from_end.val if nth_from_end else None}")
    
    # Check if palindrome (should be False)
    is_large_palindrome = fsp.is_palindrome_with_reversal(large_list)
    print(f"Is large list palindrome: {is_large_palindrome}")
    
    # Create palindromic list
    palindromic_array = [1, 2, 3, 4, 5, 4, 3, 2, 1]
    palindromic_list = fsp.create_list_from_array(palindromic_array)
    
    print(f"\nPalindrome test on {palindromic_array}:")
    methods = [
        ("Reversal method", fsp.is_palindrome_with_reversal),
        ("Stack method", fsp.is_palindrome_with_stack),
        ("Array method", fsp.is_palindrome_with_array),
    ]
    
    for method_name, method in methods:
        test_list = fsp.create_list_from_array(palindromic_array)
        result = method(test_list)
        print(f"  {method_name}: {result}")
    
    print("\n=== Demo Complete ===") 