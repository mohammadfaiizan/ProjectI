"""
Linked List Interview Challenges - Advanced Problems for Technical Interviews
This module implements the most challenging linked list problems commonly asked in technical interviews.
"""

from typing import Optional, List, Set, Dict
import math

class ListNode:
    """Basic node class for linked list"""
    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val = val
        self.next = next
    
    def __repr__(self):
        return f"ListNode({self.val})"

class LinkedListInterviewChallenges:
    """
    Implementation of advanced linked list interview challenges
    
    This class contains solutions to the most commonly asked and challenging
    linked list problems in technical interviews at top tech companies.
    """
    
    # ==================== LINKED LIST CYCLE II ====================
    
    def detect_cycle_ii(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Detect if linked list has cycle and return the node where cycle begins
        
        Problem: LeetCode 142 - Linked List Cycle II
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Algorithm: Floyd's Cycle Detection (Tortoise and Hare) + Mathematical Analysis
        
        Args:
            head: Head of the linked list
        
        Returns:
            Node where cycle begins, or None if no cycle
        """
        if not head or not head.next:
            return None
        
        # Phase 1: Detect if cycle exists using Floyd's algorithm
        slow = fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                # Cycle detected
                break
        else:
            # No cycle found
            return None
        
        # Phase 2: Find the start of the cycle
        # Mathematical proof: Distance from head to cycle start = 
        # Distance from meeting point to cycle start
        start = head
        while start != slow:
            start = start.next
            slow = slow.next
        
        return start
    
    def detect_cycle_ii_hashset(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Alternative approach using hash set
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        
        Args:
            head: Head of the linked list
        
        Returns:
            Node where cycle begins, or None if no cycle
        """
        if not head:
            return None
        
        visited = set()
        current = head
        
        while current:
            if current in visited:
                return current
            visited.add(current)
            current = current.next
        
        return None
    
    def get_cycle_length(self, head: Optional[ListNode]) -> int:
        """
        Get the length of the cycle if it exists
        
        Args:
            head: Head of the linked list
        
        Returns:
            Length of cycle, or 0 if no cycle
        """
        cycle_start = self.detect_cycle_ii(head)
        if not cycle_start:
            return 0
        
        # Count nodes in the cycle
        length = 1
        current = cycle_start.next
        
        while current != cycle_start:
            length += 1
            current = current.next
        
        return length
    
    # ==================== PALINDROME LINKED LIST ====================
    
    def is_palindrome(self, head: Optional[ListNode]) -> bool:
        """
        Check if linked list is a palindrome
        
        Problem: LeetCode 234 - Palindrome Linked List
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Algorithm: Find middle, reverse second half, compare, restore
        
        Args:
            head: Head of the linked list
        
        Returns:
            True if palindrome, False otherwise
        """
        if not head or not head.next:
            return True
        
        # Step 1: Find the middle of the linked list
        slow = fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        
        # Step 2: Reverse the second half
        second_half = self._reverse_list(slow.next)
        
        # Step 3: Compare first and second half
        first_half = head
        is_palindrome_result = True
        
        while second_half:
            if first_half.val != second_half.val:
                is_palindrome_result = False
                break
            first_half = first_half.next
            second_half = second_half.next
        
        # Step 4: Restore the original list structure
        slow.next = self._reverse_list(slow.next)
        
        return is_palindrome_result
    
    def is_palindrome_stack(self, head: Optional[ListNode]) -> bool:
        """
        Check palindrome using stack (extra space approach)
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        
        Args:
            head: Head of the linked list
        
        Returns:
            True if palindrome, False otherwise
        """
        if not head or not head.next:
            return True
        
        # Convert to list for easy comparison
        values = []
        current = head
        
        while current:
            values.append(current.val)
            current = current.next
        
        # Check if list is palindrome
        left, right = 0, len(values) - 1
        
        while left < right:
            if values[left] != values[right]:
                return False
            left += 1
            right -= 1
        
        return True
    
    def is_palindrome_recursive(self, head: Optional[ListNode]) -> bool:
        """
        Check palindrome using recursion
        
        Time Complexity: O(n)
        Space Complexity: O(n) - due to recursion stack
        
        Args:
            head: Head of the linked list
        
        Returns:
            True if palindrome, False otherwise
        """
        self.front_pointer = head
        
        def recursively_check(current_node):
            if current_node is not None:
                if not recursively_check(current_node.next):
                    return False
                if self.front_pointer.val != current_node.val:
                    return False
                self.front_pointer = self.front_pointer.next
            return True
        
        return recursively_check(head)
    
    # ==================== REORDER LIST ====================
    
    def reorder_list(self, head: Optional[ListNode]) -> None:
        """
        Reorder list as L0 → Ln → L1 → Ln-1 → L2 → Ln-2 → ...
        
        Problem: LeetCode 143 - Reorder List
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Algorithm: Find middle, reverse second half, merge alternately
        
        Args:
            head: Head of the linked list (modified in-place)
        """
        if not head or not head.next:
            return
        
        # Step 1: Find the middle of the linked list
        slow = fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        
        # Step 2: Split the list into two halves
        second_half = slow.next
        slow.next = None
        
        # Step 3: Reverse the second half
        second_half = self._reverse_list(second_half)
        
        # Step 4: Merge the two halves alternately
        first_half = head
        
        while second_half:
            temp1 = first_half.next
            temp2 = second_half.next
            
            first_half.next = second_half
            second_half.next = temp1
            
            first_half = temp1
            second_half = temp2
    
    def reorder_list_odd_even(self, head: Optional[ListNode]) -> None:
        """
        Reorder list grouping odd and even positioned nodes
        
        Problem: LeetCode 328 - Odd Even Linked List
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            head: Head of the linked list (modified in-place)
        """
        if not head or not head.next:
            return
        
        odd = head
        even = head.next
        even_head = even
        
        while even and even.next:
            odd.next = even.next
            odd = odd.next
            even.next = odd.next
            even = even.next
        
        # Connect odd and even lists
        odd.next = even_head
    
    def reorder_list_k_reverse(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        """
        Reverse every k nodes in the linked list
        
        Problem: LeetCode 25 - Reverse Nodes in k-Group
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            head: Head of the linked list
            k: Size of each group to reverse
        
        Returns:
            Head of the modified list
        """
        if not head or k == 1:
            return head
        
        # Check if we have at least k nodes
        def has_k_nodes(node, k):
            count = 0
            while node and count < k:
                count += 1
                node = node.next
            return count == k
        
        # Reverse k nodes
        def reverse_k_nodes(head, k):
            prev = None
            current = head
            
            for _ in range(k):
                next_temp = current.next
                current.next = prev
                prev = current
                current = next_temp
            
            return prev, current
        
        # Check if we have k nodes from head
        if not has_k_nodes(head, k):
            return head
        
        # Reverse first k nodes
        new_head, next_group = reverse_k_nodes(head, k)
        
        # Recursively handle remaining nodes
        head.next = self.reorder_list_k_reverse(next_group, k)
        
        return new_head
    
    # ==================== GROUP ODD AND EVEN NODES ====================
    
    def odd_even_list(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Group odd and even positioned nodes together
        
        Problem: LeetCode 328 - Odd Even Linked List
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            head: Head of the linked list
        
        Returns:
            Head of the modified list
        """
        if not head or not head.next:
            return head
        
        odd = head
        even = head.next
        even_head = even
        
        while even and even.next:
            odd.next = even.next
            odd = odd.next
            even.next = odd.next
            even = even.next
        
        odd.next = even_head
        return head
    
    def group_by_parity(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Group nodes by value parity (even values first, then odd)
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            head: Head of the linked list
        
        Returns:
            Head of the modified list
        """
        if not head:
            return head
        
        even_head = even_tail = None
        odd_head = odd_tail = None
        current = head
        
        while current:
            if current.val % 2 == 0:
                # Even value
                if not even_head:
                    even_head = even_tail = current
                else:
                    even_tail.next = current
                    even_tail = current
            else:
                # Odd value
                if not odd_head:
                    odd_head = odd_tail = current
                else:
                    odd_tail.next = current
                    odd_tail = current
            
            current = current.next
        
        # Connect even and odd parts
        if even_tail:
            even_tail.next = odd_head
        if odd_tail:
            odd_tail.next = None
        
        return even_head if even_head else odd_head
    
    def group_by_value_ranges(self, head: Optional[ListNode], ranges: List[tuple]) -> Optional[ListNode]:
        """
        Group nodes by value ranges
        
        Args:
            head: Head of the linked list
            ranges: List of (min, max) tuples defining ranges
        
        Returns:
            Head of the modified list
        """
        if not head or not ranges:
            return head
        
        # Create separate lists for each range
        range_heads = [None] * len(ranges)
        range_tails = [None] * len(ranges)
        current = head
        
        while current:
            placed = False
            
            for i, (min_val, max_val) in enumerate(ranges):
                if min_val <= current.val <= max_val:
                    if not range_heads[i]:
                        range_heads[i] = range_tails[i] = current
                    else:
                        range_tails[i].next = current
                        range_tails[i] = current
                    placed = True
                    break
            
            if not placed:
                # Handle values outside all ranges
                pass
            
            current = current.next
        
        # Connect all ranges
        result_head = None
        prev_tail = None
        
        for i, head_node in enumerate(range_heads):
            if head_node:
                if not result_head:
                    result_head = head_node
                if prev_tail:
                    prev_tail.next = head_node
                prev_tail = range_tails[i]
        
        if prev_tail:
            prev_tail.next = None
        
        return result_head
    
    # ==================== INTERSECTION OF TWO LISTS ====================
    
    def get_intersection_node(self, headA: Optional[ListNode], headB: Optional[ListNode]) -> Optional[ListNode]:
        """
        Find intersection node of two linked lists using length difference
        
        Problem: LeetCode 160 - Intersection of Two Linked Lists
        
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        
        Algorithm: Calculate length difference, align longer list, traverse together
        
        Args:
            headA: Head of first linked list
            headB: Head of second linked list
        
        Returns:
            Intersection node or None if no intersection
        """
        if not headA or not headB:
            return None
        
        # Calculate lengths of both lists
        lenA = self._get_length(headA)
        lenB = self._get_length(headB)
        
        # Align the starting points
        currA, currB = headA, headB
        
        if lenA > lenB:
            # Move currA forward by (lenA - lenB) steps
            for _ in range(lenA - lenB):
                currA = currA.next
        else:
            # Move currB forward by (lenB - lenA) steps
            for _ in range(lenB - lenA):
                currB = currB.next
        
        # Find intersection
        while currA and currB:
            if currA == currB:
                return currA
            currA = currA.next
            currB = currB.next
        
        return None
    
    def get_intersection_node_two_pointers(self, headA: Optional[ListNode], headB: Optional[ListNode]) -> Optional[ListNode]:
        """
        Find intersection using two pointers (elegant approach)
        
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        
        Algorithm: When pointer reaches end, redirect to other list's head
        
        Args:
            headA: Head of first linked list
            headB: Head of second linked list
        
        Returns:
            Intersection node or None if no intersection
        """
        if not headA or not headB:
            return None
        
        pA, pB = headA, headB
        
        while pA != pB:
            # When reaching end, redirect to other list's head
            pA = headB if pA is None else pA.next
            pB = headA if pB is None else pB.next
        
        return pA  # Either intersection node or None
    
    def get_intersection_node_hashset(self, headA: Optional[ListNode], headB: Optional[ListNode]) -> Optional[ListNode]:
        """
        Find intersection using hash set
        
        Time Complexity: O(m + n)
        Space Complexity: O(m) or O(n)
        
        Args:
            headA: Head of first linked list
            headB: Head of second linked list
        
        Returns:
            Intersection node or None if no intersection
        """
        if not headA or not headB:
            return None
        
        visited = set()
        
        # Add all nodes from list A to set
        current = headA
        while current:
            visited.add(current)
            current = current.next
        
        # Check if any node from list B is in the set
        current = headB
        while current:
            if current in visited:
                return current
            current = current.next
        
        return None
    
    def find_all_intersections(self, lists: List[Optional[ListNode]]) -> List[ListNode]:
        """
        Find all intersection nodes among multiple linked lists
        
        Args:
            lists: List of linked list heads
        
        Returns:
            List of intersection nodes
        """
        if not lists or len(lists) < 2:
            return []
        
        # Count occurrences of each node
        node_count = {}
        
        for head in lists:
            visited_in_current = set()
            current = head
            
            while current:
                if current not in visited_in_current:
                    node_count[current] = node_count.get(current, 0) + 1
                    visited_in_current.add(current)
                current = current.next
        
        # Find nodes that appear in multiple lists
        intersections = []
        for node, count in node_count.items():
            if count > 1:
                intersections.append(node)
        
        return intersections
    
    # ==================== HELPER METHODS ====================
    
    def _reverse_list(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """Reverse a linked list iteratively"""
        prev = None
        current = head
        
        while current:
            next_temp = current.next
            current.next = prev
            prev = current
            current = next_temp
        
        return prev
    
    def _get_length(self, head: Optional[ListNode]) -> int:
        """Get length of linked list"""
        length = 0
        current = head
        
        while current:
            length += 1
            current = current.next
        
        return length
    
    def _get_middle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """Get middle node of linked list"""
        if not head:
            return None
        
        slow = fast = head
        
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        
        return slow
    
    def _split_list(self, head: Optional[ListNode]) -> tuple:
        """Split list into two halves"""
        if not head:
            return None, None
        
        middle = self._get_middle(head)
        second_half = middle.next
        middle.next = None
        
        return head, second_half
    
    # ==================== ADVANCED CHALLENGES ====================
    
    def flatten_multilevel_list(self, head: Optional['Node']) -> Optional['Node']:
        """
        Flatten a multilevel doubly linked list
        
        Each node has child pointer that may point to separate doubly linked list
        
        Time Complexity: O(n)
        Space Complexity: O(d) where d is maximum depth
        """
        if not head:
            return head
        
        def flatten_dfs(node):
            last = None
            
            while node:
                if node.child:
                    # Save next node
                    next_node = node.next
                    
                    # Connect to child
                    node.next = node.child
                    node.child.prev = node
                    node.child = None
                    
                    # Recursively flatten child list and get its tail
                    child_tail = flatten_dfs(node.next)
                    
                    # Connect child tail to next node
                    if next_node:
                        child_tail.next = next_node
                        next_node.prev = child_tail
                    
                    last = child_tail
                else:
                    last = node
                
                node = node.next
            
            return last
        
        flatten_dfs(head)
        return head
    
    def copy_random_list(self, head: Optional['Node']) -> Optional['Node']:
        """
        Copy linked list with random pointers
        
        Problem: LeetCode 138 - Copy List with Random Pointer
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Algorithm: Interweave original and copied nodes
        """
        if not head:
            return None
        
        # Step 1: Create copied nodes and interweave with original
        current = head
        while current:
            copied = Node(current.val)
            copied.next = current.next
            current.next = copied
            current = copied.next
        
        # Step 2: Set random pointers for copied nodes
        current = head
        while current:
            if current.random:
                current.next.random = current.random.next
            current = current.next.next
        
        # Step 3: Separate original and copied lists
        dummy = Node(0)
        copied_current = dummy
        current = head
        
        while current:
            copied_current.next = current.next
            current.next = current.next.next
            copied_current = copied_current.next
            current = current.next
        
        return dummy.next
    
    def merge_k_sorted_lists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """
        Merge k sorted linked lists
        
        Problem: LeetCode 23 - Merge k Sorted Lists
        
        Time Complexity: O(n log k) where n is total nodes, k is number of lists
        Space Complexity: O(log k) for recursion
        
        Algorithm: Divide and conquer approach
        """
        if not lists:
            return None
        
        def merge_two_lists(l1, l2):
            dummy = ListNode(0)
            current = dummy
            
            while l1 and l2:
                if l1.val <= l2.val:
                    current.next = l1
                    l1 = l1.next
                else:
                    current.next = l2
                    l2 = l2.next
                current = current.next
            
            current.next = l1 or l2
            return dummy.next
        
        def merge_lists(lists, start, end):
            if start == end:
                return lists[start]
            if start > end:
                return None
            
            mid = (start + end) // 2
            left = merge_lists(lists, start, mid)
            right = merge_lists(lists, mid + 1, end)
            
            return merge_two_lists(left, right)
        
        return merge_lists(lists, 0, len(lists) - 1)
    
    def add_two_numbers_ii(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        """
        Add two numbers represented as linked lists (most significant digit first)
        
        Problem: LeetCode 445 - Add Two Numbers II
        
        Time Complexity: O(max(m, n))
        Space Complexity: O(max(m, n))
        """
        # Reverse both lists
        l1_rev = self._reverse_list(l1)
        l2_rev = self._reverse_list(l2)
        
        # Add numbers
        dummy = ListNode(0)
        current = dummy
        carry = 0
        
        while l1_rev or l2_rev or carry:
            val1 = l1_rev.val if l1_rev else 0
            val2 = l2_rev.val if l2_rev else 0
            
            total = val1 + val2 + carry
            carry = total // 10
            digit = total % 10
            
            current.next = ListNode(digit)
            current = current.next
            
            l1_rev = l1_rev.next if l1_rev else None
            l2_rev = l2_rev.next if l2_rev else None
        
        # Reverse result to get correct order
        return self._reverse_list(dummy.next)

# ==================== HELPER CLASSES FOR SPECIFIC PROBLEMS ====================

class Node:
    """Node class for problems requiring additional pointers"""
    def __init__(self, val: int = 0):
        self.val = val
        self.next = None
        self.prev = None
        self.child = None
        self.random = None

# ==================== EXAMPLE USAGE AND TESTING ====================

def create_list_with_cycle(values: List[int], cycle_pos: int = -1) -> Optional[ListNode]:
    """Create a linked list with cycle for testing"""
    if not values:
        return None
    
    nodes = [ListNode(val) for val in values]
    
    # Connect nodes
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    
    # Create cycle if specified
    if 0 <= cycle_pos < len(nodes):
        nodes[-1].next = nodes[cycle_pos]
    
    return nodes[0]

def create_list(values: List[int]) -> Optional[ListNode]:
    """Create a simple linked list from values"""
    if not values:
        return None
    
    head = ListNode(values[0])
    current = head
    
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    
    return head

def print_list(head: Optional[ListNode], max_nodes: int = 20) -> str:
    """Print linked list values (with cycle detection)"""
    if not head:
        return "[]"
    
    result = []
    current = head
    seen = set()
    count = 0
    
    while current and count < max_nodes:
        if current in seen:
            result.append(f"...cycle back to {current.val}")
            break
        
        seen.add(current)
        result.append(str(current.val))
        current = current.next
        count += 1
    
    if current and count >= max_nodes:
        result.append("...")
    
    return " -> ".join(result)

if __name__ == "__main__":
    print("=== Linked List Interview Challenges Demo ===\n")
    
    challenges = LinkedListInterviewChallenges()
    
    # Example 1: Linked List Cycle II
    print("1. Linked List Cycle II Detection:")
    
    # Test case 1: List with cycle
    cycle_list = create_list_with_cycle([3, 2, 0, -4], 1)
    cycle_start = challenges.detect_cycle_ii(cycle_list)
    cycle_length = challenges.get_cycle_length(cycle_list)
    
    print(f"List with cycle at position 1:")
    print(f"Cycle starts at node with value: {cycle_start.val if cycle_start else None}")
    print(f"Cycle length: {cycle_length}")
    
    # Test case 2: List without cycle
    no_cycle_list = create_list([1, 2, 3, 4, 5])
    cycle_start = challenges.detect_cycle_ii(no_cycle_list)
    
    print(f"\nList without cycle: {print_list(no_cycle_list)}")
    print(f"Cycle starts at: {cycle_start}")
    
    # Comparison of methods
    cycle_list2 = create_list_with_cycle([1, 2], 0)
    cycle_start_floyd = challenges.detect_cycle_ii(cycle_list2)
    cycle_start_hash = challenges.detect_cycle_ii_hashset(cycle_list2)
    
    print(f"\nComparison of methods:")
    print(f"Floyd's algorithm result: {cycle_start_floyd.val if cycle_start_floyd else None}")
    print(f"Hash set algorithm result: {cycle_start_hash.val if cycle_start_hash else None}")
    print()
    
    # Example 2: Palindrome Linked List
    print("2. Palindrome Linked List:")
    
    test_cases = [
        ([1, 2, 2, 1], "Even length palindrome"),
        ([1, 2, 3, 2, 1], "Odd length palindrome"),
        ([1, 2, 3, 4, 5], "Not a palindrome"),
        ([1], "Single node"),
        ([], "Empty list")
    ]
    
    for values, description in test_cases:
        if values:
            test_list = create_list(values)
            is_pal_optimal = challenges.is_palindrome(test_list)
            
            # Test other methods
            test_list2 = create_list(values)
            is_pal_stack = challenges.is_palindrome_stack(test_list2)
            
            test_list3 = create_list(values)
            is_pal_recursive = challenges.is_palindrome_recursive(test_list3)
            
            print(f"{description}: {values}")
            print(f"  Optimal (O(1) space): {is_pal_optimal}")
            print(f"  Stack method: {is_pal_stack}")
            print(f"  Recursive method: {is_pal_recursive}")
        else:
            print(f"{description}: {challenges.is_palindrome(None)}")
    
    print()
    
    # Example 3: Reorder List
    print("3. Reorder List:")
    
    # Standard reorder
    reorder_list = create_list([1, 2, 3, 4, 5])
    print(f"Original: {print_list(reorder_list)}")
    challenges.reorder_list(reorder_list)
    print(f"Reordered: {print_list(reorder_list)}")
    
    # Odd-even reorder
    odd_even_list = create_list([1, 2, 3, 4, 5, 6])
    print(f"\nOriginal: {print_list(odd_even_list)}")
    challenges.reorder_list_odd_even(odd_even_list)
    print(f"Odd-Even grouped: {print_list(odd_even_list)}")
    
    # K-group reverse
    k_reverse_list = create_list([1, 2, 3, 4, 5, 6, 7, 8])
    print(f"\nOriginal: {print_list(k_reverse_list)}")
    k_reverse_result = challenges.reorder_list_k_reverse(k_reverse_list, 3)
    print(f"Reverse in groups of 3: {print_list(k_reverse_result)}")
    print()
    
    # Example 4: Group Odd and Even Nodes
    print("4. Group Odd and Even Nodes:")
    
    # Position-based grouping
    position_list = create_list([1, 2, 3, 4, 5, 6])
    print(f"Original: {print_list(position_list)}")
    position_result = challenges.odd_even_list(position_list)
    print(f"Grouped by position: {print_list(position_result)}")
    
    # Value-based grouping
    value_list = create_list([4, 1, 8, 3, 6, 2, 7, 5])
    print(f"\nOriginal: {print_list(value_list)}")
    value_result = challenges.group_by_parity(value_list)
    print(f"Grouped by value parity: {print_list(value_result)}")
    
    # Range-based grouping
    range_list = create_list([15, 3, 25, 8, 12, 30, 1, 45])
    print(f"\nOriginal: {print_list(range_list)}")
    ranges = [(1, 10), (11, 20), (21, 50)]
    range_result = challenges.group_by_value_ranges(range_list, ranges)
    print(f"Grouped by ranges {ranges}: {print_list(range_result)}")
    print()
    
    # Example 5: Intersection of Two Lists
    print("5. Intersection of Two Linked Lists:")
    
    # Create intersecting lists
    # List A: 4 -> 1 -> 8 -> 4 -> 5
    # List B: 5 -> 6 -> 1 -> 8 -> 4 -> 5
    # Intersection at node with value 8
    
    common_part = create_list([8, 4, 5])
    
    listA = ListNode(4)
    listA.next = ListNode(1)
    listA.next.next = common_part
    
    listB = ListNode(5)
    listB.next = ListNode(6)
    listB.next.next = ListNode(1)
    listB.next.next.next = common_part
    
    print(f"List A: 4 -> 1 -> 8 -> 4 -> 5")
    print(f"List B: 5 -> 6 -> 1 -> 8 -> 4 -> 5")
    
    # Test different methods
    intersection1 = challenges.get_intersection_node(listA, listB)
    intersection2 = challenges.get_intersection_node_two_pointers(listA, listB)
    intersection3 = challenges.get_intersection_node_hashset(listA, listB)
    
    print(f"Intersection (length difference): {intersection1.val if intersection1 else None}")
    print(f"Intersection (two pointers): {intersection2.val if intersection2 else None}")
    print(f"Intersection (hash set): {intersection3.val if intersection3 else None}")
    
    # Test non-intersecting lists
    listC = create_list([1, 2, 3])
    listD = create_list([4, 5, 6])
    
    intersection4 = challenges.get_intersection_node(listC, listD)
    print(f"\nNon-intersecting lists intersection: {intersection4}")
    print()
    
    # Example 6: Advanced Challenges
    print("6. Advanced Interview Challenges:")
    
    # Merge k sorted lists
    sorted_lists = [
        create_list([1, 4, 5]),
        create_list([1, 3, 4]),
        create_list([2, 6])
    ]
    
    print("Merge k sorted lists:")
    for i, lst in enumerate(sorted_lists):
        print(f"  List {i + 1}: {print_list(lst)}")
    
    merged = challenges.merge_k_sorted_lists(sorted_lists)
    print(f"Merged result: {print_list(merged)}")
    
    # Add two numbers II
    num1 = create_list([7, 2, 4, 3])  # Represents 7243
    num2 = create_list([5, 6, 4])     # Represents 564
    
    print(f"\nAdd two numbers:")
    print(f"Number 1: {print_list(num1)} (7243)")
    print(f"Number 2: {print_list(num2)} (564)")
    
    sum_result = challenges.add_two_numbers_ii(num1, num2)
    print(f"Sum: {print_list(sum_result)} (7807)")
    print()
    
    # Example 7: Performance Analysis
    print("7. Performance Analysis:")
    
    import time
    import random
    
    # Test palindrome detection performance
    large_palindrome = list(range(1000)) + list(range(999, -1, -1))
    large_list = create_list(large_palindrome)
    
    start_time = time.time()
    is_pal = challenges.is_palindrome(large_list)
    optimal_time = time.time() - start_time
    
    large_list2 = create_list(large_palindrome)
    start_time = time.time()
    is_pal2 = challenges.is_palindrome_stack(large_list2)
    stack_time = time.time() - start_time
    
    print(f"Palindrome detection on 2000-node list:")
    print(f"  Optimal O(1) space: {optimal_time:.6f}s")
    print(f"  Stack O(n) space: {stack_time:.6f}s")
    print(f"  Both methods agree: {is_pal == is_pal2}")
    
    # Test cycle detection performance
    large_cycle_list = create_list_with_cycle(list(range(10000)), 5000)
    
    start_time = time.time()
    cycle_node1 = challenges.detect_cycle_ii(large_cycle_list)
    floyd_time = time.time() - start_time
    
    large_cycle_list2 = create_list_with_cycle(list(range(10000)), 5000)
    start_time = time.time()
    cycle_node2 = challenges.detect_cycle_ii_hashset(large_cycle_list2)
    hashset_time = time.time() - start_time
    
    print(f"\nCycle detection on 10000-node list:")
    print(f"  Floyd's algorithm: {floyd_time:.6f}s")
    print(f"  Hash set method: {hashset_time:.6f}s")
    print(f"  Both methods agree: {(cycle_node1.val if cycle_node1 else None) == (cycle_node2.val if cycle_node2 else None)}")
    
    # Memory usage analysis
    print(f"\nMemory Usage Analysis:")
    print(f"Floyd's Cycle Detection: O(1) space - only uses two pointers")
    print(f"Hash Set Cycle Detection: O(n) space - stores all visited nodes")
    print(f"Optimal Palindrome: O(1) space - reverses half the list in-place")
    print(f"Stack Palindrome: O(n) space - stores all node values")
    print(f"Two-pointer Intersection: O(1) space - elegant switching technique")
    print(f"Length Difference Intersection: O(1) space - traditional approach")
    
    # Real-world applications
    print(f"\nReal-World Applications:")
    print(f"1. Cycle Detection: Memory leak detection, infinite loop prevention")
    print(f"2. Palindrome Check: DNA sequence analysis, string validation")
    print(f"3. List Reordering: Data reorganization, optimal memory access patterns")
    print(f"4. Intersection Finding: Social network analysis, data deduplication")
    print(f"5. Merge Operations: Database joins, sorted data combination")
    
    print("\n=== Demo Complete ===") 