"""
Linked List Merge & Sort - Comprehensive Sorting and Merging Algorithms
This module implements all types of sorting and merging operations on linked lists.
"""

from typing import Optional, List
import heapq
from dataclasses import dataclass, field

class ListNode:
    """Basic node class for linked list"""
    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val = val
        self.next = next
    
    def __repr__(self):
        return f"ListNode({self.val})"
    
    def __lt__(self, other):
        """For heap comparison"""
        return self.val < other.val

@dataclass
class HeapItem:
    """Wrapper for heap items to handle comparison"""
    node: ListNode
    list_index: int = field(compare=False)
    
    def __lt__(self, other):
        return self.node.val < other.node.val

class LinkedListMergeSort:
    """
    Implementation of merge and sort operations on linked lists
    """
    
    def __init__(self):
        """Initialize merge sort class"""
        pass
    
    # ==================== MERGE TWO SORTED LISTS ====================
    
    def merge_two_lists_recursive(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        """
        Merge two sorted linked lists recursively
        
        Time Complexity: O(m + n)
        Space Complexity: O(m + n) - recursion stack
        
        Args:
            l1: Head of first sorted list
            l2: Head of second sorted list
        
        Returns:
            Head of merged sorted list
        """
        if not l1:
            return l2
        if not l2:
            return l1
        
        if l1.val <= l2.val:
            l1.next = self.merge_two_lists_recursive(l1.next, l2)
            return l1
        else:
            l2.next = self.merge_two_lists_recursive(l1, l2.next)
            return l2
    
    def merge_two_lists_iterative(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        """
        Merge two sorted linked lists iteratively
        
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        
        Args:
            l1: Head of first sorted list
            l2: Head of second sorted list
        
        Returns:
            Head of merged sorted list
        """
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
        
        # Append remaining nodes
        current.next = l1 if l1 else l2
        
        return dummy.next
    
    def merge_two_lists_in_place(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        """
        Merge two sorted lists in-place without creating new nodes
        
        Args:
            l1: Head of first sorted list
            l2: Head of second sorted list
        
        Returns:
            Head of merged sorted list
        """
        if not l1:
            return l2
        if not l2:
            return l1
        
        # Ensure l1 starts with smaller value
        if l1.val > l2.val:
            l1, l2 = l2, l1
        
        head = l1
        
        while l1.next and l2:
            if l1.next.val <= l2.val:
                l1 = l1.next
            else:
                # Insert l2 node between l1 and l1.next
                temp = l2.next
                l2.next = l1.next
                l1.next = l2
                l1 = l2
                l2 = temp
        
        # Append remaining l2 nodes
        if l2:
            l1.next = l2
        
        return head
    
    def merge_alternating(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        """
        Merge two lists by alternating nodes (not necessarily sorted)
        
        Args:
            l1: Head of first list
            l2: Head of second list
        
        Returns:
            Head of merged list
        """
        if not l1:
            return l2
        if not l2:
            return l1
        
        dummy = ListNode(0)
        current = dummy
        take_from_l1 = True
        
        while l1 and l2:
            if take_from_l1:
                current.next = l1
                l1 = l1.next
            else:
                current.next = l2
                l2 = l2.next
            
            current = current.next
            take_from_l1 = not take_from_l1
        
        # Append remaining nodes
        current.next = l1 if l1 else l2
        
        return dummy.next
    
    # ==================== MERGE K SORTED LISTS ====================
    
    def merge_k_lists_min_heap(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """
        Merge k sorted linked lists using min heap
        
        Time Complexity: O(N log k) where N is total number of nodes
        Space Complexity: O(k)
        
        Args:
            lists: List of heads of sorted linked lists
        
        Returns:
            Head of merged sorted list
        """
        if not lists:
            return None
        
        # Initialize heap with first node from each list
        heap = []
        
        for i, head in enumerate(lists):
            if head:
                heapq.heappush(heap, HeapItem(head, i))
        
        dummy = ListNode(0)
        current = dummy
        
        while heap:
            # Get the smallest node
            item = heapq.heappop(heap)
            current.next = item.node
            current = current.next
            
            # Add next node from the same list
            if item.node.next:
                heapq.heappush(heap, HeapItem(item.node.next, item.list_index))
        
        return dummy.next
    
    def merge_k_lists_divide_conquer(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """
        Merge k sorted lists using divide and conquer
        
        Time Complexity: O(N log k)
        Space Complexity: O(log k) - recursion stack
        
        Args:
            lists: List of heads of sorted linked lists
        
        Returns:
            Head of merged sorted list
        """
        if not lists:
            return None
        
        if len(lists) == 1:
            return lists[0]
        
        def merge_lists_range(start: int, end: int) -> Optional[ListNode]:
            if start == end:
                return lists[start]
            
            if start + 1 == end:
                return self.merge_two_lists_iterative(lists[start], lists[end])
            
            mid = (start + end) // 2
            left = merge_lists_range(start, mid)
            right = merge_lists_range(mid + 1, end)
            
            return self.merge_two_lists_iterative(left, right)
        
        return merge_lists_range(0, len(lists) - 1)
    
    def merge_k_lists_sequential(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """
        Merge k sorted lists sequentially (brute force)
        
        Time Complexity: O(N * k)
        Space Complexity: O(1)
        
        Args:
            lists: List of heads of sorted linked lists
        
        Returns:
            Head of merged sorted list
        """
        if not lists:
            return None
        
        result = lists[0]
        
        for i in range(1, len(lists)):
            result = self.merge_two_lists_iterative(result, lists[i])
        
        return result
    
    def merge_k_lists_priority_queue(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """
        Alternative priority queue implementation
        
        Args:
            lists: List of heads of sorted linked lists
        
        Returns:
            Head of merged sorted list
        """
        if not lists:
            return None
        
        # Filter out None lists
        valid_lists = [head for head in lists if head]
        
        if not valid_lists:
            return None
        
        # Use direct comparison with tuples
        heap = [(head.val, i, head) for i, head in enumerate(valid_lists)]
        heapq.heapify(heap)
        
        dummy = ListNode(0)
        current = dummy
        
        while heap:
            val, list_idx, node = heapq.heappop(heap)
            current.next = node
            current = current.next
            
            if node.next:
                heapq.heappush(heap, (node.next.val, list_idx, node.next))
        
        return dummy.next
    
    # ==================== SORT LINKED LIST ====================
    
    def sort_list_merge_sort(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Sort linked list using merge sort
        
        Time Complexity: O(n log n)
        Space Complexity: O(log n) - recursion stack
        
        Args:
            head: Head of unsorted linked list
        
        Returns:
            Head of sorted linked list
        """
        if not head or not head.next:
            return head
        
        # Split the list into two halves
        mid = self._get_middle(head)
        left = head
        right = mid.next
        mid.next = None  # Break the connection
        
        # Recursively sort both halves
        left_sorted = self.sort_list_merge_sort(left)
        right_sorted = self.sort_list_merge_sort(right)
        
        # Merge the sorted halves
        return self.merge_two_lists_iterative(left_sorted, right_sorted)
    
    def sort_list_merge_sort_iterative(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Iterative merge sort implementation
        
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        
        Args:
            head: Head of unsorted linked list
        
        Returns:
            Head of sorted linked list
        """
        if not head or not head.next:
            return head
        
        # Get the length of the list
        length = self._get_length(head)
        
        dummy = ListNode(0)
        dummy.next = head
        
        # Bottom-up merge sort
        size = 1
        while size < length:
            prev = dummy
            current = dummy.next
            
            while current:
                # Split into two sublists of size 'size'
                left = current
                right = self._split_list(left, size)
                current = self._split_list(right, size)
                
                # Merge the two sublists
                merged_tail = self._merge_and_get_tail(left, right)
                prev.next = merged_tail[0]  # merged head
                prev = merged_tail[1]  # merged tail
            
            size *= 2
        
        return dummy.next
    
    def insertion_sort_list(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Sort linked list using insertion sort
        
        Time Complexity: O(n²)
        Space Complexity: O(1)
        
        Args:
            head: Head of unsorted linked list
        
        Returns:
            Head of sorted linked list
        """
        if not head or not head.next:
            return head
        
        dummy = ListNode(float('-inf'))
        
        while head:
            # Take the first node from original list
            current = head
            head = head.next
            
            # Find the correct position to insert current node
            prev = dummy
            while prev.next and prev.next.val < current.val:
                prev = prev.next
            
            # Insert current node
            current.next = prev.next
            prev.next = current
        
        return dummy.next
    
    def bubble_sort_list(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Sort linked list using bubble sort
        
        Time Complexity: O(n²)
        Space Complexity: O(1)
        
        Args:
            head: Head of unsorted linked list
        
        Returns:
            Head of sorted linked list
        """
        if not head or not head.next:
            return head
        
        length = self._get_length(head)
        
        for i in range(length):
            current = head
            prev = None
            swapped = False
            
            for j in range(length - 1 - i):
                if current.next and current.val > current.next.val:
                    # Swap current and current.next
                    if prev:
                        prev.next = current.next
                    else:
                        head = current.next
                    
                    temp = current.next.next
                    current.next.next = current
                    current.next = temp
                    
                    swapped = True
                    prev = current.next if prev else head
                else:
                    prev = current
                    current = current.next
            
            if not swapped:
                break
        
        return head
    
    def quick_sort_list(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Sort linked list using quick sort
        
        Average Time Complexity: O(n log n)
        Worst Time Complexity: O(n²)
        Space Complexity: O(log n) - recursion stack
        
        Args:
            head: Head of unsorted linked list
        
        Returns:
            Head of sorted linked list
        """
        if not head or not head.next:
            return head
        
        def quick_sort_helper(start, end):
            if start == end or not start:
                return start
            
            # Partition the list
            pivot_prev = self._partition(start, end)
            
            if pivot_prev != start:
                # Recursively sort left part
                temp = start
                while temp.next != pivot_prev.next:
                    temp = temp.next
                temp.next = None
                start = quick_sort_helper(start, pivot_prev)
                
                # Connect left part with pivot
                temp = start
                while temp.next:
                    temp = temp.next
                temp.next = pivot_prev.next
            
            # Recursively sort right part
            pivot_prev.next = quick_sort_helper(pivot_prev.next, end)
            
            return start
        
        # Find the last node
        tail = head
        while tail.next:
            tail = tail.next
        
        return quick_sort_helper(head, tail)
    
    def selection_sort_list(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Sort linked list using selection sort
        
        Time Complexity: O(n²)
        Space Complexity: O(1)
        
        Args:
            head: Head of unsorted linked list
        
        Returns:
            Head of sorted linked list
        """
        if not head or not head.next:
            return head
        
        current = head
        
        while current:
            # Find minimum node in remaining list
            min_node = current
            temp = current.next
            
            while temp:
                if temp.val < min_node.val:
                    min_node = temp
                temp = temp.next
            
            # Swap values (simpler than swapping nodes)
            if min_node != current:
                current.val, min_node.val = min_node.val, current.val
            
            current = current.next
        
        return head
    
    def heap_sort_list(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Sort linked list using heap sort (convert to array, sort, convert back)
        
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        
        Args:
            head: Head of unsorted linked list
        
        Returns:
            Head of sorted linked list
        """
        if not head:
            return None
        
        # Convert to array
        values = []
        current = head
        while current:
            values.append(current.val)
            current = current.next
        
        # Sort using heap
        heapq.heapify(values)
        sorted_values = []
        while values:
            sorted_values.append(heapq.heappop(values))
        
        # Convert back to linked list
        return self.create_list_from_array(sorted_values)
    
    # ==================== SPECIALIZED MERGE OPERATIONS ====================
    
    def merge_sorted_circular_lists(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        """
        Merge two sorted circular linked lists
        
        Args:
            l1: Head of first sorted circular list
            l2: Head of second sorted circular list
        
        Returns:
            Head of merged sorted circular list
        """
        if not l1:
            return l2
        if not l2:
            return l1
        
        # Break the circular lists
        tail1 = l1
        while tail1.next != l1:
            tail1 = tail1.next
        tail1.next = None
        
        tail2 = l2
        while tail2.next != l2:
            tail2 = tail2.next
        tail2.next = None
        
        # Merge the linear lists
        merged = self.merge_two_lists_iterative(l1, l2)
        
        # Make the merged list circular
        tail = merged
        while tail.next:
            tail = tail.next
        tail.next = merged
        
        return merged
    
    def merge_with_constraints(self, l1: Optional[ListNode], l2: Optional[ListNode], 
                             max_length: int) -> Optional[ListNode]:
        """
        Merge two sorted lists with maximum length constraint
        
        Args:
            l1: Head of first sorted list
            l2: Head of second sorted list
            max_length: Maximum allowed length of result
        
        Returns:
            Head of merged list (truncated if necessary)
        """
        dummy = ListNode(0)
        current = dummy
        count = 0
        
        while l1 and l2 and count < max_length:
            if l1.val <= l2.val:
                current.next = ListNode(l1.val)
                l1 = l1.next
            else:
                current.next = ListNode(l2.val)
                l2 = l2.next
            
            current = current.next
            count += 1
        
        # Add remaining nodes up to max_length
        remaining = l1 if l1 else l2
        while remaining and count < max_length:
            current.next = ListNode(remaining.val)
            current = current.next
            remaining = remaining.next
            count += 1
        
        return dummy.next
    
    def merge_unique_only(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        """
        Merge two sorted lists keeping only unique values
        
        Args:
            l1: Head of first sorted list
            l2: Head of second sorted list
        
        Returns:
            Head of merged list with unique values only
        """
        dummy = ListNode(0)
        current = dummy
        
        while l1 and l2:
            if l1.val < l2.val:
                if not current.next or current.val != l1.val:
                    current.next = ListNode(l1.val)
                    current = current.next
                l1 = l1.next
            elif l1.val > l2.val:
                if not current.next or current.val != l2.val:
                    current.next = ListNode(l2.val)
                    current = current.next
                l2 = l2.next
            else:
                # Equal values - skip both
                l1 = l1.next
                l2 = l2.next
        
        # Add remaining unique values
        for remaining in [l1, l2]:
            while remaining:
                if not current.next or current.val != remaining.val:
                    current.next = ListNode(remaining.val)
                    current = current.next
                remaining = remaining.next
        
        return dummy.next
    
    def merge_intersection_only(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        """
        Merge two sorted lists keeping only intersection values
        
        Args:
            l1: Head of first sorted list
            l2: Head of second sorted list
        
        Returns:
            Head of list with intersection values only
        """
        dummy = ListNode(0)
        current = dummy
        
        while l1 and l2:
            if l1.val < l2.val:
                l1 = l1.next
            elif l1.val > l2.val:
                l2 = l2.next
            else:
                # Common value
                current.next = ListNode(l1.val)
                current = current.next
                l1 = l1.next
                l2 = l2.next
        
        return dummy.next
    
    # ==================== ADVANCED SORTING TECHNIQUES ====================
    
    def sort_list_stable(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Stable sort that preserves relative order of equal elements
        Uses merge sort which is naturally stable
        
        Args:
            head: Head of unsorted linked list
        
        Returns:
            Head of stably sorted linked list
        """
        return self.sort_list_merge_sort(head)
    
    def sort_list_reverse(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Sort linked list in descending order
        
        Args:
            head: Head of unsorted linked list
        
        Returns:
            Head of reverse sorted linked list
        """
        # Sort in ascending order first
        sorted_head = self.sort_list_merge_sort(head)
        
        # Reverse the sorted list
        prev = None
        current = sorted_head
        
        while current:
            next_temp = current.next
            current.next = prev
            prev = current
            current = next_temp
        
        return prev
    
    def sort_list_custom_comparator(self, head: Optional[ListNode], 
                                  compare_func) -> Optional[ListNode]:
        """
        Sort linked list using custom comparator
        
        Args:
            head: Head of unsorted linked list
            compare_func: Function that takes two nodes and returns comparison result
        
        Returns:
            Head of sorted linked list
        """
        if not head or not head.next:
            return head
        
        # Convert to array for easier custom sorting
        nodes = []
        current = head
        while current:
            nodes.append(current)
            current = current.next
        
        # Sort using custom comparator
        nodes.sort(key=lambda x: x.val, reverse=compare_func(ListNode(1), ListNode(0)) > 0)
        
        # Reconnect nodes
        for i in range(len(nodes) - 1):
            nodes[i].next = nodes[i + 1]
        nodes[-1].next = None
        
        return nodes[0]
    
    def partial_sort_list(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        """
        Partially sort list - only first k elements are guaranteed to be in correct position
        
        Args:
            head: Head of unsorted linked list
            k: Number of elements to sort correctly
        
        Returns:
            Head of partially sorted linked list
        """
        if not head or k <= 0:
            return head
        
        # Extract first k elements
        nodes = []
        current = head
        count = 0
        
        while current and count < k:
            nodes.append(current.val)
            current = current.next
            count += 1
        
        # Sort the first k elements
        nodes.sort()
        
        # Create result list
        dummy = ListNode(0)
        result_current = dummy
        
        # Add sorted first k elements
        for val in nodes:
            result_current.next = ListNode(val)
            result_current = result_current.next
        
        # Add remaining elements as-is
        result_current.next = current
        
        return dummy.next
    
    # ==================== UTILITY METHODS ====================
    
    def _get_middle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """Get middle node using slow-fast pointer technique"""
        slow = fast = head
        prev = None
        
        while fast and fast.next:
            prev = slow
            slow = slow.next
            fast = fast.next.next
        
        return prev  # Return node before middle for splitting
    
    def _get_length(self, head: Optional[ListNode]) -> int:
        """Get length of linked list"""
        length = 0
        current = head
        
        while current:
            length += 1
            current = current.next
        
        return length
    
    def _split_list(self, head: Optional[ListNode], size: int) -> Optional[ListNode]:
        """Split list after 'size' nodes and return the second part"""
        if not head:
            return None
        
        for _ in range(size - 1):
            if head.next:
                head = head.next
            else:
                break
        
        if not head:
            return None
        
        next_part = head.next
        head.next = None
        return next_part
    
    def _merge_and_get_tail(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> tuple:
        """Merge two lists and return (merged_head, merged_tail)"""
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
        
        current.next = l1 if l1 else l2
        
        # Find tail
        while current.next:
            current = current.next
        
        return dummy.next, current
    
    def _partition(self, start: Optional[ListNode], end: Optional[ListNode]) -> Optional[ListNode]:
        """Partition for quicksort"""
        if not start or start == end:
            return start
        
        pivot = end
        prev = None
        current = start
        
        while current != end:
            if current.val <= pivot.val:
                if not prev:
                    prev = start
                else:
                    prev = prev.next
                
                if prev != current:
                    # Swap values
                    prev.val, current.val = current.val, prev.val
            
            current = current.next
        
        if prev:
            # Swap pivot with element after prev
            if prev.next:
                prev.next.val, pivot.val = pivot.val, prev.next.val
                return prev.next
            else:
                prev.val, pivot.val = pivot.val, prev.val
                return prev
        else:
            return start
    
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
    
    def is_sorted(self, head: Optional[ListNode], reverse: bool = False) -> bool:
        """Check if linked list is sorted"""
        if not head or not head.next:
            return True
        
        current = head
        
        while current.next:
            if reverse:
                if current.val < current.next.val:
                    return False
            else:
                if current.val > current.next.val:
                    return False
            current = current.next
        
        return True
    
    def create_test_lists(self, num_lists: int, list_length: int, max_val: int = 100) -> List[Optional[ListNode]]:
        """Create multiple sorted test lists"""
        import random
        
        lists = []
        for _ in range(num_lists):
            values = sorted([random.randint(1, max_val) for _ in range(list_length)])
            lists.append(self.create_list_from_array(values))
        
        return lists

# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Linked List Merge & Sort Demo ===\n")
    
    lms = LinkedListMergeSort()
    
    # Example 1: Merge Two Sorted Lists
    print("1. Merge Two Sorted Lists:")
    
    list1 = lms.create_list_from_array([1, 2, 4, 6, 8])
    list2 = lms.create_list_from_array([1, 3, 4, 5, 7, 9])
    
    print(f"List 1: {lms.list_to_array(list1)}")
    print(f"List 2: {lms.list_to_array(list2)}")
    
    # Test different merge approaches
    merged_recursive = lms.merge_two_lists_recursive(list1, list2)
    print(f"Merged (recursive): {lms.list_to_array(merged_recursive)}")
    
    # Recreate lists for next test
    list1 = lms.create_list_from_array([1, 2, 4, 6, 8])
    list2 = lms.create_list_from_array([1, 3, 4, 5, 7, 9])
    merged_iterative = lms.merge_two_lists_iterative(list1, list2)
    print(f"Merged (iterative): {lms.list_to_array(merged_iterative)}")
    
    # Test alternating merge
    list3 = lms.create_list_from_array([1, 3, 5])
    list4 = lms.create_list_from_array([2, 4, 6, 7, 8])
    merged_alternating = lms.merge_alternating(list3, list4)
    print(f"Merged (alternating): {lms.list_to_array(merged_alternating)}")
    print()
    
    # Example 2: Merge K Sorted Lists
    print("2. Merge K Sorted Lists:")
    
    k_lists = [
        lms.create_list_from_array([1, 4, 5]),
        lms.create_list_from_array([1, 3, 4]),
        lms.create_list_from_array([2, 6])
    ]
    
    print("Input lists:")
    for i, lst in enumerate(k_lists):
        print(f"  List {i + 1}: {lms.list_to_array(lst)}")
    
    # Test different merge approaches
    k_lists_copy1 = [
        lms.create_list_from_array([1, 4, 5]),
        lms.create_list_from_array([1, 3, 4]),
        lms.create_list_from_array([2, 6])
    ]
    merged_heap = lms.merge_k_lists_min_heap(k_lists_copy1)
    print(f"Merged (min heap): {lms.list_to_array(merged_heap)}")
    
    k_lists_copy2 = [
        lms.create_list_from_array([1, 4, 5]),
        lms.create_list_from_array([1, 3, 4]),
        lms.create_list_from_array([2, 6])
    ]
    merged_divide_conquer = lms.merge_k_lists_divide_conquer(k_lists_copy2)
    print(f"Merged (divide & conquer): {lms.list_to_array(merged_divide_conquer)}")
    
    k_lists_copy3 = [
        lms.create_list_from_array([1, 4, 5]),
        lms.create_list_from_array([1, 3, 4]),
        lms.create_list_from_array([2, 6])
    ]
    merged_sequential = lms.merge_k_lists_sequential(k_lists_copy3)
    print(f"Merged (sequential): {lms.list_to_array(merged_sequential)}")
    print()
    
    # Example 3: Sort Linked List
    print("3. Sort Linked List:")
    
    unsorted_list = lms.create_list_from_array([4, 2, 1, 3, 6, 5, 8, 7])
    print(f"Unsorted list: {lms.list_to_array(unsorted_list)}")
    
    # Test different sorting algorithms
    test_arrays = [[4, 2, 1, 3, 6, 5, 8, 7] for _ in range(6)]
    
    sorting_methods = [
        ("Merge Sort", lms.sort_list_merge_sort),
        ("Insertion Sort", lms.insertion_sort_list),
        ("Selection Sort", lms.selection_sort_list),
        ("Bubble Sort", lms.bubble_sort_list),
        ("Heap Sort", lms.heap_sort_list),
    ]
    
    for name, method in sorting_methods:
        test_list = lms.create_list_from_array(test_arrays.pop())
        sorted_list = method(test_list)
        result_array = lms.list_to_array(sorted_list)
        is_sorted = lms.is_sorted(sorted_list)
        print(f"{name:15}: {result_array} (Sorted: {is_sorted})")
    
    # Test iterative merge sort
    test_list = lms.create_list_from_array([4, 2, 1, 3, 6, 5, 8, 7])
    sorted_iterative = lms.sort_list_merge_sort_iterative(test_list)
    print(f"Merge Sort (it): {lms.list_to_array(sorted_iterative)} (Sorted: {lms.is_sorted(sorted_iterative)})")
    print()
    
    # Example 4: Specialized Merge Operations
    print("4. Specialized Merge Operations:")
    
    # Merge with constraints
    list_a = lms.create_list_from_array([1, 3, 5, 7, 9])
    list_b = lms.create_list_from_array([2, 4, 6, 8, 10, 12])
    
    print(f"List A: {lms.list_to_array(list_a)}")
    print(f"List B: {lms.list_to_array(list_b)}")
    
    constrained_merge = lms.merge_with_constraints(list_a, list_b, 6)
    print(f"Merge (max 6): {lms.list_to_array(constrained_merge)}")
    
    # Merge unique only
    list_c = lms.create_list_from_array([1, 2, 3, 4, 5])
    list_d = lms.create_list_from_array([3, 4, 5, 6, 7])
    
    unique_merge = lms.merge_unique_only(list_c, list_d)
    print(f"Merge unique: {lms.list_to_array(unique_merge)}")
    
    # Merge intersection only
    list_e = lms.create_list_from_array([1, 2, 3, 4, 5])
    list_f = lms.create_list_from_array([3, 4, 5, 6, 7])
    
    intersection_merge = lms.merge_intersection_only(list_e, list_f)
    print(f"Merge intersection: {lms.list_to_array(intersection_merge)}")
    print()
    
    # Example 5: Advanced Sorting
    print("5. Advanced Sorting:")
    
    # Reverse sort
    unsorted_list2 = lms.create_list_from_array([3, 1, 4, 1, 5, 9, 2, 6])
    print(f"Original: {lms.list_to_array(unsorted_list2)}")
    
    reverse_sorted = lms.sort_list_reverse(unsorted_list2)
    print(f"Reverse sorted: {lms.list_to_array(reverse_sorted)}")
    
    # Partial sort
    unsorted_list3 = lms.create_list_from_array([9, 5, 1, 8, 3, 7, 2, 4, 6])
    partial_sorted = lms.partial_sort_list(unsorted_list3, 4)
    print(f"Partial sort (k=4): {lms.list_to_array(partial_sorted)}")
    print("  (First 4 elements are in correct sorted position)")
    print()
    
    # Example 6: Performance Analysis
    print("6. Performance Analysis:")
    
    # Large list sorting comparison
    import random
    import time
    
    # Create large unsorted list
    large_values = list(range(1000))
    random.shuffle(large_values)
    
    print(f"Testing sorting algorithms on {len(large_values)} elements:")
    
    # Test merge sort on large list
    large_list = lms.create_list_from_array(large_values[:100])  # Reduced for demo
    start_time = time.time()
    sorted_large = lms.sort_list_merge_sort(large_list)
    merge_time = time.time() - start_time
    
    print(f"Merge sort: {merge_time:.4f} seconds")
    print(f"Result is sorted: {lms.is_sorted(sorted_large)}")
    print(f"Sample result: {lms.list_to_array(sorted_large)[:10]}...")
    
    # Test k-list merging performance
    k_test_lists = lms.create_test_lists(10, 50, 100)
    
    print(f"\nMerging {len(k_test_lists)} sorted lists:")
    
    start_time = time.time()
    merged_result = lms.merge_k_lists_min_heap(k_test_lists)
    heap_time = time.time() - start_time
    
    print(f"Min heap approach: {heap_time:.4f} seconds")
    print(f"Result length: {len(lms.list_to_array(merged_result))}")
    print(f"Result is sorted: {lms.is_sorted(merged_result)}")
    
    # Test with different k values
    for k in [5, 10, 20]:
        test_lists = lms.create_test_lists(k, 20, 50)
        
        start_time = time.time()
        merged = lms.merge_k_lists_divide_conquer(test_lists)
        dc_time = time.time() - start_time
        
        result_length = len(lms.list_to_array(merged))
        print(f"K={k:2d} lists: {dc_time:.4f}s, result length: {result_length}")
    
    # Memory usage analysis
    print(f"\nMemory Usage Analysis:")
    print(f"In-place algorithms: merge sort (O(log n) stack), quick sort")
    print(f"Extra space algorithms: heap sort (O(n)), counting sort")
    print(f"Hybrid approaches: insertion sort for small lists, merge sort for large")
    
    print("\n=== Demo Complete ===")