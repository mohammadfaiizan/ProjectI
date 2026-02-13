# Basic Linked List Operations

## Node Class

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
```

## Create Empty List

```python
def create_empty_list():
    return None

def create_list_with_head(head):
    return head
```

## Insert at Head

```python
def insert_at_head(head, data):
    new_node = Node(data)
    new_node.next = head
    return new_node
```

## Insert at Tail

```python
def insert_at_tail(head, data):
    new_node = Node(data)
    if head is None:
        return new_node
    curr = head
    while curr.next is not None:
        curr = curr.next
    curr.next = new_node
    return head
```

## Insert at Position k

```python
def insert_at_position(head, data, k):
    if k <= 0:
        return insert_at_head(head, data)
    new_node = Node(data)
    curr = head
    for _ in range(k - 1):
        if curr is None:
            return head
        curr = curr.next
    if curr is None:
        return head
    new_node.next = curr.next
    curr.next = new_node
    return head
```

## Insert After a Given Node

```python
def insert_after_node(prev_node, data):
    if prev_node is None:
        return
    new_node = Node(data)
    new_node.next = prev_node.next
    prev_node.next = new_node
```

## Insert in Sorted List

```python
def insert_in_sorted(head, data):
    new_node = Node(data)
    if head is None or head.data >= data:
        new_node.next = head
        return new_node
    curr = head
    while curr.next is not None and curr.next.data < data:
        curr = curr.next
    new_node.next = curr.next
    curr.next = new_node
    return head
```

## Delete Head Node

```python
def delete_head(head):
    if head is None:
        return None
    return head.next
```

## Delete Tail Node

```python
def delete_tail(head):
    if head is None or head.next is None:
        return None
    curr = head
    while curr.next.next is not None:
        curr = curr.next
    curr.next = None
    return head
```

## Delete Node at Position k

```python
def delete_at_position(head, k):
    if head is None:
        return None
    if k <= 0:
        return head.next
    curr = head
    for _ in range(k - 1):
        if curr is None or curr.next is None:
            return head
        curr = curr.next
    if curr.next is not None:
        curr.next = curr.next.next
    return head
```

## Delete by Value (First Occurrence)

```python
def delete_by_value(head, value):
    if head is None:
        return None
    if head.data == value:
        return head.next
    curr = head
    while curr.next is not None:
        if curr.next.data == value:
            curr.next = curr.next.next
            return head
        curr = curr.next
    return head
```

## Delete All Occurrences

```python
def delete_all_occurrences(head, value):
    while head is not None and head.data == value:
        head = head.next
    if head is None:
        return None
    curr = head
    while curr.next is not None:
        if curr.next.data == value:
            curr.next = curr.next.next
        else:
            curr = curr.next
    return head
```

## Search for Value (Iterative)

```python
def search_iterative(head, value):
    curr = head
    while curr is not None:
        if curr.data == value:
            return True
        curr = curr.next
    return False
```

## Search for Value (Recursive)

```python
def search_recursive(head, value):
    if head is None:
        return False
    if head.data == value:
        return True
    return search_recursive(head.next, value)
```

## Traverse and Print

```python
def traverse_print(head):
    curr = head
    while curr is not None:
        print(curr.data, end=" ")
        curr = curr.next
    print()
```

## Count Nodes (Iterative)

```python
def count_nodes_iterative(head):
    count = 0
    curr = head
    while curr is not None:
        count += 1
        curr = curr.next
    return count
```

## Count Nodes (Recursive)

```python
def count_nodes_recursive(head):
    if head is None:
        return 0
    return 1 + count_nodes_recursive(head.next)
```

## Check if Empty

```python
def is_empty(head):
    return head is None
```

## Find Nth Node from Beginning

```python
def find_nth_from_beginning(head, n):
    if n < 0:
        return None
    curr = head
    for _ in range(n):
        if curr is None:
            return None
        curr = curr.next
    return curr
```

## Find Last Node

```python
def find_last_node(head):
    if head is None:
        return None
    curr = head
    while curr.next is not None:
        curr = curr.next
    return curr
```

## Clear or Destroy Entire List

```python
def clear_list(head):
    while head is not None:
        temp = head
        head = head.next
        del temp
    return None

def clear_list_recursive(head):
    if head is None:
        return None
    clear_list_recursive(head.next)
    del head
    return None
```
