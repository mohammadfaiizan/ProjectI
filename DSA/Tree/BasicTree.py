class TreeNode:
    """
    A class representing a node in a binary tree.
    """
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


class Tree:
    """
    A class representing a binary tree with different initialization and traversal methods.
    """

    def __init__(self, root=None):
        self.root = root

    # Method 1: Initialize a tree using a list of values (Level Order)
    @classmethod
    def from_list(cls, values):
        """Create a binary tree from a list using level-order traversal."""
        if not values:
            return cls()

        nodes = [TreeNode(val) if val is not None else None for val in values]
        for i in range(len(nodes)):
            if nodes[i] is not None:
                left_index = 2 * i + 1
                right_index = 2 * i + 2
                if left_index < len(nodes):
                    nodes[i].left = nodes[left_index]
                if right_index < len(nodes):
                    nodes[i].right = nodes[right_index]
        return cls(nodes[0])

    # Method 2: Manually construct a tree
    @classmethod
    def manual(cls):
        """Manually create a binary tree."""
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)
        root.left.left = TreeNode(4)
        root.left.right = TreeNode(5)
        root.right.left = TreeNode(6)
        root.right.right = TreeNode(7)
        return cls(root)

    # Traversal Methods
    def inorder(self, node=None, result=None):
        """Inorder Traversal: Left -> Root -> Right"""
        if result is None:
            result = []
        if node:
            self.inorder(node.left, result)
            result.append(node.value)
            self.inorder(node.right, result)
        return result

    def preorder(self, node=None, result=None):
        """Preorder Traversal: Root -> Left -> Right"""
        if result is None:
            result = []
        if node:
            result.append(node.value)
            self.preorder(node.left, result)
            self.preorder(node.right, result)
        return result

    def postorder(self, node=None, result=None):
        """Postorder Traversal: Left -> Right -> Root"""
        if result is None:
            result = []
        if node:
            self.postorder(node.left, result)
            self.postorder(node.right, result)
            result.append(node.value)
        return result

    def level_order(self):
        """Level Order Traversal: Breadth-First Search"""
        if not self.root:
            return []

        result = []
        queue = [self.root]

        while queue:
            node = queue.pop(0)
            if node:
                result.append(node.value)
                queue.append(node.left)
                queue.append(node.right)

        return result

# Example Usage
# Initialize a tree from a list
tree_from_list = Tree.from_list([1, 2, 3, 4, 5, 6, 7])
print("Inorder Traversal from list:", tree_from_list.inorder(tree_from_list.root))
print("Preorder Traversal from list:", tree_from_list.preorder(tree_from_list.root))
print("Postorder Traversal from list:", tree_from_list.postorder(tree_from_list.root))
print("Level Order Traversal from list:", tree_from_list.level_order())

# Manually initialize a tree
tree_manual = Tree.manual()
print("\nInorder Traversal from manual tree:", tree_manual.inorder(tree_manual.root))
print("Preorder Traversal from manual tree:", tree_manual.preorder(tree_manual.root))
print("Postorder Traversal from manual tree:", tree_manual.postorder(tree_manual.root))
print("Level Order Traversal from manual tree:", tree_manual.level_order())
