from TreeTraversal import TraversalMethod
from TreeProperties import TreeProperties
from TreeOperation import TreeOperation

class TreeNode():
    def __init__(self, val=0):
        self.val = val
        self.left = None
        self.right = None

class NArrTree():
    def __init__(self, val=0, children=None):
        self.val = val
        self.children = children if children is not None else []

class TrieNode():
    def __init__(self, val=0):
        self.data = val
        self.children = [None] * 26

def CreateBinaryTree():
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    root.right.left = TreeNode(6)
    root.right.right = TreeNode(7)
    return root

def buildNarrTree():
    root = NArrTree(1)
    child1 = NArrTree(2)
    child2 = NArrTree(3)
    child3 = NArrTree(4)
    root.children = [child1, child2, child3]
    child1.children = [NArrTree(5), NArrTree(7), NArrTree(8)]
    child2.children = [NArrTree(6)]
    child3.children = [NArrTree(9), NArrTree(10)]
    return root

def main():
    bt1 = CreateBinaryTree()
    bt2 = bt1.left
    narrTree = buildNarrTree()
    TestTraversal = False
    TestProperties = True
    TestOperation = True
    
    if TestTraversal:
        traversal = TraversalMethod()
        print("\nLevel Traversal of Binary Tree:")
        traversal.LevelOrderTraversal(bt1)
        print("\nPreOrder Traversal of Binary Tree:")
        traversal.PreOrderRecursive(bt1)
        print("\nInorder Traversal of Binary Tree:")
        traversal.InorderRecursive(bt1)
        print("\nPostOrder Traversal of Binary Tree:")
        traversal.PostOrderRecursive(bt1)
    
    if TestProperties:
        properties = TreeProperties()
        print("Number of node in the given Tree: ", end = ' ')
        print(properties.CountOfNode(bt1))
        print("Height of the given Tree:", end= ' ')
        print(properties.height_of_tree(bt1))
        print("Diameter of the given Tree:", end= ' ')
        print(properties.diameter_of_tree(bt1))
        print("Is Both Tree same:", end= ' ')
        print(properties.isIdentical(bt1, bt2))
        print("Is given tree is mirror tree:", end= ' ')
        print(properties.isIdentical(bt1.left, bt1.right))
        print("Is given tree is height balanced:", end= ' ')
        print(properties.isHeightBalanced(bt1))
        print("Is leaf sum is equal to root for all Node:", end= ' ')
        print(properties.isSumProperty(bt1))
    
    if TestOperation:
        operations = TreeOperation()
        print("Sum of the given Tree: ", end = ' ')
        print(operations.sum_of_tree(bt1))
        print("Sum of leaf node of the given Tree: ", end = ' ')
        print(operations.sum_of_leaf_nodes(bt1))

if __name__ == "__main__":
    main()
    