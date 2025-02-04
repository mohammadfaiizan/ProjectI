from collections import deque

class TraversalMethod():
    def __init__(self):
        pass
    
    def PreOrderRecursive(self, root):
        if root is None:
            return
        print(root.val, end = ' ')
        self.PreOrderRecursive(root.left)
        self.PreOrderRecursive(root.right)
    
    def InorderRecursive(self, root):
        if root is None:
            return
        self.InorderRecursive(root.left)
        print(root.val, end= ' ')
        self.InorderRecursive(root.right)
    
    def PostOrderRecursive(self, root):
        if root is None:
            return
        self.PostOrderRecursive(root.left)
        self.PostOrderRecursive(root.right)
        print(root.val, end = ' ')
    
    def LevelOrderTraversal(self, root):
        if root is None:
            return []
        q = deque()
        q.append(root)
        while q:
            node = q.popleft()
            print(node.val , end = ' ')
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
