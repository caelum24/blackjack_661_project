
class TreeNode():

    def __init__(self, state, parent=None):
        self._parent = parent
        self.state = state
        self.left = None
        self.right = None
    
    def add_left_child(self, state):
        self.left = TreeNode(state=state, parent=self)
        return self.left

    def add_right_child(self, state):
        self.right = TreeNode(state=state, parent=self)
        return self.right
    
    def is_full(self):
        return self.left is not None and self.right is not None
    
    def is_empty(self):
        return self.left is None and self.right is None

    @property
    def parent(self):
        return self._parent
    


class SplitTracker():

    def __init__(self, state) -> None:
        self.root: TreeNode = TreeNode(state=state)

        # the current node to be looked at
        self.current: TreeNode = self.root

    def split(self, state):
        # add a left child and go to that child
        self.current = self.current.add_left_child(state)

    def switch_hand(self, state):

        # traverse up until you find a parent without a right child
        while self.current.is_full() or self.current.is_empty():
            self.current = self.current.parent
        
        # add a right child to the parent (second split result state) and go to it
        self.current = self.current.add_right_child(state)
        
    def get_split_next_hands(self):
        split_next_hands = []

        def traverse(root:TreeNode):
            # print(type(root.left), type(root.right), type(root.left) == type(root.right))
            if type(root.left) != type(root.right):
                raise ValueError("Tree is incomplete... node either has 0 or 2 children, not 1")

            if root.left is None or root.right is None:
                return

            split_next_hands.append((root.state, (root.left.state, root.right.state)))
            
            traverse(root.left)
            traverse(root.right)
        
        traverse(self.root)

        return split_next_hands

if __name__ == "__main__":

    tracker = SplitTracker(1)
    tracker.split(2)
    tracker.split(3)
    tracker.switch_hand(4)
    tracker.split(5)
    tracker.switch_hand(6)
    tracker.switch_hand(7)
    print(tracker.get_split_next_hands())






