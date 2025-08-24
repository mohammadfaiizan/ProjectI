"""
353. Design Snake Game - Multiple Approaches
Difficulty: Medium

Design a Snake game that is played on a device with screen size height x width. Play the game online if you are not familiar with the game.

The snake is initially positioned at the top left corner (0, 0) with a length of 1 unit.

You are given an array of food's positions. When a snake eats food, its length and the game's score both increase by 1.

Implement the SnakeGame class:
- SnakeGame(int width, int height, int[][] food) Initializes the object with a screen of size height x width and the food positions.
- int move(String direction) Returns the game's score after applying one move by the snake in the given direction. Return -1 if the game over (the snake hits the boundary or itself).
"""

from typing import List, Tuple
from collections import deque

class SnakeGameDeque:
    """
    Approach 1: Deque Implementation (Optimal)
    
    Use deque to efficiently manage snake body.
    
    Time: O(1) for move, Space: O(snake_length)
    """
    
    def __init__(self, width: int, height: int, food: List[List[int]]):
        self.width = width
        self.height = height
        self.food = deque(food)
        self.score = 0
        
        # Snake body represented as deque of (row, col) positions
        self.snake = deque([(0, 0)])
        self.snake_set = {(0, 0)}  # For O(1) collision detection
        
        # Direction mappings
        self.directions = {
            'U': (-1, 0),
            'D': (1, 0),
            'L': (0, -1),
            'R': (0, 1)
        }
    
    def move(self, direction: str) -> int:
        """Move snake in given direction"""
        if not self.snake:
            return -1
        
        # Get current head position
        head_row, head_col = self.snake[0]
        
        # Calculate new head position
        dr, dc = self.directions[direction]
        new_head = (head_row + dr, head_col + dc)
        new_row, new_col = new_head
        
        # Check boundary collision
        if new_row < 0 or new_row >= self.height or new_col < 0 or new_col >= self.width:
            return -1
        
        # Check if food is eaten
        food_eaten = False
        if self.food and self.food[0] == [new_row, new_col]:
            self.food.popleft()
            self.score += 1
            food_eaten = True
        
        # Check self collision (excluding tail if no food eaten)
        if not food_eaten:
            # Remove tail from set before checking collision
            tail = self.snake.pop()
            self.snake_set.remove(tail)
        
        if new_head in self.snake_set:
            return -1
        
        # Add new head
        self.snake.appendleft(new_head)
        self.snake_set.add(new_head)
        
        return self.score


class SnakeGameList:
    """
    Approach 2: List Implementation
    
    Use list to manage snake body.
    
    Time: O(n) for move, Space: O(snake_length)
    """
    
    def __init__(self, width: int, height: int, food: List[List[int]]):
        self.width = width
        self.height = height
        self.food = food
        self.food_index = 0
        self.score = 0
        
        # Snake body as list of [row, col] positions
        self.snake = [[0, 0]]
        
        self.directions = {
            'U': [-1, 0],
            'D': [1, 0],
            'L': [0, -1],
            'R': [0, 1]
        }
    
    def move(self, direction: str) -> int:
        """Move snake in given direction"""
        if not self.snake:
            return -1
        
        # Get current head position
        head = self.snake[0][:]
        
        # Calculate new head position
        dr, dc = self.directions[direction]
        new_head = [head[0] + dr, head[1] + dc]
        
        # Check boundary collision
        if (new_head[0] < 0 or new_head[0] >= self.height or 
            new_head[1] < 0 or new_head[1] >= self.width):
            return -1
        
        # Check if food is eaten
        food_eaten = False
        if (self.food_index < len(self.food) and 
            self.food[self.food_index] == new_head):
            self.food_index += 1
            self.score += 1
            food_eaten = True
        
        # Remove tail if no food eaten
        if not food_eaten:
            self.snake.pop()
        
        # Check self collision
        if new_head in self.snake:
            return -1
        
        # Add new head
        self.snake.insert(0, new_head)
        
        return self.score


class SnakeGameHashSet:
    """
    Approach 3: Hash Set with List
    
    Use hash set for collision detection and list for body.
    
    Time: O(1) for move, Space: O(snake_length)
    """
    
    def __init__(self, width: int, height: int, food: List[List[int]]):
        self.width = width
        self.height = height
        self.food = food
        self.food_index = 0
        self.score = 0
        
        # Snake body and position set
        self.snake = [(0, 0)]
        self.positions = {(0, 0)}
        
        self.directions = {
            'U': (-1, 0),
            'D': (1, 0),
            'L': (0, -1),
            'R': (0, 1)
        }
    
    def move(self, direction: str) -> int:
        """Move snake in given direction"""
        if not self.snake:
            return -1
        
        # Get current head position
        head_row, head_col = self.snake[0]
        
        # Calculate new head position
        dr, dc = self.directions[direction]
        new_head = (head_row + dr, head_col + dc)
        
        # Check boundary collision
        if (new_head[0] < 0 or new_head[0] >= self.height or 
            new_head[1] < 0 or new_head[1] >= self.width):
            return -1
        
        # Check if food is eaten
        food_eaten = False
        if (self.food_index < len(self.food) and 
            self.food[self.food_index] == [new_head[0], new_head[1]]):
            self.food_index += 1
            self.score += 1
            food_eaten = True
        
        # Remove tail if no food eaten
        if not food_eaten:
            tail = self.snake.pop()
            self.positions.remove(tail)
        
        # Check self collision
        if new_head in self.positions:
            return -1
        
        # Add new head
        self.snake.insert(0, new_head)
        self.positions.add(new_head)
        
        return self.score


class SnakeGameMatrix:
    """
    Approach 4: Matrix-based Implementation
    
    Use 2D matrix to track snake positions.
    
    Time: O(1) for move, Space: O(width * height)
    """
    
    def __init__(self, width: int, height: int, food: List[List[int]]):
        self.width = width
        self.height = height
        self.food = food
        self.food_index = 0
        self.score = 0
        
        # Matrix to track snake body
        self.grid = [[0] * width for _ in range(height)]
        self.grid[0][0] = 1  # Snake head
        
        # Snake body positions
        self.snake = [(0, 0)]
        
        self.directions = {
            'U': (-1, 0),
            'D': (1, 0),
            'L': (0, -1),
            'R': (0, 1)
        }
    
    def move(self, direction: str) -> int:
        """Move snake in given direction"""
        if not self.snake:
            return -1
        
        # Get current head position
        head_row, head_col = self.snake[0]
        
        # Calculate new head position
        dr, dc = self.directions[direction]
        new_row, new_col = head_row + dr, head_col + dc
        
        # Check boundary collision
        if new_row < 0 or new_row >= self.height or new_col < 0 or new_col >= self.width:
            return -1
        
        # Check if food is eaten
        food_eaten = False
        if (self.food_index < len(self.food) and 
            self.food[self.food_index] == [new_row, new_col]):
            self.food_index += 1
            self.score += 1
            food_eaten = True
        
        # Remove tail if no food eaten
        if not food_eaten:
            tail_row, tail_col = self.snake.pop()
            self.grid[tail_row][tail_col] = 0
        
        # Check self collision
        if self.grid[new_row][new_col] == 1:
            return -1
        
        # Add new head
        self.snake.insert(0, (new_row, new_col))
        self.grid[new_row][new_col] = 1
        
        return self.score


def test_snake_game_implementations():
    """Test snake game implementations"""
    
    implementations = [
        ("Deque", SnakeGameDeque),
        ("List", SnakeGameList),
        ("HashSet", SnakeGameHashSet),
        ("Matrix", SnakeGameMatrix),
    ]
    
    test_cases = [
        {
            "width": 3,
            "height": 2,
            "food": [[1,2],[0,1]],
            "moves": ["R","D","R","U","L","U"],
            "expected": [0,0,1,1,2,-1],
            "description": "Example 1"
        },
        {
            "width": 3,
            "height": 3,
            "food": [[2,0],[0,0],[0,2],[2,2]],
            "moves": ["D","D","R","U","U","L","D","R","R","U","L","D"],
            "expected": [0,1,1,1,1,2,2,2,2,3,3,4],
            "description": "Complex path"
        },
        {
            "width": 2,
            "height": 2,
            "food": [],
            "moves": ["R","D","L","U","R"],
            "expected": [0,0,0,0,-1],
            "description": "Self collision"
        },
    ]
    
    print("=== Testing Snake Game Implementations ===")
    
    for impl_name, impl_class in implementations:
        print(f"\n--- {impl_name} Implementation ---")
        
        for test_case in test_cases:
            try:
                game = impl_class(
                    test_case["width"],
                    test_case["height"],
                    test_case["food"]
                )
                
                results = []
                for move in test_case["moves"]:
                    result = game.move(move)
                    results.append(result)
                
                expected = test_case["expected"]
                status = "✓" if results == expected else "✗"
                
                print(f"  {test_case['description']:15} | {status} | {results}")
                if results != expected:
                    print(f"    Expected: {expected}")
                
            except Exception as e:
                print(f"  {test_case['description']:15} | ERROR: {str(e)[:40]}")


def demonstrate_snake_game():
    """Demonstrate snake game step by step"""
    print("\n=== Snake Game Step-by-Step Demo ===")
    
    width, height = 3, 2
    food = [[1,2],[0,1]]
    moves = ["R","D","R","U","L","U"]
    
    print(f"Game setup: {width}x{height} grid")
    print(f"Food positions: {food}")
    print(f"Moves: {moves}")
    
    game = SnakeGameDeque(width, height, food)
    
    print(f"\nInitial state:")
    print(f"  Snake: {list(game.snake)}")
    print(f"  Score: {game.score}")
    
    for i, move in enumerate(moves):
        print(f"\nMove {i+1}: {move}")
        
        result = game.move(move)
        
        if result == -1:
            print(f"  Game Over!")
            break
        else:
            print(f"  Snake: {list(game.snake)}")
            print(f"  Score: {result}")
            print(f"  Food remaining: {list(game.food)}")


def visualize_snake_game():
    """Visualize snake game on grid"""
    print("\n=== Snake Game Visualization ===")
    
    width, height = 4, 3
    food = [[1,1],[2,2]]
    
    game = SnakeGameDeque(width, height, food)
    
    def print_grid():
        """Print current game state"""
        grid = [['.' for _ in range(width)] for _ in range(height)]
        
        # Mark food
        for f in game.food:
            if 0 <= f[0] < height and 0 <= f[1] < width:
                grid[f[0]][f[1]] = 'F'
        
        # Mark snake
        for i, (r, c) in enumerate(game.snake):
            if i == 0:
                grid[r][c] = 'H'  # Head
            else:
                grid[r][c] = 'S'  # Body
        
        print("  Grid:")
        for row in grid:
            print("    " + " ".join(row))
        print(f"  Score: {game.score}")
    
    print("Initial state:")
    print_grid()
    
    moves = ["R", "R", "D", "L", "L", "D", "R", "R"]
    
    for i, move in enumerate(moves):
        print(f"\nAfter move {move}:")
        result = game.move(move)
        
        if result == -1:
            print("  Game Over!")
            break
        
        print_grid()


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Robot path planning
    print("1. Robot Path Planning:")
    print("  - Snake body represents robot's path history")
    print("  - Collision detection prevents revisiting locations")
    print("  - Food represents target waypoints")
    
    # Application 2: Network routing
    print("\n2. Network Packet Routing:")
    print("  - Snake represents packet path through network")
    print("  - Body prevents routing loops")
    print("  - Food represents intermediate destinations")
    
    # Application 3: Resource allocation
    print("\n3. Dynamic Resource Allocation:")
    print("  - Snake represents active resource chain")
    print("  - Growth represents resource acquisition")
    print("  - Collision represents resource conflicts")


def analyze_performance():
    """Analyze performance of different implementations"""
    print("\n=== Performance Analysis ===")
    
    approaches = [
        ("Deque", "O(1)", "O(n)", "Optimal for queue operations"),
        ("List", "O(n)", "O(n)", "Insertion/deletion at head is O(n)"),
        ("HashSet + List", "O(1)", "O(n)", "Fast collision detection"),
        ("Matrix", "O(1)", "O(w*h)", "Fast access, high space usage"),
    ]
    
    print(f"{'Approach':<15} | {'Time':<8} | {'Space':<10} | {'Notes'}")
    print("-" * 60)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<15} | {time_comp:<8} | {space_comp:<10} | {notes}")
    
    print(f"\nwhere n = snake length, w = width, h = height")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    edge_cases = [
        {
            "name": "Immediate boundary hit",
            "width": 2, "height": 2, "food": [],
            "moves": ["L"], "expected": [-1]
        },
        {
            "name": "Eat all food",
            "width": 3, "height": 1, "food": [[0,1],[0,2]],
            "moves": ["R","R"], "expected": [1,2]
        },
        {
            "name": "No food available",
            "width": 3, "height": 3, "food": [],
            "moves": ["R","D","L","U"], "expected": [0,0,0,0]
        },
        {
            "name": "Self collision after growth",
            "width": 3, "height": 3, "food": [[0,1]],
            "moves": ["R","D","L","U"], "expected": [1,1,1,-1]
        },
    ]
    
    for case in edge_cases:
        try:
            game = SnakeGameDeque(case["width"], case["height"], case["food"])
            results = []
            
            for move in case["moves"]:
                result = game.move(move)
                results.append(result)
            
            status = "✓" if results == case["expected"] else "✗"
            print(f"{case['name']:25} | {status} | {results}")
            
        except Exception as e:
            print(f"{case['name']:25} | ERROR: {str(e)[:30]}")


if __name__ == "__main__":
    test_snake_game_implementations()
    demonstrate_snake_game()
    visualize_snake_game()
    demonstrate_real_world_applications()
    analyze_performance()
    test_edge_cases()

"""
Design Snake Game demonstrates advanced queue applications for game
development and simulation, including multiple data structure approaches
for efficient collision detection and dynamic body management.
"""
