"""
353. Design Snake Game - Multiple Approaches
Difficulty: Medium

Design a Snake game that is played on a device with screen size height x width. The snake is initially positioned at the top left corner (0, 0) with a length of 1 unit.

You are given an array of food's positions. When a snake eats food, its length and the game's score both increase by 1.

Implement the SnakeGame class:
- SnakeGame(int width, int height, int[][] food) Initializes the object with a screen of size height x width and the food placed at food.
- int move(String direction) Returns the game's score after applying one move by the snake in the given direction. If the game ends, return -1.
"""

from typing import List, Tuple, Deque
from collections import deque

class SnakeGameBasic:
    """
    Approach 1: Basic Implementation with List
    
    Use list to track snake body and simple collision detection.
    
    Time Complexity:
    - __init__: O(1)
    - move: O(n) where n is snake length (for collision check)
    
    Space Complexity: O(n + f) where f is food count
    """
    
    def __init__(self, width: int, height: int, food: List[List[int]]):
        self.width = width
        self.height = height
        self.food = deque(food)
        
        # Snake starts at (0, 0)
        self.snake = [(0, 0)]
        self.score = 0
        
        # Direction mappings
        self.directions = {
            'U': (-1, 0),
            'D': (1, 0),
            'L': (0, -1),
            'R': (0, 1)
        }
    
    def move(self, direction: str) -> int:
        # Get direction delta
        dr, dc = self.directions[direction]
        
        # Get current head position
        head_r, head_c = self.snake[0]
        
        # Calculate new head position
        new_head = (head_r + dr, head_c + dc)
        new_r, new_c = new_head
        
        # Check boundary collision
        if new_r < 0 or new_r >= self.height or new_c < 0 or new_c >= self.width:
            return -1
        
        # Check if food is eaten
        food_eaten = False
        if self.food and self.food[0] == [new_r, new_c]:
            food_eaten = True
            self.food.popleft()
            self.score += 1
        
        # Add new head
        self.snake.insert(0, new_head)
        
        # Remove tail if no food eaten
        if not food_eaten:
            self.snake.pop()
        
        # Check self collision (excluding head which was just added)
        if new_head in self.snake[1:]:
            return -1
        
        return self.score

class SnakeGameOptimized:
    """
    Approach 2: Optimized with Set for Fast Collision Detection
    
    Use set to track snake body for O(1) collision detection.
    
    Time Complexity:
    - __init__: O(1)
    - move: O(1)
    
    Space Complexity: O(n + f)
    """
    
    def __init__(self, width: int, height: int, food: List[List[int]]):
        self.width = width
        self.height = height
        self.food = deque(food)
        
        # Snake represented as deque for O(1) head/tail operations
        self.snake = deque([(0, 0)])
        self.snake_set = {(0, 0)}  # For O(1) collision detection
        self.score = 0
        
        self.directions = {
            'U': (-1, 0),
            'D': (1, 0),
            'L': (0, -1),
            'R': (0, 1)
        }
    
    def move(self, direction: str) -> int:
        dr, dc = self.directions[direction]
        head_r, head_c = self.snake[0]
        new_head = (head_r + dr, head_c + dc)
        new_r, new_c = new_head
        
        # Check boundary collision
        if new_r < 0 or new_r >= self.height or new_c < 0 or new_c >= self.width:
            return -1
        
        # Check food consumption
        food_eaten = False
        if self.food and self.food[0] == [new_r, new_c]:
            food_eaten = True
            self.food.popleft()
            self.score += 1
        
        # Add new head
        self.snake.appendleft(new_head)
        self.snake_set.add(new_head)
        
        # Remove tail if no food eaten
        if not food_eaten:
            tail = self.snake.pop()
            self.snake_set.remove(tail)
        
        # Check self collision (new head should not be in existing body)
        # Since we added new head to set, check if it was already there
        if len(self.snake_set) < len(self.snake):
            return -1
        
        # Alternative collision check: exclude the head we just added
        tail_removed = not food_eaten
        expected_size = len(self.snake) - (1 if not tail_removed else 0)
        if len(self.snake_set) != len(self.snake):
            return -1
        
        return self.score

class SnakeGameAdvanced:
    """
    Approach 3: Advanced with Game Features and Analytics
    
    Enhanced version with additional game features and statistics.
    
    Time Complexity:
    - __init__: O(1)
    - move: O(1)
    
    Space Complexity: O(n + f + features)
    """
    
    def __init__(self, width: int, height: int, food: List[List[int]]):
        self.width = width
        self.height = height
        self.food = deque(food)
        self.initial_food_count = len(food)
        
        # Snake state
        self.snake = deque([(0, 0)])
        self.snake_set = {(0, 0)}
        self.score = 0
        self.game_over = False
        
        # Game statistics
        self.moves_count = 0
        self.food_eaten = 0
        self.direction_stats = {'U': 0, 'D': 0, 'L': 0, 'R': 0}
        self.move_history = []
        
        # Game features
        self.max_score_achieved = 0
        self.longest_length = 1
        
        self.directions = {
            'U': (-1, 0),
            'D': (1, 0),
            'L': (0, -1),
            'R': (0, 1)
        }
    
    def move(self, direction: str) -> int:
        if self.game_over:
            return -1
        
        self.moves_count += 1
        self.direction_stats[direction] += 1
        self.move_history.append(direction)
        
        dr, dc = self.directions[direction]
        head_r, head_c = self.snake[0]
        new_head = (head_r + dr, head_c + dc)
        new_r, new_c = new_head
        
        # Check boundary collision
        if new_r < 0 or new_r >= self.height or new_c < 0 or new_c >= self.width:
            self.game_over = True
            return -1
        
        # Check self collision before adding new head
        if new_head in self.snake_set:
            self.game_over = True
            return -1
        
        # Check food consumption
        food_eaten = False
        if self.food and self.food[0] == [new_r, new_c]:
            food_eaten = True
            self.food.popleft()
            self.score += 1
            self.food_eaten += 1
            self.max_score_achieved = max(self.max_score_achieved, self.score)
        
        # Add new head
        self.snake.appendleft(new_head)
        self.snake_set.add(new_head)
        
        # Update longest length
        self.longest_length = max(self.longest_length, len(self.snake))
        
        # Remove tail if no food eaten
        if not food_eaten:
            tail = self.snake.pop()
            self.snake_set.remove(tail)
        
        return self.score
    
    def get_statistics(self) -> dict:
        """Get game statistics"""
        completion_rate = (self.food_eaten / max(1, self.initial_food_count)) * 100
        
        return {
            'score': self.score,
            'moves_count': self.moves_count,
            'food_eaten': self.food_eaten,
            'completion_rate': completion_rate,
            'snake_length': len(self.snake),
            'longest_length': self.longest_length,
            'direction_stats': self.direction_stats.copy(),
            'remaining_food': len(self.food),
            'game_over': self.game_over
        }
    
    def get_game_state(self) -> dict:
        """Get current game state"""
        return {
            'snake_head': self.snake[0] if self.snake else None,
            'snake_body': list(self.snake),
            'next_food': list(self.food[0]) if self.food else None,
            'score': self.score,
            'game_over': self.game_over
        }
    
    def get_move_history(self) -> List[str]:
        """Get history of moves"""
        return self.move_history.copy()
    
    def reset_game(self) -> None:
        """Reset game to initial state"""
        self.snake = deque([(0, 0)])
        self.snake_set = {(0, 0)}
        self.score = 0
        self.game_over = False
        self.moves_count = 0
        self.food_eaten = 0
        self.direction_stats = {'U': 0, 'D': 0, 'L': 0, 'R': 0}
        self.move_history.clear()
        
        # Reset food (would need to store original)
        # For this demo, we'll leave food as is

class SnakeGameVisualized:
    """
    Approach 4: Snake Game with Visualization Support
    
    Enhanced for debugging and visualization.
    
    Time Complexity:
    - __init__: O(1)
    - move: O(1)
    
    Space Complexity: O(n + f + visualization)
    """
    
    def __init__(self, width: int, height: int, food: List[List[int]]):
        self.width = width
        self.height = height
        self.food = deque(food)
        
        self.snake = deque([(0, 0)])
        self.snake_set = {(0, 0)}
        self.score = 0
        self.game_over = False
        
        # Visualization support
        self.board_history = []
        self.save_board_state()
        
        self.directions = {
            'U': (-1, 0),
            'D': (1, 0),
            'L': (0, -1),
            'R': (0, 1)
        }
    
    def move(self, direction: str) -> int:
        if self.game_over:
            return -1
        
        dr, dc = self.directions[direction]
        head_r, head_c = self.snake[0]
        new_head = (head_r + dr, head_c + dc)
        new_r, new_c = new_head
        
        # Check boundary collision
        if new_r < 0 or new_r >= self.height or new_c < 0 or new_c >= self.width:
            self.game_over = True
            return -1
        
        # Check self collision
        if new_head in self.snake_set:
            self.game_over = True
            return -1
        
        # Check food
        food_eaten = False
        if self.food and self.food[0] == [new_r, new_c]:
            food_eaten = True
            self.food.popleft()
            self.score += 1
        
        # Update snake
        self.snake.appendleft(new_head)
        self.snake_set.add(new_head)
        
        if not food_eaten:
            tail = self.snake.pop()
            self.snake_set.remove(tail)
        
        # Save board state for visualization
        self.save_board_state()
        
        return self.score
    
    def save_board_state(self) -> None:
        """Save current board state"""
        board = [['.' for _ in range(self.width)] for _ in range(self.height)]
        
        # Place snake
        for i, (r, c) in enumerate(self.snake):
            if i == 0:
                board[r][c] = 'H'  # Head
            else:
                board[r][c] = 'S'  # Body
        
        # Place food
        if self.food:
            fr, fc = self.food[0]
            board[fr][fc] = 'F'
        
        self.board_history.append([row[:] for row in board])
    
    def print_board(self) -> None:
        """Print current board state"""
        if not self.board_history:
            return
        
        board = self.board_history[-1]
        print(f"Score: {self.score}, Snake Length: {len(self.snake)}")
        print("+" + "-" * self.width + "+")
        
        for row in board:
            print("|" + "".join(row) + "|")
        
        print("+" + "-" * self.width + "+")
        print("H=Head, S=Snake, F=Food, .=Empty")
    
    def get_board_history(self) -> List[List[List[str]]]:
        """Get complete board history"""
        return [board[:] for board in self.board_history]

class SnakeGameMemoryOptimized:
    """
    Approach 5: Memory-Optimized for Large Games
    
    Optimized memory usage for large game boards.
    
    Time Complexity:
    - __init__: O(1)
    - move: O(1)
    
    Space Complexity: O(n + f) minimal overhead
    """
    
    def __init__(self, width: int, height: int, food: List[List[int]]):
        self.width = width
        self.height = height
        
        # Convert food to more memory-efficient format
        self.food_index = 0
        self.food_positions = food
        
        # Compact snake representation
        self.snake_positions = [(0, 0)]  # List instead of deque for memory
        self.head_index = 0  # Index of head in circular buffer if needed
        
        self.score = 0
        
        # Minimal direction mapping
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # U, D, L, R
        self.dir_map = {'U': 0, 'D': 1, 'L': 2, 'R': 3}
    
    def move(self, direction: str) -> int:
        dr, dc = self.directions[self.dir_map[direction]]
        head_r, head_c = self.snake_positions[0]
        new_head = (head_r + dr, head_c + dc)
        new_r, new_c = new_head
        
        # Boundary check
        if new_r < 0 or new_r >= self.height or new_c < 0 or new_c >= self.width:
            return -1
        
        # Self collision check
        if new_head in self.snake_positions:
            return -1
        
        # Food check
        food_eaten = False
        if (self.food_index < len(self.food_positions) and 
            self.food_positions[self.food_index] == [new_r, new_c]):
            food_eaten = True
            self.food_index += 1
            self.score += 1
        
        # Update snake
        self.snake_positions.insert(0, new_head)
        
        if not food_eaten:
            self.snake_positions.pop()
        
        return self.score


def test_snake_game_basic():
    """Test basic snake game functionality"""
    print("=== Testing Basic Snake Game Functionality ===")
    
    implementations = [
        ("Basic", SnakeGameBasic),
        ("Optimized", SnakeGameOptimized),
        ("Advanced", SnakeGameAdvanced),
        ("Visualized", SnakeGameVisualized),
        ("Memory Optimized", SnakeGameMemoryOptimized)
    ]
    
    # Test case: 3x2 board with food at (1,2) and (0,1)
    width, height = 3, 2
    food = [[1, 2], [0, 1]]
    
    for name, GameClass in implementations:
        print(f"\n{name}:")
        
        game = GameClass(width, height, food)
        
        # Test sequence: R, D, R, U, L, U
        moves = ['R', 'D', 'R', 'U', 'L', 'U']
        
        for i, move in enumerate(moves):
            score = game.move(move)
            print(f"  Move {i+1} ({move}): Score = {score}")
            
            if score == -1:
                print(f"  Game Over!")
                break

def test_snake_game_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Snake Game Edge Cases ===")
    
    # Test immediate boundary collision
    print("Immediate boundary collision:")
    game = SnakeGameOptimized(3, 3, [])
    
    score = game.move('U')  # Should hit top boundary
    print(f"  Move up from start: {score}")
    
    score = game.move('L')  # Should hit left boundary
    print(f"  Move left from start: {score}")
    
    # Test self collision
    print(f"\nSelf collision test:")
    game2 = SnakeGameAdvanced(4, 4, [[0, 1], [0, 2]])
    
    # Create a scenario for self collision: R, R, D, L, L
    moves = ['R', 'R', 'D', 'L', 'L']
    
    for i, move in enumerate(moves):
        score = game2.move(move)
        print(f"  Move {i+1} ({move}): Score = {score}")
        
        if score == -1:
            break
    
    # Test no food scenario
    print(f"\nNo food scenario:")
    game3 = SnakeGameBasic(5, 5, [])
    
    moves = ['R', 'R', 'D', 'D']
    for move in moves:
        score = game3.move(move)
        print(f"  Move {move}: Score = {score}")
    
    # Test eating all food
    print(f"\nEating all food:")
    game4 = SnakeGameMemoryOptimized(3, 3, [[0, 1], [0, 2]])
    
    moves = ['R', 'R']  # Eat both foods
    for move in moves:
        score = game4.move(move)
        print(f"  Move {move}: Score = {score}")

def test_advanced_features():
    """Test advanced features"""
    print("\n=== Testing Advanced Features ===")
    
    game = SnakeGameAdvanced(5, 4, [[1, 1], [2, 2], [3, 3]])
    
    # Play some moves
    moves = ['R', 'D', 'R', 'D', 'R', 'D', 'R']
    
    print("Playing sequence of moves:")
    for move in moves:
        score = game.move(move)
        print(f"  Move {move}: Score = {score}")
        
        if score == -1:
            break
    
    # Get statistics
    stats = game.get_statistics()
    print(f"\nGame statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}: {value}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.1f}")
        else:
            print(f"  {key}: {value}")
    
    # Get game state
    state = game.get_game_state()
    print(f"\nGame state:")
    for key, value in state.items():
        print(f"  {key}: {value}")
    
    # Get move history
    history = game.get_move_history()
    print(f"\nMove history: {history}")

def test_visualization():
    """Test visualization features"""
    print("\n=== Testing Visualization ===")
    
    game = SnakeGameVisualized(4, 3, [[1, 2], [2, 1]])
    
    print("Initial board:")
    game.print_board()
    
    # Make some moves
    moves = ['R', 'D', 'R', 'U']
    
    for move in moves:
        print(f"\nAfter move {move}:")
        score = game.move(move)
        print(f"Score: {score}")
        game.print_board()
        
        if score == -1:
            break

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: Classic Snake Game
    print("Application 1: Classic Snake Game Simulation")
    
    classic_game = SnakeGameAdvanced(6, 6, [[2, 2], [3, 4], [1, 5], [4, 1]])
    
    # Simulate a game session
    move_sequence = ['R', 'R', 'D', 'D', 'L', 'U', 'R', 'R', 'R', 'D', 'D', 'L', 'L']
    
    print("  Simulating classic game:")
    for i, move in enumerate(move_sequence):
        score = classic_game.move(move)
        
        if i % 3 == 0:  # Show progress every 3 moves
            stats = classic_game.get_statistics()
            print(f"    Move {i+1}: Score={score}, Length={stats['snake_length']}")
        
        if score == -1:
            print(f"    Game Over at move {i+1}!")
            break
    
    # Final statistics
    final_stats = classic_game.get_statistics()
    print(f"  Final stats: Score={final_stats['score']}, "
          f"Food eaten={final_stats['food_eaten']}, "
          f"Completion={final_stats['completion_rate']:.1f}%")
    
    # Application 2: Pathfinding algorithm testing
    print(f"\nApplication 2: Pathfinding Algorithm Testing")
    
    pathfinding_game = SnakeGameOptimized(8, 6, [[3, 3], [5, 6], [2, 1]])
    
    # Simulate basic pathfinding moves (simplified)
    def simple_pathfinding(game, target_food):
        """Simple pathfinding towards food"""
        if not game.snake or not target_food:
            return None
        
        head_r, head_c = game.snake[0]
        food_r, food_c = target_food
        
        # Simple strategy: move towards food
        if head_r < food_r:
            return 'D'
        elif head_r > food_r:
            return 'U'
        elif head_c < food_c:
            return 'R'
        elif head_c > food_c:
            return 'L'
        
        return 'R'  # Default
    
    print("  Testing simple pathfinding:")
    
    for step in range(10):
        if pathfinding_game.food:
            next_food = pathfinding_game.food[0]
            move = simple_pathfinding(pathfinding_game, next_food)
            
            if move:
                score = pathfinding_game.move(move)
                print(f"    Step {step+1}: Move {move} towards {next_food}, Score: {score}")
                
                if score == -1:
                    break
        else:
            print("    All food consumed!")
            break
    
    # Application 3: Game AI training environment
    print(f"\nApplication 3: AI Training Environment")
    
    ai_training_game = SnakeGameMemoryOptimized(10, 10, 
                                               [[i, j] for i in range(1, 9) for j in range(1, 9) if (i + j) % 3 == 0])
    
    # Simulate random moves for AI training
    import random
    random.seed(42)  # For reproducible results
    
    directions = ['U', 'D', 'L', 'R']
    training_scores = []
    
    print("  Simulating AI training episodes:")
    
    for episode in range(5):
        episode_score = 0
        moves_count = 0
        
        # Reset would be needed here in real scenario
        while moves_count < 20:  # Limit moves per episode
            move = random.choice(directions)
            score = ai_training_game.move(move)
            
            if score == -1:
                break
            
            episode_score = score
            moves_count += 1
        
        training_scores.append(episode_score)
        print(f"    Episode {episode+1}: Score={episode_score}, Moves={moves_count}")
    
    avg_score = sum(training_scores) / len(training_scores)
    print(f"  Training summary: Average score={avg_score:.1f}, Episodes={len(training_scores)}")

def test_performance():
    """Test performance with large games"""
    print("\n=== Testing Performance ===")
    
    import time
    
    implementations = [
        ("Basic", SnakeGameBasic),
        ("Optimized", SnakeGameOptimized),
        ("Memory Optimized", SnakeGameMemoryOptimized)
    ]
    
    # Large game setup
    width, height = 50, 50
    large_food = [[i, j] for i in range(5, 45) for j in range(5, 45) if (i + j) % 5 == 0]
    
    print(f"Performance test: {width}x{height} board, {len(large_food)} food items")
    
    for name, GameClass in implementations:
        game = GameClass(width, height, large_food)
        
        start_time = time.time()
        
        # Simulate many moves
        import random
        random.seed(42)
        
        directions = ['U', 'D', 'L', 'R']
        move_count = 0
        
        for _ in range(1000):  # 1000 moves
            move = random.choice(directions)
            score = game.move(move)
            move_count += 1
            
            if score == -1:
                break
        
        elapsed = (time.time() - start_time) * 1000
        
        print(f"  {name}: {elapsed:.2f}ms for {move_count} moves")
        print(f"    Average: {elapsed/move_count:.3f}ms per move")

def stress_test_snake_game():
    """Stress test snake game"""
    print("\n=== Stress Testing Snake Game ===")
    
    import time
    import random
    
    # Very large game
    width, height = 100, 100
    
    # Generate lots of food
    food_positions = []
    for i in range(10, 90, 5):
        for j in range(10, 90, 5):
            food_positions.append([i, j])
    
    random.shuffle(food_positions)
    food_positions = food_positions[:500]  # 500 food items
    
    print(f"Stress test: {width}x{height} board, {len(food_positions)} food items")
    
    game = SnakeGameOptimized(width, height, food_positions)
    
    start_time = time.time()
    
    directions = ['U', 'D', 'L', 'R']
    move_count = 0
    max_score = 0
    
    # Run until game over or time limit
    time_limit = 5.0  # 5 seconds
    
    while time.time() - start_time < time_limit:
        move = random.choice(directions)
        score = game.move(move)
        move_count += 1
        
        if score == -1:
            print(f"  Game over after {move_count} moves")
            break
        
        max_score = max(max_score, score)
        
        # Progress update
        if move_count % 1000 == 0:
            elapsed = time.time() - start_time
            rate = move_count / elapsed
            print(f"    {move_count} moves, {rate:.0f} moves/sec, max score: {max_score}")
    
    total_time = time.time() - start_time
    
    print(f"  Total moves: {move_count}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Rate: {move_count/total_time:.0f} moves/sec")
    print(f"  Max score achieved: {max_score}")

def test_collision_scenarios():
    """Test various collision scenarios"""
    print("\n=== Testing Collision Scenarios ===")
    
    # Boundary collision tests
    collision_tests = [
        ("Top boundary", 3, 3, [], ['U']),
        ("Bottom boundary", 3, 3, [], ['D', 'D', 'D']),
        ("Left boundary", 3, 3, [], ['L']),
        ("Right boundary", 3, 3, [], ['R', 'R', 'R']),
    ]
    
    for test_name, w, h, food, moves in collision_tests:
        game = SnakeGameOptimized(w, h, food)
        
        print(f"\n{test_name}:")
        
        for i, move in enumerate(moves):
            score = game.move(move)
            print(f"  Move {i+1} ({move}): {score}")
            
            if score == -1:
                print(f"  Expected collision occurred")
                break
    
    # Self collision test
    print(f"\nSelf collision scenario:")
    
    # Create snake that will collide with itself
    game = SnakeGameAdvanced(5, 5, [[0, 1], [0, 2], [1, 2]])
    
    # Moves: R, R, D, L, L (should cause self collision)
    self_collision_moves = ['R', 'R', 'D', 'L', 'L']
    
    for i, move in enumerate(self_collision_moves):
        score = game.move(move)
        state = game.get_game_state()
        
        print(f"  Move {i+1} ({move}): Score={score}")
        print(f"    Snake: {state['snake_body']}")
        
        if score == -1:
            print(f"  Self collision detected!")
            break

def test_food_consumption_patterns():
    """Test different food consumption patterns"""
    print("\n=== Testing Food Consumption Patterns ===")
    
    patterns = [
        ("Linear food", [[0, 1], [0, 2], [0, 3]], ['R', 'R', 'R']),
        ("Scattered food", [[1, 1], [3, 3], [2, 0]], ['R', 'D', 'R', 'D', 'D', 'L', 'L']),
        ("No food", [], ['R', 'D', 'L', 'U']),
        ("Single food", [[2, 2]], ['R', 'R', 'D', 'D'])
    ]
    
    for pattern_name, food_list, move_sequence in patterns:
        game = SnakeGameAdvanced(4, 4, food_list)
        
        print(f"\n{pattern_name}:")
        print(f"  Food locations: {food_list}")
        
        for move in move_sequence:
            score = game.move(move)
            print(f"    Move {move}: Score={score}")
            
            if score == -1:
                break
        
        stats = game.get_statistics()
        print(f"  Final: {stats['food_eaten']} food eaten, {stats['snake_length']} length")

def benchmark_memory_usage():
    """Benchmark memory usage"""
    print("\n=== Benchmarking Memory Usage ===")
    
    implementations = [
        ("Basic", SnakeGameBasic),
        ("Optimized", SnakeGameOptimized),
        ("Memory Optimized", SnakeGameMemoryOptimized)
    ]
    
    # Different game sizes
    sizes = [(10, 10, 50), (25, 25, 200), (50, 50, 500)]
    
    for width, height, food_count in sizes:
        print(f"\nGame size: {width}x{height}, {food_count} food items")
        
        # Generate food
        food = [[i, j] for i in range(1, width-1) for j in range(1, height-1)]
        food = food[:food_count]
        
        for name, GameClass in implementations:
            game = GameClass(width, height, food)
            
            # Simulate some moves to grow snake
            for _ in range(min(20, food_count)):
                game.move('R')
                if hasattr(game, 'snake') and len(game.snake) > width:
                    break
            
            # Estimate memory usage (simplified)
            if hasattr(game, 'snake_set'):
                snake_memory = len(game.snake_set) + len(game.snake)
                approach = "Set + Deque"
            elif hasattr(game, 'snake_positions'):
                snake_memory = len(game.snake_positions)
                approach = "List only"
            else:
                snake_memory = len(game.snake) * 2  # Estimate
                approach = "Basic"
            
            food_memory = len(food)
            total_memory = snake_memory + food_memory
            
            print(f"    {name} ({approach}): ~{total_memory} units")

if __name__ == "__main__":
    test_snake_game_basic()
    test_snake_game_edge_cases()
    test_advanced_features()
    test_visualization()
    demonstrate_applications()
    test_performance()
    stress_test_snake_game()
    test_collision_scenarios()
    test_food_consumption_patterns()
    benchmark_memory_usage()

"""
Snake Game Design demonstrates key concepts:

Core Approaches:
1. Basic - Simple list-based implementation with O(n) collision detection
2. Optimized - Use set for O(1) collision detection and deque for efficient operations
3. Advanced - Enhanced with game statistics, history tracking, and features
4. Visualized - Added board visualization and state history for debugging
5. Memory Optimized - Minimal overhead for large-scale games

Key Design Principles:
- Game state management with position tracking
- Efficient collision detection algorithms
- Food consumption and score management
- Boundary and self-collision handling

Performance Characteristics:
- Basic: O(n) collision check, simple implementation
- Optimized: O(1) all operations, optimal for gameplay
- Advanced: O(1) operations + analytics overhead
- Memory optimized: Minimal space overhead

Real-world Applications:
- Classic arcade game implementation
- Pathfinding algorithm testing environment
- AI training and reinforcement learning
- Game engine collision detection systems
- Educational programming projects
- Mobile game development

The optimized approach with set-based collision detection
provides the best balance of performance and functionality
for real-time gameplay while maintaining clean code structure.
"""
