import time
from collections import deque

class BehaviorAnalyzer:
    def __init__(self, target_threshold=2, window_sec=10, cooldown_sec=1):
        # Global counters
        self.left_count = 0
        self.right_count = 0
        self.center_count = 0
        
        # Sliding time window for tracking recent movements
        self.movements = deque()
        
        # Configuration
        self.threshold = target_threshold
        self.window_sec = window_sec
        self.cooldown_sec = cooldown_sec
        
        self.last_capture_time = 0.0
        
        # To track transitions without unique student IDs
        self.prev_lefts = 0
        self.prev_rights = 0
        self.prev_centers = 0

    def update(self, face_directions):
        """
        Updates the global counters based on list of head directions from the current frame.
        face_directions: list of strings like ['LEFT', 'CENTER', 'RIGHT']
        """
        curr_lefts = face_directions.count('LEFT')
        curr_rights = face_directions.count('RIGHT')
        curr_centers = face_directions.count('CENTER')
        
        # Detect transitions (basic movement detection)
        if curr_lefts > self.prev_lefts:
            diff = curr_lefts - self.prev_lefts
            self.left_count += diff
            for _ in range(diff):
                self.movements.append(time.time())
                
        if curr_rights > self.prev_rights:
            diff = curr_rights - self.prev_rights
            self.right_count += diff
            for _ in range(diff):
                self.movements.append(time.time())
                
        if curr_centers > self.prev_centers:
            diff = curr_centers - self.prev_centers
            self.center_count += diff
            
        self.prev_lefts = curr_lefts
        self.prev_rights = curr_rights
        self.prev_centers = curr_centers

    def get_recent_movements_count(self):
        """Cleans up movements older than the time window and returns the current count."""
        now = time.time()
        while self.movements and (now - self.movements[0]) > self.window_sec:
            self.movements.popleft()
        return len(self.movements)
        
    def is_suspicious(self):
        """Checks if recent left/right movements exceed the threshold."""
        return self.get_recent_movements_count() >= self.threshold
        
    def can_capture(self):
        """Enforces a cooldown between captures."""
        return (time.time() - self.last_capture_time) >= self.cooldown_sec
        
    def trigger_capture(self):
        """Registers a capture to start the cooldown timer."""
        self.last_capture_time = time.time()
