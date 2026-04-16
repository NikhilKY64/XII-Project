import csv
import os
import time

class Logger:
    def __init__(self, filename='log.csv'):
        self.filename = filename
        self._ensure_file()
        
    def _ensure_file(self):
        # Create the file with headers if it doesn't exist
        if not os.path.isfile(self.filename):
            with open(self.filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'event_type'])
                
    def log_event(self, event_type: str):
        # Log the event with the current UNIX timestamp
        with open(self.filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([int(time.time()), event_type])
