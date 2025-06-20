#!/usr/bin/env python3
"""
Launcher script for the FashionMNIST Training Agent
"""

import sys
import os
import time
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

WATCHED_DIRS = ['.']  # Add more directories as needed

class ReloadHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.last_reload = time.time()

    def on_modified(self, event):
        # Debounce rapid changes
        if time.time() - self.last_reload < 1:
            return
        self.last_reload = time.time()
        print("\nDetected code change. Reloading agent...")
        # Restart the current process
        python = sys.executable
        os.execv(python, [python] + sys.argv)

def start_watcher():
    event_handler = ReloadHandler()
    observer = Observer()
    for d in WATCHED_DIRS:
        observer.schedule(event_handler, d, recursive=True)
    observer.start()

    # Keep the thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

def main():
    try:
        from cmd_agent import TrainingAgent
    except ImportError:
        print("Error: Could not import the TrainingAgent.")
        print("Make sure you have the cmd_agent.py file in the current directory.")
        sys.exit(1)
    
    # Start the watcher in a background thread
    watcher_thread = threading.Thread(target=start_watcher, daemon=True)
    watcher_thread.start()
    
    print("Starting FashionMNIST Training Agent...")
    agent = TrainingAgent()
    agent.cmdloop()

if __name__ == "__main__":
    main() 