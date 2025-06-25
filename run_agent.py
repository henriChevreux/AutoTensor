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

# Fix for terminal echo after os.execv restart
if sys.stdin.isatty():
    try:
        import termios
        fd = sys.stdin.fileno()
        attrs = termios.tcgetattr(fd)
        attrs[3] |= termios.ECHO  # lflags
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
    except Exception:
        pass  # Not a tty or not supported, ignore

WATCHED_DIRS = ['.']  # Add more directories as needed

class ReloadHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.last_reload = 0

    def on_modified(self, event):
        # Only react to .py file changes
        if not event.is_directory and event.src_path.endswith('.py'):
            # Ignore changes in the models folder
            if 'models' in event.src_path:
                return
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