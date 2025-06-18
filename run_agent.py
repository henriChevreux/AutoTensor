#!/usr/bin/env python3
"""
Launcher script for the FashionMNIST Training Agent
"""

import sys
import os

def main():
    try:
        from cmd_agent import TrainingAgent
    except ImportError:
        print("Error: Could not import the TrainingAgent.")
        print("Make sure you have the cmd_agent.py file in the current directory.")
        sys.exit(1)
    
    print("Starting FashionMNIST Training Agent...")
    agent = TrainingAgent()
    agent.cmdloop()

if __name__ == "__main__":
    main() 