#!/usr/bin/env python3
"""
Launcher script for the FashionMNIST Training Agent
"""

import sys
import os

def main():
    try:
        from ai_agent import HyperparameterAgent
    except ImportError:
        print("Error: Could not import the HyperparameterAgent.")
        print("Make sure you have the ai_agent.py file in the current directory.")
        sys.exit(1)
    
    print("Starting FashionMNIST Training Agent...")
    agent = HyperparameterAgent()
    agent.cmdloop()

if __name__ == "__main__":
    main() 