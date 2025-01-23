#!/usr/bin/env python3
import argparse
from .interface import ChatInterface

def main():
    parser = argparse.ArgumentParser(description='CLI Chat Interface')
    parser.add_argument('--context', '-c', help='Path to context JSON file')
    args = parser.parse_args()

    chat = ChatInterface(context_file=args.context)
    chat.run()

if __name__ == "__main__":
    main() 