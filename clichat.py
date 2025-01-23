#!/usr/bin/env python3
import warnings

# Suppress pydantic warning about config keys
warnings.filterwarnings('ignore', message='Valid config keys have changed in V2')

from rich.console import Console
from rich.box import ROUNDED
from rich.panel import Panel
from rich.text import Text
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
import os
from dotenv import load_dotenv
from litellm import completion
import sys
from typing import List, Dict
from rich.live import Live
import json
from openai import OpenAI, AsyncOpenAI
import hashlib
from datetime import datetime
import os.path
from prompt_toolkit.shortcuts import prompt
from prompt_toolkit.application import get_app
import signal
import threading
import multiprocessing
from multiprocessing import Process, Queue, Event
import asyncio
import queue  # For queue.Empty exception
from chat_generator import ChatGenerator
import time

# Load environment variables
load_dotenv()

# Initialize Rich console
console = Console()

# Initialize key bindings
kb = KeyBindings()
session = PromptSession()

# Global flag for interrupt handling
interrupt_requested = threading.Event()


def handle_sigint(signum, frame):
    """Custom interrupt handler that sets a flag"""
    interrupt_requested.set()
    raise KeyboardInterrupt()


# Set up the handler
signal.signal(signal.SIGINT, handle_sigint)


class ChatMessage:
    def __init__(self, role: str, content: str, reasoning_content: str = None):
        self.role = role
        self.content = content
        self.reasoning_content = reasoning_content


class ChatInterface:
    def __init__(self, context_file: str = None):
        self.messages: List[ChatMessage] = []
        # Get model and API configuration from environment variables
        self.model = os.getenv("DEFAULT_MODEL", "anthropic/claude-3-sonnet")
        self.api_base = os.getenv("API_BASE_URL", "https://api.deepseek.com/v1")
        self.api_key = os.getenv("API_KEY", "")

        self.console = Console()
        # Make streaming configurable through environment variable
        self.streaming = os.getenv("ENABLE_STREAMING", "true").lower() == "true"

        # Initialize OpenAI client with configurable endpoint
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
        self.async_client = AsyncOpenAI(  # Add async client
            api_key=self.api_key,
            base_url=self.api_base
        )

        # Store the context file path or generate a hash-based filename
        self.log_file = context_file if context_file else self._generate_log_filename()

        if context_file:
            self.load_context(context_file)

        self.edit_mode = False  # Track if we're in edit mode
        self.edit_content = None  # Content to edit

        self.interrupt_requested = interrupt_requested  # Add reference to global flag

        self.stop_event = Event()  # For signaling the child process to stop
        self.response_queue = Queue()  # For getting responses from child process

    def _generate_log_filename(self) -> str:
        """Generate a 6-character hash-based filename using timestamp"""
        timestamp = datetime.now().isoformat()
        hash_obj = hashlib.sha256(timestamp.encode())
        return f"chat_{hash_obj.hexdigest()[:6]}.json"

    def load_context(self, filepath: str):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            if "messages" in data:
                # Clear existing messages
                self.messages.clear()
                # Load messages from file
                for msg in data["messages"]:
                    if "role" in msg and "content" in msg:
                        reasoning = msg.get("reasoning_content")  # Get reasoning if present
                        self.messages.append(ChatMessage(msg["role"], msg["content"], reasoning))
                        self.display_message(self.messages[-1])

                console.print("[green]Loaded previous context[/]")
            else:
                console.print("[red]Invalid context file format - no messages found[/]")

        except json.JSONDecodeError:
            console.print("[red]Error: Invalid JSON in context file[/]")
        except Exception as e:
            console.print(f"[red]Error loading context: {str(e)}[/]")

    def display_message(self, message: ChatMessage, live_update=False):
        # Determine style based on role
        style = "green" if message.role == "assistant" else "blue"
        title = f"[{style}]{message.role.title()}[/]"

        panels = []

        # If there's reasoning content and this is an assistant message, show it first
        if message.role == "assistant" and message.reasoning_content:
            reasoning_panel = Panel(
                Text(message.reasoning_content, style="yellow"),
                title=f"[yellow]Reasoning[/]",
                box=ROUNDED,
                border_style="yellow",
                width=min(100, console.width - 2)
            )
            panels.append(reasoning_panel)

        # For user messages, include the "You: " prefix in the content
        content = message.content
        if message.role == "user":
            content = f"You: {message.content}"

        # Create main content panel
        content_panel = Panel(
            Text(content, style="white"),
            title=title,
            box=ROUNDED,
            border_style=style,
            width=min(100, console.width - 2)
        )
        panels.append(content_panel)

        if live_update:
            from rich.console import Group
            return Group(*panels)  # Return all panels as a group for live updates
        else:
            for panel in panels:
                console.print(panel)

    def clear_last_message(self):
        # Get the last message's content
        if not self.messages:
            return

        message = self.messages[-1]

        # Calculate total lines to clear
        width = min(100, self.console.width - 2)
        num_lines = 0

        # Count lines for main content
        content = f"You: {message.content}" if message.role == "user" else message.content
        num_lines += (len(content) // width) + 3  # Add 3 for panel borders and padding

        # Count additional lines for reasoning if present
        if message.role == "assistant" and message.reasoning_content:
            num_lines += (len(message.reasoning_content) // width) + 3  # Add 3 for reasoning panel

        # Move cursor up and clear those lines
        for _ in range(num_lines):
            print("\033[A\033[K", end="")

    def regenerate_last(self):
        if not self.messages:
            return

        # Remove and clear last assistant message if it exists
        if self.messages[-1].role == "assistant":
            self.clear_last_message()
            self.messages.pop()

        self.get_completion()

    def edit_last_user_message(self, event):
        # Find last user message
        for i in reversed(range(len(self.messages))):
            if self.messages[i].role == "user":
                # Set edit mode and content
                self.edit_mode = True
                self.edit_content = self.messages[i].content
                # Exit current prompt - will restart with edit content
                event.app.exit(result="EDIT")  # Return a marker instead of None
                break

    def get_completion(self):
        try:
            # If there's already an assistant message, clear it first
            if self.messages and self.messages[-1].role == "assistant":
                self.clear_last_message()

            # Create an initial empty message
            assistant_message = ChatMessage("assistant", "", "")
            self.messages.append(assistant_message)

            def generate_response():
                """Function to run in terminal mode"""
                if self.streaming:
                    live = None
                    process = None
                    try:
                        live = Live(
                            self.display_message(assistant_message, live_update=True),
                            refresh_per_second=4,
                            vertical_overflow="visible"
                        )
                        live.start()

                        # Reset stop event and queue
                        self.stop_event.clear()
                        while not self.response_queue.empty():
                            self.response_queue.get()

                        # Start streaming process
                        messages = [{"role": m.role, "content": m.content} for m in self.messages[:-1]]
                        process = Process(
                            target=ChatGenerator.stream_in_process,
                            args=(messages, self.model, self.api_key, self.api_base,
                                  self.response_queue, self.stop_event)
                        )
                        process.start()

                        # Process responses from queue
                        while True:
                            if self.interrupt_requested.is_set():
                                raise KeyboardInterrupt

                            try:
                                # Use a very short timeout to allow for interrupt checks
                                msg_type, content = self.response_queue.get(timeout=0.05)
                                if msg_type == 'done':
                                    break
                                elif msg_type == 'error':
                                    console.print(f"[red]Error: {content}[/]")
                                    break
                                elif msg_type == 'reasoning':
                                    if assistant_message.reasoning_content is None:
                                        assistant_message.reasoning_content = ""
                                    assistant_message.reasoning_content += content
                                elif msg_type == 'content':
                                    if assistant_message.content is None:
                                        assistant_message.content = ""
                                    assistant_message.content += content
                                live.update(self.display_message(assistant_message, live_update=True))
                            except queue.Empty:
                                continue

                    except KeyboardInterrupt:
                        if process and process.is_alive():
                            self.stop_event.set()  # Signal process to stop
                            process.join(timeout=1)  # Wait for process to finish
                            if process.is_alive():
                                process.terminate()  # Force terminate if it doesn't stop

                        if live:
                            live.stop()
                        sys.stdout.write('\n')
                        console.print("[yellow]Generation aborted by user[/]")
                        return False
                    finally:
                        if live:
                            live.stop()
                        if process and process.is_alive():
                            process.terminate()
                else:
                    # Non-streaming mode
                    try:
                        response = ChatGenerator.generate_completion(
                            self.client,
                            self.model,
                            [{"role": m.role, "content": m.content} for m in self.messages[:-1]]
                        )
                        assistant_message.content = response.choices[0].message.content

                        if hasattr(response.choices[0].message, 'reasoning_content'):
                            assistant_message.reasoning_content = response.choices[0].message.reasoning_content

                        self.display_message(assistant_message)

                    except KeyboardInterrupt:
                        console.print("\n[yellow]Generation aborted by user[/]")
                        return False

                return True

            # Run generation directly without prompt_toolkit's run_in_terminal
            success = False
            try:
                success = generate_response()
            except KeyboardInterrupt:
                console.print("\n[yellow]Generation aborted by user[/]")
                success = False

            # Clean up if generation was interrupted
            if not success:
                self.clear_last_message()  # Clear the partial assistant message
                self.messages.pop()  # Remove the assistant message
                self.clear_last_message()  # Clear the user message
                self.messages.pop()  # Remove the user message

        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/]")
            if assistant_message.content == "":
                self.messages.pop()  # Remove empty message on error

    def save_context(self):
        """Save the current chat context to a file"""
        try:
            data = {
                "messages": [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "reasoning_content": msg.reasoning_content
                    } for msg in self.messages
                ]
            }

            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)

            console.print(f"[green]Chat saved to {self.log_file}[/]")
        except Exception as e:
            console.print(f"[red]Error saving chat: {str(e)}[/]")

    def undo_last_exchange(self):
        """Remove the last user-assistant message pair and show current assistant message"""
        if not self.messages:
            return

        # Remove last assistant message if present
        if self.messages[-1].role == "assistant":
            self.clear_last_message()
            self.messages.pop()

        # Remove the user message
        if self.messages and self.messages[-1].role == "user":
            self.clear_last_message()
            self.messages.pop()

        # Show the current last assistant message if it exists
        if self.messages and self.messages[-1].role == "assistant":
            self.display_message(self.messages[-1])

    def run(self):
        # Set up key bindings
        @kb.add('c-r')
        def _(event):
            self.regenerate_last()
            return False

        @kb.add('c-e')
        def _(event):
            self.edit_last_user_message(event)
            # Don't return False - we want to exit the prompt

        @kb.add('c-s')
        def _(event):
            self.save_context()
            return False

        @kb.add('c-z')  # Add Ctrl+Z for undo
        def _(event):
            self.undo_last_exchange()
            return False

        @kb.add('c-q')
        def _(event):
            event.app.exit()

        console.print("[yellow]Chat Interface Started[/]")
        console.print(f"[blue]Streaming mode: {'enabled' if self.streaming else 'disabled'}[/]")
        console.print("Press Ctrl+R to regenerate last response")
        console.print("Press Ctrl+E to edit your last message")
        console.print("Press Ctrl+S to save chat")
        console.print("Press Ctrl+Z to undo last exchange")
        console.print("Press Ctrl+C to abort generation (broken)")
        console.print("Press Ctrl+Q for instant quit")
        console.print("Press Ctrl+D to exit")
        console.print("Use -c filename.json to load a previous chat")

        while True:
            try:
                # Set up prompt with edit content if in edit mode
                default = self.edit_content if self.edit_mode else ""
                message = "Edit message: " if self.edit_mode else ""

                user_input = session.prompt(
                    message,
                    default=default,
                    key_bindings=kb,
                    handle_sigint=False  # Let Python handle Ctrl+C instead of prompt_toolkit
                )

                # Handle special return values
                if user_input == "EDIT":
                    continue  # Skip this iteration and restart with edit mode

                if user_input and user_input.strip():
                    print("\033[A\033[K", end="")

                    if self.edit_mode:
                        # Find the message we're editing and remove everything after it
                        for i in range(len(self.messages)):
                            if (self.messages[i].role == "user" and
                                    self.messages[i].content == self.edit_content):
                                # Clear all messages from this point onwards
                                for _ in range(len(self.messages) - i):
                                    self.clear_last_message()
                                    self.messages.pop()
                                break

                        self.edit_mode = False
                        self.edit_content = None

                    user_message = ChatMessage("user", user_input)
                    self.messages.append(user_message)
                    self.display_message(user_message)
                    try:
                        self.get_completion()
                    except KeyboardInterrupt:
                        console.print("\n[yellow]Debug: KeyboardInterrupt caught in run loop completion handler[/]")
                        console.print("\n[yellow]Generation aborted by user[/]")
                        # Clear and remove both messages
                        self.clear_last_message()  # Clear the partial assistant message
                        self.messages.pop()  # Remove the assistant message
                        self.clear_last_message()  # Clear the user message
                        self.messages.pop()  # Remove the user message
                        continue
                else:
                    # Cancel edit mode on empty input
                    self.edit_mode = False
                    self.edit_content = None

            except KeyboardInterrupt:
                console.print("\n[yellow]Debug: KeyboardInterrupt caught in main run loop[/]")
                self.edit_mode = False
                self.edit_content = None
                continue
            except EOFError:
                break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='CLI Chat Interface')
    parser.add_argument('--context', '-c', help='Path to context JSON file')
    args = parser.parse_args()

    chat = ChatInterface(context_file=args.context)
    chat.run() 