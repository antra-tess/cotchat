import os
from rich.console import Console
from rich.prompt import Prompt, Confirm
from pathlib import Path
from prompt_toolkit.shortcuts import confirm
from prompt_toolkit import PromptSession
from platformdirs import user_config_dir

console = Console()

def setup_environment():
    """Interactive setup for API configuration"""
    # Create config directory if it doesn't exist
    config_dir = Path(user_config_dir("cotchat"))
    config_dir.mkdir(parents=True, exist_ok=True)
    env_path = config_dir / ".env"

    console.print("\n[yellow]No API configuration found. Let's set that up![/]\n")
    
    # Create a session for consistent prompt behavior
    session = PromptSession()
    
    # Display info with rich, but use plain text for prompt
    console.print("[blue]Would you like to use DeepSeek?")
    console.print("Select 'no' to use OpenRouter instead")
    use_deepseek = confirm("Use DeepSeek?")
    
    if use_deepseek:
        console.print("\n[green]Please enter your DeepSeek API key:[/]")
        api_key = session.prompt("> ")
        api_base = "https://api.deepseek.com/v1"
        default_model = "deepseek-reasoner"
        console.print("\n[yellow]Note: Reasoning traces are available with DeepSeek models[/]")
    else:
        console.print("\n[green]Please enter your OpenRouter API key:[/]")
        api_key = session.prompt("> ")
        api_base = "https://openrouter.ai/api/v1"
        default_model = "deepseek/deepseek-r1"
        console.print("\n[yellow]Warning: Reasoning traces are not available with OpenRouter[/]")
    
    # Create .env file in config directory
    with env_path.open('w') as f:
        f.write(f"API_KEY={api_key}\n")
        f.write(f"API_BASE_URL={api_base}\n")
        f.write(f"DEFAULT_MODEL={default_model}\n")
        f.write("ENABLE_STREAMING=true\n")
    
    console.print("\n[green]Configuration saved![/]")
    console.print(f"[blue]Your settings are stored in:[/]")
    console.print(f"[yellow]{env_path}[/]")
    console.print("[dim]You can edit this file directly to change settings[/]")
    console.print("\n[blue]You can now restart cotchat to begin chatting![/]\n")
    return False  # Signal to exit after setup 