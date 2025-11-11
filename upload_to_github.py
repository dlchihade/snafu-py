#!/usr/bin/env python3
"""
GitHub Upload Script for snafu-py Repository

This script automates the process of uploading changes to GitHub.
It handles staging, committing, and pushing changes to the remote repository.
"""

import subprocess
import sys
import os
from datetime import datetime

def run_command(command, description, check=True):
    """Run a shell command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=check)
        if result.stdout:
            print(result.stdout.strip())
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {description} failed")
        print(f"Error output: {e.stderr}")
        if check:
            sys.exit(1)
        return e

def get_commit_message():
    """Get commit message from user or use default."""
    default_message = f"Update repository - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    print(f"\n📝 Enter commit message (or press Enter for default):")
    print(f"Default: {default_message}")
    
    user_message = input("Commit message: ").strip()
    return user_message if user_message else default_message

def main():
    """Main function to upload repository to GitHub."""
    print("🚀 Starting GitHub upload process for snafu-py repository...")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('.git'):
        print("❌ Error: Not in a Git repository. Please run this script from the snafu-py directory.")
        sys.exit(1)
    
    # Check current branch
    result = run_command("git branch --show-current", "Checking current branch")
    current_branch = result.stdout.strip()
    print(f"📍 Current branch: {current_branch}")
    
    # Check remote repository
    result = run_command("git remote -v", "Checking remote repository")
    if "github.com" not in result.stdout:
        print("❌ Error: No GitHub remote found. Please add a GitHub remote first.")
        sys.exit(1)
    
    # Check for uncommitted changes
    result = run_command("git status --porcelain", "Checking for changes")
    if not result.stdout.strip():
        print("✅ No changes to commit. Repository is up to date.")
        return
    
    print(f"📋 Changes detected:\n{result.stdout}")
    
    # Stage all changes
    run_command("git add .", "Staging all changes")
    
    # Get commit message
    commit_message = get_commit_message()
    
    # Commit changes
    run_command(f'git commit -m "{commit_message}"', "Committing changes")
    
    # Push to GitHub
    print(f"\n🚀 Pushing to GitHub...")
    result = run_command(f"git push origin {current_branch}", "Pushing to GitHub")
    
    if result.returncode == 0:
        print("\n✅ Successfully uploaded to GitHub!")
        print(f"📍 Repository: https://github.com/AusterweilLab/snafu-py")
        print(f"📍 Branch: {current_branch}")
    else:
        print("\n❌ Failed to push to GitHub. Please check your credentials and try again.")
        print("💡 You may need to:")
        print("   1. Set up SSH keys or use a personal access token")
        print("   2. Configure your Git credentials")
        print("   3. Check if you have write access to the repository")

if __name__ == "__main__":
    main()

