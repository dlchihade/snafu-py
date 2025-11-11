# GitHub Upload Scripts for snafu-py

This directory contains scripts to help you upload your snafu-py repository to GitHub.

## Available Scripts

### 1. Python Script (`upload_to_github.py`)
A comprehensive Python script with interactive features and detailed error handling.

**Features:**
- Interactive commit message input
- Detailed error reporting
- Progress indicators
- Automatic branch detection

**Usage:**
```bash
python upload_to_github.py
```

### 2. Shell Script (`upload_to_github.sh`)
A quick and simple shell script for fast uploads.

**Features:**
- Fast execution
- Optional commit message parameter
- Basic error handling

**Usage:**
```bash
# With default commit message
./upload_to_github.sh

# With custom commit message
./upload_to_github.sh "Your custom commit message here"
```

## Prerequisites

Before using these scripts, make sure you have:

1. **Git configured** with your credentials:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

2. **GitHub access** set up with either:
   - SSH keys configured
   - Personal access token configured
   - GitHub CLI installed and authenticated

3. **Write access** to the repository: `https://github.com/AusterweilLab/snafu-py`

## Quick Start

1. **Navigate to your repository:**
   ```bash
   cd /Users/diettachihade/snafu-py
   ```

2. **Run the upload script:**
   ```bash
   # Using Python script (recommended for first-time users)
   python upload_to_github.py
   
   # OR using shell script (for quick uploads)
   ./upload_to_github.sh
   ```

3. **Follow the prompts** to complete the upload process.

## What the Scripts Do

1. **Check repository status** - Verify you're in a Git repository
2. **Detect changes** - Show what files have been modified
3. **Stage changes** - Add all modified files to staging
4. **Commit changes** - Create a commit with your message
5. **Push to GitHub** - Upload changes to the remote repository

## Troubleshooting

### Authentication Issues
If you get authentication errors:

1. **Set up SSH keys:**
   ```bash
   ssh-keygen -t ed25519 -C "your.email@example.com"
   # Add the public key to your GitHub account
   ```

2. **Or use Personal Access Token:**
   - Go to GitHub Settings → Developer settings → Personal access tokens
   - Generate a new token with repo permissions
   - Use the token as your password when prompted

### Permission Issues
If you get permission errors:
- Ensure you have write access to the repository
- Contact the repository owner if needed

### Branch Issues
If you're on a different branch:
```bash
# Check current branch
git branch

# Switch to the desired branch
git checkout main  # or your target branch
```

## Manual Upload (Alternative)

If you prefer to upload manually:

```bash
# Stage all changes
git add .

# Commit changes
git commit -m "Your commit message"

# Push to GitHub
git push origin your-branch-name
```

## Repository Information

- **Repository URL:** https://github.com/AusterweilLab/snafu-py
- **Current Branch:** reorganized-structure
- **Remote:** origin (GitHub)

## Support

If you encounter issues with these scripts, you can:
1. Check the error messages for specific guidance
2. Review the manual upload steps above
3. Contact the repository maintainers

