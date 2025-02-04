# files-to-chat: Process Files for LLM Prompt Contexts

`files-to-chat` is a command-line tool designed to process files or folders, converting their contents into a format suitable for use as a prompt context to chat with large language models (LLMs).

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Cleaning Up the Prompt Cache](#cleaning-up-the-prompt-cache)

## Installation

To install and use `files-to-chat`, you have two options:

1. **Using pip**:
   ```sh
   pip install files-to-chat
   ```

2. **From Source**:
   ```sh
   git clone https://github.com/mzbac/files-to-chat.git
   cd files-to-chat
   pip install .
   ```

## Usage

Run the tool from the command line to chat with processed documents:

```sh
files-to-chat path/to/file_or_folder
```

For detailed options, run:

```sh
files-to-chat --help
```

### Example Usage

```sh
files-to-chat /path/to/your/file.txt
```

This command will process the specified file and generate a prompt context suitable for use in a chat session with an LLM.

## Cleaning Up the Prompt Cache

Prompt caches are stored under your home cache directory. To remove the cache, run:

```sh
rm -rf ~/.cache/files-to-chat/prompt_cache
```
