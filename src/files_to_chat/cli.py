import argparse
import json
import os
from pathlib import Path

from .prompt_cache import create_prompt_cache

from .docs import convert_file_to_markdown, convert_files_in_folder, print_as_xml
from mlx_lm.utils import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.cache import load_prompt_cache

DEFAULT_TEMP = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 1024


def setup_arg_parser():
    parser = argparse.ArgumentParser(
        description="Process files and using it as prompt cache to talk to a model"
    )
    parser.add_argument("path", nargs="?", help="Path to a file or folder to process")
    parser.add_argument(
        "-e",
        "--extensions",
        metavar="EXT",
        type=str,
        nargs="+",
        default=[],
        help="File extensions to check (e.g., .pdf .py)",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        default=False,
        help="Include hidden files in processing",
    )
    parser.add_argument(
        "--ignore_pattern",
        metavar="PATTERN",
        type=str,
        default="",
        help="Ignore files matching the specified pattern",
    )
    parser.add_argument(
        "--model",
        metavar="MODEL",
        type=str,
        default="mlx-community/Qwen2.5-7B-Instruct-1M-4bit",
        help="Model identifier (default: mlx-community/Qwen2.5-7B-Instruct-1M-4bit)",
    )
    parser.add_argument(
        "--temp", type=float, default=DEFAULT_TEMP, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=DEFAULT_TOP_P, help="Sampling top-p"
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-creating prompt cache even if it exists",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        type=str,
        help="Path to output processed documents to a single file instead of creating a prompt cache",
    )
    return parser


def cli():
    args = setup_arg_parser().parse_args()
    args.path = os.path.abspath(args.path)

    # Process documents regardless of output mode
    if args.path:
        if os.path.isdir(args.path):
            print(
                f"[INFO] Converting all files in folder: {args.path} and formatting documents"
            )
            docs = convert_files_in_folder(
                args.path, args.ignore_pattern, args.include_hidden, args.extensions
            )
        else:
            print(f"[INFO] Converting file: {args.path} and formatting document")
            docs = [convert_file_to_markdown(args.path)]

    # Format documents for output
    formatted_docs = ["<documents>"]
    for doc in docs:
        formatted_docs.append(print_as_xml(doc["path"], doc["content"]))
    formatted_docs.append("</documents>")
    formatted_content = "\n".join(formatted_docs)

    # If output file specified, write documents to file and exit
    if args.output_file:
        output_path = os.path.abspath(args.output_file)
        print(f"[INFO] Writing processed documents to {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(formatted_content)
        print(f"[INFO] Documents successfully written to {output_path}")
        return

    # Otherwise continue with prompt cache and chat functionality
    abs_name = args.path.replace(os.sep, "_")
    model_path = args.model.replace(os.sep, "_")
    cache_name = f"{model_path}{abs_name}.safetensors"

    base_cache_dir = Path.home() / ".cache"

    cache_dir = base_cache_dir / "files-to-chat" / "prompt_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_name)
    prompt_cache, model, tokenizer = None, None, None
    if args.force or not os.path.exists(cache_path):
        print(f"[INFO] Creating prompt cache for {args.model} in {cache_path}")
        prompt_cache, model, tokenizer = create_prompt_cache(
            args.model, formatted_content, cache_path
        )
    else:
        print(f"[INFO] Loading prompt cache from {cache_path}")
        prompt_cache, metadata = load_prompt_cache(
            cache_path,
            return_metadata=True,
        )
        model, tokenizer = load(
            args.model, tokenizer_config=json.loads(metadata["tokenizer_config"])
        )

    print(f"[INFO] Starting chat session with {args.model}. To exit, enter 'q'.")
    while True:
        query = input(">> ")
        if query == "q":
            break
        messages = [{"role": "user", "content": query}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            sampler=make_sampler(args.temp, args.top_p),
            prompt_cache=prompt_cache,
        ):
            print(response.text, flush=True, end="")
        print()