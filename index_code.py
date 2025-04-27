import os
import pathlib
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
import openai
import lancedb  # type: ignore
import numpy as np
from tqdm import tqdm  # type: ignore
import argparse
from typing import Any
from tiktoken import Encoding

load_dotenv()

PRESETS = {
    'python': ['py', 'pyw', 'pyi'],
    'c_cpp': ['c', 'cpp', 'cc', 'h', 'hpp'],
    'java': ['java', 'jar', 'class'],
    'web': ['html', 'css', 'js', 'ts'],
    'default': ['txt', 'log']
}


def parse_cli_args() -> argparse.Namespace:
    """Set up and parse command-line arguments for codebase indexing."""
    preset_list = ', '.join(
        f"{name}({','.join(exts)})" for name, exts in PRESETS.items()
    )
    parser = argparse.ArgumentParser(
        description='Index codebase for embeddings'
    )

    parser.add_argument(
        '--preset', '-P',
        help=(
            'Comma-separated list of presets to apply. '
            f'Available: {preset_list}'
        )
    )
    parser.add_argument(
        '--include-exts', '-I',
        help='Comma-separated list of extra file extensions to include (e.g., md,json)'
    )
    parser.add_argument(
        '--exclude-dirs', '-E',
        help='Comma-separated list of directories to exclude (e.g., build,venv)'
    )
    parser.add_argument(
        '--embedding-model', '-M',
        default='text-embedding-3-large',
        help='OpenAI embedding model ID'
    )
    parser.add_argument(
        '--proxy-http', '-X',
        help='HTTP proxy URL'
    )
    parser.add_argument(
        '--proxy-https', '-Xs',
        help='HTTPS proxy URL'
    )
    parser.add_argument(
        '--src-dir', '-D',
        default='.',
        help='Directory to search'
    )
    parser.add_argument(
        '--include-hidden', '-H',
        action='store_true',
        help='Include hidden files and directories (default excludes them)'
    )
    return parser.parse_args()


def is_hidden(path: pathlib.Path) -> bool:
    """Return True if any part of the path starts with a dot."""
    return any(part.startswith('.') for part in path.parts)


def chunk(path: pathlib.Path, tokenizer: Any, max_tokens: int = 4096):
    """Yield chunks of the input file as strings, each no longer than max_tokens tokens."""
    buf: list[str] = []
    count: int = 0
    for line in path.read_text(errors="ignore").splitlines():
        t: int = len(tokenizer.encode(line))
        if count + t > max_tokens:
            yield '\n'.join(buf)
            buf, count = [], 0
        buf.append(line)
        count += t
    if buf:
        yield '\n'.join(buf)


def index_codebase(table: Any,
                   src_dir: pathlib.Path,
                   extensions: set[str],
                   excluded_dirs: set[str],
                   tokenizer: Encoding,
                   model: str, client: Any,
                   include_hidden: bool) -> None:
    """
    Index all files in src_dir matching the given extensions, skipping directories in excluded_dirs,
    chunking and embedding their content using the specified model and tokenizer.
    """
    files: list[pathlib.Path] = []
    for ext in extensions:
        glob = src_dir.rglob(f'*.{ext}')
        ext_files = [
            f for f in glob
            if (include_hidden or not is_hidden(f))
            and not any(ex in f.parts for ex in excluded_dirs)
        ]
        files.extend(ext_files)
    total_files: int = len(files)
    with tqdm(total=total_files, desc="Indexing codebase", unit="file") as pbar:
        for f in files:
            for text in chunk(f, tokenizer):
                emb = client.embeddings.create(
                    model=model, input=text).data[0].embedding
                table.add(
                    [{"filename": str(f), "text": text, "vector": np.array(emb)}])
            pbar.update(1)


if __name__ == "__main__":
    # Parse CLI arguments
    args = parse_cli_args()
    include_hidden: bool = args.include_hidden

    # Directories to exclude
    excluded_dirs: set[str] = set()
    if args.exclude_dirs:
        excluded_dirs = {d.strip()
                         for d in args.exclude_dirs.split(',') if d.strip()}

    # Extension set from presets and custom input
    extensions: set[str] = set(PRESETS['default'])
    if args.preset:
        selected: list[str] = [p.strip()
                               for p in args.preset.split(',') if p.strip()]
        extensions = set()
        for name in selected:
            if name in PRESETS:
                extensions.update(PRESETS[name])
            else:
                print(f"Warning: preset '{name}' not recognized. Skipped.")
    if args.include_exts:
        extra_exts = {ext.strip()
                      for ext in args.include_exts.split(',') if ext.strip()}
        extensions.update(extra_exts)

    # Set up proxies and OpenAI client
    proxies = {
        'http': args.proxy_http or os.getenv('HTTP_PROXY'),
        'https': args.proxy_https or os.getenv('HTTPS_PROXY')
    }
    openai.proxy = proxies  # type: ignore[attr-defined]

    src_dir: pathlib.Path = pathlib.Path(args.src_dir)

    model: str = args.embedding_model
    tokenizer: Encoding = tiktoken.encoding_for_model(model)

    # Print which files will be included, and which directories are being excluded
    included_files: list[pathlib.Path] = []
    for ext in extensions:
        for f in src_dir.rglob(f'*.{ext}'):
            if (include_hidden or not is_hidden(f)) \
               and not any(ex in f.parts for ex in excluded_dirs):
                included_files.append(f)

    print('Included files after applying extension and exclusion logic:')
    for f in included_files:
        print(f'  {f}')
    print('Excluded directories:')
    for d in excluded_dirs:
        print(f'  {d}')

    client: Any = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    DB_DIR: pathlib.Path = src_dir / "db"
    DB_DIR.mkdir(parents=True, exist_ok=True)
    TABLE: str = "code_chunks"
    db = lancedb.connect(DB_DIR)
    if TABLE in db.table_names():
        table = db.open_table(TABLE)
    else:
        from lancedb.pydantic import LanceModel, Vector  # type: ignore

        class CodeChunk(LanceModel):
            filename: str
            text: str
            vector: Vector(3072)  # type: ignore

        table = db.create_table(TABLE, schema=CodeChunk, mode="overwrite")

    # Main processing function now uses all parameters including tokenizer and model
    index_codebase(table, src_dir, extensions,
                   excluded_dirs, tokenizer, model, client, include_hidden)
