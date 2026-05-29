# /// script
# dependencies = [
#     "httpx>=0.27.0",
#     "websocket-client>=1.8.0",
# ]
# ///
import ast
import sys
import json
import uuid
import argparse
import time
from pathlib import Path
from urllib.parse import urlparse
import httpx
from websocket import create_connection

CONFIG_FILE = Path(".jupyter_config.json")
URL_TIMEOUT_SECONDS = 3600  # 1 hour expiration window

def get_cached_url():
    """Retrieve URL from local JSON config, validating expiration."""
    if CONFIG_FILE.exists():
        try:
            config = json.loads(CONFIG_FILE.read_text())
            elapsed = time.time() - config.get("timestamp", 0)
            if elapsed > URL_TIMEOUT_SECONDS:
                print("⏳ Cached Kaggle URL might have expired. Consider providing a fresh URL.")
            return config.get("url")
        except Exception:
            pass
    return None

def save_url(url):
    """Save the fresh URL with a current timestamp."""
    config = {"url": url, "timestamp": time.time()}
    CONFIG_FILE.write_text(json.dumps(config, indent=2))
    print("💾 URL successfully cached locally.")

def execute_payload(base_url, code_payload):
    """Quietly connects to the remote kernel and pipes execution data."""
    base_url = base_url.rstrip('/')
    headers = {} # Extend here if your token handling requires headers

    # 1. Fetch active kernel ID
    try:
        with httpx.Client(verify=False) as client:
            response = client.get(f"{base_url}/api/kernels", headers=headers, timeout=5.0)
            response.raise_for_status()
            kernels = response.json()
            if not kernels:
                print("❌ Error: No active kernels found on Kaggle. Run a cell manually in the UI first.")
                return False
            kernel_id = kernels[0]['id']
    except Exception as e:
        print(f"❌ Failed to reach Jupyter API: {e}")
        return False

    # 2. Open WebSocket stream
    parsed_url = urlparse(base_url)
    ws_scheme = "wss" if parsed_url.scheme == "https" else "ws"
    ws_url = f"{ws_scheme}://{parsed_url.netloc}{parsed_url.path}/api/kernels/{kernel_id}/channels"

    try:
        ws = create_connection(ws_url, header=[f"{k}: {v}" for k, v in headers.items()])
    except Exception as e:
        print(f"❌ WebSocket Connection Failed: {e}")
        return False

    # 3. Fire-and-forget payload structure
    msg_id = str(uuid.uuid4())
    execute_request = {
        "header": {
            "msg_id": msg_id,
            "username": "local_agent",
            "session": str(uuid.uuid4()),
            "msg_type": "execute_request",
            "version": "5.3"
        },
        "metadata": {},
        "content": {
            "code": code_payload,
            "silent": False,
            "store_history": True,
            "allow_stdin": False,
            "stop_on_error": True
        },
        "parent_header": {}
    }

    ws.send(json.dumps(execute_request))

    # 4. Stream output (Only prints your code's stdout/stderr)
    success = True
    try:
        while True:
            msg = json.loads(ws.recv())
            if msg.get('parent_header', {}).get('msg_id') == msg_id:
                msg_type = msg.get('msg_type')
                content = msg.get('content', {})

                if msg_type == 'stream':
                    sys.stdout.write(content.get('text', ''))
                    sys.stdout.flush()
                elif msg_type == 'error':
                    success = False
                    print("\n❌ Execution Error:")
                    for line in content.get('traceback', []):
                        print(line)
                elif msg_type == 'execute_reply':
                    if content.get('status') == 'error':
                        success = False
                    break
    except KeyboardInterrupt:
        print("\n⚠️ Detached from stream.")
    finally:
        ws.close()
    return success

def get_module_metadata(file_path):
    """Parses a file to find what it defines and what local files it imports."""
    path = Path(file_path)
    if not path.exists():
        return set(), []

    try:
        root = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as e:
        print(f"❌ Syntax error while parsing {file_path}: {e}")
        sys.exit(1)

    defined_names = set()
    local_imports = []

    for node in ast.walk(root):
        # 1. Record things defined in this file (functions, classes, constants)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined_names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined_names.add(target.id)

        # 2. Identify import statements
        elif isinstance(node, ast.Import):
            for alias in node.names:
                local_imports.append((alias.name, node))
        elif isinstance(node, ast.ImportFrom) and node.module:
            local_imports.append((node.module, node))

    # Resolve imported names to actual local file paths if they exist
    resolved_dependencies = []
    for mod_name, import_node in local_imports:
        # Convert 'orbit_wars_data.main' or 'C.A' to physical paths
        base_rel_path = mod_name.replace('.', '/')
        possible_files = [Path(f"{base_rel_path}.py"), Path(base_rel_path) / "__init__.py"]
        
        for p in possible_files:
            if p.exists() and p.is_file():
                resolved_dependencies.append((p.resolve(), import_node))
                break

    return defined_names, resolved_dependencies

def prepare_combined_code(script_paths):
    """
    Builds a dependency DAG, orders files topologically to prevent duplicates,
    and strips out local import lines from the final bundled payload.
    """
    resolved_order = []      # Final linear order of files to concatenate
    visited = set()          # Tracks processed files to handle shared dependencies (A1)
    visiting = set()         # Tracks active recursion stack to catch infinite loops (A -> B -> A)
    all_defined_names = set() # Combined registry of everything defined so far
    nodes_to_remove = {}     # Maps specific file paths to AST import nodes that need stripping

    def dfs(file_path):
        abs_path = Path(file_path).resolve()
        if abs_path in visiting:
            print(f"❌ Circular dependency detected involving: {file_path}")
            sys.exit(1)
        if abs_path in visited:
            return

        visiting.add(abs_path)
        
        # Look ahead at what this file needs
        defined, dependencies = get_module_metadata(abs_path)
        
        for dep_path, import_node in dependencies:
            # If the imported module's elements aren't already defined by a previous script cell
            # we must crawl and inject that dependency first.
            if dep_path not in visited:
                dfs(dep_path)
            
            # Queue this import line to be stripped out so the remote kernel doesn't fail
            nodes_to_remove.setdefault(abs_path, []).append(import_node)

        visiting.remove(abs_path)
        visited.add(abs_path)
        resolved_order.append(abs_path)

    # Process all user-passed scripts in sequence (treating them like cell orders)
    for script in script_paths:
        dfs(script)

    # Build final payload by assembling the source and scrubbing obsolete imports
    final_payload_blocks = []
    
    for path in resolved_order:
        lines = path.read_text(encoding="utf-8").splitlines()
        strip_nodes = nodes_to_remove.get(path, [])
        
        # Identify lines containing local imports we want to block
        lines_to_strip = set()
        for node in strip_nodes:
            # ast node lineno attributes are 1-indexed
            for i in range(node.lineno, getattr(node, 'end_lineno', node.lineno) + 1):
                lines_to_strip.add(i - 1)

        clean_lines = [
            (f"# [Stripped local import]" if idx in lines_to_strip else line)
            for idx, line in enumerate(lines)
        ]
        
        final_payload_blocks.append(f"# === File: {path.name} ===\n" + "\n".join(clean_lines))

    return "\n\n".join(final_payload_blocks)

def main():
    parser = argparse.ArgumentParser(description="Quiet Remote Jupyter Script Runner via uv.")
    parser.add_argument("scripts", nargs="+", help="One or more Python files to run. Last file is treated as primary.")
    parser.add_argument("--url", help="Provide a fresh long Kaggle session proxy URL to reset cache.")
    args = parser.parse_args()

    # Determine URL configuration route
    target_url = args.url
    if not target_url:
        target_url = get_cached_url()
        
    if not target_url:
        print("❌ Missing target URL. Run again with the initialization parameter:\n"
              "uv run remote_run.py --url \"https://...\" your_script.py")
        sys.exit(1)

    # Cache if explicitly provided
    if args.url:
        save_url(args.url)

    # Package code blocks and map modules
    final_payload = prepare_combined_code(args.scripts)
    
    # Fire 
    execute_payload(target_url, final_payload)

if __name__ == "__main__":
    main()