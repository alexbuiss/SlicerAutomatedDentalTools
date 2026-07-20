from __future__ import annotations

import json
import os
import sys
from typing import Dict, Any, Tuple, List, Optional, TYPE_CHECKING
import yaml

# These imports are only needed for type hints. Importing them at runtime would
# create a circular import (parameter_extraction_improved / parameter_validator
# both import load_manifest from this module), so we guard them behind
# TYPE_CHECKING and rely on "from __future__ import annotations" to keep the
# annotations as plain strings at runtime.
if TYPE_CHECKING:
    from Agent_CLI_utils.parameter_extraction_improved import ImprovedParameterExtractor
    from Agent_CLI_utils.parameter_validator import ParameterValidator

# Single source of truth for the Ollama model used by the agent. Can be
# overridden with the ROUTER_MODEL environment variable, otherwise every call
# site (router, parameter extraction, repair loop) uses qwen3:8b.
DEFAULT_MODEL = "qwen3:8b"


def get_router_model() -> str:
    """Return the Ollama model name, honouring the ROUTER_MODEL override."""
    return os.environ.get("ROUTER_MODEL", DEFAULT_MODEL)


def load_manifest(manifest_path):
    """Load the manifest (YAML) holding all the tool descriptions."""
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        # Re-raise with a clearer, path-aware message so a missing or malformed
        # manifest is easy to diagnose from the CLI log.
        raise RuntimeError(f"Failed to load manifest '{manifest_path}': {e}")

_cross_encoder_model = None

def _get_cross_encoder():
    """
    Lazily load and cache the cross-encoder model for the lifetime of this
    process. Agent_CLI.py runs as a fresh subprocess per request, so this
    cache doesn't persist across requests, but it avoids reloading the model
    twice within a single one (e.g. if candidate retrieval ever runs more
    than once per run).
    """
    global _cross_encoder_model
    if _cross_encoder_model is None:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            # sentence-transformers (and its torch/transformers deps) may be
            # missing or broken; surface a clear, actionable error instead of a
            # raw ImportError deep in the retrieval path.
            raise RuntimeError(
                f"Could not load the cross-encoder model: {e}. "
                "Click the 'Check' button in the Agent_UI module to (re)install "
                "the dependencies."
            )
    return _cross_encoder_model

def cross_encoder_retrieve_candidates(manifest: Dict, user_text: str, k: int = 3) -> List[Dict]:
    """
    Rerank every tool in the manifest against the user's request with a
    cross-encoder and return the top-k tool specs (dicts from manifest
    "scripts"). Replaces the previous keyword/fuzzy-matching heuristic: a
    cross-encoder scores the (query, tool description) pair jointly through
    a small transformer, so it catches semantic/paraphrase matches that
    plain string or token overlap misses (e.g. "line up my scans" should
    still surface a "registration" tool even with zero shared words).
    """
    scripts = manifest.get("scripts", [])
    if not scripts:
        return []

    try:
        docs = []
        for s in scripts:
            tags = ", ".join(str(t) for t in (s.get("tags") or []))
            docs.append(f"{s.get('name', '')}: {s.get('description', '')} (tags: {tags})")

        model = _get_cross_encoder()
        scores = model.predict([(user_text, doc) for doc in docs])

        ranked = sorted(zip(scores, scripts), key=lambda x: x[0], reverse=True)
        return [s for _, s in ranked[:k]]
    except Exception as e:
        # If reranking fails for any reason, fall back to the first k tools so
        # the router still has candidates to choose from rather than crashing.
        print(f"Cross-encoder retrieval failed, falling back to first {k} tools: {e}")
        return scripts[:k]

def chat_with_auto_pull(model: str, messages: List[Dict[str, str]], **kwargs):
    """
    Call ollama.chat(); if the model hasn't been pulled to this machine yet
    ("model '<name>' not found", HTTP 404), pull it once (can take a while
    for large models) and retry. Any other failure (e.g. Ollama not
    installed/running at all) is re-raised unchanged - pulling a model
    can't fix that.
    """
    import ollama

    try:
        return ollama.chat(model=model, messages=messages, **kwargs)
    except Exception as e:
        if "not found" not in str(e).lower():
            raise
        print(f"Model '{model}' not found locally, pulling it now (this can take a while)...")
        last_status = None
        for progress in ollama.pull(model, stream=True):
            status = progress.get("status")
            if status and status != last_status:
                print(f"  pull {model}: {status}")
                last_status = status
        return ollama.chat(model=model, messages=messages, **kwargs)

def get_tool_def(manifest, tool_name: str):
    for tool in manifest.get("scripts", []):
        if tool.get("name") == tool_name:
            return tool
    return None

def complete_with_defaults(manifest, tool_name: str, params: dict):
    tool = get_tool_def(manifest, tool_name)
    if not tool:
        return params
    defaults = {p.get("name",""):p.get("default","") for p in tool.get("parameters", []) if not p.get("required", False) and "default" in p}
    for name,default in defaults.items():
        if name not in params:
            params[name]= default

    return params

def resolve_tool_path(rel_path: str, manifest_dir: str) -> str:
    """
    Resolve a tool's manifest path (a bare filename like "ALI_CBCT.py") to an
    absolute path on disk. Tries, in order:
      1. AGENT_CLI_TOOLS_DIR env var (lets a deployment point at any folder)
      2. "<manifest_dir>/CLI files/<rel_path>" (extension bundles its tools)
      3. "<manifest_dir>/<rel_path>" (tool sitting next to the manifest)
      4. a sibling "CLI files" folder found by walking up from manifest_dir
         (matches the current repo layout: AI_Agent/Agent_CLI/ + .../CLI files/
         at the repo root, with no env var needed)
    Returns the path unchanged if it's already absolute (back-compat with
    manifests that still have a full path).
    """
    if os.path.isabs(rel_path):
        return rel_path

    candidates = []
    env_dir = os.environ.get("AGENT_CLI_TOOLS_DIR")
    if env_dir:
        candidates.append(os.path.join(env_dir, rel_path))
    candidates.append(os.path.join(manifest_dir, "CLI files", rel_path))
    candidates.append(os.path.join(manifest_dir, rel_path))

    current = manifest_dir
    for _ in range(4):
        current = os.path.dirname(current)
        if not current or current == os.path.dirname(current):
            break
        candidates.append(os.path.join(current, "CLI files", rel_path))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    # Nothing found on disk: return the most likely candidate anyway so the
    # resulting error message points at a real, debuggable path.
    return candidates[0]

def build_tool_spec(manifest):
    """Build tool specifications with memoization support."""
    tools = []
    for s in manifest.get("scripts", []):
        # Extract parameters from "parameters" (new structure)
        params = set()
        required_params = set()
        
        # Support old structure (patterns) and new (parameters)
        if "patterns" in s:
            # Old structure
            for p in s.get("patterns", []) or []:
                for g, flag in (p.get("args") or {}).items():
                    params.add(g)
        
        if "parameters" in s:
            # New structure
            for param in s.get("parameters", []):
                param_name = param.get("name", "")
                if param_name:
                    params.add(param_name)
                    if param.get("required", False):
                        required_params.add(param_name)
        
        # Support old structure defaults
        if "defaults" in s:
            params.update(s["defaults"].keys())

        tools.append({
            "name": s["name"],
            "description": s.get("description",""),
            # "params": sorted(list(params)),           # logical names (not "--flag")
            "required_params": sorted(list(required_params)),
            # "path": s["path"],
            "tags": s.get("tags", []),
            # "priority": s.get("priority", 0),
        })
    return tools

def extract_parameters(manifest: Dict, tool_name: str, user_text: str,improved_extractor:ImprovedParameterExtractor,parameter_validator:ParameterValidator,model,history: List[Dict[str, str]] = None) -> Tuple[Dict[str, Any], float, List[str]]:
    scripts = {s["name"]: s for s in manifest.get("scripts", [])}
    tool_spec = scripts.get(tool_name)
    if not tool_spec:
        # print(f"Tool {tool_name} not found in manifest")
        return {}, 0.0, []
    
    name_temp = ["temp_fold","tmp_folder","temp_folder","log_path","logPath"]

    tool_spec_clear = tool_spec.copy()

    # Hide temp/log parameters from the model so it doesn't try to fill them;
    # they are injected afterwards from the caller-provided temp folder.
    params = tool_spec.get("parameters", [])
    tool_spec_clear["parameters"] = [p for p in params if p.get("name") not in name_temp]

    removed = [p["name"] for p in tool_spec.get("parameters", []) if p.get("name") in name_temp]

    try:
        # Build the extraction prompt (with few-shot style instructions).
        prompt = improved_extractor.build_prompt(tool_name, user_text, tool_spec_clear)
        router_system = "You are a parameter extraction expert. Output ONLY valid JSON on one line."

        try:
            response = chat_with_auto_pull(
                model,
                messages=[
                    {"role": "system", "content": router_system},
                    *(history or []),
                    {"role": "user", "content": prompt}
                ],
                format="json"
            )
        except Exception as e:
            raise RuntimeError(f"Router error: {e}")

        data = json.loads(response["message"]["content"])
        extracted_raw = data.get("extracted", {})
        confidence = float(data.get("confidence", 0.0))

        # Validation (already converts types via _check_types, so there is no
        # need to also run improved_extractor.convert_types beforehand).
        validation_result = parameter_validator.validate(tool_name, extracted_raw)

        # Return the validated parameters.
        final_params = validation_result["params"]
        final_confidence = confidence if validation_result["valid"] else confidence * 0.6
        missing_required = validation_result["missing_required"]

        return final_params, final_confidence, missing_required, removed

    except Exception as e:
        # Never let extraction crash the whole request: return empty params so
        # the caller reports "missing parameters" instead of failing hard. The
        # required parameters become the missing list.
        print(f"Parameter extraction failed: {e}")
        required = [
            p.get("name") for p in tool_spec.get("parameters", [])
            if p.get("required", False) and p.get("name") not in name_temp
        ]
        return {}, 0.0, required, removed

def build_repair_prompt(tool_name: str, tool_spec: Dict, params: Dict[str, Any], cli_args: List[str], stderr: str) -> str:
    """
    Build a prompt asking the LLM to propose corrected parameters for a tool
    invocation that was just executed and failed. Used by the post-execution
    repair loop (Agent_UI.onCliUpdated) once subprocess.run() returns a
    non-zero exit code.
    """
    params_desc = []
    for p in tool_spec.get("parameters", []):
        name = p.get("name", "")
        ptype = p.get("type", "string")
        description = p.get("description", "")
        params_desc.append(f"  - {name} ({ptype}): {description}")
    params_section = "\n".join(params_desc) if params_desc else "  (no parameters)"

    params_used = "\n".join(f"  {k} = {v!r}" for k, v in (params or {}).items())
    command_str = " ".join(str(a) for a in cli_args)
    stderr_tail = (stderr or "")[-2000:]

    return f"""The tool "{tool_name}" was just executed and FAILED. Your job is to propose corrected parameters.

PARAMETER DEFINITIONS:
{params_section}

PARAMETERS USED IN THE FAILED RUN:
{params_used}

FAILED COMMAND:
{command_str}

ERROR OUTPUT (stderr, possibly truncated):
{stderr_tail}

Look at the error and decide which parameter(s) caused it (e.g. wrong path, wrong type, wrong format) and
propose corrected values for ONLY the parameters that need to change. Omit any parameter that should stay
the same - it will be kept as-is automatically.

Return ONLY valid JSON on one line (no markdown, no explanation):
{{"extracted": {{"param_name": "corrected_value", ...}}, "confidence": 0.0-1.0, "explanation": "..."}}
"""

def _encode_list_value(value, encode: str) -> str:
    """
    Serialize a "list"-typed parameter back into the plain CLI string the
    underlying tool script expects. The validator already turned a
    user-supplied string into a real Python list via _convert_list (and
    manifest defaults are plain YAML lists), so by the time this runs the
    value is normally a list - join it back per the manifest's "encode"
    (the CLI scripts split on "," or " ", never on a Python list repr).
    """
    if isinstance(value, str):
        return value
    items = [str(x) for x in value]
    if encode == "space_separated":
        return " ".join(items)
    return ",".join(items)


def _format_cli_value(value, param_def: Optional[Dict]) -> str:
    ptype = (param_def or {}).get("type", "")
    encode = (param_def or {}).get("encode", "")
    if ptype.startswith("list"):
        return _encode_list_value(value, encode)
    if ptype == "bool" and encode == "lower_str_bool" and not isinstance(value, str):
        return "true" if value else "false"
    return str(value)


def build_cli_args(tool_name: str, params: Dict[str, Any], manifest: Dict, manifest_dir: str = None) -> List[str]:
    """Build the command-line argument list for the underlying tool script."""

    scripts = {s["name"]: s for s in manifest.get("scripts", [])}
    if tool_name not in scripts:
        raise KeyError(f"Tool '{tool_name}' not found in manifest")
    spec = scripts[tool_name]

    tool_path = spec["path"]
    if manifest_dir:
        tool_path = resolve_tool_path(tool_path, manifest_dir)

    param_list = spec.get("parameters", [])
    param_by_name = {p.get("name", ""): p for p in param_list}

    # Merge extracted params with the manifest defaults.
    defaults = {
        p["name"]: p["default"]
        for p in param_list
        if "default" in p
    }
    merged = dict(defaults)
    merged.update({k: v for k, v in (params or {}).items() if v is not None})
    cli_style = spec.get("cli_style", "positional")

    if cli_style == "positional":
        # Positional arguments
        positional_order = spec.get("positional_order", [])

        cli = []
        if positional_order:
            for param_name in positional_order:
                if param_name in merged:
                    cli.append(_format_cli_value(merged[param_name], param_by_name.get(param_name)))
        else:
            for param in param_list:
                param_name = param.get("name", "")
                if param_name in merged:
                    cli.append(_format_cli_value(merged[param_name], param))

        return [sys.executable, tool_path] + cli
    else:
        # Named arguments (--flag style)
        param2flag = {}
        for p in param_list:
            param_name = p.get("name", "")
            if param_name:
                flag = p.get("flag", f"--{param_name}")
                param2flag[param_name] = flag

        cli = []
        for k, v in merged.items():
            flag = param2flag.get(k, f"--{k}")
            cli += [flag, _format_cli_value(v, param_by_name.get(k))]

        return [sys.executable, tool_path] + cli