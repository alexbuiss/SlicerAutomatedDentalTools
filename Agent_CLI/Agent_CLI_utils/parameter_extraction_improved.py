#!/usr/bin/env python3
"""
Improved parameter extraction.

Builds the prompt that asks the LLM to extract a tool's parameters from the
user request. Type conversion / validation of the extracted values is handled
separately by ParameterValidator (see parameter_validator.py), so this module
is only responsible for prompt construction.
"""

from typing import Dict

from Agent_CLI_utils.utils import load_manifest


class ImprovedParameterExtractor:
    """
    Build the parameter-extraction prompt for a given tool, listing every
    parameter with its type, whether it is required, and its encoding so the
    model returns values in the exact format the CLI scripts expect.
    """

    def __init__(self, manifest_path: str = "manifest.yaml"):
        try:
            self.manifest = load_manifest(manifest_path)
        except Exception as e:
            raise RuntimeError(
                f"ImprovedParameterExtractor could not load manifest: {e}"
            )

        self.scripts = {s["name"]: s for s in self.manifest.get("scripts", [])}

    def build_prompt(self, tool_name: str, user_text: str, tool_spec) -> str:
        """Build the extraction prompt for `tool_name`."""

        # 1. Build the parameter section (name, type, required flag, encoding).
        params_section = self._build_params_section(tool_spec)

        # 2. Build the final prompt.
        prompt = f"""Extract parameters from the user request for tool: {tool_name}

PARAMETER DEFINITIONS:
{params_section}

EXTRACTION RULES:
1. For lists: Use the correct format based on 'encode':
   - comma_no_brackets: "1,2,3" (no brackets, no quotes around items)
   - space_separated: "1 2 3" (space separated, no brackets)
   - comma_separated: "1,2,3" (comma separated, no brackets)
   - python_literal: "[1.0, 0.3]" (keep Python array format as string)

2. For booleans: Use JSON format without quotes: true or false

3. For integers: Use JSON format without quotes: 10, 0, 64

4. For floats: Use JSON format without quotes: 1.0, 0.95

5. For paths:
   - If parameter name contains "folder" or "dir": Use folder path only (no filename)
   - Otherwise: Can be file path (with extension)

6. For strings: Use normal string format with quotes: "value"

7. ONLY extract parameters mentioned in the user request
   - Do NOT invent values
   - Leave unmentioned optional parameters absent

USER REQUEST: "{user_text}"

Return ONLY valid JSON on one line (no markdown, no explanation):
{{"extracted": {{...}}, "confidence": 0.0-1.0, "missing_required": [...], "notes": "..."}}
"""

        return prompt

    def _build_params_section(self, tool_spec: Dict) -> str:
        """Build the human-readable parameter list embedded in the prompt."""

        lines = []
        for param in tool_spec.get("parameters", []):
            name = param["name"]
            ptype = param.get("type", "string")
            required = "REQUIRED" if param.get("required") else "optional"
            encode = param.get("encode", "")
            description = param.get("description", "")

            encode_info = f" (encode: {encode})" if encode else ""
            lines.append(f"  - {name} ({ptype}) [{required}]{encode_info}: {description}")

        return "\n".join(lines)
