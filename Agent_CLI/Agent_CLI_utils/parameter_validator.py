#!/usr/bin/env python3
"""
Parameter validator - validate and coerce the parameters extracted by the LLM.

This is the second stage after extraction: it checks required parameters,
converts values to their declared types, validates paths / enums / ranges and
reports anything missing or out of spec.
"""

from typing import Dict, List, Tuple, Any
from pathlib import Path
from collections.abc import Sequence

from Agent_CLI_utils.utils import load_manifest


class ParameterValidator:
    """Validate and coerce the parameters extracted for a tool."""

    def __init__(self, manifest_path: str = "manifest.yaml"):
        try:
            self.manifest = load_manifest(manifest_path)
        except Exception as e:
            raise RuntimeError(f"ParameterValidator could not load manifest: {e}")

        self.scripts = {s["name"]: s for s in self.manifest.get("scripts", [])}

    def validate(self, tool_name: str, extracted_params: Dict) -> Dict:
        """
        Validate the extracted parameters.

        Returns a dict with:
            {
                "valid": bool,
                "params": dict (validated/coerced params),
                "errors": list,
                "warnings": list,
                "missing_required": list,
                "extra_params": list
            }
        """

        tool_spec = self.scripts.get(tool_name)
        if not tool_spec:
            return {
                "valid": False,
                "params": {},
                "errors": [f"Tool '{tool_name}' not found in manifest"],
                "warnings": [],
                "missing_required": [],
                "extra_params": []
            }

        errors = []
        warnings = []
        corrected_params = extracted_params.copy()

        # 1. Check required parameters.
        required_errors, missing_required = self._check_required_params(
            tool_spec, extracted_params
        )
        errors.extend(required_errors)

        # 2. Check / coerce types.
        type_errors, corrected_params = self._check_types(
            tool_spec, corrected_params
        )
        errors.extend(type_errors)

        # 3. Check path syntax.
        path_errors = self._check_paths(tool_spec, corrected_params)
        errors.extend(path_errors)

        # 4. Check enums (choices).
        enum_errors = self._check_enums(tool_spec, corrected_params)
        errors.extend(enum_errors)

        # 5. Check ranges (min/max).
        range_errors = self._check_ranges(tool_spec, corrected_params)
        errors.extend(range_errors)

        # 6. Check for extra parameters not in the spec.
        extra_params = self._check_extra_params(tool_spec, extracted_params)
        if extra_params:
            warnings.append(f"Extra parameters not in spec: {extra_params}")

        return {
            "valid": len(errors) == 0,
            "params": corrected_params,
            "errors": errors,
            "warnings": warnings,
            "missing_required": missing_required,
            "extra_params": extra_params
        }

    def _check_required_params(
        self,
        tool_spec: Dict,
        params: Dict
    ) -> Tuple[List[str], List[str]]:
        """Check that all required parameters are present."""

        errors = []
        missing = []

        for param in tool_spec.get("parameters", []):
            name = param["name"]
            if param.get("required", False) and name not in params:
                msg = f"MISSING REQUIRED PARAMETER: {name}"
                errors.append(msg)
                missing.append(name)

        return errors, missing

    def _check_types(
        self,
        tool_spec: Dict,
        params: Dict
    ) -> Tuple[List[str], Dict]:
        """
        Check and convert types.
        Returns (errors, corrected_params).
        """

        errors = []
        corrected = params.copy()

        for param in tool_spec.get("parameters", []):
            name = param["name"]
            if name not in params:
                continue

            ptype = param.get("type", "string")
            value = params[name]

            try:
                # Attempt conversion.
                if ptype == "bool":
                    converted = self._convert_bool(value)
                    encode = param.get("encode", "")
                    if encode == "lower_str_bool":
                        # Some CLI scripts compare the raw arg to "true"/"false"
                        # case-sensitively (no .lower() on their side), so
                        # Python's str(bool) ("True"/"False") would never match.
                        corrected[name] = "true" if converted else "false"
                    else:
                        corrected[name] = converted

                elif ptype == "int":
                    corrected[name] = self._convert_int(value)

                elif ptype == "float":
                    corrected[name] = self._convert_float(value)

                elif ptype == "list":
                    encode = param.get("encode", "")
                    corrected[name] = self._convert_list(value, encode)

                elif ptype == "path":
                    corrected[name] = self._convert_path(value)

                else:  # string
                    corrected[name] = str(value)

            except ValueError as e:
                errors.append(f"TYPE ERROR in '{name}': {str(e)}")

        return errors, corrected

    def _check_paths(self, tool_spec: Dict, params: Dict) -> List[str]:
        """Check path syntax."""

        errors = []

        for param in tool_spec.get("parameters", []):
            name = param["name"]
            if name not in params or param.get("type") != "path":
                continue

            value = params[name]

            # Check that it is a valid string path.
            try:
                Path(str(value))
            except Exception:
                errors.append(f"INVALID PATH '{name}': {value}")

            # Folder paths should not carry a file extension.
            if "folder" in name or "dir" in name:
                str_value = str(value)
                if str_value.endswith((".nii.gz", ".nii", ".stl", ".ply", ".vtk", ".json", ".txt", ".log")):
                    errors.append(
                        f"FOLDER PATH SHOULD NOT HAVE EXTENSION '{name}': {value}. "
                        f"Remove the filename part."
                    )

        return errors

    def _check_enums(self, tool_spec: Dict, params: Dict) -> List[str]:
        """Check that values belong to their allowed choices."""

        errors = []

        for param in tool_spec.get("parameters", []):
            name = param["name"]
            if name not in params or "choices" not in param:
                continue

            value = params[name]
            choices = param["choices"]

            if value not in choices:
                errors.append(
                    f"INVALID VALUE '{name}': {value}. "
                    f"Must be one of: {choices}"
                )

        return errors

    def _check_ranges(self, tool_spec: Dict, params: Dict) -> List[str]:
        """Check numeric ranges (min/max)."""

        errors = []

        for param in tool_spec.get("parameters", []):
            name = param["name"]
            if name not in params:
                continue

            value = params[name]

            # Check min.
            if "min" in param:
                try:
                    if float(value) < float(param["min"]):
                        errors.append(
                            f"VALUE TOO SMALL '{name}': {value} < {param['min']}"
                        )
                except (TypeError, ValueError):
                    pass  # Skip if not a number.

            # Check max.
            if "max" in param:
                try:
                    if float(value) > float(param["max"]):
                        errors.append(
                            f"VALUE TOO LARGE '{name}': {value} > {param['max']}"
                        )
                except (TypeError, ValueError):
                    pass  # Skip if not a number.

        return errors

    def _check_extra_params(self, tool_spec: Dict, params: Dict) -> List[str]:
        """Return parameters supplied that are not declared in the spec."""

        spec_names = {p["name"] for p in tool_spec.get("parameters", [])}
        param_names = set(params.keys())

        return list(param_names - spec_names)

    # ===== CONVERSION METHODS =====

    @staticmethod
    def _convert_bool(value) -> bool:
        """Convert a value to a boolean."""
        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            if value.lower() in ("true", "yes", "1", "on"):
                return True
            elif value.lower() in ("false", "no", "0", "off"):
                return False

        if isinstance(value, int):
            return value != 0

        raise ValueError(f"Cannot convert '{value}' to boolean")

    @staticmethod
    def _convert_int(value) -> int:
        """Convert a value to an integer."""
        if isinstance(value, int):
            return value

        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            raise ValueError(f"Cannot convert '{value}' to integer")

    @staticmethod
    def _convert_float(value) -> float:
        """Convert a value to a float."""
        if isinstance(value, (int, float)):
            return float(value)

        try:
            return float(str(value))
        except (TypeError, ValueError):
            raise ValueError(f"Cannot convert '{value}' to float")

    @staticmethod
    def _convert_list(value, encode: str) -> Any:
        """Convert a value to a list according to its encoding."""

        # Accept list-like values (ObservedList, tuple, etc.) but not strings.
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, dict)):
            return [str(x) for x in list(value)]

        if isinstance(value, str):
            s = value.strip()

            if encode == "python_literal":
                return s

            if encode == "space_separated":
                items = s.split()
            else:
                if s.startswith("[") and s.endswith("]"):
                    s = s[1:-1]
                items = [x.strip() for x in s.split(",")]

            return items

        raise ValueError(f"Cannot convert '{value}' to list")

    @staticmethod
    def _convert_path(value) -> str:
        """Convert a value to a path (just trim whitespace)."""
        return str(value).strip()


class ValidationReport:
    """Generate a human-readable validation report."""

    @staticmethod
    def generate_report(tool_name: str, validation_result: Dict) -> str:
        """Generate a readable report from a validation result."""

        report = []
        report.append(f"\n{'='*80}")
        report.append(f"VALIDATION REPORT FOR: {tool_name}")
        report.append(f"{'='*80}")

        # Status
        status = "VALID" if validation_result["valid"] else "INVALID"
        report.append(f"\nStatus: {status}")

        # Missing required
        if validation_result["missing_required"]:
            report.append(f"\nMissing Required Parameters:")
            for param in validation_result["missing_required"]:
                report.append(f"   - {param}")

        # Errors
        if validation_result["errors"]:
            report.append(f"\nErrors:")
            for error in validation_result["errors"]:
                report.append(f"   - {error}")

        # Warnings
        if validation_result["warnings"]:
            report.append(f"\nWarnings:")
            for warning in validation_result["warnings"]:
                report.append(f"   - {warning}")

        # Extra params
        if validation_result["extra_params"]:
            report.append(f"\nExtra Parameters (not in spec):")
            for param in validation_result["extra_params"]:
                report.append(f"   - {param}")

        # Validated params
        if validation_result["params"]:
            report.append(f"\nValidated Parameters:")
            for key, value in validation_result["params"].items():
                if isinstance(value, list):
                    report.append(f"   {key}: {value} (list with {len(value)} items)")
                else:
                    report.append(f"   {key}: {value}")

        report.append(f"\n{'='*80}\n")

        return "\n".join(report)
