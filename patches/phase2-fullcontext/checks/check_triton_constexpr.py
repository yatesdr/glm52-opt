#!/usr/bin/env python3
"""Reject Triton kernels that close over Python module globals."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_ROOT = PACKAGE_ROOT / "transport" / "overlays"
REQUIRED_CONSTEXPR = {
    "_remap_ckv_topk_kernel": {
        "VIRTUAL_BLOCK",
        "WORLD_SIZE",
        "PAGE_RECORDS",
    },
    "_remap_ckv_pool_topk_kernel": {
        "VIRTUAL_BLOCK",
        "WORLD_SIZE",
        "PAGE_RECORDS",
    },
}


def _bound_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        result: set[str] = set()
        for item in target.elts:
            result.update(_bound_names(item))
        return result
    return set()


def _module_bindings(tree: ast.Module) -> set[str]:
    result: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                result.update(_bound_names(target))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                result.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                result.add(alias.asname or alias.name)
    return result


def _is_triton_jit(decorator: ast.expr) -> bool:
    return (
        isinstance(decorator, ast.Attribute)
        and decorator.attr == "jit"
        and isinstance(decorator.value, ast.Name)
        and decorator.value.id == "triton"
    )


def _constexpr_args(node: ast.FunctionDef) -> set[str]:
    result: set[str] = set()
    for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
        annotation = arg.annotation
        if (
            isinstance(annotation, ast.Attribute)
            and annotation.attr == "constexpr"
            and isinstance(annotation.value, ast.Name)
            and annotation.value.id == "tl"
        ):
            result.add(arg.arg)
    return result


def _audit_file(path: Path) -> tuple[list[str], list[str]]:
    tree = ast.parse(path.read_text(), filename=str(path))
    module_bindings = _module_bindings(tree)
    kernel_names: list[str] = []
    errors: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not any(
            _is_triton_jit(decorator) for decorator in node.decorator_list
        ):
            continue
        kernel_names.append(node.name)
        parameters = {
            arg.arg
            for arg in (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            )
        }
        local_bindings = set(parameters)
        loaded: set[str] = set()
        for statement in node.body:
            for child in ast.walk(statement):
                if isinstance(child, ast.Name):
                    if isinstance(child.ctx, (ast.Store, ast.Del)):
                        local_bindings.add(child.id)
                    elif isinstance(child.ctx, ast.Load):
                        loaded.add(child.id)
        closed_over = sorted(
            (loaded & module_bindings) - local_bindings - {"tl"}
        )
        if closed_over:
            errors.append(
                f"{path}:{node.lineno}: {node.name} closes over module "
                f"globals {closed_over}; pass them as kernel parameters"
            )
        required = REQUIRED_CONSTEXPR.get(node.name, set())
        missing = sorted(required - _constexpr_args(node))
        if missing:
            errors.append(
                f"{path}:{node.lineno}: {node.name} lacks tl.constexpr "
                f"geometry parameters {missing}"
            )
    return kernel_names, errors


def main() -> None:
    paths = (
        tuple(Path(arg) for arg in sys.argv[1:])
        if len(sys.argv) > 1
        else tuple(sorted(OVERLAY_ROOT.rglob("*.py")))
    )
    kernels: list[str] = []
    errors: list[str] = []
    for path in paths:
        file_kernels, file_errors = _audit_file(path)
        kernels.extend(file_kernels)
        errors.extend(file_errors)
    missing_required = sorted(set(REQUIRED_CONSTEXPR) - set(kernels))
    if missing_required:
        errors.append(f"required remap kernels not found: {missing_required}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(
        "PASS Triton constexpr audit: "
        f"files={len(paths)} kernels={len(kernels)} "
        f"names={','.join(kernels)}"
    )


if __name__ == "__main__":
    main()
