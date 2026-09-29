"""Final architecture audit for the AJD/DSL application structure.

The audit protects the architecture established by the DSL migration:
- AJD responsibilities and memory contracts must cover all workflows.
- Workflow and topology DSL definitions must validate as a catalog.
- compatibility facades remain import-only shims.
- new package code must import concern-specific capability modules, not facades.
- raw LLM retry mechanics remain confined to the common execution layer.

This is intentionally a structural audit, not another workflow runtime.
"""
from __future__ import annotations

import ast
from importlib import resources
from typing import Any, Iterable

from research_fellow.application.dsl import (
    load_topology_definition,
    topology_graph,
    validate_capability_catalog,
    validate_workflow_catalog,
)


COMPATIBILITY_FACADES = {
    "advising.py",
    "advising_research_questions.py",
    "claim_curation.py",
    "progressive_curation.py",
    "paper_coauthor.py",
    "search_profiles.py",
    "literature_discovery.py",
    "ontology_curation.py",
}

_ALLOWED_RAW_RETRY_MODULES = {"llm_retry.py", "llm_execution.py"}
_FACADE_IMPORT_PREFIXES = tuple(
    f"research_fellow.application.{name[:-3]}" for name in sorted(COMPATIBILITY_FACADES)
)


def _iter_files(root: Any, suffix: str) -> Iterable[Any]:
    for child in root.iterdir():
        if child.is_dir():
            yield from _iter_files(child, suffix)
        elif child.name.endswith(suffix):
            yield child


def _is_import_only_facade(source: str) -> bool:
    tree = ast.parse(source)
    for index, node in enumerate(tree.body):
        if isinstance(node, ast.Expr) and index == 0 and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        return False
    return True




def _calls_named(source: str, function_name: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == function_name:
            return True
        if isinstance(node.func, ast.Attribute) and node.func.attr == function_name:
            return True
    return False

def _facade_imports(source: str) -> list[str]:
    hits: list[str] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if node.module in _FACADE_IMPORT_PREFIXES:
            hits.append(node.module)
    return sorted(set(hits))


def audit_application_architecture() -> dict[str, Any]:
    """Run static and DSL-level architecture checks.

    Raises ``ValueError`` when a structural regression is found.  A successful
    return value is a compact report suitable for tests or diagnostics.
    """
    workflow_report = validate_workflow_catalog(strict_ajd_coverage=True)
    capability_report = validate_capability_catalog()

    package_root = resources.files("research_fellow")
    topology_root = package_root.joinpath("topologies")
    topologies: list[dict[str, Any]] = []
    if topology_root.is_dir():
        for resource in _iter_files(topology_root, ".yaml"):
            definition = load_topology_definition(resource.name)
            graph = topology_graph(resource.name)
            topologies.append({
                "id": definition.topology_id,
                "workflow_count": len(graph.get("workflows") or []),
                "loop_count": len(graph.get("loops") or []),
            })

    app_root = package_root.joinpath("application")
    facade_errors: list[str] = []
    facade_import_violations: list[dict[str, Any]] = []
    retry_violations: list[str] = []
    module_count = 0

    for resource in _iter_files(app_root, ".py"):
        module_count += 1
        source = resource.read_text(encoding="utf-8")
        name = resource.name
        if name in COMPATIBILITY_FACADES:
            if not _is_import_only_facade(source):
                facade_errors.append(name)
            continue

        imported_facades = _facade_imports(source)
        if imported_facades:
            facade_import_violations.append({
                "module": name,
                "facades": imported_facades,
            })

        if _calls_named(source, "call_with_retry") and name not in _ALLOWED_RAW_RETRY_MODULES:
            retry_violations.append(name)

    if facade_errors:
        raise ValueError(f"Compatibility facades contain implementation logic: {sorted(facade_errors)}")
    if facade_import_violations:
        raise ValueError(f"Internal package code imports compatibility facades: {facade_import_violations}")
    if retry_violations:
        raise ValueError(f"Raw LLM retry leaked outside common execution layer: {sorted(retry_violations)}")

    traceability = workflow_report.get("ajd_traceability") or {}
    return {
        "application_modules": module_count,
        "capabilities": capability_report["capability_count"],
        "semantic_types": capability_report["semantic_type_count"],
        "semantic_flow_checks": capability_report["semantic_flow_checks"],
        "python_capabilities": capability_report["python_capabilities"],
        "external_capabilities": capability_report["external_capabilities"],
        "binding_profile": capability_report["binding_profile"],
        "missing_bindings": list(capability_report["missing_bindings"]),
        "orphan_capabilities": list(capability_report["orphan_capabilities"]),
        "compatibility_facades": sorted(COMPATIBILITY_FACADES),
        "ajd_links": len(traceability.get("responsibility_links") or []),
        "orphan_workflows": list(traceability.get("orphan_workflows") or []),
        "topologies": topologies,
        "shared_phenomenon_links": len(workflow_report.get("links") or []),
        "boundary_triggers": len(workflow_report.get("boundary_triggers") or []),
        "boundary_emits": len(workflow_report.get("boundary_emits") or []),
        "facade_import_violations": [],
        "retry_violations": [],
    }
