#!/usr/bin/env python3

import argparse
import ast
import json
import shutil
import subprocess
import tarfile
import time
from collections import defaultdict
from pathlib import Path, PurePosixPath


DEFAULT_ARCHIVE_ROOT = Path("/work2/10446/tchang85/stampede3/dllm/experiment_dir")
DEFAULT_TMP_ROOT = Path("/scratch/10446/tchang85/tmp")
DEFAULT_OUTPUT_ROOT = Path("/scratch/10446/tchang85/accuracy_benchmark_results")
SYSTEM_TAR = shutil.which("tar")


def log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract only the experiment directories referenced by an accuracy "
            "benchmark plotting script."
        )
    )
    parser.add_argument(
        "plot_script",
        type=Path,
        help="Path to the plotting script that contains the benchmarks dict.",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=DEFAULT_ARCHIVE_ROOT,
        help="Directory containing per-day tar.gz archives and/or extracted day dirs.",
    )
    parser.add_argument(
        "--tmp-root",
        type=Path,
        default=DEFAULT_TMP_ROOT,
        help="Temporary extraction directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Final combined output directory.",
    )
    parser.add_argument(
        "--archive-suffix",
        default=".tar.gz",
        help="Archive suffix for each day file. Default: .tar.gz",
    )
    parser.add_argument(
        "--manifest-name",
        default="manifest.json",
        help="Manifest filename to write inside the output root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already-copied destination directories.",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep temporary extracted directories instead of deleting them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be extracted and copied without changing files.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only print the required directories and exit.",
    )
    return parser.parse_args()


def is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False

    test = node.test
    if not isinstance(test, ast.Compare):
        return False
    if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    if len(test.comparators) != 1:
        return False

    comparator = test.comparators[0]
    return isinstance(comparator, ast.Constant) and comparator.value == "__main__"


def collect_literal_assignments(block: list[ast.stmt]) -> dict[str, ast.AST]:
    assignments: dict[str, ast.AST] = {}

    for node in block:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                assignments[node.target.id] = node.value

    return assignments


def resolve_literal_expr(expr: ast.AST, assignments: dict[str, ast.AST], seen: set[str] | None = None):
    try:
        return ast.literal_eval(expr)
    except Exception:
        pass

    if isinstance(expr, ast.Name):
        if seen is None:
            seen = set()
        if expr.id in seen:
            raise ValueError(f"Circular literal reference detected for {expr.id!r}")
        if expr.id not in assignments:
            raise ValueError(f"Could not resolve literal value for name {expr.id!r}")
        return resolve_literal_expr(assignments[expr.id], assignments, seen | {expr.id})

    raise ValueError(f"Could not resolve literal value for AST node {type(expr).__name__}")


def literal_assignment(block: list[ast.stmt], name: str):
    assignments = collect_literal_assignments(block)
    if name not in assignments:
        raise ValueError(f"Could not find literal assignment for {name!r}")
    return resolve_literal_expr(assignments[name], assignments)


def find_plot_call(block: list[ast.stmt]) -> ast.Call | None:
    for node in block:
        if not isinstance(node, ast.Expr):
            continue
        if not isinstance(node.value, ast.Call):
            continue

        func = node.value.func
        if isinstance(func, ast.Name) and func.id == "plot_multi_benchmarks":
            return node.value

    return None


def resolve_call_argument(call: ast.Call, assignments: dict[str, ast.AST], name: str, position: int):
    for keyword in call.keywords:
        if keyword.arg == name:
            return resolve_literal_expr(keyword.value, assignments)

    if len(call.args) > position:
        return resolve_literal_expr(call.args[position], assignments)

    raise ValueError(f"Could not resolve argument {name!r} from plot_multi_benchmarks(...) call")


def load_benchmarks_config(plot_script: Path) -> tuple[dict, str]:
    tree = ast.parse(plot_script.read_text())

    main_block = None
    for node in tree.body:
        if is_main_guard(node):
            main_block = node.body
            break

    search_block = main_block if main_block is not None else tree.body
    assignments = collect_literal_assignments(search_block)

    try:
        benchmarks = resolve_literal_expr(assignments["benchmarks"], assignments)
    except Exception:
        plot_call = find_plot_call(search_block)
        if plot_call is None:
            raise ValueError("Could not find literal assignment for 'benchmarks' or a plot_multi_benchmarks(...) call")
        benchmarks = resolve_call_argument(plot_call, assignments, "benchmarks", 1)

    try:
        exp_dir_name = resolve_literal_expr(assignments["exp_dir_name"], assignments)
    except Exception:
        plot_call = find_plot_call(search_block)
        if plot_call is not None:
            try:
                exp_dir_name = resolve_call_argument(plot_call, assignments, "exp_dir_name", 4)
            except Exception:
                exp_dir_name = "experiment_dir"
        else:
            exp_dir_name = "experiment_dir"

    if not isinstance(benchmarks, dict):
        raise ValueError("Resolved benchmarks value is not a dict")
    if not isinstance(exp_dir_name, str):
        exp_dir_name = "experiment_dir"

    return benchmarks, exp_dir_name


def collect_required_dirs(benchmarks: dict) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)

    for bench_cfg in benchmarks.values():
        for key in ("rps_dirs", "slo_dirs"):
            system_map = bench_cfg.get(key, {})
            for dirs in system_map.values():
                for rel_dir in dirs:
                    parts = PurePosixPath(rel_dir).parts
                    if len(parts) != 2:
                        raise ValueError(
                            f"Expected relative experiment dir like YYYYMMDD/HHMMSS, got {rel_dir!r}"
                        )
                    grouped[parts[0]].add(rel_dir)

    return {day: set(sorted(rel_dirs)) for day, rel_dirs in grouped.items()}


def path_contains_rel_dir(member_name: str, rel_dir: str) -> bool:
    member = PurePosixPath(member_name)
    rel = PurePosixPath(rel_dir)

    if len(member.parts) < len(rel.parts):
        return False

    member_parts = member.parts
    rel_parts = rel.parts
    window = len(rel_parts)

    for start in range(len(member_parts) - window + 1):
        if member_parts[start : start + window] == rel_parts:
            return True
    return False


def split_after_rel_dir(path_like: str, rel_dir: str) -> tuple[str, ...] | None:
    path_parts = PurePosixPath(path_like).parts
    rel_parts = PurePosixPath(rel_dir).parts
    window = len(rel_parts)

    for start in range(len(path_parts) - window + 1):
        if path_parts[start : start + window] == rel_parts:
            return path_parts[start + window :]
    return None


def is_required_experiment_member(member_name: str, rel_dir: str) -> bool:
    suffix = split_after_rel_dir(member_name, rel_dir)
    if suffix is None:
        return False

    if suffix == ("run_lmeval.sh",):
        return True

    if len(suffix) >= 2 and suffix[0] == "results" and suffix[-1].endswith(".json"):
        return True

    return False


def list_archive_members(archive_path: Path) -> list[str]:
    if SYSTEM_TAR is not None:
        log(f"Listing archive members with system tar: {archive_path}")
        result = subprocess.run(
            [SYSTEM_TAR, "-tzf", str(archive_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return [line for line in result.stdout.splitlines() if line]

    log(f"Listing archive members with python tarfile: {archive_path}")
    with tarfile.open(archive_path, "r:gz") as tar:
        return tar.getnames()


def find_matching_member_names(member_names: list[str], rel_dir: str, archive_label: str) -> list[str]:
    matches = [
        member_name
        for member_name in member_names
        if is_required_experiment_member(member_name, rel_dir)
    ]
    if not matches:
        raise FileNotFoundError(f"Could not find {rel_dir} inside archive {archive_label}")
    return matches


def collect_members_for_rel_dirs(
    member_names: list[str],
    rel_dirs: list[str],
) -> tuple[dict[str, list[str]], list[str]]:
    members_by_rel_dir = {
        rel_dir: find_matching_member_names(member_names, rel_dir, "member list")
        for rel_dir in rel_dirs
    }

    unique_members: dict[str, str] = {}
    for members in members_by_rel_dir.values():
        for member_name in members:
            unique_members[member_name] = member_name

    return members_by_rel_dir, list(unique_members.values())


def extract_members_with_system_tar(
    archive_path: Path,
    tmp_root: Path,
    member_names: list[str],
) -> None:
    if SYSTEM_TAR is None:
        raise RuntimeError("System tar is not available")

    payload = "".join(f"{member_name}\n" for member_name in member_names)
    log(f"Starting system tar extraction from {archive_path.name} into {tmp_root}")
    subprocess.run(
        [SYSTEM_TAR, "-xzf", str(archive_path), "-C", str(tmp_root), "-T", "-"],
        input=payload,
        text=True,
        check=True,
    )
    log(f"Finished system tar extraction from {archive_path.name}")


def extract_members_with_python_tar(
    archive_path: Path,
    tmp_root: Path,
    member_names: list[str],
) -> None:
    wanted = set(member_names)
    log(f"Starting python tarfile extraction from {archive_path.name} into {tmp_root}")
    with tarfile.open(archive_path, "r:gz") as tar:
        selected = [member for member in tar.getmembers() if member.name in wanted]
        tar.extractall(tmp_root, members=selected)
    log(f"Finished python tarfile extraction from {archive_path.name}")


def extract_selected_members(
    archive_path: Path,
    tmp_root: Path,
    member_names: list[str],
) -> None:
    if SYSTEM_TAR is not None:
        extract_members_with_system_tar(archive_path, tmp_root, member_names)
        return

    extract_members_with_python_tar(archive_path, tmp_root, member_names)


def find_matching_path(root: Path, rel_dir: str) -> Path:
    rel_parts = PurePosixPath(rel_dir).parts
    if len(rel_parts) != 2:
        raise ValueError(f"Unexpected relative directory {rel_dir!r}")

    day, run_id = rel_parts

    direct = root / day / run_id
    if direct.exists():
        return direct

    for candidate in root.rglob(run_id):
        if not candidate.is_dir():
            continue
        parts = candidate.parts
        if len(parts) >= 2 and parts[-2:] == rel_parts:
            return candidate

    raise FileNotFoundError(f"Could not locate extracted path for {rel_dir} under {root}")


def gather_required_files(exp_dir: Path) -> list[Path]:
    required: list[Path] = []

    run_script = exp_dir / "run_lmeval.sh"
    if run_script.exists():
        required.append(run_script)

    results_dir = exp_dir / "results"
    if results_dir.exists():
        required.extend(sorted(results_dir.rglob("*.json")))

    return required


def copy_required_experiment_files(src: Path, dst: Path, overwrite: bool, dry_run: bool) -> list[str]:
    if dst.exists():
        if not overwrite:
            print(f"[skip] destination exists: {dst}")
            return []
        if dry_run:
            print(f"[dry-run] would remove existing destination: {dst}")
        else:
            shutil.rmtree(dst)

    required_files = gather_required_files(src)
    if not required_files:
        raise FileNotFoundError(f"No required files found in extracted experiment directory {src}")

    if dry_run:
        log(f"[dry-run] would copy {len(required_files)} required files from {src} -> {dst}")
        return [str(path.relative_to(src)) for path in required_files]

    copied_rel_paths: list[str] = []
    for path in required_files:
        rel_path = path.relative_to(src)
        target = dst / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied_rel_paths.append(str(rel_path))

    log(f"Copied {len(required_files)} required files from {src} -> {dst}")
    return copied_rel_paths


def extract_from_archive(
    archive_path: Path,
    rel_dirs: list[str],
    tmp_root: Path,
    output_root: Path,
    overwrite: bool,
    keep_tmp: bool,
    dry_run: bool,
) -> list[dict]:
    records = []
    started = time.time()
    archive_member_names = list_archive_members(archive_path)
    members_by_rel_dir, unique_members = collect_members_for_rel_dirs(archive_member_names, rel_dirs)

    if dry_run:
        log(
            f"[dry-run] would extract {len(unique_members)} unique tar members "
            f"covering {len(rel_dirs)} directories from {archive_path.name}"
        )
    else:
        log(
            f"[extract] {archive_path.name}: extracting {len(unique_members)} unique tar members "
            f"for {len(rel_dirs)} directories using {'system tar' if SYSTEM_TAR else 'python tarfile'}"
        )
        tmp_root.mkdir(parents=True, exist_ok=True)
        extract_selected_members(archive_path, tmp_root, unique_members)

    for index, rel_dir in enumerate(rel_dirs, start=1):
        log(
            f"[prepare] {rel_dir} from {archive_path.name} "
            f"({len(members_by_rel_dir[rel_dir])} tar members, {index}/{len(rel_dirs)})"
        )

        extracted_dir = find_matching_path(tmp_root, rel_dir) if not dry_run else tmp_root / rel_dir
        dst = output_root / rel_dir
        copied_rel_paths = copy_required_experiment_files(
            extracted_dir,
            dst,
            overwrite=overwrite,
            dry_run=dry_run,
        )

        if not keep_tmp and not dry_run:
            shutil.rmtree(extracted_dir)
            log(f"[cleanup] removed tmp directory {extracted_dir}")

        records.append(
            {
                "relative_dir": rel_dir,
                "source_type": "archive",
                "source": str(archive_path),
                "output": str(dst),
                "copied_files": copied_rel_paths,
            }
        )

    log(f"Completed archive {archive_path.name} in {time.time() - started:.1f}s")

    return records


def copy_from_extracted_day(
    day_root: Path,
    rel_dirs: list[str],
    output_root: Path,
    overwrite: bool,
    dry_run: bool,
) -> list[dict]:
    records = []
    started = time.time()
    for index, rel_dir in enumerate(rel_dirs, start=1):
        log(f"[copy-source] preparing {rel_dir} from extracted directory source ({index}/{len(rel_dirs)})")
        src = day_root / PurePosixPath(rel_dir).parts[-1]
        if not src.exists():
            src = find_matching_path(day_root.parent, rel_dir)
        dst = output_root / rel_dir
        copied_rel_paths = copy_required_experiment_files(
            src,
            dst,
            overwrite=overwrite,
            dry_run=dry_run,
        )
        records.append(
            {
                "relative_dir": rel_dir,
                "source_type": "directory",
                "source": str(src),
                "output": str(dst),
                "copied_files": copied_rel_paths,
            }
        )
    log(f"Completed extracted directory source {day_root} in {time.time() - started:.1f}s")
    return records


def resolve_day_source(archive_root: Path, day: str, archive_suffix: str) -> tuple[str, Path]:
    archive_path = archive_root / f"{day}{archive_suffix}"
    if archive_path.exists():
        return "archive", archive_path

    day_dir = archive_root / day
    if day_dir.exists():
        return "directory", day_dir

    raise FileNotFoundError(
        f"Could not find either {archive_path} or extracted directory {day_dir}"
    )


def write_manifest(output_root: Path, manifest_name: str, payload: dict, dry_run: bool) -> None:
    manifest_path = output_root / manifest_name
    if dry_run:
        print(f"[dry-run] would write manifest to {manifest_path}")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[manifest] wrote {manifest_path}")


def main() -> None:
    args = parse_args()
    overall_started = time.time()
    benchmarks, exp_dir_name = load_benchmarks_config(args.plot_script)
    grouped_dirs = collect_required_dirs(benchmarks)

    total_dirs = sum(len(rel_dirs) for rel_dirs in grouped_dirs.values())
    log(f"Plot script: {args.plot_script}")
    log(f"Experiment directory name: {exp_dir_name}")
    log(f"Unique referenced experiment directories: {total_dirs}")

    for day in sorted(grouped_dirs):
        log(f"Day {day}: {len(grouped_dirs[day])} dirs")
        for rel_dir in sorted(grouped_dirs[day]):
            print(f"    {rel_dir}", flush=True)

    if args.list_only:
        return

    all_records = []
    sorted_days = sorted(grouped_dirs)
    for day_index, day in enumerate(sorted_days, start=1):
        rel_dirs = sorted(grouped_dirs[day])
        source_type, source_path = resolve_day_source(
            args.archive_root,
            day,
            args.archive_suffix,
        )
        log(
            f"Starting day {day} ({day_index}/{len(sorted_days)}): "
            f"{len(rel_dirs)} directories from {source_type} source {source_path}"
        )

        if source_type == "archive":
            records = extract_from_archive(
                archive_path=source_path,
                rel_dirs=rel_dirs,
                tmp_root=args.tmp_root,
                output_root=args.output_root,
                overwrite=args.overwrite,
                keep_tmp=args.keep_tmp,
                dry_run=args.dry_run,
            )
        else:
            records = copy_from_extracted_day(
                day_root=source_path,
                rel_dirs=rel_dirs,
                output_root=args.output_root,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )

        all_records.extend(records)
        log(f"Finished day {day} ({day_index}/{len(sorted_days)})")

    manifest = {
        "plot_script": str(args.plot_script),
        "archive_root": str(args.archive_root),
        "tmp_root": str(args.tmp_root),
        "output_root": str(args.output_root),
        "total_dirs": total_dirs,
        "days": {day: sorted(grouped_dirs[day]) for day in sorted(grouped_dirs)},
        "copied": all_records,
    }
    write_manifest(args.output_root, args.manifest_name, manifest, args.dry_run)
    log(f"All work completed in {time.time() - overall_started:.1f}s")


if __name__ == "__main__":
    main()