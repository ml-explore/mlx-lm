import argparse
import fnmatch
from typing import List, Union

from huggingface_hub import list_repo_files, scan_cache_dir


def tabulate(rows: List[List[Union[str, int]]], headers: List[str]) -> str:
    """
    Inspired by:
    - stackoverflow.com/a/8356620/593036
    - stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    """
    col_widths = [max(len(str(x)) for x in col) for col in zip(*rows, headers)]
    row_format = ("{{:{}}} " * len(headers)).format(*col_widths)
    lines = []
    lines.append(row_format.format(*headers))
    lines.append(row_format.format(*["-" * w for w in col_widths]))
    for row in rows:
        lines.append(row_format.format(*row))
    return "\n".join(lines)


def ask_for_confirmation(message: str) -> bool:
    """Ask user for confirmation with Y/N prompt.
    Returns True for Y/yes, False for N/no/empty."""
    y = ("y", "yes", "1")
    n = ("n", "no", "0", "")
    full_message = f"{message} (y/n) "
    while True:
        answer = input(full_message).lower()
        if answer in y:
            return True
        if answer in n:
            return False
        print(f"Invalid input. Must be one of: yes/no/y/n or empty for no")


def main():
    parser = argparse.ArgumentParser(description="MLX Model Cache.")
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan Hugging Face cache for mlx models.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete models matching the given pattern.",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Keep only the latest snapshot per repo, delete older ones.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Model repos contain the pattern.",
        default="mlx",
    )

    args = parser.parse_args()

    if args.scan:
        print(f'Scanning Hugging Face cache for models with pattern "{args.pattern}".')
        hf_cache_info = scan_cache_dir()
        print(
            tabulate(
                rows=[
                    [
                        repo.repo_id,
                        repo.repo_type,
                        "{:>12}".format(repo.size_on_disk_str),
                        repo.nb_files,
                        repo.last_accessed_str,
                        repo.last_modified_str,
                        str(repo.repo_path),
                    ]
                    for repo in sorted(
                        hf_cache_info.repos, key=lambda repo: repo.repo_path
                    )
                    if args.pattern in repo.repo_id
                ],
                headers=[
                    "REPO ID",
                    "REPO TYPE",
                    "SIZE ON DISK",
                    "NB FILES",
                    "LAST_ACCESSED",
                    "LAST_MODIFIED",
                    "LOCAL PATH",
                ],
            )
        )

    if args.delete:
        print(f'Deleting models matching pattern "{args.pattern}"')
        hf_cache_info = scan_cache_dir()

        repos = [
            repo
            for repo in sorted(hf_cache_info.repos, key=lambda repo: repo.repo_path)
            if args.pattern in repo.repo_id
        ]
        if repos:
            print("\nFound the following models:")
            print(
                tabulate(
                    rows=[
                        [
                            repo.repo_id,
                            repo.size_on_disk_str,  # Added size information
                            str(repo.repo_path),
                        ]
                        for repo in repos
                    ],
                    headers=[
                        "REPO ID",
                        "SIZE",  # Added size header
                        "LOCAL PATH",
                    ],
                )
            )

            confirmed = ask_for_confirmation(
                "\nAre you sure you want to delete these models?"
            )
            if confirmed:
                for model_info in repos:
                    print(f"\nDeleting {model_info.repo_id}...")
                    for revision in sorted(
                        model_info.revisions, key=lambda revision: revision.commit_hash
                    ):
                        strategy = hf_cache_info.delete_revisions(revision.commit_hash)
                        strategy.execute()
                print("\nModel(s) deleted successfully.")
            else:
                print("\nDeletion cancelled - no changes made.")
        else:
            print(f'No models found matching pattern "{args.pattern}"')

    if args.prune:
        print(f'Pruning old snapshots for models matching pattern "{args.pattern}"')
        hf_cache_info = scan_cache_dir()

        all_repos = [
            repo
            for repo in sorted(hf_cache_info.repos, key=lambda repo: repo.repo_path)
            if args.pattern in repo.repo_id
        ]

        # Find repos with missing files compared to what mlx-lm would download
        allow_patterns = [
            "*.json",
            "model*.safetensors",
            "*.py",
            "tokenizer.model",
            "*.tiktoken",
            "tiktoken.model",
            "*.txt",
            "*.jsonl",
            "*.jinja",
        ]
        incomplete_repos = []
        for repo in all_repos:
            rev = max(repo.revisions, key=lambda r: r.last_modified)
            local_files = {
                str(f.file_path.relative_to(rev.snapshot_path)) for f in rev.files
            }
            try:
                remote_files = list_repo_files(repo.repo_id, revision=rev.commit_hash)
            except Exception:
                continue
            expected = {
                f
                for f in remote_files
                if any(fnmatch.fnmatch(f, p) for p in allow_patterns)
            }
            missing = expected - local_files
            if missing:
                incomplete_repos.append((repo, missing))

        if incomplete_repos:
            print("\nFound repos with missing files:")
            for repo, missing in incomplete_repos:
                print(f"\n  {repo.repo_id} ({len(missing)} missing):")
                for f in sorted(missing):
                    print(f"    - {f}")
            if ask_for_confirmation("\nDelete these incomplete repos?"):
                for repo, _ in incomplete_repos:
                    for revision in repo.revisions:
                        strategy = hf_cache_info.delete_revisions(revision.commit_hash)
                        strategy.execute()
                    print(f"  Deleted {repo.repo_id}")
                print("\nIncomplete repos deleted.")
            else:
                print("\nSkipping incomplete repos.")

        incomplete_repo_ids = {repo.repo_id for repo, _ in incomplete_repos}
        repos = [
            repo
            for repo in all_repos
            if repo.repo_id not in incomplete_repo_ids and len(repo.revisions) > 1
        ]
        if repos:
            rows = []
            old_revisions = {}
            for repo in repos:
                revisions = sorted(
                    repo.revisions, key=lambda r: r.last_modified, reverse=True
                )
                keep = revisions[0]
                old = revisions[1:]
                old_revisions[repo.repo_id] = old
                rows.append(
                    [
                        repo.repo_id,
                        f"{len(old)}",
                        keep.commit_hash[:8],
                    ]
                )
            print("\nRepos with old snapshots to prune:")
            print(
                tabulate(
                    rows=rows,
                    headers=["REPO ID", "OLD SNAPSHOTS", "KEEPING"],
                )
            )

            confirmed = ask_for_confirmation(
                "\nAre you sure you want to delete old snapshots?"
            )
            if confirmed:
                for repo in repos:
                    for revision in old_revisions[repo.repo_id]:
                        strategy = hf_cache_info.delete_revisions(revision.commit_hash)
                        strategy.execute()
                    print(
                        f"  Pruned {len(old_revisions[repo.repo_id])} old snapshot(s)"
                        f" from {repo.repo_id}"
                    )
                print("\nPrune complete.")
            else:
                print("\nPrune cancelled - no changes made.")
        if not repos and not incomplete_repos:
            print(f'Nothing to prune for repos matching "{args.pattern}"')


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.manage...` directly is deprecated."
        " Use `mlx_lm.manage...` or `python -m mlx_lm manage ...` instead."
    )
    main()
