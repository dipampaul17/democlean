"""democlean CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VERSION = "0.1.3"


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        prog="democlean",
        description="Score robot demonstrations by motion quality.",
    )
    parser.add_argument("-v", "--version", action="version", version=VERSION)

    sub = parser.add_subparsers(dest="cmd")

    # analyze
    p = sub.add_parser("analyze", help="Score episodes in a dataset")
    p.add_argument("dataset", help="HuggingFace repo or local path")
    p.add_argument("--keep", type=float, metavar="R", help="Keep top ratio (0-1)")
    p.add_argument("--top-k", type=int, metavar="K", help="Keep top K episodes")
    p.add_argument("-r", "--report", type=Path, help="Save JSON report")
    p.add_argument("-n", "--max-episodes", type=int, help="Limit episodes")
    p.add_argument("-k", type=int, default=3, help="KSG neighbors (default: 3)")
    p.add_argument("--max-dim", type=int, help="PCA reduce states to this dim")
    p.add_argument("--ci", action="store_true", help="Compute 95%% CI")
    p.add_argument("-q", "--quiet", action="store_true", help="Minimal output")
    p.add_argument(
        "--merge",
        type=Path,
        metavar="FILE",
        help="Merge with score_lerobot_episodes JSON",
    )
    p.add_argument(
        "--min-mi", type=float, metavar="T", help="Drop episodes below MI threshold"
    )
    p.add_argument(
        "--normalize-length", action="store_true", help="Normalize MI by episode length"
    )
    p.add_argument(
        "--explain", action="store_true", help="Show detailed explanation of results"
    )

    args = parser.parse_args(argv)

    if not args.cmd:
        parser.print_help()
        return 0

    # Import console for early warnings
    from rich.console import Console

    con = Console()

    # Warn when n=0 (loads all episodes)
    if hasattr(args, "max_episodes") and args.max_episodes == 0:
        con.print("[yellow]Warning:[/] -n 0 will load ALL episodes")

    return run_analyze(args)


def run_analyze(args: argparse.Namespace) -> int:
    """Analyze dataset episodes."""
    from rich import box
    from rich.console import Console
    from rich.table import Table

    from democlean.scorer import DemoScorer

    con = Console()
    quiet = args.quiet

    # Validate k >= 1
    if args.k < 1:
        con.print("[red]Error:[/] -k must be >= 1")
        return 1

    # Validate --keep in (0, 1]
    if args.keep is not None and not (0 < args.keep <= 1):
        con.print("[red]Error:[/] --keep must be between 0 and 1")
        return 1

    if not quiet:
        con.print(f"\n[bold blue]democlean[/] [dim]{VERSION}[/]\n")

    con.print(f"[bold]Dataset[/] {args.dataset}")

    # Score
    scorer = DemoScorer(
        k=args.k,
        max_state_dim=args.max_dim,
        bootstrap_ci=args.ci,
    )

    try:
        scores = scorer.score_dataset(
            args.dataset,
            show_progress=not quiet,
            max_episodes=args.max_episodes,
        )
    except Exception as e:
        con.print(f"[red]Error:[/] {e}")
        return 1

    if not scores:
        con.print("[yellow]No episodes found[/]")
        return 1

    # Quality assessment
    assessment = scorer.get_quality_assessment(scores)
    n = len(scores)

    if not quiet:
        dims = f"{scores[0].state_dim}->{scores[0].action_dim}"
        con.print(f"[dim]Episodes: {n} | Dims: {dims}[/]\n")

    # Apply --min-mi filtering if specified
    if args.min_mi is not None:
        keep = [s.episode_index for s in scores if s.mi_score >= args.min_mi]
        drop = [s.episode_index for s in scores if s.mi_score < args.min_mi]
        if not quiet:
            msg = f"[bold]--min-mi {args.min_mi}:[/] Keep {len(keep)}, drop {len(drop)}"
            con.print(msg)
            if drop:
                preview = ", ".join(map(str, drop[:6]))
                if len(drop) > 6:
                    preview += f" +{len(drop) - 6}"
                con.print(f"  [red]Dropped:[/] [{preview}]\n")
        if quiet:
            print(json.dumps({"keep": keep, "drop": drop}))
        # Filter scores for subsequent processing
        scores = [s for s in scores if s.mi_score >= args.min_mi]
        n = len(scores)
        if n == 0:
            con.print("[yellow]No episodes remaining after --min-mi filter[/]")
            return 0

    # Distribution - use normalized_score if --normalize-length is set
    if args.normalize_length:
        ranked = sorted(scores, key=lambda s: s.normalized_score, reverse=True)
    else:
        ranked = sorted(scores, key=lambda s: s.mi_score, reverse=True)
    n_hi, n_md = int(n * 0.6), int(n * 0.3)
    hi, md, lo = ranked[:n_hi], ranked[n_hi : n_hi + n_md], ranked[n_hi + n_md :]

    if not quiet:
        w = 25
        con.print("[bold]Distribution[/]")
        con.print(f"  [green]{'█' * int(w * len(hi) / n):<{w}}[/] High   {len(hi):>3}")
        con.print(f"  [yellow]{'█' * int(w * len(md) / n):<{w}}[/] Medium {len(md):>3}")
        con.print(f"  [red]{'█' * int(w * len(lo) / n):<{w}}[/] Low    {len(lo):>3}\n")

        # Stats with MI interpretation
        mi_mean = assessment["mi_mean"]
        if mi_mean > 3.0:
            interpretation = "(very high - clean data)"
        elif mi_mean > 2.0:
            interpretation = "(typical for human teleop)"
        elif mi_mean > 1.0:
            interpretation = "(moderate - check quality)"
        else:
            interpretation = "(low - noisy data)"

        tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        tbl.add_column(style="dim")
        tbl.add_column(justify="right")
        tbl.add_row("Mean", f"{mi_mean:.3f} {interpretation}")
        tbl.add_row("Std", f"{assessment['mi_std']:.3f}")
        mi_range = f"[{assessment['mi_min']:.3f}, {assessment['mi_max']:.3f}]"
        tbl.add_row("Range", mi_range)
        con.print(tbl)
        con.print()

        # Warnings
        for warning in assessment["warnings"]:
            con.print(f"[yellow]Warning:[/] {warning}\n")

    # Filter
    keep, drop = None, None
    if args.keep or args.top_k:
        n_keep = max(1, int(n * args.keep)) if args.keep else min(args.top_k, n)
        keep = [s.episode_index for s in ranked[:n_keep]]
        drop = [s.episode_index for s in ranked[n_keep:]]

        if not quiet:
            con.print(f"[green]Keep[/] {len(keep)} ({100 * len(keep) // n}%)")
            if drop:
                preview = ", ".join(map(str, drop[:6]))
                if len(drop) > 6:
                    preview += f" +{len(drop) - 6}"
                con.print(f"[red]Drop[/] {len(drop)}: [{preview}]")

                # Show why dropped episodes are bad
                drop_set = set(drop[:3])
                dropped_scores = [s for s in ranked if s.episode_index in drop_set]
                con.print("[dim]Dropped episodes tend to have:[/]")
                for s in dropped_scores:
                    mi_val = f"MI={s.mi_score:.2f}"
                    con.print(f"  ep {s.episode_index}: {mi_val}, len={s.length}")
                con.print()

        if quiet:
            print(json.dumps(keep))

    else:
        # Show flagged
        if not quiet and lo:
            con.print("[bold]Flagged[/] [dim](lowest MI)[/]")
            for s in lo[:5]:
                ci_str = ""
                if s.ci_lower is not None:
                    ci_str = f" [{s.ci_lower:.2f}-{s.ci_upper:.2f}]"
                con.print(f"  ep {s.episode_index:>3}  {s.mi_score:.3f}{ci_str}")
            con.print()

    # Show explanation if --explain flag is set
    if not quiet and args.explain:
        con.print("[bold]What is MI (Mutual Information)?[/]")
        con.print(
            "  MI measures how predictable actions are given states. High MI means"
        )
        con.print(
            "  the robot's actions are consistent and purposeful. Low MI suggests"
        )
        con.print("  random or hesitant behavior.\n")
        con.print("[bold]How to interpret results:[/]")
        con.print("  - MI > 3.0: Excellent demos, very consistent behavior")
        con.print("  - MI 2.0-3.0: Good quality, typical for human teleoperation")
        con.print("  - MI 1.0-2.0: Moderate quality, review for issues")
        con.print("  - MI < 1.0: Poor quality, actions may be nearly random\n")
        con.print("[bold]Recommendations:[/]")
        con.print("  - Use --keep 0.8 to keep top 80% of episodes by quality")
        con.print("  - Use --min-mi 1.5 to drop episodes below a threshold")
        con.print("  - High std suggests mixed quality - filtering can help")
        con.print("  - Low std means uniform quality - filtering won't help much\n")

    # Merge with score_lerobot_episodes
    merged_scores = None
    if args.merge:
        merged_scores = _merge_scores(scores, args.merge, con, quiet)

    # Report
    if args.report:
        report = {
            "version": VERSION,
            "dataset": args.dataset,
            "n_episodes": n,
            "assessment": assessment,
            "scores": [s.to_dict() for s in scores],
        }

        if keep is not None:
            report["keep"] = keep
            report["drop"] = drop

        if merged_scores:
            report["merged_scores"] = merged_scores

        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))

        if not quiet:
            con.print(f"[green]Saved[/] {args.report}")

    return 0


def _merge_scores(
    mi_scores: list,
    other_path: Path,
    con,
    quiet: bool,
) -> list[dict] | None:
    """Merge MI scores with score_lerobot_episodes output."""
    if not other_path.exists():
        con.print(f"[yellow]Warning:[/] {other_path} not found, skipping merge")
        return None

    try:
        other_data = json.loads(other_path.read_text())
    except json.JSONDecodeError:
        con.print(f"[yellow]Warning:[/] Invalid JSON in {other_path}")
        return None

    # score_lerobot_episodes format: list of {episode_id, scores: {...}}
    other_by_ep = {}
    if isinstance(other_data, list):
        for item in other_data:
            ep_id = item.get("episode_id") or item.get("episode_index")
            if ep_id is not None:
                other_by_ep[ep_id] = item.get("scores", item)
    elif isinstance(other_data, dict) and "scores" in other_data:
        # Wrapped format
        for item in other_data["scores"]:
            ep_id = item.get("episode_id") or item.get("episode_index")
            if ep_id is not None:
                other_by_ep[ep_id] = item.get("scores", item)

    # Merge
    merged = []
    for s in mi_scores:
        entry = {"episode_index": s.episode_index, "mi_score": s.mi_score}
        if s.episode_index in other_by_ep:
            entry["visual_scores"] = other_by_ep[s.episode_index]
        merged.append(entry)

    if not quiet:
        n_matched = sum(1 for m in merged if "visual_scores" in m)
        con.print(f"[dim]Merged {n_matched}/{len(merged)} with visual scores[/]")

    return merged


if __name__ == "__main__":
    sys.exit(main())
