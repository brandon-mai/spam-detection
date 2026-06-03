import os
import sys
import glob
import random
import contextlib
from typing import List, Tuple

# 1. Silently import make from kaggle-environments
@contextlib.contextmanager
def silence_outputs():
    """Silences both Python-level and C-level stdout/stderr using fd duplication."""
    stdout_fd = 1
    stderr_fd = 2
    
    try:
        dup_stdout = os.dup(stdout_fd)
        dup_stderr = os.dup(stderr_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        
        os.dup2(devnull, stdout_fd)
        os.dup2(devnull, stderr_fd)
    except Exception:
        # Fallback to python-level redirection if fd operations fail
        class Dummy:
            def write(self, x): pass
            def flush(self): pass
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Dummy(), Dummy()
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err
        return

    try:
        yield
    finally:
        os.dup2(dup_stdout, stdout_fd)
        os.dup2(dup_stderr, stderr_fd)
        os.close(devnull)
        os.close(dup_stdout)
        os.close(dup_stderr)

with silence_outputs():
    from kaggle_environments import make

# Import rich elements for premium terminal visuals
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

console = Console()

def run_match(target_agent: str, opponent_agent: str, seed: int) -> Tuple[float, int]:
    """Runs a 2-player Orbit Wars match and returns target agent's reward and steps taken."""
    env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": 500})
    # Target is always player 0 (slot 0), opponent is player 1 (slot 1)
    env.run([target_agent, opponent_agent])
    
    # Extract results
    final_step = env.steps[-1]
    target_reward = final_step[0].reward if final_step[0].reward is not None else 0.0
    steps_taken = len(env.steps)
    return target_reward, steps_taken

def get_available_opponents(exclude_agent: str) -> List[str]:
    """Finds all valid rule-based agents in the agents/ folder."""
    all_files = glob.glob(os.path.join("agents", "*.py"))
    valid_opponents = []
    
    for f in all_files:
        name = os.path.basename(f)
        if name.startswith("__") or f == exclude_agent:
            continue
        valid_opponents.append(f)
        
    return valid_opponents

def main():
    # Configure target agent and number of evaluation episodes
    TARGET_AGENT = os.path.join("agents", "hellburner_v2.py")
    NUM_EPISODES = 30
    # Set to a path to force all matches against one opponent, or None for random
    HEAD_TO_HEAD_OPPONENT = os.path.join("agents", "hellburner.py")
    
    console.print(Panel.fit(
        f"[bold cyan]Orbit Wars Heuristic Agent Benchmark[/bold cyan]\n"
        f"[dim]Target Agent:[/dim] [green]{TARGET_AGENT}[/green]\n"
        f"[dim]Episodes:[/dim] [yellow]{NUM_EPISODES}[/yellow]",
        border_style="cyan"
    ))
    
    opponents = get_available_opponents(TARGET_AGENT)
    if not opponents:
        console.print("[bold red]Error: No opponent agents found in agents/ directory![/bold red]")
        sys.exit(1)
        
    wins = 0
    draws = 0
    losses = 0
    total_steps = 0
    
    opponent_stats = {}
    
    # Set up progress bar using rich
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Running matches...", total=NUM_EPISODES)
        
        for i in range(NUM_EPISODES):
            opp = HEAD_TO_HEAD_OPPONENT if HEAD_TO_HEAD_OPPONENT else random.choice(opponents)
            opp_name = os.path.basename(opp)
            
            # Update description with current match-up
            progress.update(task, description=f"[cyan]vs {opp_name}...")
            
            seed = random.randint(1, 1000000)
            try:
                reward, steps = run_match(TARGET_AGENT, opp, seed)
                total_steps += steps
                
                # Initialize opponent specific stats
                if opp_name not in opponent_stats:
                    opponent_stats[opp_name] = {"wins": 0, "losses": 0, "draws": 0, "total": 0}
                
                opponent_stats[opp_name]["total"] += 1
                
                outcome = "DRAW"
                outcome_style = "yellow"
                if reward > 0:
                    wins += 1
                    opponent_stats[opp_name]["wins"] += 1
                    outcome = "WIN"
                    outcome_style = "green"
                elif reward < 0:
                    losses += 1
                    opponent_stats[opp_name]["losses"] += 1
                    outcome = "LOSS"
                    outcome_style = "red"
                else:
                    draws += 1
                    opponent_stats[opp_name]["draws"] += 1
                
                # Real-time print (instantly visible in logs)
                console.print(
                    f"[dim][{i+1:02d}/{NUM_EPISODES:02d}][/dim] "
                    f"vs [cyan]{opp_name:<28}[/cyan] -> "
                    f"[{outcome_style}]{outcome:<4}[/{outcome_style}] "
                    f"in {steps:>3} steps"
                )
                    
            except Exception as e:
                console.print(f"[bold red]Match {i+1} failed: {e}[/bold red]")
                
            progress.advance(task)
            
    # Print overall results summary table
    win_rate = (wins / NUM_EPISODES) * 100 if NUM_EPISODES > 0 else 0
    avg_steps = total_steps / NUM_EPISODES if NUM_EPISODES > 0 else 0
    
    summary_table = Table(title="Overall Benchmark Summary", border_style="cyan")
    summary_table.add_column("Metric", style="bold white")
    summary_table.add_column("Value", justify="right")
    
    summary_table.add_row("Total Matches", str(NUM_EPISODES))
    summary_table.add_row("Wins", f"[green]{wins}[/green]")
    summary_table.add_row("Losses", f"[red]{losses}[/red]")
    summary_table.add_row("Draws", f"[yellow]{draws}[/yellow]")
    summary_table.add_row("Win Rate", f"[bold green]{win_rate:.1f}%[/bold green]")
    summary_table.add_row("Avg Steps / Match", f"{avg_steps:.1f}")
    
    console.print("\n")
    console.print(summary_table)
    
    # Break down by opponent
    opp_table = Table(title="Detailed Breakdowns by Opponent", border_style="dim")
    opp_table.add_column("Opponent Agent", style="cyan")
    opp_table.add_column("Matches", justify="right")
    opp_table.add_column("W - L - D", justify="center")
    opp_table.add_column("Win Rate", justify="right")
    
    for opp_name, stats in sorted(opponent_stats.items()):
        o_win_rate = (stats["wins"] / stats["total"]) * 100
        opp_table.add_row(
            opp_name,
            str(stats["total"]),
            f"[green]{stats['wins']}[/green]-[red]{stats['losses']}[/red]-[yellow]{stats['draws']}[/yellow]",
            f"[bold]{o_win_rate:.1f}%[/bold]"
        )
        
    console.print("\n")
    console.print(opp_table)

if __name__ == "__main__":
    main()
