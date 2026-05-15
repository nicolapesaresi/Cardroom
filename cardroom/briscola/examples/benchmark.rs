use briscola::agents::utils::make_agent;
use briscola::utils::benchmark::benchmark_n_games;

const DEFAULT_N_ROUNDS: usize = 100;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: benchmark <agent0> <agent1> [n_rounds]");
        eprintln!("Available agents: bot, random, mcts, human");
        std::process::exit(1);
    }

    let n_rounds: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_N_ROUNDS);
    let agents = vec![make_agent(&args[1]), make_agent(&args[2])];

    benchmark_n_games(agents, n_rounds);
}
