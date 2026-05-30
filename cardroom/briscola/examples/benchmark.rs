use briscola::agents::utils::make_agent;
use briscola::utils::benchmark::benchmark_n_games;

const DEFAULT_N_ROUNDS: usize = 100;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: benchmark <agent0> <agent1> [n_rounds] [--csv]");
        eprintln!("Available agents: bot, random, mcts, human");
        std::process::exit(1);
    }

    let csv = args.iter().any(|a| a == "--csv");
    let n_rounds: usize = args.iter()
        .skip(3)
        .filter_map(|s| s.parse().ok())
        .next()
        .unwrap_or(DEFAULT_N_ROUNDS);

    let agents = vec![make_agent(&args[1]), make_agent(&args[2])];
    benchmark_n_games(agents, n_rounds, csv);
}
