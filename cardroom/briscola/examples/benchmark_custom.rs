// Edit the agents vec below to choose which agents play, then run with:
//   cargo run --example benchmark_custom -- [n_rounds]
#![allow(unused_imports)]

use briscola::agents::bot::BotAgent;
use briscola::agents::generic::BriscolaAgent;
use briscola::agents::human::HumanAgent;
use briscola::agents::mcts::MCTSAgent;
use briscola::agents::random::RandomAgent;
use briscola::utils::benchmark::benchmark_n_games;

const DEFAULT_N_ROUNDS: usize = 100;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_rounds: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_N_ROUNDS);

    let agents: Vec<Box<dyn BriscolaAgent>> = vec![
        Box::new(MCTSAgent::new("mcts100".to_string(), 100)),
        Box::new(MCTSAgent::new("mcts1k".to_string(), 1000)),
        // Box::new(BotAgent::default()),
        // Box::new(RandomAgent::default()),
        // Box::new(HumanAgent::default()),
    ];

    benchmark_n_games(agents, n_rounds);
}
