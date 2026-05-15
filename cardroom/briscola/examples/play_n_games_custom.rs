//this script uses agents with custom parameters instead of the default ones
//simply instantiate the agents with the preferred parameters and comment the others

#![allow(unused_imports)]
use briscola::utils::playing::play_n_games;
use briscola::agents::human::HumanAgent;
use briscola::agents::random::RandomAgent;
use briscola::agents::bot::BotAgent;
use briscola::agents::mcts::MCTSAgent;

const DEFAULT_N_GAMES: usize = 100;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_games: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_N_GAMES);
    
    // *** SET AGENTS HERE ***

    // let human = HumanAgent::default();
    // let bot = BotAgent::default();
    // let random = RandomAgent::default();
    // let mcts = MCTSAgent::default();
    let mcts100 = MCTSAgent::new(format!("mcts100"), 100);
    let mcts1k = MCTSAgent::new(format!("mcts1k"), 1000);

    let agents: Vec<Box<dyn briscola::agents::generic::BriscolaAgent>> = vec![
        Box::new(mcts100), // P0
        Box::new(mcts1k), // P1
    ];

    play_n_games(agents, n_games);
}
