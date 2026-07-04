use super::bot::BotAgent;
use super::generic::BriscolaAgent;
use super::mcts::MCTSAgent;
use super::random::RandomAgent;
use crate::agents::human::HumanAgent;


pub fn make_agent(name: &str) -> Box<dyn BriscolaAgent> {
    match name {
        "human" => Box::new(HumanAgent::default()),
        "bot" => Box::new(BotAgent::default()),
        "random" => Box::new(RandomAgent::default()),
        "mcts" => Box::new(MCTSAgent::default()),
        _ => {
            eprintln!("Unknown agent '{}'. Available: bot, random, mcts, minimax", name);
            std::process::exit(1);
        }
    }
}