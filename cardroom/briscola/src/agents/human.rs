use std::io;
use std::io::Write;
use super::generic::BriscolaAgent;
use crate::env::env::BriscolaObs;
use crate::env::player::BriscolaAction;
use crate::env::utils::env_from_obs;

pub struct HumanAgent {
    name: String,
}
impl Default for HumanAgent {
    fn default() -> Self {
        Self { name: format!("Human") }
    }
}

impl HumanAgent {
    pub fn new(name: String) -> Self {
        Self { name: name }
    }
}

impl BriscolaAgent for HumanAgent {
    fn name(&self) -> &str { &self.name }

    fn select_action(&self, obs: BriscolaObs) -> BriscolaAction {
        let env = env_from_obs(&obs);
        let legal_actions = env.get_legal_actions();

        loop {
            print!("Enter action ");
            for action in &legal_actions {
                print!("[{}]-{} ", action.idx(), env.players[env.current_player_id].hand[action.idx()].name());
            }
            print!(": ");
            io::stdout().flush().expect("Failed to flush stdout");

            let mut input = String::new();
            io::stdin().read_line(&mut input).expect("Failed to read line");

            match input.trim().parse::<usize>() {
                Ok(idx) if legal_actions.iter().any(|a| a.idx() == idx) => {
                    return BriscolaAction::from_idx(idx);
                }
                _ => {
                    let valid: Vec<String> = legal_actions.iter().map(|a| a.idx().to_string()).collect();
                    println!("Invalid input. Please enter one of: {}", valid.join(", "));
                }
            }
        }
    }
}

