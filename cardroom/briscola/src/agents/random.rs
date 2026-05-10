use rand::prelude::IndexedRandom;
use super::generic::BriscolaAgent;
use crate::env::env::BriscolaEnv;
use crate::env::player::BriscolaAction;  

pub struct RandomAgent {
    name: String,
}

impl RandomAgent {
    pub fn new() -> Self {
        Self{name: format!("Random")}
    }
}

impl BriscolaAgent for RandomAgent {
    fn name(&self) -> &str { &self.name }

    fn select_action(&self, env: &BriscolaEnv) -> BriscolaAction {
        let legal_actions = env.get_legal_actions();
        *legal_actions.choose(&mut rand::rng()).unwrap()
    }
}

