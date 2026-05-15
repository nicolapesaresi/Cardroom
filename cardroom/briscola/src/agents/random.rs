use rand::prelude::IndexedRandom;
use super::generic::BriscolaAgent;
use crate::env::env::BriscolaObs;
use crate::env::player::BriscolaAction;  
use crate::env::utils::env_from_obs;

pub struct RandomAgent {
    name: String,
}

impl Default for RandomAgent {
    fn default() -> Self {
        Self{name: format!("Random")}
    }
}

impl RandomAgent {
    pub fn new(name: String) -> Self {
        Self { name: name }
    }
}

impl BriscolaAgent for RandomAgent {
    fn name(&self) -> &str { &self.name }

    fn select_action(&self, obs: BriscolaObs) -> BriscolaAction {
        let env = env_from_obs(&obs);
        let legal_actions = env.get_legal_actions();
        *legal_actions.choose(&mut rand::rng()).unwrap()
    }
}

