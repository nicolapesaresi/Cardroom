use std::collections::HashMap;
use super::generic::BriscolaAgent;
use crate::env::env::{BriscolaEnv, BriscolaObs, BriscolaResult};
use crate::env::player::BriscolaAction;
use crate::env::utils::env_from_obs;

pub struct MCTSAgent {
    name: String,
    simulations: usize,
}

impl MCTSAgent {
    pub fn new(simulations: usize) -> Self {
        Self{name: format!("MCTS"), simulations}
    }
}

impl BriscolaAgent for MCTSAgent {
    fn name(&self) -> &str { &self.name }

    fn select_action(&self, obs: BriscolaObs) -> BriscolaAction {
        naive_mcts(obs, self.simulations)
    }
}

struct MCTSNode {
    obs: BriscolaObs,
    to_play: usize,
    legal_actions: Vec<BriscolaAction>,
    visits: i32,
    tot_value: f32,
    children_visits: HashMap<BriscolaAction, i32>,
    children_value: HashMap<BriscolaAction, f32>,
}

impl MCTSNode {
    pub fn new(obs: BriscolaObs) -> Self {
        let env = env_from_obs(&obs);
        let to_play = env.current_player_id;
        let legal_actions = env.get_legal_actions();
        let children_visits = legal_actions.iter().map(|&a| (a, 0)).collect();
        let children_value = legal_actions.iter().map(|&a| (a, 0.0)).collect();
        Self {
            obs,
            to_play,
            legal_actions,
            visits: 0,
            tot_value: 0.0,
            children_visits,
            children_value,
        }
    }

    pub fn avg_value(&self, action:BriscolaAction) -> f32 {
        if self.visits == 0 { return f32::NEG_INFINITY; }
        self.children_value[&action] / self.children_visits[&action] as f32
    }

    pub fn update_stats(&mut self, action:BriscolaAction, value: f32) -> () {
        self.visits += 1;
        self.tot_value += value;
        *self.children_visits.get_mut(&action).unwrap() += 1;
        *self.children_value.get_mut(&action).unwrap() += value;
    }

    pub fn best_child(&self) -> BriscolaAction {
        let mut best_action = self.legal_actions[0];
        let mut best_value = f32::NEG_INFINITY;

        for &action in &self.legal_actions {
            let value = self.avg_value(action); // however you compute UCB/avg value
            if value > best_value {
                best_value = value;
                best_action = action;
            }
        }

        best_action
    }
}


fn naive_mcts(obs: BriscolaObs, simulations: usize) -> BriscolaAction {
    let mut root = MCTSNode::new(obs);

    for _ in 0..simulations {
        let env = env_from_obs(&root.obs);
        let (result, first_action) = rollout(env);

        let mut value = match result {
            BriscolaResult::P0Win => 1.0,
            BriscolaResult::Draw => 0.0,
            BriscolaResult::P1Win => -1.0,
            _ => panic!("Result returned to agent when simulation not finished.")
        };
        if root.to_play == 1 { value = -value; }

        root.update_stats(first_action, value);
    }

    root.best_child()
}

fn rollout(mut env: BriscolaEnv) -> (BriscolaResult, BriscolaAction) {
    let first_action = random_action(&env);
    env.step(first_action);
    while !env.done {
        env.step(random_action(&env));
    }
    (env.result, first_action)
}

#[inline]
fn random_action(env: &BriscolaEnv) -> BriscolaAction {
    let n = env.players[env.current_player_id].hand.len();
    BriscolaAction::from_idx(rand::random_range(0..n))
}
    
