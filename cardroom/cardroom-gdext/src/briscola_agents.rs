use godot::prelude::*;
use briscola::agents::random::RandomAgent;
use briscola::agents::bot::BotAgent;
use briscola::agents::generic::BriscolaAgent;
use crate::briscola_game::BriscolaEnvNode;

#[derive(GodotClass)]
#[class(base=Node)]
pub struct RandomAgentNode {
    inner: RandomAgent,
    base: Base<Node>,
}

#[godot_api]
impl INode for RandomAgentNode {
    fn init(base: Base<Node>) -> Self {
        Self { inner: RandomAgent::new(), base }
    }
}

#[godot_api]
impl RandomAgentNode {
    #[func]
    fn select_action(&self, env: Gd<BriscolaEnvNode>) -> i32 {
        self.inner.select_action(&env.bind().inner).idx() as i32
    }
}

#[derive(GodotClass)]
#[class(base=Node)]
pub struct BotAgentNode {
    inner: BotAgent,
    base: Base<Node>,
}

#[godot_api]
impl INode for BotAgentNode {
    fn init(base: Base<Node>) -> Self {
        Self { inner: BotAgent::new(), base }
    }
}

#[godot_api]
impl BotAgentNode {
    #[func]
    fn select_action(&self, env: Gd<BriscolaEnvNode>) -> i32 {
        self.inner.select_action(&env.bind().inner).idx() as i32
    }
}
