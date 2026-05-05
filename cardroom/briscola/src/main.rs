use briscola::agents::generic::BriscolaAgent;
use briscola::agents::human::HumanAgent;
use briscola::agents::random::RandomAgent;
use briscola::env::env::BriscolaEnv;

fn main() {
    let mut env = BriscolaEnv::new();
    let agents: Vec<Box<dyn BriscolaAgent>> = vec![
        Box::new(HumanAgent::new()),
        Box::new(RandomAgent::new()),
    ];

    env.render();

    while !env.done {
        let action = agents[env.current_player_id].select_action(&env);
        env.step(action);
        env.render();
    }
}
