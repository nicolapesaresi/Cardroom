use briscola::agents::bot::BotAgent;
use briscola::agents::generic::BriscolaAgent;
use briscola::agents::random::RandomAgent;
use briscola::env::env::{BriscolaEnv, BriscolaResult};

const DEFAULT_N_GAMES: usize = 100;

fn main() {
    let n_games: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_N_GAMES);

    let agents: Vec<Box<dyn BriscolaAgent>> = vec![
        Box::new(BotAgent::new()),
        Box::new(RandomAgent::new()),
    ];

    let mut env = BriscolaEnv::new();
    let mut bot_wins = 0usize;
    let mut random_wins = 0usize;
    let mut draws = 0usize;
    let mut bot_points_total = 0i64;
    let mut random_points_total = 0i64;

    for _ in 0..n_games {
        env.reset();
        while !env.done {
            let action = agents[env.current_player_id].select_action(&env);
            env.step(action);
        }
        match env.result {
            BriscolaResult::P0Win => bot_wins += 1,
            BriscolaResult::P1Win => random_wins += 1,
            BriscolaResult::Draw => draws += 1,
            BriscolaResult::InProgress => unreachable!(),
        }
        bot_points_total += env.players[0].points as i64;
        random_points_total += env.players[1].points as i64;
    }

    let n = n_games as f64;
    println!("Results over {} games  (Bot=P0  vs  Random=P1)", n_games);
    println!("  Bot wins:    {:4}  ({:.1}%)", bot_wins, bot_wins as f64 / n * 100.0);
    println!("  Random wins: {:4}  ({:.1}%)", random_wins, random_wins as f64 / n * 100.0);
    println!("  Draws:       {:4}  ({:.1}%)", draws, draws as f64 / n * 100.0);
    println!("  Avg bot points:    {:.1}", bot_points_total as f64 / n);
    println!("  Avg random points: {:.1}", random_points_total as f64 / n);
}
