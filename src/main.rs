use env_logger::Env;
use log::{error, info};

use blitz_bot::bot::Bot;
use blitz_bot::client::WebSocketGameClient;
use blitz_bot::solvers::{AntColonyOptimizationSolver, Solver};

fn new_solver() -> Box<dyn Solver> {
    // TODO: support sweep hyperparams.json use-case.
    // let hyperparams = if let Ok(hyperparam_data) = fs::read_to_string("hyperparams.json") {
    //     info!("[MACRO] Loading hyperparams from hyperparams.json.");
    //     let parsed: Value = serde_json::from_str(&hyperparam_data).expect("invalid json");
    //     serde_json::from_value(parsed).expect("invalid hyperparams")
    // }
    info!("[ACO] Using default hyperparams.");
    Box::new(AntColonyOptimizationSolver::default())
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    // Load .env file
    dotenvy::dotenv().ok();
    // Init logger with default value of info
    // This can be overriden with RUST_LOG env var
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    // TODO: parse solver from args
    let solver = new_solver();
    let bot = Bot::new(solver);
    let token = dotenvy::var("TOKEN");

    if let Err(err) = WebSocketGameClient::new(bot, token.ok()).run().await {
        error!("Error while running bot with underlying error:");
        error!("  {}", err);
    }
}
