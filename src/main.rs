use clap::{Parser, ValueEnum};
use env_logger::Env;
use log::{error, info};

use blitz_bot::bot::Bot;
use blitz_bot::client::WebSocketGameClient;
use blitz_bot::solvers::{AntColonyOptimizationSolver, ExactTspSolver, ExactTspSomeStartsSolver, NearestNeighborSolver, Solver};

#[derive(ValueEnum, Clone)]
enum SolverName {
    /// Ant colony optimization
    AntColonyOptimization,
    /// Exact TSP solver, tries all possible starts.
    ExactTsp,
    /// Exact TSP solver, only try the configured amount of starts.
    ExactTspSomeStarts,
    /// Greedy nearest-neighbor solver.
    NearestNeighbor,
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Solver implementation to use to find a solution.
    #[arg(short, long, value_enum,
          default_value_t = SolverName::AntColonyOptimization)]
    solver: SolverName,

    /// When using exact_tsp_some_starts, maximum number of starts to try.
    #[arg(long, default_value_t = 4)]
    exact_tsp_max_starts: usize,
}

fn new_solver(cli: Cli) -> Box<dyn Solver> {
    match cli.solver {
        SolverName::AntColonyOptimization => {
            // TODO: support sweep hyperparams.json use-case.
            // let hyperparams = if let Ok(hyperparam_data) = fs::read_to_string("hyperparams.json") {
            //     info!("[MACRO] Loading hyperparams from hyperparams.json.");
            //     let parsed: Value = serde_json::from_str(&hyperparam_data).expect("invalid json");
            //     serde_json::from_value(parsed).expect("invalid hyperparams")
            // }
            info!("[ACO] Using default hyperparams.");
            Box::new(AntColonyOptimizationSolver::default())
        },
        SolverName::ExactTsp => Box::new(ExactTspSolver{}),
        SolverName::ExactTspSomeStarts => Box::new(
            ExactTspSomeStartsSolver { max_starts: cli.exact_tsp_max_starts }),
        SolverName::NearestNeighbor => Box::new(NearestNeighborSolver::new()),
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    // Load .env file
    dotenvy::dotenv().ok();
    // Init logger with default value of info
    // This can be overriden with RUST_LOG env var
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();

    let solver = new_solver(cli);
    let bot = Bot::new(solver);
    let token = dotenvy::var("TOKEN");

    if let Err(err) = WebSocketGameClient::new(bot, token.ok()).run().await {
        error!("Error while running bot with underlying error:");
        error!("  {}", err);
    }
}
