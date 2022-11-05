use clap::{Parser, ValueEnum};
use env_logger::Env;
use log::{error, info};
use serde_json::Value;

use blitz_bot::bot::Bot;
use blitz_bot::client::WebSocketGameClient;
use blitz_bot::solvers::{AntColonyOptimizationSolver, ExactTspSolver, ExactTspSomeStartsSolver, NearestNeighborSolver, OptimalSolver, Solver};

#[derive(ValueEnum, Clone)]
enum SolverName {
    /// Ant colony optimization
    AntColonyOptimization,
    /// Exact TSP solver, tries all possible starts for a full tour.
    ExactTsp,
    /// Exact TSP solver, only tries the configured # of starts for a full tour.
    ExactTspSomeStarts,
    /// Greedy nearest-neighbor solver.
    NearestNeighbor,
    /// Optimal solver, does exact TSP for all starts, for all ports subsets.
    Optimal,
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Solver implementation to use to find a solution.
    #[arg(short, long, value_enum,
          default_value_t = SolverName::AntColonyOptimization)]
    solver: SolverName,

    /// When using exact-tsp-some-starts, maximum number of starts to try.
    #[arg(long, default_value_t = 4)]
    exact_tsp_max_starts: usize,

    /// When using ant-colony-optimization, hyperparams JSON file to use.
    #[arg(long)]
    aco_hyperparams_file: Option<String>,
}

fn new_solver(cli: Cli) -> Box<dyn Solver> {
    match cli.solver {
        SolverName::AntColonyOptimization => {
            match &cli.aco_hyperparams_file {
                Some(filename) => {
                    let hyperparams_data = std::fs::read_to_string(filename)
                        .expect("failed to read aco_hyperparams_file");
                    info!("[ACO] Loading hyperparams from {filename}");
                    let parsed: Value = serde_json::from_str(&hyperparams_data)
                        .expect("invalid json");
                    let hyperparams = serde_json::from_value(parsed)
                        .expect("invalid hyperparams");
                    Box::new(AntColonyOptimizationSolver::new(hyperparams))
                },
                None => {
                    info!("[ACO] Using default hyperparams.");
                    Box::new(AntColonyOptimizationSolver::default())
                },
            }
        },
        SolverName::ExactTsp => Box::new(ExactTspSolver{}),
        SolverName::ExactTspSomeStarts => Box::new(
            ExactTspSomeStartsSolver { max_starts: cli.exact_tsp_max_starts }),
        SolverName::NearestNeighbor => Box::new(NearestNeighborSolver::new()),
        SolverName::Optimal => Box::new(OptimalSolver{}),
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
