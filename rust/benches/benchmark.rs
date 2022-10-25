use criterion::{criterion_group, criterion_main, Criterion};
use serde_json::{Value};
use std::fs;

use blitz_bot::ant_colony_optimization::{Colony, HyperParams};
use blitz_bot::held_karp::{held_karp};
use blitz_bot::game_interface::{GameTick};
use blitz_bot::graph::{Graph};
use blitz_bot::pathfinding::{Pathfinder};


fn make_game() -> GameTick {
    // Note this isn't great, we ideally shouldn't read from disk here.
    let game_file = "../games/3226.json";
    let game_json = fs::read_to_string(game_file).expect("Couldn't read game file");
    let parsed: Value = serde_json::from_str(&game_json).expect("Couldn't parse JSON in game file");
    serde_json::from_value(parsed).expect("Couldn't parse game tick in game file")
}

fn make_pathfinder(game: &GameTick) -> Pathfinder {
    let mut pathfinder = Pathfinder::new();
    let schedule: Vec<u8> = game.tide_schedule.iter().map(|&e| e as u8).collect();
    pathfinder.grid.init(&game.map, &schedule, game.current_tick.into());
    pathfinder
}

fn make_hyperparams() -> HyperParams {
    HyperParams {
        iterations: 342,
        ants: 461,
        evaporation_rate: 0.6,
        exploitation_probability: 0.077,
        heuristic_power: 3.34,
        base_pheromones: 0.7635,
        local_evaporation_rate: 0.50,
    }
}

fn bench_pathfind_only(c: &mut Criterion) {
    let game = make_game();
    c.bench_function("pathfind_only", |b| b.iter(|| {
        let mut pathfinder = make_pathfinder(&game);
        let graph = Graph::new(&mut pathfinder, &game);
        graph
    }));
}

fn bench_ant_colony_optimization_only(c: &mut Criterion) {
    let game = make_game();
    let mut pathfinder = make_pathfinder(&game);
    let graph = Graph::new(&mut pathfinder, &game);
    c.bench_function("ant_colony_optimization_only", |b| b.iter(|| {
        let hyperparams = make_hyperparams();
        let mut colony = Colony::new(graph.clone(), hyperparams, /*seed=*/42);
        colony.run()
    }));
}

fn bench_full_tick0(c: &mut Criterion) {
    let game = make_game();
    c.bench_function("full_tick0", |b| b.iter(|| {
        let mut pathfinder = make_pathfinder(&game);
        let graph = Graph::new(&mut pathfinder, &game);
        let hyperparams = make_hyperparams();
        let mut colony = Colony::new(graph.clone(), hyperparams, /*seed=*/42);
        colony.run()
    }));
}

fn bench_held_karp(c: &mut Criterion) {
    let game = make_game();
    let mut pathfinder = make_pathfinder(&game);
    let mut graph = Graph::new(&mut pathfinder, &game);
    graph.ports.truncate(18);
    c.bench_function("held_karp", |b| b.iter(|| {
        held_karp(&graph)
    }));
}

criterion_group!{
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_pathfind_only, bench_ant_colony_optimization_only, bench_full_tick0, bench_held_karp
}
criterion_main!(benches);
