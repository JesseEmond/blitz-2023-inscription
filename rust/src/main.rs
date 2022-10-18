use env_logger::Env;
use log::error;

use blitz_bot::bot::Bot;
use blitz_bot::client::WebSocketGameClient;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    // Load .env file
    dotenvy::dotenv().ok();
    // Init logger with default value of info
    // This can be overriden with RUST_LOG env var
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    let bot = Bot::default();
    let token = dotenvy::var("TOKEN");

    if let Err(err) = WebSocketGameClient::new(bot, token.ok()).run().await {
        error!("Error while running bot with underlying error:");
        error!("  {}", err);
    }
}
