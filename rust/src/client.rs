use futures_util::{SinkExt, StreamExt};
use log::{debug, error, info};
use serde_json::Error as JSONError;
use serde_json::{json, Value};
use thiserror::Error;
use tokio_tungstenite::tungstenite::Error as TungsteniteError;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use crate::bot::{Bot, Error as BotError};
use crate::game_interface::GameTick;

pub struct WebSocketGameClient {
    bot: Bot,
    uri: String,
    token: Option<String>,
}

#[derive(Error, Debug)]
pub enum Error {
    #[error("Could not connect to the game ({0})")]
    WebSocketError(#[from] TungsteniteError),
    #[error("The server did not respond to our registration request")]
    EmptyRegistrationResponse,
    #[error("Unable to (de)serialize payload from/to server ({0})")]
    JSONError(#[from] JSONError),
    #[error("Received error from server ({0})")]
    ServerError(String),
    #[error("Error while execution bot's code ({0})")]
    BotError(#[from] BotError),
}

impl WebSocketGameClient {
    pub fn new(bot: Bot, token: Option<String>) -> Self {
        WebSocketGameClient {
            bot,
            uri: "ws://127.0.0.1:8765".to_string(),
            token,
        }
    }

    pub async fn run(&mut self) -> Result<(), Error> {
        let (mut stream, _resp) = connect_async(&self.uri).await?;

        let registration = match &self.token {
            Some(token_value) => json!({"type": "REGISTER", "token": token_value}),
            None => json!({"type": "REGISTER", "teamName": "Rusty Bot"}),
        };

        stream.send(Message::text(registration.to_string())).await?;

        loop {
            if let Some(message) = stream.next().await {
                let message = message?;
                let message_text = message.to_text()?;

                debug!("Payload: {}", message_text);

                if message_text.is_empty() {
                    return Err(Error::EmptyRegistrationResponse);
                }

                let parsed: Value = serde_json::from_str(message_text)?;
                if parsed["type"] == "ERROR" {
                    return Err(Error::ServerError(parsed.to_string()));
                }

                let game_tick: GameTick = serde_json::from_value(parsed)?;
                info!(
                    "Playing tick {} of {}",
                    game_tick.current_tick, game_tick.total_ticks
                );

                // Copy current tick in order to send the command response
                let current_tick = game_tick.current_tick;
                // Fetch the next action from the bot
                let action = self.bot.get_next_move(&game_tick)?;

                info!("Action of bot is: {:?}", action);
                let response = json!({"type": "COMMAND", "tick": current_tick, "action": action});
                debug!("Response payload: {}", response.to_string());
                stream.send(Message::Text(response.to_string())).await?;

                if self.bot.is_done(&game_tick) {
                    return Ok(())
                }
            }
        }
    }
}
