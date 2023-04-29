use qrcode::{render::unicode, QrCode};
use rand::RngCore;
use sqlx::{sqlite::SqliteConnectOptions, ConnectOptions};
use std::{error::Error, str::FromStr};
use structopt::StructOpt;
use tokio::io::{self, AsyncBufReadExt, AsyncWriteExt};
use totp_rs::{Algorithm, TOTP};

#[derive(StructOpt, Debug)]
struct Opt {
    #[structopt(subcommand)] // Note that we mark a field as a subcommand
    cmd: Command,
}

#[derive(StructOpt, Debug)]
enum Command {
    /// Add a new user
    AddUser { name: String },
    /// List users
    QueryUsers { name_filter: String },
    /// Remove a user
    RmUser { id: i32 },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let opt = Opt::from_args();
    let db_url = dotenvy::var("DATABASE_URL").unwrap_or("sqlite:db.sqlite".to_string());
    let mut db = SqliteConnectOptions::from_str(&db_url)?
        .create_if_missing(true)
        .connect()
        .await?;

    sqlx::migrate!().run(&mut db).await?;

    match opt.cmd {
        Command::AddUser { name } => {
            let mut secret = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut secret);
            let secret = secret.to_vec();
            let totp = TOTP::new(
                Algorithm::SHA256,
                6,
                1,
                30,
                secret.clone(),
                Some("morselock".to_string()),
                name.clone(),
            )
            .unwrap();
            let code = QrCode::new(totp.get_url()).unwrap();
            let image = code
                .render::<unicode::Dense1x2>()
                .dark_color(unicode::Dense1x2::Light)
                .light_color(unicode::Dense1x2::Dark)
                .build();
            println!("{}", image);
            println!("Enter TOTP code to continue: ");
            println!("Current: {}", totp.generate_current()?);
            let stdin = io::stdin();
            let mut reader = io::BufReader::new(stdin);
            let mut buffer = String::new();
            reader.read_line(&mut buffer).await?;


            if totp.check_current(buffer.trim())? {
                println!("User added.");
                sqlx::query!(
                    "INSERT INTO users (name, secret) VALUES (?, ?)",
                    name,
                    secret
                )
                .execute(&mut db)
                .await?;
            } else {
                println!("Aborted.");
            }
        }
        Command::QueryUsers { name_filter } => {
            let results = sqlx::query!(
                "SELECT id, name FROM users WHERE name LIKE \"%\" || ? || \"%\"",
                name_filter
            )
            .fetch_all(&mut db)
            .await?;
            println!("id,name");
            for user in results {
                println!("{},{}", user.id, user.name);
            }
        }
        Command::RmUser { id } => {
            sqlx::query!("DELETE FROM users WHERE id = ?", id)
                .execute(&mut db)
                .await?;
        }
    };
    Ok(())
}
