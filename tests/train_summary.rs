use clap::Parser;
use menace::cli::commands::train::{TrainArgs, execute};
use tempfile::tempdir;

fn parse_args<I, T>(args: I) -> TrainArgs
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    TrainArgs::parse_from(args)
}

#[test]
fn summary_without_extension_appends_json() {
    let tmp = tempdir().unwrap();
    let summary_stem = tmp.path().join("run_overview");

    let args = parse_args([
        "menace-train",
        "menace",
        "--games",
        "5",
        "--opponent",
        "random",
        "--summary",
        summary_stem.to_str().unwrap(),
        "--validation-games",
        "0",
    ]);

    execute(args).expect("training with summary should succeed");

    let expected_path = summary_stem.with_extension("json");
    assert!(
        expected_path.exists(),
        "expected summary at {}",
        expected_path.display()
    );

    let contents = std::fs::read_to_string(&expected_path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();
    assert_eq!(parsed["training"]["total_games"], 5);
    assert_eq!(parsed["regimen"], "optimal");
}

#[test]
fn summary_directory_argument_creates_default_file() {
    let tmp = tempdir().unwrap();
    let summary_dir = tmp.path().join("summaries");
    let summary_arg = format!("{}/", summary_dir.display());

    let args = parse_args([
        "menace-train",
        "menace",
        "--games",
        "3",
        "--opponent",
        "random",
        "--summary",
        &summary_arg,
        "--validation-games",
        "0",
    ]);

    execute(args).expect("training with directory summary should succeed");

    let expected_path = summary_dir.join("training_summary.json");
    assert!(
        expected_path.exists(),
        "expected summary at {}",
        expected_path.display()
    );

    let contents = std::fs::read_to_string(&expected_path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();
    assert_eq!(parsed["training"]["total_games"], 3);
}
