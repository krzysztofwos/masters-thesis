//! Output formatting and progress bars for CLI

use indicatif::{ProgressBar, ProgressStyle};

/// Create a progress bar for training
pub fn create_training_progress(total_games: u64) -> ProgressBar {
    let pb = ProgressBar::new(total_games);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} games ({msg})")
            .expect("Invalid progress bar template")
            .progress_chars("=>-"),
    );
    pb
}

/// Create a spinner for analysis tasks
pub fn create_spinner(message: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .expect("Invalid spinner template"),
    );
    pb.set_message(message.to_string());
    pb
}

/// Print a section header
pub fn print_section(title: &str) {
    println!("\n{}", "=".repeat(60));
    println!("{title}");
    println!("{}", "=".repeat(60));
}

/// Print a subsection header
pub fn print_subsection(title: &str) {
    println!("\n{title}");
    println!("{}", "-".repeat(40));
}

/// Format a number with thousands separators
pub fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i.is_multiple_of(3) {
            result.insert(0, ',');
        }
        result.insert(0, c);
    }
    result
}

/// Print a key-value pair
pub fn print_kv(key: &str, value: &str) {
    println!("  {:20} {}", format!("{}:", key), value);
}

/// Print statistics table
pub fn print_stats_table(stats: &[(&str, &str)]) {
    for (key, value) in stats {
        print_kv(key, value);
    }
}
