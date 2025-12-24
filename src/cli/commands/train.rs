//! Train command - Train learners (MENACE, Active Inference, etc.)

use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use anyhow::{Result, anyhow};
use clap::{Parser, ValueEnum};
use serde::Serialize;
use serde_json::to_writer_pretty;

use crate::{
    active_inference::{PolicyPrior, PreferenceModel},
    menace::{MenaceAgent, ReinforcementValues, SavedMenaceAgent},
    pipeline::{
        CurriculumConfig, JsonlObserver, Learner, MenaceLearner, MetricsObserver,
        MilestoneObserver, ProgressObserver, SharedMenaceLearner, TrainingConfig, TrainingPipeline,
    },
    tictactoe::Player,
    workspace::{InitialBeadSchedule, StateFilter},
};

#[derive(Debug, Serialize)]
struct SummaryStats {
    total_games: usize,
    wins: usize,
    draws: usize,
    losses: usize,
    win_rate: f64,
    draw_rate: f64,
    loss_rate: f64,
}

impl From<&crate::pipeline::TrainingResult> for SummaryStats {
    fn from(result: &crate::pipeline::TrainingResult) -> Self {
        Self {
            total_games: result.total_games,
            wins: result.wins,
            draws: result.draws,
            losses: result.losses,
            win_rate: result.win_rate,
            draw_rate: result.draw_rate,
            loss_rate: result.loss_rate,
        }
    }
}

#[derive(Debug, Serialize)]
struct ScheduleBlock {
    opponent: String,
    games: usize,
}

#[derive(Debug, Serialize)]
struct TrainingSummaryFile {
    training: SummaryStats,
    validation: Option<SummaryStats>,
    regimen: String,
    schedule: Vec<ScheduleBlock>,
    metadata: SummaryMetadata,
}

#[derive(Debug, Serialize)]
struct SummaryMetadata {
    filter: String,
    restock_mode: String,
    agent_player: String,
    first_player: String,
    seed: Option<u64>,
}

pub(crate) fn parse_player_token(value: &str, flag: &str) -> Result<Player> {
    match value.trim().to_ascii_lowercase().as_str() {
        "x" | "first" | "player1" | "p1" => Ok(Player::X),
        "o" | "second" | "player2" | "p2" => Ok(Player::O),
        other => Err(anyhow!(
            "Invalid value '{other}' for {flag} (expected 'x' or 'o')"
        )),
    }
}

fn sanitize_summary_path(raw: &Path) -> PathBuf {
    let mut normalized = raw.to_path_buf();
    let raw_str = raw.as_os_str().to_string_lossy();

    // Treat trailing separators or missing filename as a directory target.
    if raw_str.ends_with(std::path::MAIN_SEPARATOR) || normalized.file_name().is_none() {
        normalized.push("training_summary.json");
        return normalized;
    }

    match normalized.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("json") => normalized,
        _ => {
            normalized.set_extension("json");
            normalized
        }
    }
}

#[derive(Parser, Debug)]
#[command(about = "Train a learner", allow_negative_numbers = true)]
pub struct TrainArgs {
    /// Type of learner to train
    #[arg(value_enum)]
    pub learner: LearnerType,

    /// Opponent to train against
    #[arg(long, short = 'o', default_value = "random")]
    pub opponent: String,

    /// Number of training games
    #[arg(long, short = 'g', default_value_t = 500)]
    pub games: usize,

    /// Output file for trained agent
    #[arg(long, short = 'O')]
    pub output: Option<PathBuf>,

    /// Optional file for JSONL observations
    #[arg(long)]
    pub observations: Option<PathBuf>,

    /// Optional path for writing a summary JSON file
    #[arg(long)]
    pub summary: Option<PathBuf>,

    /// Random seed for reproducibility
    #[arg(long)]
    pub seed: Option<u64>,

    /// Which token the agent controls (`x` or `o`)
    #[arg(long, default_value = "x")]
    pub agent_player: String,

    /// Which token makes the first move (`x` or `o`)
    #[arg(long, default_value = "x")]
    pub first_player: String,

    /// Show progress bar
    #[arg(long, default_value_t = true)]
    pub progress: bool,

    /// Reward schedule (win=3,draw=1,loss=-1)
    #[arg(long)]
    pub reward: Option<String>,

    /// Initial bead schedule per ply (e.g., 4,3,2,1)
    #[arg(long)]
    pub init_beads: Option<String>,

    /// Epistemic weight for Active Inference (0.0-1.0, balance exploration vs exploitation)
    #[arg(long = "ai-epistemic-weight", alias = "ai-beta", default_value_t = 0.5)]
    pub ai_beta: f64,

    /// Ambiguity weight for Active Inference (penalize/reward outcome uncertainty)
    /// Positive: prefer deterministic outcomes (risk-averse)
    /// Negative: prefer uncertain outcomes (keep options open)
    /// Zero disables ambiguity sensitivity (default: 0.0)
    #[arg(
        long = "ai-ambiguity-weight",
        alias = "ai-ambiguity",
        default_value_t = 0.0
    )]
    pub ai_ambiguity: f64,

    /// Active Inference opponent model (uniform, adversarial, or minimax)
    #[arg(long, default_value = "uniform")]
    pub ai_opponent: String,

    /// Restock mode when matchbox runs out of beads (none, move, or box)
    #[arg(long, default_value = "box")]
    pub restock: String,

    /// State filter strategy (all, decision-only, michie, or both)
    #[arg(long, default_value = "michie")]
    pub filter: String,

    /// Skip expensive MENACE policy metrics for performance
    #[arg(long, default_value_t = false)]
    pub skip_menace_policy_metrics: bool,

    /// Training curriculum regimen (optimal, random, defensive, or mixed)
    #[arg(long, short = 'r', default_value = "optimal")]
    pub regimen: String,

    /// Override: random-opponent games when using mixed regimen
    #[arg(long)]
    pub mixed_random_games: Option<usize>,

    /// Override: defensive-opponent games when using mixed regimen
    #[arg(long)]
    pub mixed_defensive_games: Option<usize>,

    /// Override: optimal-opponent games when using mixed regimen
    #[arg(long)]
    pub mixed_optimal_games: Option<usize>,

    /// Number of post-training validation games vs optimal play
    #[arg(long, short = 'v', default_value_t = 50)]
    pub validation_games: usize,

    /// Seed for stochastic elements during validation (defaults to seed+1)
    #[arg(long)]
    pub validation_seed: Option<u64>,

    /// Track learning milestones (first draw, last loss) during validation
    #[arg(long, default_value_t = false)]
    pub track_milestones: bool,

    /// Show historical comparison with Michie's 1961 MENACE
    #[arg(long, default_value_t = true)]
    pub historical_comparison: bool,

    /// Phase-based training: comma-separated game counts per phase (e.g., "50,100,150,100,100")
    #[arg(long)]
    pub phases: Option<String>,

    /// Show detailed report after each training phase
    #[arg(long, default_value_t = false)]
    pub phase_report: bool,

    /// Self-play: agent plays both X and O to learn from both perspectives
    #[arg(long, default_value_t = false)]
    pub self_play: bool,

    /// Agent vs agent: train against another saved MENACE agent
    #[arg(long)]
    pub opponent_agent: Option<PathBuf>,

    /// Show convergence metrics (entropy reduction, high-confidence positions)
    #[arg(long, default_value_t = false)]
    pub convergence_report: bool,

    /// Expected free energy mode for Active Inference (approx or exact)
    #[arg(long, default_value = "approx")]
    pub efe: String,

    /// Policy temperature / precision parameter for KL-regularised policy selection: q(a) ∝ p(a) exp(-EFE(a)/λ)
    /// Smaller λ => sharper (more exploitative) policies; larger λ => closer to the prior
    #[arg(long, default_value_t = 0.25)]
    pub policy_lambda: f64,

    /// Scale factor converting policy probabilities to bead weights
    #[arg(long, default_value_t = 40.0)]
    pub policy_beads_scale: f64,

    /// Dirichlet α for the opponent model in exact EFE mode
    #[arg(long, default_value_t = 1.0)]
    pub opponent_dirichlet_alpha: f64,

    /// Policy prior for KL regularization (uniform, menace, or menace-initial)
    #[arg(long, default_value = "uniform")]
    pub policy_prior: String,

    /// Outcome preferences for Active Inference (e.g., probs:0.9,0.09,0.01 or utils:1,0,-1)
    #[arg(long)]
    pub prefs: Option<String>,

    /// Export exact Active Inference decompositions to CSV
    #[arg(long)]
    pub export_aif_exact_csv: Option<PathBuf>,

    /// Comma-separated list of canonical states to include in the export
    #[arg(long)]
    pub export_states: Option<String>,

    /// Include O-to-move states in the exact AIF CSV export
    #[arg(long, default_value_t = false)]
    pub aif_include_o: bool,

    /// Q-learning/SARSA learning rate α (0.0-1.0)
    #[arg(long, default_value_t = 0.5)]
    pub ql_learning_rate: f64,

    /// Q-learning/SARSA discount factor γ (0.0-1.0)
    #[arg(long, default_value_t = 0.99)]
    pub ql_discount: f64,

    /// Q-learning/SARSA initial epsilon (exploration rate)
    #[arg(long, default_value_t = 0.5)]
    pub ql_epsilon: f64,

    /// Q-learning/SARSA epsilon decay per episode
    #[arg(long, default_value_t = 0.995)]
    pub ql_epsilon_decay: f64,

    /// Q-learning/SARSA minimum epsilon
    #[arg(long, default_value_t = 0.01)]
    pub ql_min_epsilon: f64,

    /// Q-learning/SARSA initial Q-value
    #[arg(long, default_value_t = 0.0)]
    pub ql_q_init: f64,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum LearnerType {
    /// Classic MENACE (Matchbox Educable Noughts And Crosses Engine)
    Menace,
    /// Hybrid Active Inference (EFE-based policy + Bayesian opponent beliefs)
    ActiveInference,
    /// Oracle Active Inference (uses perfect game tree knowledge, no learning)
    OracleActiveInference,
    /// Pure Active Inference (Bayesian beliefs for both opponent and action outcomes)
    PureActiveInference,
    /// Q-learning (off-policy TD control)
    QLearning,
    /// SARSA (on-policy TD control)
    Sarsa,
}

/// Parse reward schedule from string (e.g., "win=3,draw=1,loss=-1")
fn parse_reward_schedule(s: &str) -> Result<ReinforcementValues> {
    let mut win = None;
    let mut draw = None;
    let mut loss = None;

    for part in s.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut iter = trimmed.splitn(2, '=');
        let key = iter
            .next()
            .ok_or_else(|| anyhow!("Invalid reward entry: '{trimmed}'"))?;
        let value_str = iter
            .next()
            .ok_or_else(|| anyhow!("Invalid reward entry '{trimmed}'. Expected key=value"))?;
        let value: i16 = value_str
            .parse()
            .map_err(|_| anyhow!("Invalid numeric reward '{value_str}' in '{trimmed}'"))?;
        match key.trim().to_ascii_lowercase().as_str() {
            "win" => win = Some(value),
            "draw" => draw = Some(value),
            "loss" => loss = Some(value),
            other => {
                return Err(anyhow!(
                    "Unknown reward key '{other}'. Expected win, draw, or loss"
                ));
            }
        }
    }

    Ok(ReinforcementValues {
        win: win.unwrap_or(3),
        draw: draw.unwrap_or(1),
        loss: loss.unwrap_or(-1),
    })
}

/// Parse initial bead schedule from string (e.g., "4,3,2,1")
fn parse_initial_beads(s: &str) -> Result<InitialBeadSchedule> {
    let values: Vec<f64> = s
        .split(',')
        .map(|v| v.trim().parse::<f64>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| anyhow!("Invalid bead schedule '{s}': {e}"))?;

    if values.len() != 4 {
        return Err(anyhow!(
            "Bead schedule must have exactly 4 values (ply 0,2,4,6), got {}",
            values.len()
        ));
    }

    Ok(InitialBeadSchedule::new([
        values[0], values[1], values[2], values[3],
    ]))
}

/// Parse restock mode from string (e.g., "none", "move", "box")
fn parse_restock_mode(s: &str) -> Result<crate::workspace::RestockMode> {
    use crate::workspace::RestockMode;
    match s.to_lowercase().as_str() {
        "none" => Ok(RestockMode::None),
        "move" => Ok(RestockMode::Move),
        "box" => Ok(RestockMode::Box),
        other => Err(anyhow!(
            "Unknown restock mode '{other}'. Use 'none', 'move', or 'box'"
        )),
    }
}

/// Parse state filter from string (e.g., "all", "decision-only", "michie", "both")
fn parse_state_filter(s: &str) -> Result<crate::workspace::StateFilter> {
    use crate::workspace::StateFilter;
    match s.to_lowercase().as_str() {
        "all" => Ok(StateFilter::All),
        "decision-only" => Ok(StateFilter::DecisionOnly),
        "michie" => Ok(StateFilter::Michie),
        "both" => Ok(StateFilter::Both),
        other => Err(anyhow!(
            "Unknown state filter '{other}'. Use 'all', 'decision-only', 'michie', or 'both'"
        )),
    }
}

/// Parse training regimen from string (e.g., "optimal", "random", "defensive", "mixed")
fn parse_training_regimen(s: &str) -> Result<crate::pipeline::TrainingRegimen> {
    use crate::pipeline::TrainingRegimen;
    match s.to_lowercase().as_str() {
        "optimal" => Ok(TrainingRegimen::Optimal),
        "random" => Ok(TrainingRegimen::Random),
        "defensive" => Ok(TrainingRegimen::Defensive),
        "mixed" => Ok(TrainingRegimen::Mixed),
        other => Err(anyhow!(
            "Unknown training regimen '{other}'. Use 'optimal', 'random', 'defensive', or 'mixed'"
        )),
    }
}

/// Outcome preferences for Active Inference
#[derive(Debug, Clone, Copy)]
enum PreferencesArg {
    Probs(f64, f64, f64),
    Utils(f64, f64, f64),
}

/// Parse outcome preferences from string (e.g., "probs:0.9,0.09,0.01" or "utils:1,0,-1")
fn parse_preferences(s: &str) -> Result<PreferencesArg> {
    let (kind, rest) = s
        .split_once(':')
        .ok_or_else(|| anyhow!("Expected format probs:w,d,l or utils:w,d,l"))?;

    let values: Vec<&str> = rest.split(',').map(|t| t.trim()).collect();
    if values.len() != 3 {
        return Err(anyhow!(
            "Provide three comma-separated numbers for win,draw,loss preferences"
        ));
    }

    let w = values[0]
        .parse::<f64>()
        .map_err(|_| anyhow!("Invalid preference value '{}'", values[0]))?;
    let d = values[1]
        .parse::<f64>()
        .map_err(|_| anyhow!("Invalid preference value '{}'", values[1]))?;
    let l = values[2]
        .parse::<f64>()
        .map_err(|_| anyhow!("Invalid preference value '{}'", values[2]))?;

    match kind.trim().to_ascii_lowercase().as_str() {
        "probs" | "prob" => Ok(PreferencesArg::Probs(w, d, l)),
        "utils" | "util" => Ok(PreferencesArg::Utils(w, d, l)),
        other => Err(anyhow!(
            "Preference kind '{other}' must be 'probs' or 'utils'"
        )),
    }
}

/// Parse EFE mode from string (e.g., "approx" or "exact")
fn parse_efe_mode(s: &str) -> Result<crate::active_inference::EFEMode> {
    use crate::active_inference::EFEMode;
    match s.to_lowercase().as_str() {
        "approx" => Ok(EFEMode::Approx),
        "exact" => Ok(EFEMode::Exact),
        other => Err(anyhow!(
            "Unknown EFE mode '{other}'. Use 'approx' or 'exact'"
        )),
    }
}

/// Policy prior type
#[derive(Debug, Clone)]
enum PolicyPriorArg {
    Uniform,
    Menace,
    MenaceInitial,
}

/// Parse policy prior from string (e.g., "uniform", "menace", "menace-initial")
fn parse_policy_prior(s: &str) -> Result<PolicyPriorArg> {
    match s.to_lowercase().as_str() {
        "uniform" => Ok(PolicyPriorArg::Uniform),
        "menace" => Ok(PolicyPriorArg::Menace),
        "menace-initial" => Ok(PolicyPriorArg::MenaceInitial),
        other => Err(anyhow!(
            "Unknown policy prior '{other}'. Use 'uniform', 'menace', or 'menace-initial'"
        )),
    }
}

/// Parse phase specification from string (e.g., "50,100,150,100,100")
fn parse_phases(s: &str) -> Result<Vec<usize>> {
    s.split(',')
        .map(|phase| {
            phase
                .trim()
                .parse::<usize>()
                .map_err(|e| anyhow!("Invalid phase game count '{phase}': {e}"))
        })
        .collect()
}

/// Display convergence metrics for a trained agent
fn display_convergence_metrics(agent: &crate::menace::MenaceAgent) {
    println!("\n=== Convergence Metrics ===");

    let stats = agent.stats();
    println!(
        "Total beads: {}",
        format_number(stats.total_beads.round() as usize)
    );
    println!("Average entropy: {:.3}", stats.avg_entropy);

    // Calculate high-confidence positions
    let mut high_confidence = 0;
    let mut total_positions = 0;

    for label_str in agent.decision_labels() {
        if let Ok(label) = crate::types::CanonicalLabel::parse(label_str)
            && let Some(weights) = agent.workspace().move_weights(&label)
        {
            let total_weight: f64 = weights.iter().map(|(_, w)| *w).sum();
            if total_weight > 0.0 {
                total_positions += 1;
                let max_weight = weights.iter().map(|(_, w)| *w).fold(0.0, f64::max);
                if max_weight / total_weight > 0.5 {
                    high_confidence += 1;
                }
            }
        }
    }

    if total_positions > 0 {
        println!(
            "High-confidence positions: {}/{} ({:.1}%)",
            high_confidence,
            total_positions,
            high_confidence as f64 / total_positions as f64 * 100.0
        );
    }

    // Entropy reduction
    if stats.total_matchboxes > 0 {
        let initial_entropy = (stats.total_matchboxes as f64).ln();
        if initial_entropy > 0.0 {
            let entropy_reduction = (initial_entropy - stats.avg_entropy) / initial_entropy * 100.0;
            println!("Entropy reduction: {entropy_reduction:.1}%");
        }
    }
}

/// Format numbers with thousand separators
fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    let mut count = 0;

    for c in s.chars().rev() {
        if count == 3 {
            result.insert(0, ',');
            count = 0;
        }
        result.insert(0, c);
        count += 1;
    }

    result
}

/// Display historical comparison with Michie's 1961 MENACE
fn display_historical_comparison(
    state_filter: crate::workspace::StateFilter,
    training_games: usize,
    milestone_observer: Option<&MilestoneObserver>,
) {
    use crate::workspace::{MenaceWorkspace, StateFilter};

    println!("\n=== Historical Comparison: Michie's MENACE (1961) ===");
    println!("Donald Michie's original MENACE implementation:");
    println!("  Configuration: 287 matchboxes (states)");
    println!("  Initial beads: 4 per opening, decreasing by ply");
    println!("  Reinforcement: +3 win, +1 draw, -1 loss");
    println!("  Opponent: Optimal (minimax) play");
    println!("  Games to competence: ~20 games");
    println!("  Final performance: Consistent draws vs optimal");

    println!("\nYour implementation:");
    let state_count = MenaceWorkspace::new(state_filter)
        .map(|workspace| workspace.decision_labels().count())
        .unwrap_or_else(|_| match state_filter {
            StateFilter::Michie => 287,
            StateFilter::DecisionOnly => 304,
            StateFilter::All => 338,
            StateFilter::Both => 1042,
        });
    println!("  Configuration: {state_count} states ({state_filter:?} filter)");
    println!("  Training games: {training_games}");

    if let Some(observer) = milestone_observer {
        if let Some(first_draw) = observer.first_draw() {
            println!("  Games to first draw: {}", first_draw + 1);

            // Compare to Michie's ~20 games
            if first_draw < 20 {
                println!("  ✓ Achieved first draw faster than Michie's 20 games");
            } else if first_draw <= 30 {
                println!("  ≈ Achieved first draw in similar timeframe to Michie");
            } else {
                println!("  ⚠ Took longer than Michie's ~20 games to achieve first draw");
            }
        }

        if let Some(last_loss) = observer.last_loss()
            && let Some(first_draw) = observer.first_draw()
        {
            if last_loss < first_draw {
                println!("  ✓ Achieved competent play (no losses after first draw)");
            } else {
                let games_after = observer.games_vs_opponent() - last_loss;
                println!(
                    "  Games without loss: {} (after game #{})",
                    games_after,
                    last_loss + 1
                );
            }
        }
    }

    println!("\nNote: Michie used manual play with physical matchboxes.");
    println!("      Your implementation uses exact optimal play simulation.");
}

pub fn execute(args: TrainArgs) -> Result<()> {
    // Parse reward schedule if provided
    let reward_values = if let Some(ref reward_str) = args.reward {
        parse_reward_schedule(reward_str)?
    } else {
        ReinforcementValues::default()
    };

    let agent_player = parse_player_token(&args.agent_player, "--agent-player")?;
    let first_player = parse_player_token(&args.first_player, "--first-player")?;

    // Parse initial beads if provided
    let initial_beads = if let Some(ref beads_str) = args.init_beads {
        parse_initial_beads(beads_str)?
    } else {
        InitialBeadSchedule::default()
    };

    // Parse restock mode
    let restock_mode = parse_restock_mode(&args.restock)?;

    // Parse state filter
    let mut state_filter = parse_state_filter(&args.filter)?;

    let summary_spec = args.summary.as_ref().map(|raw| {
        let sanitized = sanitize_summary_path(raw);
        let normalized = sanitized != *raw;
        (sanitized, normalized)
    });

    if !args.self_play && agent_player == Player::O && !matches!(state_filter, StateFilter::Both) {
        println!(
            "⚠️  Agent plays as O; overriding state filter {state_filter:?} → Both to include O decision states"
        );
        state_filter = StateFilter::Both;
    }

    // Parse training regimen
    let regimen = parse_training_regimen(&args.regimen)?;

    // Create curriculum config
    let curriculum_config = CurriculumConfig {
        mixed_random_games: args.mixed_random_games,
        mixed_defensive_games: args.mixed_defensive_games,
        mixed_optimal_games: args.mixed_optimal_games,
    };

    // Generate training schedule
    let schedule = regimen.schedule(args.games, &curriculum_config);

    // Create training config
    let mut config = TrainingConfig {
        num_games: args.games,
        seed: args.seed,
        agent_player,
        first_player,
    };

    let mut validation_result_opt: Option<crate::pipeline::TrainingResult> = None;

    // Check for conflicting options
    if args.self_play && args.opponent_agent.is_some() {
        return Err(anyhow!(
            "Cannot use both --self-play and --opponent-agent. Choose one training mode."
        ));
    }

    // Handle self-play mode BEFORE creating the regular agent
    if args.self_play {
        println!("\n=== Self-Play Training ===");
        println!("Single agent learning from both X and O perspectives");
        println!("Games: {}", config.num_games);
        println!();

        // Only MENACE agents support self-play currently
        if !matches!(args.learner, LearnerType::Menace) {
            return Err(anyhow!(
                "Self-play is currently only supported for MENACE agents"
            ));
        }

        // Self-play requires Both filter (X-to-move AND O-to-move states)
        // Override the filter if necessary and warn the user
        let self_play_filter = if !matches!(state_filter, StateFilter::Both) {
            println!("⚠️  Self-play requires StateFilter::Both (X and O perspectives)");
            println!("   Overriding --filter {state_filter:?} → Both");
            println!();
            StateFilter::Both
        } else {
            state_filter
        };

        // Create agent specifically for self-play
        let mut builder = MenaceAgent::builder();
        if let Some(seed) = args.seed {
            builder = builder.seed(seed);
        }
        builder = builder
            .filter(self_play_filter)
            .reinforcement(reward_values)
            .initial_beads(initial_beads)
            .restock_mode(restock_mode);

        let shared_agent = Arc::new(Mutex::new(builder.build()?));

        if agent_player != Player::X {
            println!(
                "⚠️  Self-play uses separate X/O learners; ignoring --agent-player={} for metric reporting.",
                args.agent_player
            );
        }
        config.agent_player = Player::X;
        config.first_player = first_player;

        // Create two learners sharing the same agent
        let mut agent_x =
            SharedMenaceLearner::new(Arc::clone(&shared_agent), "MENACE-X".to_string(), Player::X);
        let mut agent_o =
            SharedMenaceLearner::new(Arc::clone(&shared_agent), "MENACE-O".to_string(), Player::O);

        // Create pipeline
        let mut pipeline = TrainingPipeline::new(config.clone());
        if args.progress {
            pipeline = pipeline.with_observer(Box::new(ProgressObserver::new()));
        }
        let metrics_observer = MetricsObserver::new();
        pipeline = pipeline.with_observer(Box::new(metrics_observer));

        // Run self-play training
        let result = pipeline.run(&mut agent_x, &mut agent_o)?;

        println!("\n=== Self-Play Complete ===");
        println!("Total games: {}", result.total_games);
        println!("X wins: {} ({:.1}%)", result.wins, result.win_rate * 100.0);
        println!("Draws: {} ({:.1}%)", result.draws, result.draw_rate * 100.0);
        println!(
            "O wins: {} ({:.1}%)",
            result.losses,
            result.loss_rate * 100.0
        );

        // Display convergence metrics if requested
        if args.convergence_report {
            display_convergence_metrics(&shared_agent.lock().unwrap());
        }

        // Save agent if output path provided
        if let Some(output_path) = args.output {
            println!("\nSaving self-play trained agent...");
            let metadata = crate::menace::TrainingMetadata {
                games_trained: Some(result.total_games),
                opponents: vec!["self-play".to_string()],
                seed: args.seed,
                saved_at: None,
                agent_player: None,
                first_player: Some(config.first_player),
            };

            match SavedMenaceAgent::from_agent(&shared_agent.lock().unwrap(), metadata) {
                Ok(saved) => match saved.save_to_file(&output_path) {
                    Ok(()) => {
                        println!("✓ Agent saved to: {}", output_path.display());
                        println!("  Learned from {} self-play games", result.total_games);
                    }
                    Err(e) => eprintln!("Error saving agent: {e:#}"),
                },
                Err(e) => eprintln!("Error creating saved agent: {e}"),
            }
        }

        return Ok(());
    }

    // Create agent for non-self-play modes
    let mut agent: Box<dyn Learner> = match args.learner {
        LearnerType::Menace => {
            let mut builder = MenaceAgent::builder();

            if let Some(seed) = args.seed {
                builder = builder.seed(seed);
            }

            builder = builder
                .agent_player(agent_player)
                .filter(state_filter)
                .reinforcement(reward_values)
                .initial_beads(initial_beads)
                .restock_mode(restock_mode);

            let menace = builder.build()?;
            Box::new(MenaceLearner::new(menace, "MENACE".to_string()))
        }
        LearnerType::ActiveInference => {
            use crate::active_inference::OpponentKind;

            // Parse opponent kind
            let opponent_kind = match args.ai_opponent.to_lowercase().as_str() {
                "uniform" => OpponentKind::Uniform,
                "adversarial" => OpponentKind::Adversarial,
                "minimax" => OpponentKind::Minimax,
                other => {
                    return Err(anyhow!(
                        "Unknown Active Inference opponent model: '{other}'. Use 'uniform', 'adversarial', or 'minimax'"
                    ));
                }
            };

            // Build base preference model
            let base_preferences = if let Some(ref prefs_str) = args.prefs {
                let parsed_prefs = parse_preferences(prefs_str)?;
                match parsed_prefs {
                    PreferencesArg::Probs(w, d, l) => PreferenceModel::from_probabilities(w, d, l),
                    PreferencesArg::Utils(w, d, l) => PreferenceModel::from_utilities(w, d, l),
                }
            } else {
                let (win, draw, loss) =
                    crate::active_inference::preferences::CANONICAL_PREFERENCE_PROBS;
                PreferenceModel::from_probabilities(win, draw, loss)
            };

            let policy_prior_arg = parse_policy_prior(&args.policy_prior)?;
            let policy_prior = match policy_prior_arg {
                PolicyPriorArg::Uniform => PolicyPrior::Uniform,
                PolicyPriorArg::Menace => PolicyPrior::MenacePositional,
                PolicyPriorArg::MenaceInitial => PolicyPrior::MenaceInitial(initial_beads),
            };

            let preferences = base_preferences
                .with_efe_mode(parse_efe_mode(&args.efe)?)
                .with_epistemic_weight(args.ai_beta)
                .with_ambiguity_weight(args.ai_ambiguity)
                .with_policy_lambda(args.policy_lambda)
                .with_policy_to_beads_scale(args.policy_beads_scale)
                .with_opponent_dirichlet_alpha(args.opponent_dirichlet_alpha)
                .with_policy_prior(policy_prior);

            // Build agent with custom Active Inference configuration
            let mut builder = MenaceAgent::builder();

            if let Some(seed) = args.seed {
                builder = builder.seed(seed);
            }

            builder = builder
                .agent_player(agent_player)
                .filter(state_filter)
                .initial_beads(initial_beads)
                .active_inference_custom(opponent_kind, preferences);

            let aif_agent = builder.build()?;
            Box::new(MenaceLearner::new(
                aif_agent,
                format!("ActiveInference({})", args.ai_opponent),
            ))
        }
        LearnerType::OracleActiveInference => {
            use crate::active_inference::OpponentKind;

            // Parse opponent kind
            let opponent_kind = match args.ai_opponent.to_lowercase().as_str() {
                "uniform" => OpponentKind::Uniform,
                "adversarial" => OpponentKind::Adversarial,
                "minimax" => OpponentKind::Minimax,
                other => {
                    return Err(anyhow!(
                        "Unknown Active Inference opponent model: '{other}'. Use 'uniform', 'adversarial', or 'minimax'"
                    ));
                }
            };

            // Create preference model
            let base_preferences = if let Some(ref prefs_str) = args.prefs {
                let parsed_prefs = parse_preferences(prefs_str)?;
                match parsed_prefs {
                    PreferencesArg::Probs(w, d, l) => PreferenceModel::from_probabilities(w, d, l),
                    PreferencesArg::Utils(w, d, l) => PreferenceModel::from_utilities(w, d, l),
                }
            } else {
                let (win, draw, loss) =
                    crate::active_inference::preferences::CANONICAL_PREFERENCE_PROBS;
                PreferenceModel::from_probabilities(win, draw, loss)
            };

            let policy_prior_arg = parse_policy_prior(&args.policy_prior)?;
            let policy_prior = match policy_prior_arg {
                PolicyPriorArg::Uniform => PolicyPrior::Uniform,
                PolicyPriorArg::Menace => PolicyPrior::MenacePositional,
                PolicyPriorArg::MenaceInitial => PolicyPrior::MenaceInitial(initial_beads),
            };

            let preferences = base_preferences
                .with_efe_mode(parse_efe_mode(&args.efe)?)
                .with_epistemic_weight(args.ai_beta)
                .with_ambiguity_weight(args.ai_ambiguity)
                .with_policy_lambda(args.policy_lambda)
                .with_policy_to_beads_scale(args.policy_beads_scale)
                .with_opponent_dirichlet_alpha(args.opponent_dirichlet_alpha)
                .with_policy_prior(policy_prior);

            // Build Oracle Active Inference agent
            let mut builder = MenaceAgent::builder();

            if let Some(seed) = args.seed {
                builder = builder.seed(seed);
            }

            builder = builder
                .agent_player(agent_player)
                .filter(state_filter)
                .initial_beads(initial_beads)
                .oracle_active_inference_custom(opponent_kind, preferences);

            let oracle_agent = builder.build()?;
            Box::new(MenaceLearner::new(
                oracle_agent,
                format!("OracleAIF({})", args.ai_opponent),
            ))
        }
        LearnerType::PureActiveInference => {
            use crate::active_inference::OpponentKind;

            // Parse opponent kind
            let opponent_kind = match args.ai_opponent.to_lowercase().as_str() {
                "uniform" => OpponentKind::Uniform,
                "adversarial" => OpponentKind::Adversarial,
                "minimax" => OpponentKind::Minimax,
                other => {
                    return Err(anyhow!(
                        "Unknown Active Inference opponent model: '{other}'. Use 'uniform', 'adversarial', or 'minimax'"
                    ));
                }
            };

            // Create preference model
            let base_preferences = if let Some(ref prefs_str) = args.prefs {
                let parsed_prefs = parse_preferences(prefs_str)?;
                match parsed_prefs {
                    PreferencesArg::Probs(w, d, l) => PreferenceModel::from_probabilities(w, d, l),
                    PreferencesArg::Utils(w, d, l) => PreferenceModel::from_utilities(w, d, l),
                }
            } else {
                let (win, draw, loss) =
                    crate::active_inference::preferences::CANONICAL_PREFERENCE_PROBS;
                PreferenceModel::from_probabilities(win, draw, loss)
            };

            let policy_prior_arg = parse_policy_prior(&args.policy_prior)?;
            let policy_prior = match policy_prior_arg {
                PolicyPriorArg::Uniform => PolicyPrior::Uniform,
                PolicyPriorArg::Menace => PolicyPrior::MenacePositional,
                PolicyPriorArg::MenaceInitial => PolicyPrior::MenaceInitial(initial_beads),
            };

            let preferences = base_preferences
                .with_efe_mode(parse_efe_mode(&args.efe)?)
                .with_epistemic_weight(args.ai_beta)
                .with_ambiguity_weight(args.ai_ambiguity)
                .with_policy_lambda(args.policy_lambda)
                .with_policy_to_beads_scale(args.policy_beads_scale)
                .with_opponent_dirichlet_alpha(args.opponent_dirichlet_alpha)
                .with_policy_prior(policy_prior);

            // Build Pure Active Inference agent
            let mut builder = MenaceAgent::builder();

            if let Some(seed) = args.seed {
                builder = builder.seed(seed);
            }

            builder = builder
                .agent_player(agent_player)
                .filter(state_filter)
                .initial_beads(initial_beads)
                .pure_active_inference_custom(opponent_kind, preferences);

            let pure_agent = builder.build()?;
            Box::new(MenaceLearner::new(
                pure_agent,
                format!("PureAIF({})", args.ai_opponent),
            ))
        }
        LearnerType::QLearning => {
            use crate::q_learning::QLearningAgent;

            let mut q_agent = QLearningAgent::new(
                args.ql_learning_rate,
                args.ql_discount,
                args.ql_epsilon,
                args.ql_epsilon_decay,
                args.ql_min_epsilon,
                args.ql_q_init,
            );

            if let Some(seed) = args.seed {
                q_agent = q_agent.with_seed(seed);
            }

            Box::new(q_agent)
        }
        LearnerType::Sarsa => {
            use crate::q_learning::SarsaAgent;

            let mut sarsa_agent = SarsaAgent::new(
                args.ql_learning_rate,
                args.ql_discount,
                args.ql_epsilon,
                args.ql_epsilon_decay,
                args.ql_min_epsilon,
                args.ql_q_init,
            );

            if let Some(seed) = args.seed {
                sarsa_agent = sarsa_agent.with_seed(seed);
            }

            Box::new(sarsa_agent)
        }
    };

    // Handle agent-vs-agent mode
    if let Some(ref opponent_path) = args.opponent_agent {
        println!("\n=== Agent vs Agent Training ===");
        println!("Training agent against loaded opponent");
        println!("Opponent: {}", opponent_path.display());
        println!("Games: {}", config.num_games);
        println!();

        // Only MENACE agents support agent-vs-agent currently
        if !matches!(args.learner, LearnerType::Menace) {
            return Err(anyhow!(
                "Agent-vs-agent training is currently only supported for MENACE agents"
            ));
        }

        // Load opponent agent
        let saved_opponent = SavedMenaceAgent::load_from_file(opponent_path)?;
        let opponent_agent = saved_opponent.to_agent()?;
        let mut opponent = MenaceLearner::new(opponent_agent, "Opponent-MENACE".to_string());

        println!("Opponent loaded:");
        println!("  Filter: {:?}", saved_opponent.state_filter);
        println!("  Algorithm: {:?}", saved_opponent.algorithm_type);

        // Create pipeline
        let mut pipeline = TrainingPipeline::new(config.clone());
        if args.progress {
            pipeline = pipeline.with_observer(Box::new(ProgressObserver::new()));
        }
        let metrics_observer = MetricsObserver::new();
        pipeline = pipeline.with_observer(Box::new(metrics_observer));

        // Run agent-vs-agent training
        let result = pipeline.run(agent.as_mut(), &mut opponent)?;

        println!("\n=== Agent vs Agent Complete ===");
        println!("Total games: {}", result.total_games);
        println!(
            "Agent wins: {} ({:.1}%)",
            result.wins,
            result.win_rate * 100.0
        );
        println!("Draws: {} ({:.1}%)", result.draws, result.draw_rate * 100.0);
        println!(
            "Agent losses: {} ({:.1}%)",
            result.losses,
            result.loss_rate * 100.0
        );

        // Display convergence metrics if requested
        if args.convergence_report
            && let Some(menace_learner) = agent.as_any().downcast_ref::<MenaceLearner>()
        {
            display_convergence_metrics(menace_learner.agent());
        }

        // Save both agents if output path provided
        if let Some(output_path) = args.output {
            println!("\nSaving trained agents...");

            // Save main agent
            if let Some(menace_learner) = agent.as_any().downcast_ref::<MenaceLearner>() {
                let metadata = crate::menace::TrainingMetadata {
                    games_trained: Some(result.total_games),
                    opponents: vec!["menace-agent".to_string()],
                    seed: args.seed,
                    saved_at: None,
                    agent_player: Some(config.agent_player),
                    first_player: Some(config.first_player),
                };

                match SavedMenaceAgent::from_agent(menace_learner.agent(), metadata) {
                    Ok(saved) => match saved.save_to_file(&output_path) {
                        Ok(()) => {
                            println!("✓ Main agent saved to: {}", output_path.display());
                        }
                        Err(e) => eprintln!("Error saving main agent: {e:#}"),
                    },
                    Err(e) => eprintln!("Error creating saved agent: {e}"),
                }
            }

            // Save opponent agent too
            let opponent_output = output_path
                .with_file_name(format!(
                    "{}-opponent",
                    output_path.file_stem().unwrap().to_str().unwrap()
                ))
                .with_extension("json");

            let opponent_metadata = crate::menace::TrainingMetadata {
                games_trained: Some(result.total_games),
                opponents: vec!["menace-agent".to_string()],
                seed: args.seed,
                saved_at: None,
                agent_player: Some(config.agent_player.opponent()),
                first_player: Some(config.first_player),
            };

            match SavedMenaceAgent::from_agent(opponent.agent(), opponent_metadata) {
                Ok(saved) => match saved.save_to_file(&opponent_output) {
                    Ok(()) => {
                        println!("✓ Opponent agent saved to: {}", opponent_output.display());
                    }
                    Err(e) => eprintln!("Error saving opponent agent: {e:#}"),
                },
                Err(e) => eprintln!("Error creating saved opponent: {e}"),
            }
        }

        return Ok(());
    }

    // Run training - either in phases or as single run
    let result = if let Some(ref phases_str) = args.phases {
        // Phase-based training
        let phase_counts = parse_phases(phases_str)?;
        println!("\n=== Phase-based Training ===");
        println!("Total phases: {}", phase_counts.len());
        println!("Total games: {}", phase_counts.iter().sum::<usize>());
        println!();

        // Track cumulative statistics
        let mut cumulative_wins = 0;
        let mut cumulative_draws = 0;
        let mut cumulative_losses = 0;
        let mut cumulative_games = 0;

        for (i, &phase_games) in phase_counts.iter().enumerate() {
            println!("Phase {} ({} games)", i + 1, phase_games);

            // Create phase-specific pipeline
            let mut phase_config = config.clone();
            phase_config.num_games = phase_games;

            let mut phase_pipeline = TrainingPipeline::new(phase_config);

            // Add progress bar for this phase if requested
            if args.progress {
                phase_pipeline = phase_pipeline.with_observer(Box::new(ProgressObserver::new()));
            }

            // Add metrics observer for this phase
            let phase_metrics = MetricsObserver::new();
            phase_pipeline = phase_pipeline.with_observer(Box::new(phase_metrics));

            // Generate schedule for this phase
            let phase_schedule = regimen.schedule(phase_games, &curriculum_config);

            // Run training for this phase
            let phase_result = phase_pipeline.run_curriculum(agent.as_mut(), &phase_schedule)?;

            // Update cumulative stats
            cumulative_wins += phase_result.wins;
            cumulative_draws += phase_result.draws;
            cumulative_losses += phase_result.losses;
            cumulative_games += phase_result.total_games;

            // Display phase report if requested
            if args.phase_report {
                println!(
                    "  Wins:   {} ({:.1}%)",
                    phase_result.wins,
                    phase_result.win_rate * 100.0
                );
                println!(
                    "  Draws:  {} ({:.1}%)",
                    phase_result.draws,
                    phase_result.draw_rate * 100.0
                );
                println!(
                    "  Losses: {} ({:.1}%)",
                    phase_result.losses,
                    phase_result.loss_rate * 100.0
                );

                let cumulative_win_rate = cumulative_wins as f64 / cumulative_games as f64 * 100.0;
                println!("  Cumulative win rate: {cumulative_win_rate:.1}%");

                // Show convergence metrics between phases if requested
                if args.convergence_report {
                    use crate::pipeline::MenaceLearner;
                    if let Some(menace_learner) = agent.as_any().downcast_ref::<MenaceLearner>() {
                        display_convergence_metrics(menace_learner.agent());
                    }
                }
                println!();
            }
        }

        // Create cumulative result
        use crate::pipeline::TrainingResult;
        TrainingResult {
            total_games: cumulative_games,
            wins: cumulative_wins,
            draws: cumulative_draws,
            losses: cumulative_losses,
            win_rate: cumulative_wins as f64 / cumulative_games as f64,
            draw_rate: cumulative_draws as f64 / cumulative_games as f64,
            loss_rate: cumulative_losses as f64 / cumulative_games as f64,
        }
    } else {
        // Normal single-run training
        let mut pipeline = TrainingPipeline::new(config.clone());

        // Add progress bar observer if requested
        if args.progress {
            pipeline = pipeline.with_observer(Box::new(ProgressObserver::new()));
        }

        // Add metrics observer
        let metrics_observer = MetricsObserver::new();
        pipeline = pipeline.with_observer(Box::new(metrics_observer));

        // Add JSONL observer if requested
        if let Some(observations_path) = &args.observations {
            let jsonl_observer = JsonlObserver::new(observations_path)?;
            pipeline = pipeline.with_observer(Box::new(jsonl_observer));
        }

        pipeline.run_curriculum(agent.as_mut(), &schedule)?
    };

    // Print results
    println!("\n=== Training Complete ===");
    println!("Total games: {}", result.total_games);
    println!("Wins: {} ({:.1}%)", result.wins, result.win_rate * 100.0);
    println!("Draws: {} ({:.1}%)", result.draws, result.draw_rate * 100.0);
    println!(
        "Losses: {} ({:.1}%)",
        result.losses,
        result.loss_rate * 100.0
    );

    // Display convergence metrics if requested
    if args.convergence_report {
        use crate::pipeline::MenaceLearner;
        if let Some(menace_learner) = agent.as_any().downcast_ref::<MenaceLearner>() {
            display_convergence_metrics(menace_learner.agent());
        }
    }

    // Export Active Inference CSV if requested
    if let Some(ref csv_path) = args.export_aif_exact_csv {
        // Only export for Active Inference agents
        if matches!(
            args.learner,
            LearnerType::ActiveInference
                | LearnerType::OracleActiveInference
                | LearnerType::PureActiveInference
        ) {
            use crate::{
                active_inference::{GenerativeModel, OpponentKind},
                export::{AifCsvExporter, AifExportConfig},
            };

            println!("\n=== Active Inference Export ===");
            println!("Collecting states to export...");

            // Parse selected states if provided
            let selected_states = args.export_states.as_ref().map(|s| {
                s.split(',')
                    .map(|t| t.trim().to_string())
                    .filter(|t| !t.is_empty())
                    .collect::<Vec<String>>()
            });

            // Collect states based on filter
            let states =
                AifCsvExporter::collect_states(state_filter, selected_states, args.aif_include_o)?;

            println!(
                "Exporting {} state(s) to {}...",
                states.len(),
                csv_path.display()
            );

            // Re-create preferences (same as used for training)
            let mut preferences = if let Some(ref prefs_str) = args.prefs {
                let parsed_prefs = parse_preferences(prefs_str)?;
                match parsed_prefs {
                    PreferencesArg::Probs(w, d, l) => {
                        crate::active_inference::PreferenceModel::from_probabilities(w, d, l)
                    }
                    PreferencesArg::Utils(w, d, l) => {
                        crate::active_inference::PreferenceModel::from_utilities(w, d, l)
                    }
                }
            } else {
                let (win, draw, loss) =
                    crate::active_inference::preferences::CANONICAL_PREFERENCE_PROBS;
                crate::active_inference::PreferenceModel::from_probabilities(win, draw, loss)
            };

            preferences.efe_mode = parse_efe_mode(&args.efe)?;
            preferences.epistemic_weight = args.ai_beta;
            preferences.ambiguity_weight = args.ai_ambiguity;
            preferences.policy_lambda = args.policy_lambda;
            preferences.opponent_dirichlet_alpha = args.opponent_dirichlet_alpha;

            let policy_prior_arg = parse_policy_prior(&args.policy_prior)?;
            preferences.policy_prior = match policy_prior_arg {
                PolicyPriorArg::Uniform => crate::active_inference::PolicyPrior::Uniform,
                PolicyPriorArg::Menace => crate::active_inference::PolicyPrior::MenacePositional,
                PolicyPriorArg::MenaceInitial => {
                    crate::active_inference::PolicyPrior::MenaceInitial(initial_beads)
                }
            };

            // Create opponent
            let opponent_kind = match args.ai_opponent.to_lowercase().as_str() {
                "uniform" => OpponentKind::Uniform,
                "adversarial" => OpponentKind::Adversarial,
                "minimax" => OpponentKind::Minimax,
                _ => OpponentKind::Uniform, // Default
            };
            let opponent = opponent_kind.into_boxed_opponent();

            // Create beliefs
            let beliefs = crate::beliefs::Beliefs::symmetric(preferences.opponent_dirichlet_alpha);

            // Create generative model
            let model = GenerativeModel::new();

            // Create export config
            let config = AifExportConfig {
                state_filter,
                selected_states: None, // Already collected
                include_o_states: args.aif_include_o,
                generative_model: &model,
                preferences: &preferences,
                opponent: opponent.as_ref(),
                beliefs: &beliefs,
            };

            // Export
            let exported_count = AifCsvExporter::export(&config, csv_path, states)?;

            println!(
                "✓ Exported {} state(s) to {}",
                exported_count,
                csv_path.display()
            );
        } else {
            println!("\n⚠️  AIF CSV export is only available for Active Inference learners.");
            println!("    Use: menace train active-inference --export-aif-exact-csv <path>");
        }
    }

    // Post-training validation if requested
    if args.validation_games > 0 {
        use std::sync::{Arc, Mutex};

        use crate::pipeline::{FrozenLearner, OptimalLearner};

        println!("\n=== Post-Training Validation ===");
        println!(
            "Playing {} games vs optimal opponent...",
            args.validation_games
        );

        // Use validation seed (defaults to training seed + 1)
        let validation_seed = args
            .validation_seed
            .or_else(|| args.seed.map(|s| s.wrapping_add(1)));

        let validation_config = TrainingConfig {
            num_games: args.validation_games,
            seed: validation_seed,
            agent_player: config.agent_player,
            first_player: config.first_player,
        };

        let mut validation_pipeline = TrainingPipeline::new(validation_config);
        if args.progress {
            validation_pipeline =
                validation_pipeline.with_observer(Box::new(ProgressObserver::new()));
        }

        // Add milestone observer if requested (wrapped in Arc<Mutex<>> to retrieve data later)
        let milestone_observer = if args.track_milestones {
            let observer = Arc::new(Mutex::new(MilestoneObserver::new("Optimal".to_string())));
            let observer_clone = observer.clone();

            // Create a wrapper that delegates to the Arc<Mutex<>> observer
            struct MilestoneWrapper {
                inner: Arc<Mutex<MilestoneObserver>>,
            }

            impl crate::pipeline::Observer for MilestoneWrapper {
                fn on_game_start(&mut self, game_num: usize) -> crate::Result<()> {
                    self.inner.lock().unwrap().on_game_start(game_num)
                }

                fn on_game_end(
                    &mut self,
                    game_num: usize,
                    outcome: crate::tictactoe::GameOutcome,
                ) -> crate::Result<()> {
                    self.inner.lock().unwrap().on_game_end(game_num, outcome)
                }
            }

            validation_pipeline =
                validation_pipeline.with_observer(Box::new(MilestoneWrapper { inner: observer }));
            Some(observer_clone)
        } else {
            None
        };

        let mut optimal_opponent = OptimalLearner::new("Optimal".to_string());

        // Wrap agent in FrozenLearner to prevent learning during validation
        // This ensures we test the policy as it exists after training,
        // without belief/workspace updates during validation games
        let mut frozen_agent = FrozenLearner::new(agent.as_mut());
        let validation_result =
            validation_pipeline.run(&mut frozen_agent, &mut optimal_opponent)?;
        validation_result_opt = Some(validation_result.clone());

        println!("\n=== Validation Results ===");
        println!("Total games: {}", validation_result.total_games);
        println!(
            "Wins: {} ({:.1}%)",
            validation_result.wins,
            validation_result.win_rate * 100.0
        );
        println!(
            "Draws: {} ({:.1}%)",
            validation_result.draws,
            validation_result.draw_rate * 100.0
        );
        println!(
            "Losses: {} ({:.1}%)",
            validation_result.losses,
            validation_result.loss_rate * 100.0
        );

        // Display milestone summary if tracking was enabled
        let milestone_obs_ref = if let Some(ref observer) = milestone_observer {
            let obs = observer.lock().unwrap();
            obs.display_summary();
            Some(observer)
        } else {
            None
        };

        // Display historical comparison if requested
        if args.historical_comparison {
            let milestone_ref = milestone_obs_ref.map(|arc| arc.lock().unwrap());
            display_historical_comparison(
                state_filter,
                result.total_games,
                milestone_ref.as_deref(),
            );
        }
    }

    // Save agent if output path provided
    if let Some(output_path) = args.output {
        println!("\nSaving trained agent...");

        // Extract the trained agent (downcast from Learner trait)
        use crate::{
            menace::{SavedMenaceAgent, TrainingMetadata},
            pipeline::MenaceLearner,
            q_learning::{QLearningAgent, SarsaAgent, SavedTdAgent},
        };

        // Build opponent description from training schedule
        let opponent_desc = if schedule.len() == 1 {
            schedule[0].opponent.label().to_string()
        } else {
            format!("curriculum-{}", regimen.label())
        };

        let metadata = TrainingMetadata {
            games_trained: Some(result.total_games),
            opponents: vec![opponent_desc],
            seed: args.seed,
            saved_at: None,
            agent_player: Some(config.agent_player),
            first_player: Some(config.first_player),
        };

        if let Some(menace_learner) = agent.as_any().downcast_ref::<MenaceLearner>() {
            match SavedMenaceAgent::from_agent(menace_learner.agent(), metadata) {
                Ok(saved) => match saved.save_to_file(&output_path) {
                    Ok(()) => {
                        println!("✓ Agent saved to: {}", output_path.display());
                        println!("  Filter: {:?}", saved.state_filter);
                        println!("  Algorithm: {:?}", saved.algorithm_type);
                        println!("  Games trained: {}", result.total_games);

                        // Display algorithm-specific details
                        match &saved.algorithm_params {
                            crate::menace::AlgorithmParams::ClassicMenace { reinforcement } => {
                                println!(
                                    "  Reinforcement: win={}, draw={}, loss={}",
                                    reinforcement.win, reinforcement.draw, reinforcement.loss
                                );
                            }
                            crate::menace::AlgorithmParams::ActiveInference {
                                opponent_kind,
                                preferences,
                                beliefs,
                                ..
                            } => {
                                println!("  Opponent model: {opponent_kind:?}");
                                println!("  Epistemic weight: {:.2}", preferences.epistemic_weight);
                                println!("  EFE mode: {:?}", preferences.efe_mode);
                                println!("  Stored opponent states: {}", beliefs.tracked_states());
                            }
                            crate::menace::AlgorithmParams::PureActiveInference {
                                opponent_kind,
                                preferences,
                                action_beliefs,
                                ..
                            } => {
                                println!("  Opponent model: {opponent_kind:?}");
                                println!("  Epistemic weight: {:.2}", preferences.epistemic_weight);
                                println!("  EFE mode: {:?}", preferences.efe_mode);
                                println!(
                                    "  Stored action-outcome pairs: {}",
                                    action_beliefs.tracked_pairs()
                                );
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Error saving agent: {e:#}");
                    }
                },
                Err(e) => {
                    eprintln!("Error creating saved agent: {e}");
                }
            }
        } else if let Some(q_agent) = agent.as_any().downcast_ref::<QLearningAgent>() {
            let saved = SavedTdAgent::from_q_learning(q_agent, metadata);
            match saved.save_to_file(&output_path) {
                Ok(()) => {
                    println!("✓ Agent saved to: {}", output_path.display());
                    println!("  Algorithm: Q-learning");
                    println!("  Stored Q-values: {}", q_agent.q_table_size());
                }
                Err(e) => eprintln!("Error saving agent: {e:#}"),
            }
        } else if let Some(sarsa_agent) = agent.as_any().downcast_ref::<SarsaAgent>() {
            let saved = SavedTdAgent::from_sarsa(sarsa_agent, metadata);
            match saved.save_to_file(&output_path) {
                Ok(()) => {
                    println!("✓ Agent saved to: {}", output_path.display());
                    println!("  Algorithm: SARSA");
                    println!("  Stored Q-values: {}", sarsa_agent.q_table_size());
                }
                Err(e) => eprintln!("Error saving agent: {e:#}"),
            }
        } else {
            eprintln!("Warning: Could not extract agent for serialization.");
            eprintln!("         This is unexpected - please report this issue.");
        }
    }

    if let Some((summary_path, normalized)) = summary_spec {
        if normalized {
            println!(
                "\n⚠️  Normalizing summary path to {}",
                summary_path.display()
            );
        }

        if let Some(parent) = summary_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let schedule_summary = schedule
            .iter()
            .map(|block| ScheduleBlock {
                opponent: block.opponent.label().to_string(),
                games: block.games,
            })
            .collect();

        let summary = TrainingSummaryFile {
            training: SummaryStats::from(&result),
            validation: validation_result_opt.as_ref().map(SummaryStats::from),
            regimen: args.regimen.clone(),
            schedule: schedule_summary,
            metadata: SummaryMetadata {
                filter: format!("{state_filter:?}"),
                restock_mode: format!("{restock_mode:?}"),
                agent_player: format!("{agent_player:?}"),
                first_player: format!("{first_player:?}"),
                seed: args.seed,
            },
        };

        let file = File::create(&summary_path)?;
        to_writer_pretty(file, &summary)?;
        println!("\nSummary written to {}", summary_path.display());
    }

    Ok(())
}
