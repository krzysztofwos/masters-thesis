//! Active Inference learning algorithms
//!
//! Provides three types of Active Inference agents:
//! 1. **OracleActiveInference** - Uses perfect game tree knowledge (no learning needed)
//! 2. **ActiveInference** - EFE-based policy with learned opponent beliefs (Bayesian belief updates)
//! 3. **PureActiveInference** - Purely Bayesian (learns beliefs about action outcomes, computes EFE from beliefs)
//!
//! ## Oracle Active Inference
//!
//! The Oracle agent has perfect knowledge of action-outcomes (via game tree)
//! but requires an **opponent model** to compute Expected Free Energy:
//!
//! - **OpponentKind::Uniform**: Assumes opponent plays randomly → FAILS vs optimal opponent
//! - **OpponentKind::Minimax**: Uses optimal opponent policy → Should achieve optimal performance
//! - **OpponentKind::Adversarial**: Assumes worst-case opponent → Pessimistic but robust
//!
//! **Key Insight**: Perfect action-outcome knowledge is insufficient without correct opponent modeling.
//! This explains why Oracle with Uniform opponent performs worse than adaptive learning agents.
//!
//! ## Hybrid Active Inference (ActiveInference)
//!
//! The ActiveInference agent combines:
//! - EFE-based policy selection (Expected Free Energy minimization)
//! - Bayesian belief updates about opponent behavior (learned from observations)
//! - Perfect game tree knowledge for state transitions
//!
//! ## Pure Active Inference
//!
//! The Pure Active Inference agent uses:
//! - Bayesian belief updates about action-outcome distributions
//! - EFE-based policy computed from learned beliefs
//! - Should converge toward Oracle performance as beliefs improve

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use super::learning::LearningAlgorithm;
use crate::{
    active_inference::{GenerativeModel, Opponent, OpponentKind, PreferenceModel},
    beliefs::Beliefs,
    efe::{dirichlet_categorical_mi, optimal_policy_with_kl},
    tictactoe::{BoardState, GameOutcome, Player},
    utils::normalize_weights,
    workspace::MenaceWorkspace,
};

/// Oracle Active Inference agent (uses perfect game tree knowledge)
///
/// This agent computes Active Inference policy using complete knowledge of the game tree.
/// It has perfect action-outcome knowledge but **requires specification of opponent model**.
///
/// ## Critical Parameter: opponent_kind
///
/// The `opponent_kind` parameter determines how the Oracle models opponent behavior:
///
/// - **OpponentKind::Uniform**: Assumes random opponent (1/N probability for all moves)
///   - Results in **poor performance** vs optimal opponents (17% draws)
///   - Demonstrates: perfect knowledge + wrong model = failure
///
/// - **OpponentKind::Minimax**: Uses optimal opponent policy from game tree
///   - Should achieve **near-optimal performance** (~95% draws predicted)
///   - Demonstrates: perfect knowledge + correct model = success
///
/// - **OpponentKind::Adversarial**: Assumes worst-case opponent behavior
///   - Provides **pessimistic but robust** performance
///   - Useful for risk-averse strategies
///
/// ## Key Insight
///
/// Perfect action-outcome knowledge is **insufficient** without correct opponent modeling.
/// This explains the "Oracle Paradox": why Oracle (Uniform) performs worse than adaptive
/// learning agents that have incomplete knowledge but learn the correct opponent model.
///
/// See `docs/ORACLE_PARADOX_EXPLAINED.md` for detailed analysis.
pub struct OracleActiveInference {
    /// Generative model for game tree evaluation
    generative_model: GenerativeModel,
    /// Preference model encoding utilities
    preferences: PreferenceModel,
    /// Opponent model for policy computation
    opponent: Box<dyn Opponent>,
    /// Weight for epistemic (information-seeking) value
    epistemic_weight: f64,
    /// Which player this agent controls (X or O)
    agent_player: Player,
    /// Number of games played (for statistics)
    games_played: usize,
}

impl std::fmt::Debug for OracleActiveInference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OracleActiveInference")
            .field("opponent_kind", &self.opponent.kind())
            .field("epistemic_weight", &self.epistemic_weight)
            .field("agent_player", &self.agent_player)
            .field("games_played", &self.games_played)
            .finish()
    }
}

impl OracleActiveInference {
    /// Create a new Oracle agent with the specified opponent model and epistemic weight
    pub fn new(opponent_kind: OpponentKind, epistemic_weight: f64) -> Self {
        Self::new_for_player(opponent_kind, epistemic_weight, Player::X)
    }

    /// Create a new Oracle agent controlling the provided player token
    pub fn new_for_player(
        opponent_kind: OpponentKind,
        epistemic_weight: f64,
        agent_player: Player,
    ) -> Self {
        let generative_model = GenerativeModel::new();
        let (win_pref, draw_pref, loss_pref) =
            crate::active_inference::preferences::CANONICAL_PREFERENCE_PROBS;
        let mut preferences = PreferenceModel::from_probabilities(win_pref, draw_pref, loss_pref);
        preferences.epistemic_weight = epistemic_weight;
        let opponent = opponent_kind.into_boxed_opponent();

        Self {
            generative_model,
            preferences,
            opponent,
            epistemic_weight,
            agent_player,
            games_played: 0,
        }
    }

    /// Create with custom preferences
    pub fn with_custom_preferences(
        opponent_kind: OpponentKind,
        preferences: PreferenceModel,
        agent_player: Player,
    ) -> Self {
        let generative_model = GenerativeModel::new();
        let epistemic_weight = preferences.epistemic_weight;
        let opponent = opponent_kind.into_boxed_opponent();

        Self {
            generative_model,
            preferences,
            opponent,
            epistemic_weight,
            agent_player,
            games_played: 0,
        }
    }

    /// Get the opponent kind
    pub fn opponent_kind(&self) -> OpponentKind {
        self.opponent.kind()
    }

    /// Get epistemic weight
    pub fn epistemic_weight(&self) -> f64 {
        self.epistemic_weight
    }

    /// Get the player controlled by this agent
    pub fn agent_player(&self) -> Player {
        self.agent_player
    }

    /// Access the current preference model
    pub fn preference_model(&self) -> &PreferenceModel {
        &self.preferences
    }

    /// Update which player the agent controls.
    pub fn set_agent_player(&mut self, player: Player) {
        if self.agent_player == player {
            return;
        }
        self.agent_player = player;
        self.games_played = 0;
    }

    /// Update workspace weights for a given state based on Expected Free Energy evaluation.
    ///
    /// The Oracle uses perfect opponent model based on `opponent_kind`:
    /// - **Uniform**: Uses symmetric beliefs (assumes random opponent)
    /// - **Minimax**: Uses optimal policy from game tree (beliefs ignored)
    /// - **Adversarial**: Uses worst-case analysis (beliefs ignored)
    ///
    /// Note: For Minimax/Adversarial opponents, the `beliefs` parameter is not used by
    /// the generative model - it derives opponent behavior from game tree analysis.
    fn update_workspace_from_state(&self, workspace: &mut MenaceWorkspace, state: &BoardState) {
        let (_, orig_label) = state.canonical_context_and_label();
        workspace.ensure_box_weighted(&orig_label);

        // Beliefs are only used for OpponentKind::Uniform
        // For Minimax and Adversarial, the generative model uses game tree analysis instead
        let beliefs = Beliefs::symmetric(1.0);

        let evaluations = self.generative_model.evaluate_actions(
            orig_label.as_str(),
            &self.preferences,
            self.opponent.as_ref(), // This determines opponent behavior!
            &beliefs,
            self.agent_player,
        );

        if evaluations.is_empty() {
            return;
        }

        let efe_values: Vec<f64> = evaluations.iter().map(|e| e.free_energy).collect();
        let mut prior_weights: Vec<f64> = evaluations
            .iter()
            .map(|e| e.policy_prior.max(0.0))
            .collect();
        if prior_weights.iter().all(|w| *w <= 0.0) {
            prior_weights = vec![1.0; prior_weights.len()];
        }

        let policy =
            optimal_policy_with_kl(&efe_values, &prior_weights, self.preferences.policy_lambda);

        let bead_counts: HashMap<usize, f64> = evaluations
            .iter()
            .zip(policy.iter())
            .map(|(evaluation, &prob)| {
                (
                    evaluation.action,
                    prob.max(0.0) * self.preferences.policy_to_beads_scale,
                )
            })
            .collect();

        workspace.set_move_weights(&orig_label, &bead_counts);
    }
}

impl LearningAlgorithm for OracleActiveInference {
    fn train_from_game(
        &mut self,
        workspace: &mut MenaceWorkspace,
        states: &[BoardState],
        _moves: &[usize],
        _outcome: GameOutcome,
        player: Player,
    ) -> crate::Result<()> {
        if player != self.agent_player {
            self.games_played += 1;
            return Ok(());
        }

        // Oracle computes optimal policy from scratch - no learning from outcomes needed
        // We update workspace to reflect the theoretically optimal policy at each state
        for state in states.iter() {
            if state.to_move == self.agent_player && !state.is_terminal() {
                self.update_workspace_from_state(workspace, state);
            }
        }

        self.games_played += 1;
        Ok(())
    }

    fn name(&self) -> &str {
        match self.opponent.kind() {
            OpponentKind::Uniform => "Oracle Active Inference (Uniform)",
            OpponentKind::Adversarial => "Oracle Active Inference (Adversarial)",
            OpponentKind::Minimax => "Oracle Active Inference (Minimax)",
        }
    }

    fn stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        stats.insert("games_played".to_string(), self.games_played as f64);
        stats.insert("epistemic_weight".to_string(), self.epistemic_weight);
        stats.insert(
            "opponent_type".to_string(),
            self.opponent.kind() as u8 as f64,
        );
        stats.insert("is_oracle".to_string(), 1.0);
        stats
    }

    fn reset(&mut self) {
        self.games_played = 0;
        // No belief state to reset for Oracle
    }

    // Oracle doesn't use reinforcement values
}

/// Hybrid Active Inference learning algorithm
///
/// Combines EFE-based policy with learned opponent beliefs:
///
/// 1. Maintains beliefs about opponent behavior using Dirichlet distributions
/// 2. Uses perfect game tree knowledge (like Oracle) for state transitions
/// 3. Computes Expected Free Energy (EFE) using learned opponent beliefs
/// 4. Updates policy weights based on EFE minimization
/// 5. Learns by observing opponent moves and updating beliefs
///
/// Unlike Oracle (which assumes perfect opponent model), this agent
/// learns the opponent model from observations. Unlike Pure AIF (which
/// learns action-outcome beliefs), this agent has perfect transition knowledge.
pub struct ActiveInference {
    /// Generative model for game tree evaluation (perfect transitions)
    generative_model: GenerativeModel,
    /// Preference model encoding utilities
    preferences: PreferenceModel,
    /// Beliefs about opponent behavior (learned from observations)
    beliefs: Beliefs,
    /// Cached opponent model (to avoid repeated allocations)
    opponent: Box<dyn Opponent>,
    /// Weight for epistemic (information-seeking) value
    epistemic_weight: f64,
    /// Number of games played
    games_played: usize,
    /// Which player this agent controls (X or O)
    agent_player: Player,
}

impl std::fmt::Debug for ActiveInference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActiveInference")
            .field("opponent_kind", &self.opponent.kind())
            .field("epistemic_weight", &self.epistemic_weight)
            .field("agent_player", &self.agent_player)
            .field("games_played", &self.games_played)
            .finish()
    }
}

impl ActiveInference {
    /// Default preference probabilities for win, draw, loss
    const DEFAULT_PREFERENCES: (f64, f64, f64) =
        crate::active_inference::preferences::CANONICAL_PREFERENCE_PROBS;

    /// Create preferences with the given epistemic weight
    fn create_preferences(epistemic_weight: f64) -> PreferenceModel {
        let mut prefs = PreferenceModel::from_probabilities(
            Self::DEFAULT_PREFERENCES.0,
            Self::DEFAULT_PREFERENCES.1,
            Self::DEFAULT_PREFERENCES.2,
        );
        prefs.epistemic_weight = epistemic_weight;
        prefs
    }

    /// Create a new ActiveInference agent with the specified opponent model and epistemic weight
    pub fn new(opponent_kind: OpponentKind, epistemic_weight: f64) -> Self {
        Self::new_for_player(opponent_kind, epistemic_weight, Player::X)
    }

    /// Create a new ActiveInference agent controlling the provided player token.
    pub fn new_for_player(
        opponent_kind: OpponentKind,
        epistemic_weight: f64,
        agent_player: Player,
    ) -> Self {
        let generative_model = GenerativeModel::new();
        let preferences = Self::create_preferences(epistemic_weight);
        let beliefs = Beliefs::symmetric(1.0);
        let opponent = opponent_kind.into_boxed_opponent();

        Self {
            generative_model,
            preferences,
            beliefs,
            opponent,
            epistemic_weight,
            games_played: 0,
            agent_player,
        }
    }

    /// Create a new ActiveInference agent with fully customizable preferences
    pub fn with_custom_preferences(
        opponent_kind: OpponentKind,
        preferences: PreferenceModel,
    ) -> Self {
        Self::with_custom_preferences_for_player(opponent_kind, preferences, Player::X)
    }

    /// Create an Active Inference agent with custom preferences controlling the provided player.
    pub fn with_custom_preferences_for_player(
        opponent_kind: OpponentKind,
        preferences: PreferenceModel,
        agent_player: Player,
    ) -> Self {
        let beliefs = Beliefs::symmetric(preferences.opponent_dirichlet_alpha);
        Self::with_custom_preferences_and_beliefs_for_player(
            opponent_kind,
            preferences,
            beliefs,
            agent_player,
        )
    }

    /// Create an ActiveInference agent with explicit beliefs (e.g., from serialization)
    pub fn with_custom_preferences_and_beliefs(
        opponent_kind: OpponentKind,
        preferences: PreferenceModel,
        beliefs: Beliefs,
    ) -> Self {
        Self::with_custom_preferences_and_beliefs_for_player(
            opponent_kind,
            preferences,
            beliefs,
            Player::X,
        )
    }

    /// Create an ActiveInference agent with explicit beliefs for the provided player.
    pub fn with_custom_preferences_and_beliefs_for_player(
        opponent_kind: OpponentKind,
        preferences: PreferenceModel,
        beliefs: Beliefs,
        agent_player: Player,
    ) -> Self {
        let generative_model = GenerativeModel::new();
        let epistemic_weight = preferences.epistemic_weight;
        let opponent = opponent_kind.into_boxed_opponent();

        Self {
            generative_model,
            preferences,
            beliefs,
            opponent,
            epistemic_weight,
            games_played: 0,
            agent_player,
        }
    }

    /// Create with uniform opponent model (convenience wrapper)
    pub fn with_uniform_opponent(epistemic_weight: f64) -> Self {
        Self::new_for_player(OpponentKind::Uniform, epistemic_weight, Player::X)
    }

    /// Create with uniform opponent model for a specific player token.
    pub fn with_uniform_opponent_for_player(epistemic_weight: f64, agent_player: Player) -> Self {
        Self::new_for_player(OpponentKind::Uniform, epistemic_weight, agent_player)
    }

    /// Create with adversarial opponent model (convenience wrapper)
    pub fn with_adversarial_opponent(epistemic_weight: f64) -> Self {
        Self::new_for_player(OpponentKind::Adversarial, epistemic_weight, Player::X)
    }

    /// Create with adversarial opponent model for a specific player token.
    pub fn with_adversarial_opponent_for_player(
        epistemic_weight: f64,
        agent_player: Player,
    ) -> Self {
        Self::new_for_player(OpponentKind::Adversarial, epistemic_weight, agent_player)
    }

    /// Create with minimax opponent model (convenience wrapper)
    pub fn with_minimax_opponent() -> Self {
        Self::new_for_player(OpponentKind::Minimax, 0.0, Player::X)
    }

    /// Create with minimax opponent model for a specific player token.
    pub fn with_minimax_opponent_for_player(agent_player: Player) -> Self {
        Self::new_for_player(OpponentKind::Minimax, 0.0, agent_player)
    }

    /// Get the opponent kind
    pub fn opponent_kind(&self) -> OpponentKind {
        self.opponent.kind()
    }

    /// Get epistemic weight
    pub fn epistemic_weight(&self) -> f64 {
        self.epistemic_weight
    }

    /// Get the player controlled by this agent.
    pub fn agent_player(&self) -> Player {
        self.agent_player
    }

    /// Update which player the agent controls.
    pub fn set_agent_player(&mut self, player: Player) {
        if self.agent_player == player {
            return;
        }
        self.agent_player = player;
        // Reset belief state so opponent modelling stays aligned with the new perspective.
        self.beliefs = Beliefs::symmetric(self.preferences.opponent_dirichlet_alpha);
        self.games_played = 0;
    }

    /// Access the current preference model
    pub fn preference_model(&self) -> &PreferenceModel {
        &self.preferences
    }

    /// Access the current opponent-belief state
    pub fn beliefs(&self) -> &Beliefs {
        &self.beliefs
    }

    /// Update beliefs based on an observed opponent move
    fn update_beliefs_from_opponent_move(&mut self, state: &BoardState, opponent_move: usize) {
        let (ctx, label) = state.canonical_context_and_label();
        let legal_moves = ctx.state.legal_moves();
        if let Some(idx) = legal_moves
            .iter()
            .position(|&m| m == ctx.map_move_to_canonical(opponent_move))
        {
            self.beliefs
                .observe_opponent_action(label.as_str(), legal_moves.len(), idx);
        }
    }

    /// Update workspace weights for a given state based on Expected Free Energy evaluation.
    ///
    /// Unlike Oracle (which uses assumed opponent model), this uses learned beliefs
    /// about opponent behavior to compute EFE.
    fn update_workspace_from_state(&self, workspace: &mut MenaceWorkspace, state: &BoardState) {
        let (_, orig_label) = state.canonical_context_and_label();
        workspace.ensure_box_weighted(&orig_label);

        // Use learned beliefs about opponent (key difference from Oracle!)
        let evaluations = self.generative_model.evaluate_actions(
            orig_label.as_str(),
            &self.preferences,
            self.opponent.as_ref(),
            &self.beliefs, // Learned beliefs, not symmetric
            self.agent_player,
        );

        if evaluations.is_empty() {
            return;
        }

        let efe_values: Vec<f64> = evaluations.iter().map(|e| e.free_energy).collect();

        let mut prior_weights: Vec<f64> = evaluations
            .iter()
            .map(|e| e.policy_prior.max(0.0))
            .collect();
        if prior_weights.iter().all(|w| *w <= 0.0) {
            prior_weights = vec![1.0; prior_weights.len()];
        }

        let policy =
            optimal_policy_with_kl(&efe_values, &prior_weights, self.preferences.policy_lambda);

        let bead_counts: HashMap<usize, f64> = evaluations
            .iter()
            .zip(policy.iter())
            .map(|(evaluation, &prob)| {
                (
                    evaluation.action,
                    prob.max(0.0) * self.preferences.policy_to_beads_scale,
                )
            })
            .collect();

        workspace.set_move_weights(&orig_label, &bead_counts);
    }
}

impl Default for ActiveInference {
    fn default() -> Self {
        Self::with_uniform_opponent(1.0)
    }
}

impl LearningAlgorithm for ActiveInference {
    fn train_from_game(
        &mut self,
        workspace: &mut MenaceWorkspace,
        states: &[BoardState],
        moves: &[usize],
        _outcome: GameOutcome,
        player: Player,
    ) -> crate::Result<()> {
        if player != self.agent_player {
            self.games_played += 1;
            return Ok(());
        }

        // Update beliefs about opponent behavior
        for (state, &mv) in states.iter().zip(moves.iter()) {
            if state.to_move != self.agent_player {
                self.update_beliefs_from_opponent_move(state, mv);
            }
        }

        // Update workspace using EFE-based policy with learned beliefs
        // This is the key change - we use EFE instead of RL!
        for state in states.iter() {
            if state.to_move == self.agent_player && !state.is_terminal() {
                self.update_workspace_from_state(workspace, state);
            }
        }

        self.games_played += 1;
        Ok(())
    }

    fn name(&self) -> &str {
        match self.opponent.kind() {
            OpponentKind::Uniform => "Active Inference (Uniform)",
            OpponentKind::Adversarial => "Active Inference (Adversarial)",
            OpponentKind::Minimax => "Active Inference (Minimax)",
        }
    }

    fn stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        stats.insert("games_played".to_string(), self.games_played as f64);
        stats.insert("epistemic_weight".to_string(), self.epistemic_weight);
        stats.insert(
            "opponent_type".to_string(),
            self.opponent.kind() as u8 as f64,
        );
        stats.insert("beliefs_version".to_string(), self.beliefs.version() as f64);
        stats
    }

    fn reset(&mut self) {
        self.games_played = 0;
        self.beliefs = Beliefs::symmetric(self.preferences.opponent_dirichlet_alpha);
    }

    // ActiveInference doesn't use reinforcement values, so it uses the default
    // implementation that returns None
}

/// Tracks beliefs about what outcomes result from taking actions in states.
///
/// For each (state, action) pair, maintains a Dirichlet distribution over
/// three possible outcomes: Win, Draw, Loss.
///
/// This enables pure Active Inference where the agent learns beliefs about
/// the consequences of its actions and uses those beliefs to compute EFE.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionOutcomeBeliefs {
    /// Maps "state_label_action" to [α_win, α_draw, α_loss]
    action_outcomes: BTreeMap<String, Vec<f64>>,
    /// Default Dirichlet parameter (symmetric prior)
    default_alpha: f64,
    /// Version counter for tracking updates
    version: usize,
}

impl ActionOutcomeBeliefs {
    /// Create new action-outcome beliefs with symmetric Dirichlet prior.
    ///
    /// # Arguments
    /// * `default_alpha` - Symmetric Dirichlet parameter for [Win, Draw, Loss]
    pub fn new(default_alpha: f64) -> Self {
        assert!(
            default_alpha.is_finite() && default_alpha > 0.0,
            "Dirichlet α must be positive"
        );
        Self {
            action_outcomes: BTreeMap::new(),
            default_alpha,
            version: 0,
        }
    }

    /// Get the belief key for a (state, action) pair
    fn belief_key(state_label: &str, action: usize) -> String {
        format!("{state_label}_a{action}")
    }

    /// Get Dirichlet parameters [α_win, α_draw, α_loss] for a (state, action) pair
    pub fn alpha_for(&self, state_label: &str, action: usize) -> Vec<f64> {
        let key = Self::belief_key(state_label, action);
        self.action_outcomes
            .get(&key)
            .cloned()
            .unwrap_or_else(|| vec![self.default_alpha; 3])
    }

    /// Get predictive distribution over outcomes [P(Win), P(Draw), P(Loss)]
    pub fn predictive(&self, state_label: &str, action: usize) -> Vec<f64> {
        let alpha = self.alpha_for(state_label, action);
        normalize_weights(alpha).unwrap_or_else(|| vec![1.0 / 3.0; 3])
    }

    /// Observe an outcome after taking an action in a state.
    ///
    /// Updates the Dirichlet distribution for this (state, action) pair.
    pub fn observe_outcome(
        &mut self,
        state_label: &str,
        action: usize,
        outcome: GameOutcome,
        agent_player: Player,
    ) {
        let key = Self::belief_key(state_label, action);
        let entry = self
            .action_outcomes
            .entry(key)
            .or_insert_with(|| vec![self.default_alpha; 3]);

        // Ensure we have exactly 3 parameters
        if entry.len() != 3 {
            *entry = vec![self.default_alpha; 3];
        }

        // Update the appropriate parameter based on outcome
        match outcome {
            GameOutcome::Win(winner) if winner == agent_player => {
                entry[0] += 1.0; // Win
            }
            GameOutcome::Win(_) => {
                entry[2] += 1.0; // Loss
            }
            GameOutcome::Draw => {
                entry[1] += 1.0; // Draw
            }
        }

        self.version = self.version.wrapping_add(1);
    }

    /// Get version counter (changes whenever beliefs are updated)
    pub fn version(&self) -> usize {
        self.version
    }

    /// Number of (state, action) pairs with explicit belief counts
    pub fn tracked_pairs(&self) -> usize {
        self.action_outcomes.len()
    }
}

/// Pure Active Inference learning algorithm
///
/// This agent implements theoretically pure Active Inference by:
/// 1. Maintaining Dirichlet beliefs about action-outcome distributions
/// 2. Updating beliefs when outcomes are observed (Bayesian learning)
/// 3. Computing EFE from learned beliefs (not from perfect game tree)
/// 4. Selecting actions that minimize expected free energy
///
/// Unlike the hybrid ActiveInference agent (which has perfect game tree knowledge),
/// this agent learns purely through belief updates. It should gradually converge
/// toward Oracle performance as its beliefs about action effectiveness improve.
pub struct PureActiveInference {
    /// Preference model encoding utilities for the current player perspective
    preferences: PreferenceModel,
    /// Canonical (X-perspective) preference model used to re-orient when swapping players
    base_preferences: PreferenceModel,
    /// Beliefs about opponent behavior
    opponent_beliefs: Beliefs,
    /// Beliefs about action-outcome distributions
    action_beliefs: ActionOutcomeBeliefs,
    /// Opponent model for policy computation
    opponent: Box<dyn Opponent>,
    /// Weight for epistemic (information-seeking) value
    epistemic_weight: f64,
    /// Number of games played
    games_played: usize,
    /// Which player this agent controls (X or O)
    agent_player: Player,
}

impl std::fmt::Debug for PureActiveInference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PureActiveInference")
            .field("opponent_kind", &self.opponent.kind())
            .field("epistemic_weight", &self.epistemic_weight)
            .field("agent_player", &self.agent_player)
            .field("games_played", &self.games_played)
            .field("tracked_action_pairs", &self.action_beliefs.tracked_pairs())
            .finish()
    }
}

impl PureActiveInference {
    /// Default preference probabilities for win, draw, loss
    const DEFAULT_PREFERENCES: (f64, f64, f64) =
        crate::active_inference::preferences::CANONICAL_PREFERENCE_PROBS;

    /// Create a new PureActiveInference agent
    pub fn new(opponent_kind: OpponentKind, epistemic_weight: f64) -> Self {
        Self::new_for_player(opponent_kind, epistemic_weight, Player::X)
    }

    /// Create a new PureActiveInference agent controlling the provided player
    pub fn new_for_player(
        opponent_kind: OpponentKind,
        epistemic_weight: f64,
        agent_player: Player,
    ) -> Self {
        let mut base_preferences = PreferenceModel::from_probabilities(
            Self::DEFAULT_PREFERENCES.0,
            Self::DEFAULT_PREFERENCES.1,
            Self::DEFAULT_PREFERENCES.2,
        );
        base_preferences.epistemic_weight = epistemic_weight;
        let preferences = base_preferences.for_player(agent_player);
        let opponent_beliefs = Beliefs::symmetric(preferences.opponent_dirichlet_alpha);
        let action_beliefs = ActionOutcomeBeliefs::new(1.0);
        let opponent = opponent_kind.into_boxed_opponent();

        Self {
            preferences,
            base_preferences,
            opponent_beliefs,
            action_beliefs,
            opponent,
            epistemic_weight,
            games_played: 0,
            agent_player,
        }
    }

    /// Create with custom preferences
    pub fn with_custom_preferences(
        opponent_kind: OpponentKind,
        preferences: PreferenceModel,
        agent_player: Player,
    ) -> Self {
        let base_preferences = preferences;
        let preferences = base_preferences.for_player(agent_player);
        let epistemic_weight = preferences.epistemic_weight;
        let opponent_beliefs = Beliefs::symmetric(preferences.opponent_dirichlet_alpha);
        let action_beliefs = ActionOutcomeBeliefs::new(1.0);
        let opponent = opponent_kind.into_boxed_opponent();

        Self {
            preferences,
            base_preferences,
            opponent_beliefs,
            action_beliefs,
            opponent,
            epistemic_weight,
            games_played: 0,
            agent_player,
        }
    }

    /// Create with custom beliefs (e.g., from serialization)
    ///
    /// This constructor allows full control over both opponent beliefs and action-outcome beliefs,
    /// enabling proper deserialization of trained agents.
    pub fn with_custom_beliefs(
        opponent_kind: OpponentKind,
        preferences: PreferenceModel,
        opponent_beliefs: Beliefs,
        action_beliefs: ActionOutcomeBeliefs,
        agent_player: Player,
    ) -> Self {
        let base_preferences = preferences;
        let preferences = base_preferences.for_player(agent_player);
        let epistemic_weight = preferences.epistemic_weight;
        let opponent = opponent_kind.into_boxed_opponent();

        Self {
            preferences,
            base_preferences,
            opponent_beliefs,
            action_beliefs,
            opponent,
            epistemic_weight,
            games_played: 0,
            agent_player,
        }
    }

    /// Get the opponent kind
    pub fn opponent_kind(&self) -> OpponentKind {
        self.opponent.kind()
    }

    /// Get epistemic weight
    pub fn epistemic_weight(&self) -> f64 {
        self.epistemic_weight
    }

    /// Get the player controlled by this agent
    pub fn agent_player(&self) -> Player {
        self.agent_player
    }

    /// Access the current preference model
    pub fn preference_model(&self) -> &PreferenceModel {
        &self.preferences
    }

    /// Update which player the agent controls.
    pub fn set_agent_player(&mut self, player: Player) {
        if self.agent_player == player {
            return;
        }
        self.agent_player = player;
        self.games_played = 0;
        self.preferences = self.base_preferences.for_player(player);
        self.opponent_beliefs = Beliefs::symmetric(self.preferences.opponent_dirichlet_alpha);
        let default_alpha = self.action_beliefs.default_alpha;
        self.action_beliefs = ActionOutcomeBeliefs::new(default_alpha);
    }

    /// Access action-outcome beliefs
    pub fn action_beliefs(&self) -> &ActionOutcomeBeliefs {
        &self.action_beliefs
    }

    /// Access opponent beliefs
    pub fn opponent_beliefs(&self) -> &Beliefs {
        &self.opponent_beliefs
    }

    /// Update beliefs based on an observed opponent move
    fn update_beliefs_from_opponent_move(&mut self, state: &BoardState, opponent_move: usize) {
        let (ctx, label) = state.canonical_context_and_label();
        let legal_moves = ctx.state.legal_moves();
        if let Some(idx) = legal_moves
            .iter()
            .position(|&m| m == ctx.map_move_to_canonical(opponent_move))
        {
            self.opponent_beliefs
                .observe_opponent_action(label.as_str(), legal_moves.len(), idx);
        }
    }

    /// Compute EFE for an action using learned beliefs about outcomes.
    ///
    /// This is the key difference from Oracle (which uses perfect game tree knowledge)
    /// and hybrid ActiveInference (which has perfect transition knowledge but learns opponent beliefs).
    ///
    /// Formula: EFE = Risk + β_ambiguity × Ambiguity - β_epistemic × Epistemic
    /// where:
    /// - Risk = KL[Q(o)||P(o)] (pragmatic value - divergence from preferences)
    /// - Ambiguity = H[Q(o)] (entropy of outcome distribution)
    /// - Epistemic = I[O; θ] (Dirichlet–categorical information gain about action–outcome parameters; implemented via dirichlet_categorical_mi)
    fn compute_learned_efe(&self, state_label: &str, action: usize) -> f64 {
        // Get predictive distribution over outcomes from learned beliefs
        let outcome_probs = self.action_beliefs.predictive(state_label, action);
        let alpha = self.action_beliefs.alpha_for(state_label, action);
        // [P(Win), P(Draw), P(Loss)]

        let q_win = outcome_probs[0];
        let q_draw = outcome_probs[1];
        let q_loss = outcome_probs[2];

        // Learned outcome beliefs are in agent-relative order [Win, Draw, Loss], while
        // preferences are stored over absolute terminal outcomes (XWin/Draw/OWin).
        let p_win = self.preferences.agent_win_preference_for(self.agent_player);
        let p_loss = self
            .preferences
            .agent_loss_preference_for(self.agent_player);
        let p_draw = self.preferences.draw_preference();

        // Risk (pragmatic value): KL divergence between predicted (Q) and preferred (P) distributions
        // KL[Q||P] = Σ Q(i) * ln(Q(i) / P(i))
        let mut risk = 0.0;
        if q_win > 0.0 && p_win > 0.0 {
            risk += q_win * (q_win / p_win).ln();
        }
        if q_draw > 0.0 && p_draw > 0.0 {
            risk += q_draw * (q_draw / p_draw).ln();
        }
        if q_loss > 0.0 && p_loss > 0.0 {
            risk += q_loss * (q_loss / p_loss).ln();
        }

        // Ambiguity: entropy of outcome distribution H[Q(o)]
        // H[Q] = -Σ Q(i) * ln(Q(i))
        // High entropy = uncertain about outcomes (many possibilities)
        // Low entropy = certain about outcomes (one likely outcome)
        let ambiguity = -outcome_probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum::<f64>();

        // Epistemic value: information gain about action-outcome parameters
        // Approximated via Dirichlet-categorical mutual information.
        let epistemic = dirichlet_categorical_mi(&alpha);

        // EFE = Risk + β_ambiguity × Ambiguity - β_epistemic × Epistemic
        // Positive β_ambiguity penalizes uncertain outcomes (risk-averse)
        // Negative β_ambiguity rewards uncertain outcomes (risk-seeking)
        risk + self.preferences.ambiguity_weight * ambiguity
            - self.preferences.epistemic_weight * epistemic
    }

    #[cfg(test)]
    pub(crate) fn compute_learned_efe_for_test(&self, state_label: &str, action: usize) -> f64 {
        self.compute_learned_efe(state_label, action)
    }

    /// Update workspace weights for a given state using learned EFE values.
    fn update_workspace_from_state(&mut self, workspace: &mut MenaceWorkspace, state: &BoardState) {
        let (ctx, orig_label) = state.canonical_context_and_label();
        workspace.ensure_box_weighted(&orig_label);

        let legal_moves = ctx.state.legal_moves();
        if legal_moves.is_empty() {
            return;
        }

        // Compute EFE for each legal action using learned beliefs
        let efe_values: Vec<f64> = legal_moves
            .iter()
            .map(|&canonical_action| {
                self.compute_learned_efe(orig_label.as_str(), canonical_action)
            })
            .collect();

        // Get policy priors (can use existing infrastructure)
        let mut prior_weights: Vec<f64> = legal_moves
            .iter()
            .map(|_| 1.0) // Uniform prior for now
            .collect();
        if prior_weights.iter().all(|w| *w <= 0.0) {
            prior_weights = vec![1.0; prior_weights.len()];
        }

        // Compute optimal policy given EFE values
        let policy =
            optimal_policy_with_kl(&efe_values, &prior_weights, self.preferences.policy_lambda);

        // Convert policy to bead counts
        let bead_counts: HashMap<usize, f64> = legal_moves
            .iter()
            .zip(policy.iter())
            .map(|(&action, &prob)| {
                (
                    action,
                    prob.max(0.0) * self.preferences.policy_to_beads_scale,
                )
            })
            .collect();

        workspace.set_move_weights(&orig_label, &bead_counts);
    }
}

impl LearningAlgorithm for PureActiveInference {
    fn train_from_game(
        &mut self,
        workspace: &mut MenaceWorkspace,
        states: &[BoardState],
        moves: &[usize],
        outcome: GameOutcome,
        player: Player,
    ) -> crate::Result<()> {
        if player != self.agent_player {
            self.games_played += 1;
            return Ok(());
        }

        // Update action-outcome beliefs for each (state, action) pair we played
        for (state, &mv) in states.iter().zip(moves.iter()) {
            if state.to_move == self.agent_player && !state.is_terminal() {
                let (ctx, label) = state.canonical_context_and_label();
                let canonical_action = ctx.map_move_to_canonical(mv);

                // Observe that taking this action in this state led to this outcome
                self.action_beliefs.observe_outcome(
                    label.as_str(),
                    canonical_action,
                    outcome,
                    self.agent_player,
                );
            }
        }

        // Update opponent beliefs
        for (state, &mv) in states.iter().zip(moves.iter()) {
            if state.to_move != self.agent_player {
                self.update_beliefs_from_opponent_move(state, mv);
            }
        }

        // Now update workspace based on new learned beliefs
        // We recompute EFE for all states we visited using updated beliefs
        for state in states.iter() {
            if state.to_move == self.agent_player && !state.is_terminal() {
                self.update_workspace_from_state(workspace, state);
            }
        }

        self.games_played += 1;
        Ok(())
    }

    fn name(&self) -> &str {
        match self.opponent.kind() {
            OpponentKind::Uniform => "Pure Active Inference (Uniform)",
            OpponentKind::Adversarial => "Pure Active Inference (Adversarial)",
            OpponentKind::Minimax => "Pure Active Inference (Minimax)",
        }
    }

    fn stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        stats.insert("games_played".to_string(), self.games_played as f64);
        stats.insert("epistemic_weight".to_string(), self.epistemic_weight);
        stats.insert(
            "opponent_type".to_string(),
            self.opponent.kind() as u8 as f64,
        );
        stats.insert(
            "opponent_beliefs_version".to_string(),
            self.opponent_beliefs.version() as f64,
        );
        stats.insert(
            "action_beliefs_version".to_string(),
            self.action_beliefs.version() as f64,
        );
        stats.insert(
            "tracked_action_pairs".to_string(),
            self.action_beliefs.tracked_pairs() as f64,
        );
        stats.insert("is_pure_aif".to_string(), 1.0);
        stats
    }

    fn reset(&mut self) {
        self.games_played = 0;
        self.opponent_beliefs = Beliefs::symmetric(self.preferences.opponent_dirichlet_alpha);
        let default_alpha = self.action_beliefs.default_alpha;
        self.action_beliefs = ActionOutcomeBeliefs::new(default_alpha);
    }

    // Pure Active Inference doesn't use reinforcement values
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::{
        active_inference::PolicyPrior,
        tictactoe::{BoardState, GameOutcome, Player},
        workspace::{InitialBeadSchedule, MenaceWorkspace, RestockMode, StateFilter},
    };

    fn root_distribution_for_prior(prior: PolicyPrior) -> HashMap<usize, f64> {
        let mut workspace = MenaceWorkspace::with_config(
            StateFilter::Michie,
            RestockMode::default(),
            InitialBeadSchedule::default(),
        )
        .expect("workspace construction should succeed");

        let initial_state = BoardState::new();
        let (_, label) = initial_state.canonical_context_and_label();

        let preferences = PreferenceModel::from_probabilities(0.9, 0.09, 0.01)
            .with_epistemic_weight(0.0)
            .with_policy_lambda(1e6)
            .with_policy_prior(prior);

        // Use Oracle agent since it uses EFE-based policy updates
        // Learning-based agent uses outcome reinforcement, not EFE updates
        let mut alg = OracleActiveInference::with_custom_preferences(
            OpponentKind::Uniform,
            preferences,
            Player::X,
        );

        let states = vec![initial_state];
        let moves = vec![0usize];
        alg.train_from_game(
            &mut workspace,
            &states,
            &moves,
            GameOutcome::Draw,
            Player::X,
        )
        .unwrap();

        workspace
            .move_distribution(&label)
            .expect("root state should have distribution")
            .into_iter()
            .collect()
    }

    #[test]
    fn positional_prior_shapes_oracle_distribution() {
        let distribution = root_distribution_for_prior(PolicyPrior::MenacePositional);
        let center = distribution.get(&4).copied().unwrap_or_default();
        let corner = distribution.get(&0).copied().unwrap_or_default();
        let edge = distribution.get(&1).copied().unwrap_or_default();

        assert!(
            center > corner && corner > edge,
            "Expected center ({center}) > corner ({corner}) > edge ({edge}) when using positional prior"
        );
    }

    #[test]
    fn uniform_prior_produces_flat_oracle_distribution() {
        let distribution = root_distribution_for_prior(PolicyPrior::Uniform);
        let mut values: Vec<f64> = distribution.values().copied().collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min = values.first().copied().unwrap_or(0.0);
        let max = values.last().copied().unwrap_or(0.0);
        assert!(
            (max - min).abs() < 1e-6,
            "Uniform prior should yield near-uniform distribution; min={min}, max={max}"
        );
    }

    #[test]
    fn pure_active_inference_o_respects_perspective() {
        let preferences =
            PreferenceModel::from_probabilities(0.9, 0.05, 0.05).with_epistemic_weight(0.0);
        let mut agent = PureActiveInference::with_custom_preferences(
            OpponentKind::Uniform,
            preferences,
            Player::O,
        );
        let board = BoardState::new_with_player(Player::O);
        let (_, label) = board.canonical_context_and_label();
        let label = label.into_string();

        for _ in 0..3 {
            agent
                .action_beliefs
                .observe_outcome(&label, 0, GameOutcome::Win(Player::O), Player::O);
        }
        for _ in 0..3 {
            agent
                .action_beliefs
                .observe_outcome(&label, 1, GameOutcome::Win(Player::X), Player::O);
        }

        let win_efe = agent.compute_learned_efe_for_test(&label, 0);
        let loss_efe = agent.compute_learned_efe_for_test(&label, 1);

        assert!(
            win_efe < loss_efe,
            "EFE should favour agent-winning trajectories for O; win {win_efe}, loss {loss_efe}"
        );
    }

    #[test]
    fn pure_active_inference_x_respects_perspective() {
        let preferences =
            PreferenceModel::from_probabilities(0.9, 0.05, 0.05).with_epistemic_weight(0.0);
        let mut agent = PureActiveInference::with_custom_preferences(
            OpponentKind::Uniform,
            preferences,
            Player::X,
        );
        let board = BoardState::new();
        let (_, label) = board.canonical_context_and_label();
        let label = label.into_string();

        for _ in 0..3 {
            agent
                .action_beliefs
                .observe_outcome(&label, 0, GameOutcome::Win(Player::X), Player::X);
        }
        for _ in 0..3 {
            agent
                .action_beliefs
                .observe_outcome(&label, 1, GameOutcome::Win(Player::O), Player::X);
        }

        let win_efe = agent.compute_learned_efe_for_test(&label, 0);
        let loss_efe = agent.compute_learned_efe_for_test(&label, 1);

        assert!(
            win_efe < loss_efe,
            "EFE should favour agent-winning trajectories for X; win {win_efe}, loss {loss_efe}"
        );
    }
}
