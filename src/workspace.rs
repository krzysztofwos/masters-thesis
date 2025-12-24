//! MENACE workspace and learning infrastructure.
//!
//! This module provides the core workspace types for MENACE training:
//! - `MenaceWorkspace`: Central structure managing matchbox learning
//! - `StateFilter`: Strategies for filtering decision states
//! - `RestockMode`: Matchbox restocking behaviors
//! - `Reinforcement`: Learning signal types

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt,
    str::FromStr,
};

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::identifiers::{MoveId, StateId};

/// Reinforcement signal used to adjust matchbox weights.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Reinforcement {
    /// Strengthen the probability of moves along the path.
    Positive(f64),
    /// Weaken the probability of moves along the path.
    Negative(f64),
    /// No update.
    Neutral,
    /// Custom multiplier applied directly.
    Custom(f64),
}

impl Reinforcement {
    fn as_delta(&self) -> f64 {
        match self {
            Reinforcement::Positive(strength) => *strength,
            Reinforcement::Negative(strength) => -*strength,
            Reinforcement::Neutral => 0.0,
            Reinforcement::Custom(multiplier) => *multiplier,
        }
    }
}

/// Context describing a single learning update.
#[derive(Debug, Clone)]
pub struct LearningContext {
    pub reinforcement: Reinforcement,
    pub path_taken: Vec<MoveId>,
    pub timestep: usize,
}

use crate::{
    tictactoe::{
        Player,
        board::BoardState,
        game_tree::{analyze_menace_positions, build_reduced_game_tree},
    },
    types::{MoveWeights, Position, SampledMove, Weight},
    utils::{normalize_weighted_pairs, weighted_sample},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum StateFilter {
    /// Include all X-to-move canonical states as decision points (historical MENACE).
    All,
    /// Exclude X-to-move positions with a single legal move (forced plays).
    DecisionOnly,
    /// Reproduce Michie's original 287 matchboxes (exclude forced + double-threat positions).
    #[default]
    Michie,
    /// Include both X-to-move and O-to-move states (player-agnostic).
    /// Excludes forced moves and double-threat positions for both players.
    Both,
}

/// Strategy describing how depleted matchboxes are restocked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RestockMode {
    /// Do not restock depleted beads; empty boxes stay empty.
    None,
    /// Restock the specific move immediately when its weight reaches zero.
    Move,
    /// Restock the entire matchbox when all weights have been exhausted.
    #[default]
    Box,
}

impl fmt::Display for StateFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            StateFilter::All => "all",
            StateFilter::DecisionOnly => "decision-only",
            StateFilter::Michie => "michie",
            StateFilter::Both => "both",
        };
        f.write_str(label)
    }
}

impl FromStr for StateFilter {
    type Err = crate::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let normalised = s.trim().to_ascii_lowercase();
        match normalised.as_str() {
            "all" | "338" => Ok(StateFilter::All),
            "decision-only" | "decision_only" | "304" => Ok(StateFilter::DecisionOnly),
            "michie" | "287" => Ok(StateFilter::Michie),
            "both" | "player-agnostic" | "bi-player" => Ok(StateFilter::Both),
            _ => Err(crate::Error::ParseStateFilter {
                input: s.to_string(),
                expected: "all/338, decision-only/304, michie/287, both".to_string(),
            }),
        }
    }
}

impl fmt::Display for RestockMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            RestockMode::None => "none",
            RestockMode::Move => "move",
            RestockMode::Box => "box",
        };
        f.write_str(label)
    }
}

impl FromStr for RestockMode {
    type Err = crate::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(RestockMode::None),
            "move" => Ok(RestockMode::Move),
            "box" => Ok(RestockMode::Box),
            _ => Err(crate::Error::ParseRestockMode {
                input: s.to_string(),
                expected: "none, move, box".to_string(),
            }),
        }
    }
}

/// Initial bead weights assigned to MENACE matchboxes per ply.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct InitialBeadSchedule {
    /// Weights for plies 0, 2, 4, and 6 respectively.
    pub per_ply: [f64; 4],
}

impl InitialBeadSchedule {
    /// Creates a new schedule. Values are clamped to be non-negative.
    pub fn new(per_ply: [f64; 4]) -> Self {
        Self {
            per_ply: per_ply.map(|value| value.max(0.0)),
        }
    }

    /// Returns the default MENACE schedule (4/3/2/1 beads by ply).
    pub fn menace() -> Self {
        Self::new([4.0, 3.0, 2.0, 1.0])
    }

    /// Returns the initial weight for the supplied piece count (number of occupied cells).
    ///
    /// Odd piece counts (O-to-move states) share the same schedule bucket as the
    /// preceding even ply, ensuring both players receive comparable initialization.
    pub fn weight_for_piece_count(&self, count: usize) -> f64 {
        let index = (count / 2).min(3);
        self.per_ply[index]
    }

    /// Returns the initial bead count for a given ply level (as used in original MENACE).
    ///
    /// The ply level represents the move number divided by 2, grouping moves in pairs:
    /// - ply 0, 1: early game (piece_count 0-3)
    /// - ply 2, 3: mid game (piece_count 4-7)
    /// - ply 4, 5: late game (piece_count 8-11)
    /// - ply 6+: endgame
    pub fn beads_for_ply(&self, ply: usize) -> u32 {
        let index = (ply / 2).min(3);
        self.per_ply[index] as u32
    }
}

impl Default for InitialBeadSchedule {
    fn default() -> Self {
        InitialBeadSchedule::menace()
    }
}

impl fmt::Display for InitialBeadSchedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{},{},{},{}",
            self.per_ply[0], self.per_ply[1], self.per_ply[2], self.per_ply[3]
        )
    }
}

impl FromStr for InitialBeadSchedule {
    type Err = crate::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut values = Vec::new();
        for part in s
            .split([',', ';', ' ', '\t'])
            .filter(|token| !token.is_empty())
        {
            let value: f64 = part.parse().map_err(|_| crate::Error::ParseBeadSchedule {
                input: s.to_string(),
                reason: format!(
                    "Invalid bead weight '{part}'. Expected four numeric values (e.g., 4,3,2,1)"
                ),
            })?;
            values.push(value);
        }

        if values.len() != 4 {
            return Err(crate::Error::ParseBeadSchedule {
                input: s.to_string(),
                reason: format!(
                    "Expected exactly four bead weights (ply 0/2/4/6). Got {} value(s)",
                    values.len()
                ),
            });
        }

        Ok(InitialBeadSchedule::new([
            values[0], values[1], values[2], values[3],
        ]))
    }
}

/// Strategy for restocking depleted matchboxes.
///
/// Consolidates the restocking logic used throughout MENACE training,
/// providing a single point of control for how empty matchboxes are refilled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestockingStrategy {
    mode: RestockMode,
    schedule: InitialBeadSchedule,
}

impl RestockingStrategy {
    /// Create a new restocking strategy with the given mode and schedule.
    pub fn new(mode: RestockMode, schedule: InitialBeadSchedule) -> Self {
        Self { mode, schedule }
    }

    /// Check if this state needs box-level restocking and apply if necessary.
    ///
    /// This method only applies for `RestockMode::Box`, restocking the entire
    /// matchbox when all weights are depleted. Individual weight restocking
    /// (RestockMode::Move) is handled by `restock_single_move()`.
    ///
    /// Returns true if restocking was applied, false otherwise.
    pub(crate) fn apply(
        &self,
        weights: &mut WeightStore,
        outgoing: &BTreeMap<String, Vec<MoveId>>,
        state_label: &StateId,
    ) -> bool {
        // Only restock entire boxes when mode is Box
        if self.mode != RestockMode::Box {
            return false;
        }

        // Check if the matchbox has any positive weights
        if self.has_positive_weight(weights, outgoing, state_label) {
            return false;
        }

        // Get the default weight for this state
        let Some(default_weight) = self.initial_weight_for_state(state_label) else {
            return false;
        };

        // Restock all moves from this state
        self.restock_moves(weights, outgoing, state_label, default_weight);
        true
    }

    /// Get the mode of this strategy.
    pub fn mode(&self) -> RestockMode {
        self.mode
    }

    /// Get the schedule of this strategy.
    pub fn schedule(&self) -> InitialBeadSchedule {
        self.schedule
    }

    /// Restock a single move if move-level restocking is enabled.
    ///
    /// This is used during weight updates to immediately restock individual moves
    /// that reach zero (RestockMode::Move). Returns true if restocking was applied.
    ///
    /// # Arguments
    ///
    /// * `weight` - Mutable reference to the weight to potentially restock
    /// * `source_label` - The source state of the move
    ///
    /// # Returns
    ///
    /// Returns true if the weight was restocked, false otherwise
    pub fn restock_single_move(&self, weight: &mut f64, source_label: &StateId) -> bool {
        if self.mode != RestockMode::Move || *weight > 0.0 {
            return false;
        }

        if let Some(base) = Self::compute_initial_weight(source_label, &self.schedule) {
            *weight = base;
            true
        } else {
            false
        }
    }

    /// Compute the initial weight for a state based on the bead schedule.
    ///
    /// This is a public helper that can be used during workspace construction
    /// or for querying what the initial weight should be for a given state.
    pub fn compute_initial_weight(label: &StateId, schedule: &InitialBeadSchedule) -> Option<f64> {
        let state = BoardState::from_label(label.as_str()).ok()?;
        let count = state.occupied_count();
        let weight = schedule.weight_for_piece_count(count);
        Some(weight)
    }

    // Private helper methods

    fn initial_weight_for_state(&self, label: &StateId) -> Option<f64> {
        Self::compute_initial_weight(label, &self.schedule)
    }

    fn has_positive_weight(
        &self,
        weights: &WeightStore,
        outgoing: &BTreeMap<String, Vec<MoveId>>,
        state_label: &StateId,
    ) -> bool {
        if let Some(moves) = outgoing.get(state_label.as_str()) {
            moves
                .iter()
                .any(|label| weights.get(label).map(|w| w > 0.0).unwrap_or(false))
        } else {
            false
        }
    }

    fn restock_moves(
        &self,
        weights: &mut WeightStore,
        outgoing: &BTreeMap<String, Vec<MoveId>>,
        state_label: &StateId,
        default_weight: f64,
    ) {
        if let Some(moves) = outgoing.get(state_label.as_str()) {
            for label in moves {
                weights.set(label, default_weight);
            }
        }
    }
}

/// Lightweight storage for move weights.
///
/// Map-based storage for move probabilities at runtime.
///
/// Note: This is `pub(crate)` to allow its use in public trait methods
/// while keeping it internal to the crate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WeightStore {
    weights: BTreeMap<MoveId, f64>,
}

impl WeightStore {
    fn new() -> Self {
        Self {
            weights: BTreeMap::new(),
        }
    }

    /// Get weight for a move
    fn get(&self, label: &MoveId) -> Option<f64> {
        self.weights.get(label).copied()
    }

    /// Set weight for a move.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if weight is negative, NaN, or infinite.
    fn set(&mut self, label: &MoveId, weight: f64) {
        debug_assert!(
            weight.is_finite() && weight >= 0.0,
            "WeightStore::set received invalid weight: {weight} (must be finite and non-negative)"
        );
        self.weights.insert(label.clone(), weight);
    }

    /// Get mutable reference to weight
    fn get_mut(&mut self, label: &MoveId) -> Option<&mut f64> {
        self.weights.get_mut(label)
    }
}

/// MENACE learning rule that clamps, restocks, and restricts updates to the
/// configured move set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MenaceLearningRule {
    learnable_moves: Option<BTreeSet<MoveId>>,
    strategy: RestockingStrategy,
}

impl MenaceLearningRule {
    /// Restrict learning to the supplied move set (typically X-to-move decisions).
    pub fn for_learnable(learnable_moves: BTreeSet<MoveId>, restock_mode: RestockMode) -> Self {
        Self::for_learnable_with_schedule(
            learnable_moves,
            restock_mode,
            InitialBeadSchedule::default(),
        )
    }

    /// Restrict learning with a custom initial bead schedule.
    pub fn for_learnable_with_schedule(
        learnable_moves: BTreeSet<MoveId>,
        restock_mode: RestockMode,
        initial_beads: InitialBeadSchedule,
    ) -> Self {
        Self {
            learnable_moves: Some(learnable_moves),
            strategy: RestockingStrategy::new(restock_mode, initial_beads),
        }
    }

    /// Allow updates on every move encountered in the learning trace.
    pub fn for_all(restock_mode: RestockMode) -> Self {
        Self::for_all_with_schedule(restock_mode, InitialBeadSchedule::default())
    }

    /// Allow updates with a custom initial bead schedule.
    pub fn for_all_with_schedule(
        restock_mode: RestockMode,
        initial_beads: InitialBeadSchedule,
    ) -> Self {
        Self {
            learnable_moves: None,
            strategy: RestockingStrategy::new(restock_mode, initial_beads),
        }
    }

    pub fn initial_beads(&self) -> InitialBeadSchedule {
        self.strategy.schedule()
    }

    /// Get the restocking strategy used by this learning rule.
    pub fn restocking_strategy(&self) -> &RestockingStrategy {
        &self.strategy
    }

    /// Apply reinforcement learning update to weights.
    ///
    /// This is the preferred method for use with MenaceWorkspace which uses WeightStore.
    pub(crate) fn apply_reinforcement(
        &self,
        weights: &mut WeightStore,
        outgoing: &BTreeMap<String, Vec<MoveId>>,
        move_sources: &BTreeMap<MoveId, StateId>,
        context: LearningContext,
    ) -> crate::Result<()> {
        let delta = context.reinforcement.as_delta();

        let mut affected_states = BTreeSet::new();

        for move_label in &context.path_taken {
            if let Some(learnable) = &self.learnable_moves
                && !learnable.contains(move_label)
            {
                continue;
            }

            if let Some(weight) = weights.get_mut(move_label) {
                let source_label = move_sources.get(move_label).ok_or_else(|| {
                    crate::Error::MissingMoveSource {
                        move_id: move_label.to_string(),
                    }
                })?;

                *weight += delta;
                if *weight < 0.0 {
                    *weight = 0.0;
                }
                // Restock individual move if it hit zero (RestockMode::Move)
                self.strategy.restock_single_move(weight, source_label);

                affected_states.insert(source_label.clone());
            }
        }

        // Restock entire matchboxes if all weights depleted
        // The strategy handles mode checking internally (only applies for RestockMode::Box)
        for state_label in affected_states {
            self.strategy.apply(weights, outgoing, &state_label);
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MenaceWorkspace {
    /// Move weights
    weights: WeightStore,
    /// Track which state each move comes from (needed for restocking)
    move_sources: BTreeMap<MoveId, StateId>,
    canonical_states: BTreeMap<String, BoardState>,
    decision_states: BTreeSet<String>,
    outgoing: BTreeMap<String, Vec<MoveId>>,
    move_lookup: BTreeMap<(String, usize), MoveId>,
    move_positions: BTreeMap<MoveId, usize>,
    learnable_moves: BTreeSet<MoveId>,
    learning_rule: MenaceLearningRule,
    timestep: usize,
    strategy: RestockingStrategy,
}

impl MenaceWorkspace {
    pub fn new(filter: StateFilter) -> crate::Result<Self> {
        Self::with_config(
            filter,
            RestockMode::default(),
            InitialBeadSchedule::default(),
        )
    }

    pub fn with_options(filter: StateFilter, restock_mode: RestockMode) -> crate::Result<Self> {
        Self::with_config(filter, restock_mode, InitialBeadSchedule::default())
    }

    pub fn with_config(
        filter: StateFilter,
        restock_mode: RestockMode,
        initial_beads: InitialBeadSchedule,
    ) -> crate::Result<Self> {
        // For StateFilter::Both, we need both X and O states in the game tree
        let x_only = filter != StateFilter::Both;
        let tree = build_reduced_game_tree(x_only, false);
        let stats = analyze_menace_positions(&tree);
        let forced: BTreeSet<_> = stats.forced_positions.into_iter().collect();
        let double: BTreeSet<_> = stats.double_threat_positions.into_iter().collect();
        let canonical_states: BTreeMap<String, BoardState> =
            tree.canonical_states.into_iter().collect();

        let mut decision_states = BTreeSet::new();
        let mut learnable_moves = BTreeSet::new();
        let mut outgoing: BTreeMap<String, Vec<MoveId>> = BTreeMap::new();
        let mut move_lookup: BTreeMap<(String, usize), MoveId> = BTreeMap::new();
        let mut move_positions: BTreeMap<MoveId, usize> = BTreeMap::new();
        let mut weights = WeightStore::new();
        let mut move_sources = BTreeMap::new();

        for (label, state) in &canonical_states {
            if state.is_terminal() {
                continue;
            }

            let is_forced = if state.to_move == Player::X {
                forced.contains(label)
            } else {
                state.has_forced_move()
            };
            let is_double = if state.to_move == Player::X {
                if state.occupied_count().is_multiple_of(2) {
                    double.contains(label)
                } else {
                    state.opponent_has_double_threat()
                }
            } else {
                state.opponent_has_double_threat()
            };

            let is_decision = match filter {
                StateFilter::All => state.to_move == Player::X,
                StateFilter::DecisionOnly => state.to_move == Player::X && !is_forced,
                StateFilter::Michie => state.to_move == Player::X && !is_forced && !is_double,
                StateFilter::Both => {
                    // Both X and O states can be decision states
                    !is_forced && !is_double
                }
            };

            if is_decision {
                decision_states.insert(label.clone());
            }

            let source_state_id = StateId::from(label.as_str());

            for move_pos in state.legal_moves() {
                let next_state = state
                    .make_move_force(move_pos)
                    .map_err(|e| crate::Error::LegalMoveFailed {
                        message: e.to_string(),
                    })?
                    .canonical();
                let _next_label = next_state.encode();

                let move_id = move_label(label, move_pos);
                let weight =
                    RestockingStrategy::compute_initial_weight(&source_state_id, &initial_beads)
                        .unwrap_or(1.0);

                weights.set(&move_id, weight);
                move_sources.insert(move_id.clone(), source_state_id.clone());

                outgoing
                    .entry(label.clone())
                    .or_default()
                    .push(move_id.clone());
                move_lookup.insert((label.clone(), move_pos), move_id.clone());
                move_positions.insert(move_id.clone(), move_pos);

                // Mark moves as learnable only for decision states.
                // For StateFilter::Both, both X and O decision state moves are learnable.
                // For other filters, only X-to-move decision moves are learnable.
                if is_decision {
                    learnable_moves.insert(move_id.clone());
                }
            }
        }

        let learnable_moves_clone = learnable_moves.clone();
        let learning_rule = MenaceLearningRule::for_learnable_with_schedule(
            learnable_moves_clone,
            restock_mode,
            initial_beads,
        );

        Ok(Self {
            weights,
            move_sources,
            canonical_states,
            decision_states,
            outgoing,
            move_lookup,
            move_positions,
            learnable_moves,
            learning_rule,
            timestep: 0,
            strategy: RestockingStrategy::new(restock_mode, initial_beads),
        })
    }

    pub fn canonical_states(&self) -> &BTreeMap<String, BoardState> {
        &self.canonical_states
    }

    pub fn decision_labels(&self) -> impl Iterator<Item = &String> {
        self.decision_states.iter()
    }

    /// Get the board state for a canonical label.
    pub fn state(&self, label: &crate::types::CanonicalLabel) -> Option<&BoardState> {
        self.canonical_states.get(label.as_str())
    }

    /// Get the move identifier for a specific move from a canonical state.
    pub fn move_for_position(
        &self,
        label: &crate::types::CanonicalLabel,
        move_index: usize,
    ) -> Option<&MoveId> {
        self.move_lookup
            .get(&(label.as_str().to_string(), move_index))
    }

    /// Sample a weighted move from the given canonical state.
    ///
    /// Returns an error if the matchbox exists but all moves have been depleted
    /// while operating in `RestockMode::None`.
    pub fn sample_move<R: Rng>(
        &mut self,
        canonical_label: &crate::types::CanonicalLabel,
        rng: &mut R,
    ) -> crate::Result<SampledMove> {
        // Check if matchbox exists - if not, handle according to missing box policy
        if !self.has_matchbox(canonical_label) {
            return self.handle_missing_matchbox(canonical_label, rng);
        }

        // Extract moves with positive weights using the shared helper.
        // If all moves have been exhausted and we're in Box restock mode,
        // restock the entire box and retry.
        let mut entries = self.extract_weighted_morphisms(canonical_label, |w| w > 0.0);
        if entries.is_none() {
            match self.restock_mode() {
                RestockMode::Box => {
                    // Restock all morphisms from this state then retry
                    self.ensure_box_weighted(canonical_label);
                    entries = self.extract_weighted_morphisms(canonical_label, |w| w > 0.0);
                }
                RestockMode::None => {
                    return Err(crate::Error::DepletedMatchbox {
                        label: canonical_label.as_str().to_string(),
                    });
                }
                RestockMode::Move => {}
            }
        }
        let entries = entries.ok_or(crate::Error::NoValidMoves)?;

        // Convert to owned data for sorting and sampling
        let mut weighted_moves: Vec<_> = entries
            .into_iter()
            .map(|(label, mv, w)| (label.clone(), mv, w))
            .collect();

        // Sort by move index for deterministic ordering
        weighted_moves.sort_by(|a, b| a.1.cmp(&b.1));

        // Prepare items for weighted sampling (tuple -> weight pairs)
        let items_for_sampling: Vec<_> = weighted_moves
            .iter()
            .map(|(label, move_index, weight)| ((label.clone(), *move_index), *weight))
            .collect();

        // Sample using the shared weighted sampling utility
        let (label, move_index) =
            weighted_sample(rng, &items_for_sampling).ok_or(crate::Error::NoValidMoves)?;

        // Build the distribution for the SampledMove
        let distribution = weighted_moves
            .iter()
            .filter_map(|(_, mv, w)| {
                let pos = Position::new(*mv).ok()?;
                let weight = Weight::new_or_zero(*w);
                Some((pos, weight))
            })
            .collect();

        let position = Position::new(move_index)?;

        Ok(SampledMove::new(
            position,
            label,
            MoveWeights::new(distribution),
        ))
    }

    /// Handle sampling from a state without a matchbox.
    ///
    /// States can lack matchboxes for legitimate reasons:
    /// 1. **Forced moves** (1 legal move) - No decision to make, no learning opportunity
    /// 2. **Filtered states** - Deliberately excluded by the StateFilter policy:
    ///    - Double-threat positions (already lost, nothing to learn)
    ///    - States deemed not worth learning from
    /// 3. **Wrong player** - O-to-move states when using X-only filters
    ///
    /// # Strategy
    ///
    /// Play through these states WITHOUT learning:
    /// - If forced (1 move): play that move
    /// - If multiple moves: play randomly
    /// - Return a dummy morphism label that's not in `learnable_morphisms`
    /// - When `apply_reinforcement` is called, these moves will be skipped
    ///
    /// # Example: Double-Threat Position
    ///
    /// State: `...XOXOXO` (O has two diagonal winning threats)
    /// - Michie filter excludes this (nothing to learn, game already lost)
    /// - Agent encounters it during gameplay
    /// - We play randomly and don't learn from it
    /// - Reinforcement from this game still applies to earlier decisions
    ///
    /// # Why This Is Correct
    ///
    /// The absence of a matchbox is a policy decision, not a bug:
    /// - The workspace constructor deliberately chose not to create one
    /// - The agent should be able to play through the entire game
    /// - Learning happens only at designated decision points
    /// - This enables flexible filtering strategies without breaking gameplay
    fn handle_missing_matchbox<R: Rng>(
        &mut self,
        canonical_label: &crate::types::CanonicalLabel,
        rng: &mut R,
    ) -> crate::Result<SampledMove> {
        // Get the board state to determine legal moves
        let state = self
            .canonical_states
            .get(canonical_label.as_str())
            .ok_or_else(|| crate::Error::InvalidConfiguration {
                message: format!(
                    "canonical state '{}' not found while sampling move",
                    canonical_label.as_str()
                ),
            })?;
        let legal_moves = state.legal_moves();

        if legal_moves.is_empty() {
            // Terminal state or invalid state - this should never happen
            return Err(crate::Error::NoValidMoves);
        }

        // Pick a move: forced if only one option, random otherwise
        let move_index = if legal_moves.len() == 1 {
            legal_moves[0]
        } else {
            legal_moves[rng.random_range(0..legal_moves.len())]
        };

        // Play this move without learning
        self.play_unlearnable_move(canonical_label, move_index)
    }

    /// Play a move without learning (forced move or filtered state).
    ///
    /// Returns a SampledMove with a dummy move identifier that's NOT in `learnable_moves`,
    /// so when `apply_reinforcement` is called, this move will be skipped.
    ///
    /// This preserves the learning semantics:
    /// - Moves from states with matchboxes get reinforced
    /// - Moves from states without matchboxes don't affect learning
    /// - The agent can still complete the game and learn from earlier decisions
    fn play_unlearnable_move(
        &self,
        canonical_label: &crate::types::CanonicalLabel,
        move_index: usize,
    ) -> crate::Result<SampledMove> {
        // Create dummy move identifier (not in learnable_moves, so won't be updated)
        let dummy_label = MoveId::new(format!("unlearnable_{}", canonical_label.as_str()));

        // Single move with weight 1.0
        let position = Position::new(move_index)?;
        let distribution = vec![(position, Weight::new_or_zero(1.0))];

        Ok(SampledMove::new(
            position,
            dummy_label,
            MoveWeights::new(distribution),
        ))
    }

    pub fn apply_reinforcement(
        &mut self,
        path: Vec<MoveId>,
        reinforcement: Reinforcement,
    ) -> crate::Result<()> {
        if path.is_empty() {
            return Ok(());
        }

        let context = LearningContext {
            reinforcement,
            path_taken: path,
            timestep: self.timestep,
        };

        self.learning_rule.apply_reinforcement(
            &mut self.weights,
            &self.outgoing,
            &self.move_sources,
            context,
        )?;
        self.timestep += 1;
        Ok(())
    }

    pub fn restock_mode(&self) -> RestockMode {
        self.strategy.mode()
    }

    pub fn set_restock_mode(&mut self, restock_mode: RestockMode) {
        if self.strategy.mode() == restock_mode {
            return;
        }
        self.strategy = RestockingStrategy::new(restock_mode, self.strategy.schedule());
        let learnable = self.learnable_moves.clone();
        self.learning_rule = MenaceLearningRule::for_learnable_with_schedule(
            learnable,
            restock_mode,
            self.strategy.schedule(),
        );
    }

    /// Check if a matchbox exists for this canonical state
    pub fn has_matchbox(&self, label: &crate::types::CanonicalLabel) -> bool {
        self.decision_states.contains(label.as_str())
    }

    /// Extract weighted moves for a canonical state with optional filtering.
    ///
    /// This is a private helper to avoid duplication between move_weights and sample_move.
    fn extract_weighted_morphisms<F>(
        &self,
        label: &crate::types::CanonicalLabel,
        filter: F,
    ) -> Option<Vec<(&MoveId, usize, f64)>>
    where
        F: Fn(f64) -> bool,
    {
        let moves = self.outgoing.get(label.as_str())?;
        let entries: Vec<_> = moves
            .iter()
            .filter_map(|move_id| {
                let weight = self.weights.get(move_id)?;
                if filter(weight) {
                    let move_index = *self.move_positions.get(move_id)?;
                    return Some((move_id, move_index, weight));
                }
                None
            })
            .collect();

        if entries.is_empty() {
            None
        } else {
            Some(entries)
        }
    }

    /// Get the raw weights for all moves from a canonical state.
    pub fn move_weights(&self, label: &crate::types::CanonicalLabel) -> Option<Vec<(usize, f64)>> {
        let entries = self.extract_weighted_morphisms(label, |_| true)?;
        Some(entries.into_iter().map(|(_, mv, w)| (mv, w)).collect())
    }

    /// Get the normalized probability distribution for moves from a canonical state.
    pub fn move_distribution(
        &self,
        label: &crate::types::CanonicalLabel,
    ) -> Option<Vec<(usize, f64)>> {
        let weights = self.move_weights(label)?;
        normalize_weighted_pairs(weights)
    }

    /// Ensure the matchbox for this canonical state has non-zero weights (restock if needed).
    pub fn ensure_box_weighted(&mut self, label: &crate::types::CanonicalLabel) {
        let source = StateId::from(label.as_str());
        self.strategy
            .apply(&mut self.weights, &self.outgoing, &source);
    }

    /// Set the weights for all moves from a canonical state.
    pub fn set_move_weights(
        &mut self,
        label: &crate::types::CanonicalLabel,
        new_weights: &HashMap<usize, f64>,
    ) {
        if let Some(moves) = self.outgoing.get(label.as_str()) {
            for move_id in moves {
                if !self.learnable_moves.contains(move_id) {
                    continue;
                }

                if let Some(move_index) = self.move_positions.get(move_id) {
                    let weight = new_weights.get(move_index).copied().unwrap_or(0.0);
                    self.weights.set(move_id, weight.max(0.0));
                }
            }
        }
        self.timestep += 1;
    }
}

fn move_label(label: &str, move_index: usize) -> MoveId {
    MoveId::new(format!("{label}_{move_index}"))
}
