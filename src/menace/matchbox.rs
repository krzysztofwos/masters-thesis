//! Matchbox implementation for MENACE

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{types::Entropy, utils::weighted_sample, workspace::InitialBeadSchedule};

/// A matchbox containing beads for a specific board state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Matchbox {
    /// State identifier (canonical board encoding)
    state_id: String,
    /// Beads for each possible move (position -> count)
    beads: HashMap<usize, u32>,
    /// Ply level (0 = opening, 1 = second move, etc.)
    ply: usize,
    /// Base bead count for this ply level
    base_beads: u32,
}

impl Matchbox {
    /// Create a new matchbox for a state using the default MENACE bead schedule.
    pub fn new(state_id: String, valid_moves: Vec<usize>, ply: usize) -> Self {
        Self::with_schedule(state_id, valid_moves, ply, &InitialBeadSchedule::menace())
    }

    /// Create a new matchbox for a state with a custom bead schedule.
    pub fn with_schedule(
        state_id: String,
        valid_moves: Vec<usize>,
        ply: usize,
        schedule: &InitialBeadSchedule,
    ) -> Self {
        let base_beads = schedule.beads_for_ply(ply);

        let mut beads = HashMap::new();
        for pos in valid_moves {
            beads.insert(pos, base_beads);
        }

        Matchbox {
            state_id,
            beads,
            ply,
            base_beads,
        }
    }

    /// Get the state identifier
    pub fn state_id(&self) -> &str {
        &self.state_id
    }

    /// Get the ply level
    pub fn ply(&self) -> usize {
        self.ply
    }

    /// Get the base bead count for this ply level
    pub fn base_beads(&self) -> u32 {
        self.base_beads
    }

    /// Get the bead count for a specific position
    pub fn bead_count(&self, position: usize) -> Option<u32> {
        self.beads.get(&position).copied()
    }

    /// Get an iterator over all position-bead pairs
    pub fn all_beads(&self) -> impl Iterator<Item = (usize, u32)> + '_ {
        self.beads.iter().map(|(&pos, &count)| (pos, count))
    }

    /// Sample a move from the matchbox
    pub fn sample_move(&self, rng: &mut impl rand::Rng) -> Option<usize> {
        let mut items: Vec<(usize, u32)> = self
            .beads
            .iter()
            .map(|(&pos, &count)| (pos, count))
            .collect();

        // Sort by position for deterministic ordering
        items.sort_by(|a, b| a.0.cmp(&b.0));

        weighted_sample(rng, &items)
    }

    /// Reinforce a move (positive or negative)
    pub fn reinforce(&mut self, position: usize, delta: i16) {
        if let Some(count) = self.beads.get_mut(&position) {
            if delta > 0 {
                *count = count.saturating_add(delta as u32);
            } else {
                *count = count.saturating_sub((-delta) as u32);
            }
        }

        if self.total_beads() == 0 {
            for count in self.beads.values_mut() {
                *count = self.base_beads;
            }
        }
    }

    /// Get total bead count
    pub fn total_beads(&self) -> u32 {
        self.beads.values().sum()
    }

    /// Calculate entropy of the bead distribution
    pub fn entropy(&self) -> Entropy {
        Entropy::from_weights(self.beads.values().map(|&count| count as f64))
    }

    /// Reset to initial bead counts
    pub fn reset(&mut self) {
        for count in self.beads.values_mut() {
            *count = self.base_beads;
        }
    }
}
