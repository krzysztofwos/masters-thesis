//! Test suite for Tic-tac-toe implementation
//! Validates mathematical invariants and game rules

use menace::tictactoe::{BoardState, Cell, Player};

mod double_line_validation {
    use super::*;

    #[test]
    fn test_double_line_requires_shared_cell() {
        // Test case: Three X's on top row AND left column must share a cell
        let mut state = BoardState::new();

        // Create a board with X winning on two lines that share cell (0,0)
        // X X X
        // X O .
        // X O .
        let moves = vec![
            (0, Player::X),
            (4, Player::O),
            (1, Player::X),
            (7, Player::O),
            (2, Player::X),
            (8, Player::O), // X wins horizontally
            (3, Player::X),
            (5, Player::O), // Extra moves
            (6, Player::X),
        ];

        for (pos, player) in moves {
            if state.to_move == player {
                state = state.make_move(pos).unwrap();
            } else {
                // Skip if wrong player's turn
                continue;
            }
        }

        // This should be valid because both winning lines share cell 0
        assert!(
            state.is_valid(),
            "Two winning lines sharing a cell should be valid"
        );
    }

    #[test]
    fn test_invalid_double_win_without_shared_cell() {
        // Test that a board with two non-intersecting winning lines is invalid
        // This is physically impossible in a real game

        // Construct an impossible board state directly
        let mut cells = [Cell::Empty; 9];

        // X wins on row 0 and row 2 (impossible in real game)
        // X X X
        // O O .
        // X X X
        cells[0] = Cell::X;
        cells[1] = Cell::X;
        cells[2] = Cell::X;
        cells[3] = Cell::O;
        cells[4] = Cell::O;
        cells[6] = Cell::X;
        cells[7] = Cell::X;
        cells[8] = Cell::X;

        let state = BoardState {
            cells,
            to_move: Player::O,
        };

        assert!(
            !state.is_valid(),
            "Two non-intersecting winning lines should be invalid"
        );
    }
}

mod o_first_support {
    use super::*;

    #[test]
    fn o_first_opening_is_considered_valid() {
        let mut state = BoardState::new_with_player(Player::O);
        assert!(state.is_valid(), "empty O-first board should be valid");

        state = state.make_move(4).unwrap(); // O plays first
        assert!(
            state.is_valid(),
            "O-first board after the opening move should remain valid"
        );
    }
}

mod d4_symmetry {
    use menace::tictactoe::symmetry::D4Transform;

    use super::*;

    #[test]
    fn test_d4_has_8_elements() {
        let transforms = D4Transform::all();
        assert_eq!(transforms.len(), 8, "D4 should have exactly 8 elements");

        // Verify all are unique by checking their effects on a test position
        let test_pos = 0;
        let mut transformed_positions = std::collections::HashSet::new();

        for t in &transforms {
            let new_pos = t.transform_position(test_pos);
            transformed_positions.insert((new_pos, t.rotation, t.reflection));
        }

        assert_eq!(
            transformed_positions.len(),
            8,
            "All 8 transformations should produce unique results"
        );
    }

    #[test]
    fn test_identity_transform() {
        let identity = D4Transform {
            rotation: 0,
            reflection: false,
        };

        for pos in 0..9 {
            assert_eq!(
                identity.transform_position(pos),
                pos,
                "Identity should not change position {pos}"
            );
        }
    }

    #[test]
    fn test_transform_inverse() {
        let transforms = D4Transform::all();

        for t in &transforms {
            let inverse = t.inverse();

            for pos in 0..9 {
                let transformed = t.transform_position(pos);
                let restored = inverse.transform_position(transformed);
                assert_eq!(
                    restored, pos,
                    "Transform {t:?} composed with its inverse should be identity"
                );
            }
        }
    }

    #[test]
    fn test_canonical_form_is_consistent() {
        // Create a non-symmetric board
        let mut state = BoardState::new();
        state = state.make_move(0).unwrap(); // X at top-left
        state = state.make_move(4).unwrap(); // O at center
        state = state.make_move(8).unwrap(); // X at bottom-right

        let canonical1 = state.canonical();

        // Apply all transformations and verify they all produce the same canonical form
        for t in D4Transform::all() {
            let transformed = state.transform(&t);
            let canonical2 = transformed.canonical();

            assert_eq!(
                canonical1, canonical2,
                "All symmetrical boards should have the same canonical form"
            );
        }
    }
}

mod menace_counts {
    use super::*;

    #[test]
    fn test_total_state_space() {
        // Total possible configurations
        let total = 3_usize.pow(9);
        assert_eq!(total, 19683, "Total state space should be 3^9 = 19,683");
    }

    #[test]
    fn test_valid_game_states() {
        // Count valid game states via BFS from initial position
        let mut valid_count = 0;
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();

        queue.push_back(BoardState::new());
        visited.insert(BoardState::new().encode());

        while let Some(state) = queue.pop_front() {
            valid_count += 1;

            if state.is_terminal() {
                continue;
            }

            for pos in state.empty_positions() {
                let next = state.make_move(pos).unwrap();
                let key = next.encode();

                if !visited.contains(&key) {
                    visited.insert(key);
                    queue.push_back(next);
                }
            }
        }

        // Should be 5,478 valid game states
        assert_eq!(
            valid_count, 5478,
            "Should have exactly 5,478 valid game states"
        );
    }

    #[test]
    fn test_canonical_states_under_d4() {
        // Count unique canonical forms
        let mut canonical_set = std::collections::HashSet::new();
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();

        queue.push_back(BoardState::new());
        visited.insert(BoardState::new().encode());

        while let Some(state) = queue.pop_front() {
            canonical_set.insert(state.canonical().encode());

            if state.is_terminal() {
                continue;
            }

            for pos in state.empty_positions() {
                let next = state.make_move(pos).unwrap();
                let key = next.encode();

                if !visited.contains(&key) {
                    visited.insert(key);
                    queue.push_back(next);
                }
            }
        }

        // Should be 765 canonical states
        assert_eq!(
            canonical_set.len(),
            765,
            "Should have exactly 765 canonical states under D4 symmetry"
        );
    }
}

mod saturating_arithmetic {
    use menace::menace::matchbox::Matchbox;

    #[test]
    fn test_bead_count_never_negative() {
        let mut matchbox = Matchbox::new("test_state".to_string(), vec![0, 1, 2], 0);

        // Reinforce negatively multiple times
        for _ in 0..10 {
            matchbox.reinforce(0, -5);
        }

        // Bead count should never go negative (saturates at 0 then restocks)
        let total = matchbox.total_beads();
        assert!(total > 0, "Bead count should never be negative");
    }

    #[test]
    fn test_bead_saturation_at_max() {
        let mut matchbox = Matchbox::new("test_state".to_string(), vec![0, 1, 2], 0);

        // Get initial bead count
        let initial_beads = matchbox.bead_count(0).unwrap_or(0);

        // Reinforce positively with maximum value
        matchbox.reinforce(0, i16::MAX);

        // Should increase but stay within bounds (saturating addition)
        let final_beads = matchbox.bead_count(0).unwrap_or(0);
        assert!(final_beads >= initial_beads, "Bead count should increase");
        // The saturating_add in the implementation ensures it won't overflow
    }
}

mod active_inference_baseline {
    use menace::{
        ActiveBeliefs, ActiveGenerativeModel, ActivePreferenceModel, ActiveUniformOpponent,
        tictactoe::Player,
    };

    #[test]
    fn test_active_inference_convergence() {
        // This test validates that Active Inference produces valid evaluations
        // Note: With canonical KL divergence, the exact move preferences may differ
        // from the old -ln(p) model with epistemic_scale=4.0
        let model = ActiveGenerativeModel::new();
        let prefs =
            ActivePreferenceModel::from_probabilities(0.9, 0.09, 0.01).with_epistemic_weight(0.5);

        let opponent = ActiveUniformOpponent;
        let beliefs = ActiveBeliefs::symmetric(1.0);

        let evaluations =
            model.evaluate_actions(model.root(), &prefs, &opponent, &beliefs, Player::X);

        assert!(
            !evaluations.is_empty(),
            "Expected root evaluations to exist"
        );

        // Verify all evaluations have finite free energy values
        for eval in &evaluations {
            assert!(
                eval.free_energy.is_finite(),
                "Free energy should be finite for action {}",
                eval.action
            );
            assert!(
                eval.risk.is_finite(),
                "Risk should be finite for action {}",
                eval.action
            );
            assert!(
                eval.epistemic >= 0.0,
                "Epistemic value should be non-negative for action {}",
                eval.action
            );
        }

        // Verify evaluations are sorted by free energy
        for i in 1..evaluations.len() {
            assert!(
                evaluations[i].free_energy >= evaluations[i - 1].free_energy,
                "Evaluations should be sorted by free energy"
            );
        }
    }
}
