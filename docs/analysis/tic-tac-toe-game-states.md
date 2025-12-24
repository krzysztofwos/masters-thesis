# Tic-Tac-Toe State Space Analysis: The Mathematics of 287, 304, and 338

## Abstract

Donald Michie's 1963 paper states that MENACE contained "every one of the 287 essentially distinct positions which the opening player can encounter," with one matchbox per position. This document provides the precise combinatorial derivation of that figure and clarifies related state counts that appear in the literature.

## Summary of State Counts

| Count | Description                                                      |
| ----- | ---------------------------------------------------------------- |
| 338   | All canonical X-to-move states (including forced final moves)    |
| 304   | Decision points only (excluding forced single-move positions)    |
| 287   | Michie's count (excluding forced moves and double-threat losses) |

## Definition of Essential Distinctness

Michie identifies board positions up to the 8 geometrical symmetries of the 3×3 grid (the dihedral group D₄: 4 rotations + 4 reflections). Two boards that are rotations or reflections of each other count as one position. This is why each matchbox face shows a single "standard" orientation.

## Enumeration of Opening Player Choice-Points

MENACE plays as the opening player and requires a matchbox only when it faces a genuine decision (at least two legal moves). Enumerating the legal game tree (stopping as soon as a win occurs), reducing by symmetry, and considering only positions where the opener is to move (before moves 1, 3, 5, 7) yields:

| Decision Point  | Ply | Positions |
| --------------- | --- | --------- |
| Before 1st move | 0   | 1         |
| Before 3rd move | 2   | 12        |
| Before 5th move | 4   | 108       |
| Before 7th move | 6   | 183       |
| **Total**       |     | **304**   |

These four rows sum to 304. This breakdown (1, 12, 108, 183 = 304) is widely reproduced in the literature, including by Rodney Brooks.

### Exclusion of 9th Move Positions

With eight marks already placed, there is only one empty square, so there is no decision to make. MENACE does not require a matchbox for such forced final moves. Including these positions while reducing by symmetry would add 34 more positions, reaching 338, but these are not choice-points.

## The Final Subtraction: From 304 to 287

Among the 183 "before 7th-move" positions, there are 17 symmetry-classes where the opponent already has two distinct immediate winning threats (a true fork) and the opener has no immediate winning move. In these states, all legal moves lose on the next turn—nothing MENACE does can change the outcome.

Modern analyses count exactly 17 such inevitable-loss classes. Subtracting these yields:

$$\underbrace{1 + 12 + 108 + 183}_{\text{all choice-points}} \;-\; \underbrace{17}_{\text{inevitable-loss classes}} \;=\; \boxed{287}$$

Equivalently, from the full set of opener-to-move positions up to symmetry (338): removing 34 "no-choice" last-move states and the 17 inevitable-loss states yields 287.

## Explanation of Variant Counts in Literature

Different sources report different counts:

- **304**: Most reconstructions include all genuine decision points (even those doomed ones), hence 1 + 12 + 108 + 183 = 304 matchboxes. This is now the standard build count in tutorials and implementations.
- **287**: Michie's 1963 paper is consistent with excluding the 17 "no-escape" cases as not worth separate boxes, yielding his reported figure.
- **288**: In later writings Michie sometimes mentioned 288; the literature is not fully consistent on exactly which fringe cases were included. Brooks notes this discrepancy and that most modern implementations settle on 304.

## Interpretation

Michie's 287 represents all opener-to-move, symmetry-unique decision positions except:

1. The trivial last-move cases (34 positions)
2. The 17 positions where loss is already forced next turn

This yields one matchbox per usefully distinct situation the opening player can face during play.

## The 17 Double-Threat Classes

The following table enumerates the 17 symmetry-classes of double-threat positions. These are positions at ply 6 where:

- It is X's turn to move
- X has no immediate winning move
- O has two or more distinct immediate winning moves

Board notation: rows shown as row1/row2/row3, with X = opener, O = opponent, · = empty. Position indices are row-major 0-8:

```
0 1 2
3 4 5
6 7 8
```

| ID  | Canonical Key | Board (X=opener, O=opponent, ·=empty) | O Immediate Wins (indices) |
| --- | ------------- | ------------------------------------- | -------------------------- |
| 01  | `...XOXOXO`   | `OXO/XOX/···`                         | [6, 8]                     |
| 02  | `..OO.XOXX`   | `XXO/X·O/O··`                         | [4, 8]                     |
| 03  | `..OXOX.OX`   | `XO·/XOX/O··`                         | [2, 7]                     |
| 04  | `..OXOX.XO`   | `OXO/XO·/·X·`                         | [6, 8]                     |
| 05  | `..X.OOXOX`   | `XOX/OO·/X··`                         | [5, 7]                     |
| 06  | `..X.OOXXO`   | `XOO/·OX/··X`                         | [6, 7]                     |
| 07  | `..XO.OOXX`   | `XOX/X··/OO·`                         | [4, 8]                     |
| 08  | `..XOO.XXO`   | `XO·/XO·/O·X`                         | [2, 7]                     |
| 09  | `..XOXXO.O`   | `XXO/·X·/·OO`                         | [5, 6]                     |
| 10  | `..XXOX.OO`   | `XXO/·OO/·X·`                         | [3, 6, 8]                  |
| 11  | `..XXOXO.O`   | `XXO/·O·/·XO`                         | [5, 6]                     |
| 12  | `.OOX..OXX`   | `XXO/··X/OO·`                         | [4, 8]                     |
| 13  | `.OOXOX.X.`   | `OX·/OOX/·X·`                         | [6, 8]                     |
| 14  | `.OOXXO.X.`   | `·X·/OXX/OO·`                         | [0, 8]                     |
| 15  | `.X.XOXO.O`   | `OX·/·OX/OX·`                         | [2, 3, 8]                  |
| 16  | `O.O..XOXX`   | `XXO/X··/O·O`                         | [4, 5, 7]                  |
| 17  | `O.O..XXXO`   | `XXO/··X/O·O`                         | [4, 7]                     |

Each line represents one canonical representative of its symmetry-class. Applying rotations and reflections generates all raw positions within the same class.

## Implementation Reference

The MENACE implementation verifies these counts programmatically. The state enumeration proceeds as follows:

1. Enumerate the full legal game tree from the empty board, stopping immediately at wins
2. Reduce by symmetry (all 8 dihedral symmetries), using the lexicographically smallest image as the class key
3. Count opener-to-move positions at plies 0, 2, 4, 6, 8 (before moves 1, 3, 5, 7, 9)
4. Flag "double-threat" states for the opener at ply 6 by checking:
   - The opener has no immediate winning move
   - The opponent has two or more distinct immediate winning moves

The implementation produces:

```json
{
  "total_reachable_positions_raw": 5478,
  "xturn_essential_by_ply": { "0": 1, "2": 12, "4": 108, "6": 183, "8": 34 },
  "sum_ply_0_2_4_6": 304,
  "num_forced_loss_classes_at_ply6": 17,
  "derived_287": 287,
  "num_ply8_nochoice_classes": 34,
  "total_xturn_essential_any_ply": 338
}
```

## Implementation Notes

- Board encoding uses a 9-character tuple of X, O, or empty; parity determines whose turn it is (X starts)
- Symmetry group is the full D₄ (4 rotations, 4 reflections)
- "Essentially distinct" means "up to D₄ symmetry"; the class key is the minimal image
- The 34 "no-choice" classes at ply 8 correspond to positions with exactly one legal move

## Mathematical Summary

The count of 287 is computed as:

1. Count all symmetry-unique opener-to-move positions before moves 1, 3, 5, 7: 1 + 12 + 108 + 183 = 304
2. Exclude classes where loss next turn is already forced (opponent has two distinct immediate wins; opener has no immediate win): 17 classes
3. Result: 304 − 17 = 287

This matches Michie's statement about MENACE's matchboxes.

## References

- Michie, D. (1963). Experiments on the Mechanization of Game-Learning Part I. Characterization of the Model and its Parameters. The Computer Journal, 6(3), 232-236.
- Brooks, R. Analysis of MENACE state counts.
