# Resources

Reference materials, primary sources, and background reading for the MENACE project.

## Structure

### `papers/`

Academic papers related to MENACE, Active Inference, reinforcement learning, and bandit problems. All files follow the naming convention `Author - Year - Title.pdf`.

**MENACE and Game Learning:**

- Michie - 1963 - Experiments on the Mechanization of Game-Learning
- Michie - 1966 - Game-playing and game-learning automata
- Fox (ed.) - 1966 - Advances in Programming and Non-Numerical Computation (contains Michie's chapter)

**Active Inference:**

- Friston et al. - 2009 - Reinforcement Learning or Active Inference
- Friston et al. - 2016 - Active inference and learning
- Friston et al. - 2017 - Active inference a process theory
- Da Costa et al. - 2020 - Active inference on discrete state-spaces A synthesis
- Parr and Friston - 2019 - Generalised free energy and active inference
- Isomura - 2022 - Active inference leads to Bayesian neurophysiology
- Isomura and Friston - 2022 - Canonical neural networks perform active inference
- Fields et al. - 2022 - A free energy principle for generic quantum systems
- Fields et al. - 2024 - Making the thermodynamic cost of active inference explicit
- Nuijten et al. - 2025 - A Message Passing Realization of Expected Free Energy Minimization

**Reinforcement Learning:**

- Bellemare et al. - 2017 - A Distributional Perspective on Reinforcement Learning

**Bandit Problems:**

- Thompson - 1933 - On the likelihood that one unknown probability exceeds another
- Auer et al. - 2002 - Finite-time Analysis of the Multiarmed Bandit Problem
- Agrawal and Goyal - 2012 - Analysis of Thompson Sampling for the multi-armed bandit problem

**Stability Theory:**

- LaSalle - 1960 - Some extensions of Liapunov's second method

### `MENACE/`

Primary source materials from Donald Michie's original work:

- `Experiments on the mechanization of game-learning.md` - Transcript of Michie's 1963 paper
- `Game-Playing and Game-Learning Automata.md` - Transcript of Michie's 1966 book chapter
- `Figure *.png`, `Table *.png` - Scanned figures and tables from the original papers

### `mscroggs-menace/`

Matthew Scroggs' JavaScript implementation of MENACE (reference implementation).
See: http://mscroggs.co.uk/menace

### Root Files

- `double_threat_positions.txt` - List of double-threat (fork) positions in tic-tac-toe
- `two-armed-bandit-problem.md` - Background reading on the bandit problem and its historical context
