# Contributing to OpenLanguageModel (OLM)

Thank you for your interest in contributing to OLM! We welcome contributions from everyone, whether you're fixing a typo, adding extensive documentation, or implementing a new model architecture.

## Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/your-username/openlanguagemodel.git
    cd openlanguagemodel
    ```
3.  **Set up your environment**:
    We recommend using a virtual environment (venv or conda).
    ```bash
    # Create a virtual environment
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate

    # Install dependencies in editable mode with dev tools
    pip install -e .
    # Install additional dev dependencies if needed (e.g. pytest, pre-commit)
    pip install pytest pre-commit
    ```
4.  **Install pre-commit hooks**:
    This ensures your code is formatted and checked before committing.
    ```bash
    pre-commit install
    ```

## Development Workflow

1.  **Create a branch** for your feature or fix:
    ```bash
    git checkout -b feature/my-new-feature
    ```
2.  **Make your changes**.
    *   **Code Style**: We follow standard Python conventions. Please ensure your code is readable and well-documented.
    *   **Docstrings**: All public functions and classes should have Google-style docstrings.
    *   **Type Hints**: Use type hints for function arguments and return values.

3.  **Run Tests**:
    Ensure that your changes don't break existing functionality.
    ```bash
    pytest tests/
    ```
    If you are adding a new feature, please add a corresponding test case in `tests/`.

4.  **Commit your changes**:
    ```bash
    git add .
    git commit -m "feat: add support for XYZ architecture"
    ```
    We follow [Conventional Commits](https://www.conventionalcommits.org/). Common types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`.

## Project Structure

*   `src/olm/`: The core library code.
    *   `models/`: High-level model definitions (e.g., GPT, OLMo).
    *   `nn/`: Reusable neural network blocks (attention, feedforward, etc.).
    *   `data/`: Data loading and processing utilities.
    *   `train/`: Training loop implementation.
*   `tests/`: Unit and integration tests.
*   `examples/`: Example scripts and usage demos.

## Pull Request Process

1.  Push your branch to GitHub.
2.  Open a Pull Request (PR) against the `dev` branch of the original repository.
3.  Describe your changes clearly in the PR description. Link to any relevant issues.
4.  Wait for review. We may ask for changes or clarifications.
5.  Once approved, your PR will be merged!

## Adding a New Model

If you are adding a new model architecture:

1.  Create a new file in `src/olm/models/` (e.g., `mynewmodel.py`).
2.  Implement the model class, inheriting from `torch.nn.Module`.
3.  Reuse existing components from `olm.nn` wherever possible (e.g., `Attention`, `FeedForward`).
4.  Add a configuration class or dictionary if needed.
5.  Add a test in `tests/` ensuring the model can run a forward pass and backward pass.

Thank you for helping make OLM better!
