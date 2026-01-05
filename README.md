# Project Setup
Project initializes using `uv`.

1. Install UV

    ```
    sudo snap install astral-uv --classic
    ```

2. Initialize in the project directory.

    ```
    uv init
    ```

3. Initialize the virtual environment.

    ```
    uv venv
    source .venv/bin/activate
    ```

3. Run some tests.

    ```
    uv run pytest
    ```