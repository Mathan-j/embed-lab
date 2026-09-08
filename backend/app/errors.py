class StageError(Exception):
    """A user-fixable input error (empty text, k out of range, an uncached model).
    Renders as 409 with `detail` + `hint`, never as an unhandled 500 stack trace."""

    def __init__(self, detail: str, hint: str):
        self.detail = detail
        self.hint = hint
        super().__init__(f"{detail} — {hint}")
