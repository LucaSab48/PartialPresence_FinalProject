import tkinter as tk
from tkinter import messagebox

from openai_auth import (
    PLACEHOLDER_API_KEY,
    load_openai_settings,
    save_openai_token,
)


class OpenAISettingsWindow:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("OpenAI Token")
        self.root.geometry("720x220")
        self.root.configure(bg="#111111")

        settings = load_openai_settings(required=False)

        wrapper = tk.Frame(self.root, bg="#111111", padx=24, pady=24)
        wrapper.pack(fill="both", expand=True)

        title = tk.Label(
            wrapper,
            text="OpenAI API Key",
            font=("Segoe UI", 16, "bold"),
            fg="#f5f5f5",
            bg="#111111",
        )
        title.pack(anchor="w", pady=(0, 18))

        self.key_var = tk.StringVar(value=str(settings.get("api_key", "")))
        self._build_field(wrapper, "API Key", self.key_var, show="*")

        hint = tk.Label(
            wrapper,
            text="Saved locally to openai_token.txt. OPENAI_API_KEY still overrides the saved key if set.",
            fg="#b9b9b9",
            bg="#111111",
            font=("Segoe UI", 10),
        )
        hint.pack(anchor="w", pady=(14, 18))

        button_row = tk.Frame(wrapper, bg="#111111")
        button_row.pack(anchor="e", fill="x")

        save_button = tk.Button(
            button_row,
            text="Save",
            command=self.save,
            font=("Segoe UI", 11, "bold"),
            bg="#e8e8e8",
            fg="#111111",
            padx=18,
            pady=8,
        )
        save_button.pack(side="right")

    def _build_field(self, parent: tk.Widget, label: str, variable: tk.StringVar, show: str | None = None) -> None:
        field_label = tk.Label(
            parent,
            text=label,
            fg="#d5d5d5",
            bg="#111111",
            font=("Segoe UI", 11),
        )
        field_label.pack(anchor="w")

        entry = tk.Entry(
            parent,
            textvariable=variable,
            font=("Consolas", 11),
            width=72,
            show=show or "",
            bg="#1c1c1c",
            fg="#f5f5f5",
            insertbackground="#f5f5f5",
            relief="flat",
        )
        entry.pack(anchor="w", fill="x", pady=(6, 14), ipady=7)

    def save(self) -> None:
        api_key = self.key_var.get().strip()

        if not api_key:
            api_key = PLACEHOLDER_API_KEY

        save_openai_token(api_key)
        messagebox.showinfo("Saved", "OpenAI token saved to openai_token.txt")


def main() -> None:
    root = tk.Tk()
    OpenAISettingsWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
