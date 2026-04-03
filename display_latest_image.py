"""Display the newest generated image fullscreen with a separate live inference panel."""

import json
import sys
from pathlib import Path
import tkinter as tk

from PIL import Image, ImageDraw, ImageFont, ImageTk


IMAGE_FOLDER = Path(__file__).resolve().parent / "generated_images"
STATE_FILE = IMAGE_FOLDER / "current_interpretation_state.json"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
POLL_INTERVAL_MS = 700

# Layout controls for projector-friendly presentation.
LAYOUT_MODE = "side_panel"  # stacked or side_panel
PANEL_TITLE = "LIVE INFERENCE"
BODY_FONT_SIZE = 20
TITLE_FONT_SIZE = 15
LINE_SPACING = 12
SCREEN_MARGIN = 48
CONTENT_GAP = 8
TEXT_PANEL_RATIO = 0.26
SIDE_PANEL_RATIO = 0.28
TEXT_PANEL_FILL = "#050505"
TEXT_PANEL_BORDER = "#1a1a1a"
TEXT_PANEL_TEXT_COLOR = "#f2f2f2"
TEXT_PANEL_TITLE_COLOR = "#9d9d9d"
TEXT_PANEL_PADDING_X = 60
TEXT_PANEL_PADDING_Y = 28
TEXT_PANEL_BORDER_WIDTH = 1
IMAGE_BACKGROUND = "#000000"
MAX_TEXT_LINES = 5
MAX_ERROR_LINES = 2
MAX_WRAPPED_TEXT_LINES = 5

PillowFont = ImageFont.ImageFont | ImageFont.FreeTypeFont


def load_shared_state() -> dict | None:
    if not STATE_FILE.exists():
        return None

    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def get_sorted_image_files() -> list[Path]:
    if not IMAGE_FOLDER.exists():
        return []

    image_files = [
        path
        for path in IMAGE_FOLDER.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(image_files, key=lambda path: path.stat().st_mtime, reverse=True)


def delete_two_generations_ago_image() -> Path | None:
    image_files = get_sorted_image_files()
    if len(image_files) < 3:
        return None

    stale_image_path = image_files[2]
    stale_log_path = stale_image_path.with_suffix(".txt")

    try:
        stale_image_path.unlink()
        print(f"Deleted stale image: {stale_image_path.name}")
    except FileNotFoundError:
        return None
    except OSError as error:
        print(f"Could not delete stale image {stale_image_path.name}: {error}", file=sys.stderr)
        return None

    if stale_log_path.exists():
        try:
            stale_log_path.unlink()
            print(f"Deleted stale log: {stale_log_path.name}")
        except OSError as error:
            print(f"Could not delete stale log {stale_log_path.name}: {error}", file=sys.stderr)

    return stale_image_path


class LatestImageDisplay:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.configure(bg="black")
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", self.close)

        self.label = tk.Label(self.root, bg="black", borderwidth=0, highlightthickness=0)
        self.label.pack(fill="both", expand=True)

        self.current_key = None
        self.current_photo = None
        self.waiting_message_shown = False
        self.missing_folder_message_shown = False
        self.last_logged_selection = None

        self.body_font: PillowFont = ImageFont.load_default()
        self.title_font: PillowFont = ImageFont.load_default()
        self.root.after(0, self.load_fonts)
        self.root.after(0, self.check_for_updates)

    def load_fonts(self) -> None:
        # Prefer a Windows font if available, then fall back to Pillow's default bitmap font.
        font_candidates = [
            "C:/Windows/Fonts/consola.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
        for font_path in font_candidates:
            try:
                self.body_font = ImageFont.truetype(font_path, BODY_FONT_SIZE)
                self.title_font = ImageFont.truetype(font_path, TITLE_FONT_SIZE)
                return
            except OSError:
                continue

    def close(self, event=None) -> None:
        self.root.destroy()

    def check_for_updates(self) -> None:
        shared_state = load_shared_state()
        latest_image = self.get_latest_image(shared_state)

        if latest_image is None:
            self.show_waiting_state(shared_state)
        else:
            image_key = self.build_image_key(latest_image, shared_state)
            if image_key != self.current_key:
                self.display_image(latest_image, image_key, shared_state)

        self.root.after(POLL_INTERVAL_MS, self.check_for_updates)

    def get_latest_image(self, shared_state: dict | None) -> Path | None:
        current_image_path = None
        selected_path = None
        log_message = None
        if shared_state and not shared_state.get("current_image_path"):
            return None
        if shared_state:
            current_image_path = shared_state.get("current_image_path")
            if current_image_path:
                candidate = Path(current_image_path)
                if candidate.exists() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
                    resolved = candidate.resolve()
                    selected_path = resolved
                    log_message = f"[DISPLAY_SELECT] source=shared_state current_image_path path={resolved}"

        if selected_path is None and not IMAGE_FOLDER.exists():
            if not self.missing_folder_message_shown:
                print(f"Waiting for images... Folder not found: {IMAGE_FOLDER}")
                self.missing_folder_message_shown = True
            invalid_text = f" invalid_current_image_path={current_image_path}" if current_image_path else ""
            log_message = f"[DISPLAY_SELECT] source=none path=None reason=missing_folder{invalid_text}"

        if selected_path is None:
            self.missing_folder_message_shown = False

            image_files = get_sorted_image_files()
            if not image_files:
                invalid_text = f" invalid_current_image_path={current_image_path}" if current_image_path else ""
                log_message = f"[DISPLAY_SELECT] source=none path=None reason=no_images{invalid_text}"
            else:
                selected = image_files[0].resolve()
                selected_path = selected
                if current_image_path:
                    log_message = (
                        f"[DISPLAY_SELECT] source=fallback_newest_file path={selected} "
                        f"invalid_current_image_path={current_image_path}"
                    )
                else:
                    log_message = f"[DISPLAY_SELECT] source=fallback_newest_file path={selected}"

        if selected_path != self.last_logged_selection and log_message:
            print(log_message)
            self.last_logged_selection = selected_path

        return selected_path

    def get_current_state_image(self, shared_state: dict | None) -> Path | None:
        if shared_state:
            current_image_path = shared_state.get("current_image_path")
            if current_image_path:
                candidate = Path(current_image_path)
                if candidate.exists() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
                    return candidate
        return None

    def show_waiting_state(self, shared_state: dict | None) -> None:
        state_key = self.build_state_key(shared_state)
        waiting_key = ("waiting", state_key)
        if self.waiting_message_shown and self.current_key == waiting_key:
            return

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        canvas = Image.new("RGB", (screen_width, screen_height), "black")
        composed = self.compose_layout(canvas, None, shared_state)

        self.current_photo = ImageTk.PhotoImage(composed)
        self.label.configure(image=self.current_photo, text="")
        self.current_key = waiting_key
        self.waiting_message_shown = True
        print("Waiting for images...")

    def build_image_key(self, image_path: Path, shared_state: dict | None) -> tuple[str, int, float, str]:
        stat = image_path.stat()
        state_key = self.build_state_key(shared_state)
        return (str(image_path.resolve()), stat.st_size, stat.st_mtime, state_key)

    def build_state_key(self, shared_state: dict | None) -> str:
        if not shared_state:
            return ""

        state_snapshot = {
            "live_inference_lines": shared_state.get("live_inference_lines", []),
            "last_image_error": shared_state.get("last_image_error"),
        }
        return json.dumps(state_snapshot, ensure_ascii=True)

    def display_image(self, image_path: Path, image_key: tuple[str, int, float, str], shared_state: dict | None) -> None:
        try:
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")
        except (OSError, PermissionError) as error:
            print(f"Error displaying image: {error}")
            return

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        canvas = Image.new("RGB", (screen_width, screen_height), "black")
        composed = self.compose_layout(canvas, image, shared_state)

        self.current_photo = ImageTk.PhotoImage(composed)
        self.label.configure(image=self.current_photo, text="")
        self.current_key = image_key
        self.waiting_message_shown = False
        print(f"Displayed: {image_path.name}")
        delete_two_generations_ago_image()

    def compose_layout(
        self, canvas: Image.Image, source_image: Image.Image | None, shared_state: dict | None
    ) -> Image.Image:
        draw = ImageDraw.Draw(canvas, "RGBA")

        if LAYOUT_MODE == "side_panel":
            image_box, panel_box = self.resolve_side_panel_boxes(canvas.width, canvas.height)
        else:
            image_box, panel_box = self.resolve_stacked_boxes(canvas.width, canvas.height)

        draw.rectangle(panel_box, fill=TEXT_PANEL_FILL, outline=TEXT_PANEL_BORDER, width=TEXT_PANEL_BORDER_WIDTH)

        if source_image is not None:
            self.paste_fitted_image(canvas, source_image, image_box)
        else:
            draw.rectangle(image_box, fill=IMAGE_BACKGROUND)

        self.draw_text_panel(canvas, panel_box, shared_state)
        return canvas

    def get_overlay_lines(self, shared_state: dict | None) -> list[str]:
        lines = []
        if shared_state:
            lines = list(shared_state.get("live_inference_lines", []))[:MAX_TEXT_LINES]
            last_image_error = shared_state.get("last_image_error")
            last_openai_error = shared_state.get("last_openai_error")
            if shared_state.get("image_generation_in_progress"):
                lines.append("image refresh active")
            if last_openai_error:
                error_text = str(last_openai_error).replace("\n", " ").strip()
                if len(error_text) > 140:
                    error_text = f"{error_text[:137]}..."
                lines.extend([f"OpenAI fallback active: {error_text}"][:MAX_ERROR_LINES])
            if last_image_error:
                error_text = str(last_image_error).replace("\n", " ").strip()
                if len(error_text) > 140:
                    error_text = f"{error_text[:137]}..."
                lines.extend(
                    [f"image generation error: {error_text}"][:MAX_ERROR_LINES]
                )

        lines = [line for line in lines if line][:MAX_TEXT_LINES]
        if not lines:
            return ["waiting for interpreted sensor state"]
        return lines

    def resolve_stacked_boxes(
        self, screen_width: int, screen_height: int
    ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
        panel_height = int((screen_height - (SCREEN_MARGIN * 2) - CONTENT_GAP) * TEXT_PANEL_RATIO)
        panel_height = max(180, panel_height)
        image_height = screen_height - (SCREEN_MARGIN * 2) - CONTENT_GAP - panel_height
        image_box = (
            SCREEN_MARGIN,
            SCREEN_MARGIN,
            screen_width - SCREEN_MARGIN,
            SCREEN_MARGIN + max(1, image_height),
        )
        panel_box = (
            SCREEN_MARGIN,
            image_box[3] + CONTENT_GAP,
            screen_width - SCREEN_MARGIN,
            screen_height - SCREEN_MARGIN,
        )
        return image_box, panel_box

    def resolve_side_panel_boxes(
        self, screen_width: int, screen_height: int
    ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
        panel_width = int((screen_width - (SCREEN_MARGIN * 2) - CONTENT_GAP) * SIDE_PANEL_RATIO)
        panel_width = max(280, panel_width)
        panel_box = (
            SCREEN_MARGIN,
            SCREEN_MARGIN,
            SCREEN_MARGIN + panel_width,
            screen_height - SCREEN_MARGIN,
        )
        image_box = (
            panel_box[2] + CONTENT_GAP,
            SCREEN_MARGIN,
            screen_width - SCREEN_MARGIN,
            screen_height - SCREEN_MARGIN,
        )
        return image_box, panel_box

    def paste_fitted_image(
        self, canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int]
    ) -> None:
        box_width = max(1, box[2] - box[0])
        box_height = max(1, box[3] - box[1])
        image_width, image_height = image.size
        scale = max(box_width / image_width, box_height / image_height)
        new_size = (
            max(1, int(image_width * scale)),
            max(1, int(image_height * scale)),
        )
        resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
        offset_x = box[0] + (box_width - new_size[0]) // 2
        offset_y = box[1] + (box_height - new_size[1]) // 2
        canvas.paste(resized_image.crop((
            max(0, -offset_x),
            max(0, -offset_y),
            max(0, -offset_x) + box_width,
            max(0, -offset_y) + box_height,
        )), (box[0], box[1]))

    def draw_text_panel(
        self, canvas: Image.Image, panel_box: tuple[int, int, int, int], shared_state: dict | None
    ) -> None:
        draw = ImageDraw.Draw(canvas, "RGBA")
        text_width = max(1, panel_box[2] - panel_box[0] - (TEXT_PANEL_PADDING_X * 2))
        raw_lines = self.get_overlay_lines(shared_state)
        lines = self.wrap_text_lines(draw, raw_lines, self.body_font, text_width)
        text_x = panel_box[0] + TEXT_PANEL_PADDING_X
        text_y = panel_box[1] + TEXT_PANEL_PADDING_Y

        title_box = draw.textbbox((0, 0), PANEL_TITLE, font=self.title_font)
        title_height = title_box[3] - title_box[1]
        draw.text((text_x, text_y), PANEL_TITLE, font=self.title_font, fill=TEXT_PANEL_TITLE_COLOR)
        text_y += title_height + LINE_SPACING

        for line in lines:
            draw.text((text_x, text_y), line, font=self.body_font, fill=TEXT_PANEL_TEXT_COLOR)
            line_box = draw.textbbox((0, 0), line, font=self.body_font)
            line_height = line_box[3] - line_box[1]
            text_y += line_height + LINE_SPACING

    def wrap_text_lines(
        self, draw: ImageDraw.ImageDraw, lines: list[str], font: PillowFont, max_width: int
    ) -> list[str]:
        wrapped_lines: list[str] = []
        for line in lines:
            words = line.split()
            if not words:
                wrapped_lines.append("")
                continue

            current_line = words[0]
            for word in words[1:]:
                trial_line = f"{current_line} {word}"
                trial_box = draw.textbbox((0, 0), trial_line, font=font)
                trial_width = trial_box[2] - trial_box[0]
                if trial_width <= max_width:
                    current_line = trial_line
                else:
                    wrapped_lines.append(current_line)
                    if len(wrapped_lines) >= MAX_WRAPPED_TEXT_LINES:
                        return wrapped_lines[:MAX_WRAPPED_TEXT_LINES]
                    current_line = word
            wrapped_lines.append(current_line)
            if len(wrapped_lines) >= MAX_WRAPPED_TEXT_LINES:
                return wrapped_lines[:MAX_WRAPPED_TEXT_LINES]

        return wrapped_lines[:MAX_WRAPPED_TEXT_LINES]

    def hex_to_rgba(self, hex_value: str, alpha: int) -> tuple[int, int, int, int]:
        hex_value = hex_value.lstrip("#")
        red = int(hex_value[0:2], 16)
        green = int(hex_value[2:4], 16)
        blue = int(hex_value[4:6], 16)
        return red, green, blue, alpha


def main() -> None:
    root = tk.Tk()
    root.title("Latest Image Display")
    app = LatestImageDisplay(root)
    root.mainloop()


if __name__ == "__main__":
    main()
