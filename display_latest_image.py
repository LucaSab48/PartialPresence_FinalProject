"""Display the newest generated image fullscreen with a live inference text overlay."""

import json
import sys
from pathlib import Path
import tkinter as tk

from PIL import Image, ImageDraw, ImageFont, ImageTk


IMAGE_FOLDER = Path(__file__).resolve().parent / "generated_images"
STATE_FILE = IMAGE_FOLDER / "current_interpretation_state.json"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
POLL_INTERVAL_MS = 700

# Overlay controls for quick iteration during prototyping.
SHOW_OVERLAY_TITLE = True
OVERLAY_TITLE = "LIVE INFERENCE"
OVERLAY_FONT_SIZE = 18
OVERLAY_TITLE_SIZE = 16
OVERLAY_LINE_SPACING = 12
OVERLAY_MARGIN_X = 70
OVERLAY_MARGIN_Y = 70
OVERLAY_POSITION = "bottom_left"  # top_left, top_right, bottom_left, bottom_right
OVERLAY_TEXT_COLOR = "#f2f2f2"
OVERLAY_TITLE_COLOR = "#b8b8b8"
OVERLAY_BOX_FILL = "#050505"
OVERLAY_BOX_ALPHA = 10
OVERLAY_BOX_PADDING_X = 24
OVERLAY_BOX_PADDING_Y = 18
MAX_TEXT_LINES = 5
MAX_ERROR_LINES = 2


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

        self.body_font = ImageFont.load_default()
        self.title_font = ImageFont.load_default()
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
                self.body_font = ImageFont.truetype(font_path, OVERLAY_FONT_SIZE)
                self.title_font = ImageFont.truetype(font_path, OVERLAY_TITLE_SIZE)
                return
            except OSError:
                continue

    def close(self, event=None) -> None:
        self.root.destroy()

    def check_for_updates(self) -> None:
        latest_image = self.get_latest_image()
        shared_state = load_shared_state()

        if latest_image is None:
            self.show_waiting_state(shared_state)
        else:
            image_key = self.build_image_key(latest_image, shared_state)
            if image_key != self.current_key:
                self.display_image(latest_image, image_key, shared_state)

        self.root.after(POLL_INTERVAL_MS, self.check_for_updates)

    def get_latest_image(self) -> Path | None:
        if not IMAGE_FOLDER.exists():
            if not self.missing_folder_message_shown:
                print(f"Waiting for images... Folder not found: {IMAGE_FOLDER}")
                self.missing_folder_message_shown = True
            return None

        self.missing_folder_message_shown = False

        image_files = get_sorted_image_files()
        if not image_files:
            return None

        return image_files[0]

    def show_waiting_state(self, shared_state: dict | None) -> None:
        state_key = self.build_state_key(shared_state)
        waiting_key = ("waiting", state_key)
        if self.waiting_message_shown and self.current_key == waiting_key:
            return

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        canvas = Image.new("RGB", (screen_width, screen_height), "black")
        composed = self.draw_overlay(canvas, shared_state)

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
        image_width, image_height = image.size

        scale = min(screen_width / image_width, screen_height / image_height)
        new_size = (
            max(1, int(image_width * scale)),
            max(1, int(image_height * scale)),
        )
        resized_image = image.resize(new_size, Image.Resampling.LANCZOS)

        canvas = Image.new("RGB", (screen_width, screen_height), "black")
        offset_x = (screen_width - new_size[0]) // 2
        offset_y = (screen_height - new_size[1]) // 2
        canvas.paste(resized_image, (offset_x, offset_y))

        composed = self.draw_overlay(canvas, shared_state)

        self.current_photo = ImageTk.PhotoImage(composed)
        self.label.configure(image=self.current_photo, text="")
        self.current_key = image_key
        self.waiting_message_shown = False
        print(f"Displayed: {image_path.name}")
        delete_two_generations_ago_image()

    def draw_overlay(self, image: Image.Image, shared_state: dict | None) -> Image.Image:
        draw = ImageDraw.Draw(image, "RGBA")
        lines = self.get_overlay_lines(shared_state)

        title_height = 0
        line_heights = []
        text_width = 0
        text_height = 0

        if SHOW_OVERLAY_TITLE:
            title_box = draw.textbbox((0, 0), OVERLAY_TITLE, font=self.title_font)
            title_height = title_box[3] - title_box[1]
            text_width = max(text_width, title_box[2] - title_box[0])
            text_height += title_height + OVERLAY_LINE_SPACING

        for line in lines:
            box = draw.textbbox((0, 0), line, font=self.body_font)
            line_width = box[2] - box[0]
            line_height = box[3] - box[1]
            line_heights.append(line_height)
            text_width = max(text_width, line_width)
            text_height += line_height

        if len(lines) > 1:
            text_height += OVERLAY_LINE_SPACING * (len(lines) - 1)

        box_width = int(text_width + (OVERLAY_BOX_PADDING_X * 2))
        box_height = int(text_height + (OVERLAY_BOX_PADDING_Y * 2))
        x, y = self.resolve_overlay_position(image.width, image.height, box_width, box_height)

        draw.rounded_rectangle(
            (x, y, x + box_width, y + box_height),
            radius=18,
            fill=self.hex_to_rgba(OVERLAY_BOX_FILL, OVERLAY_BOX_ALPHA),
        )

        text_x = x + OVERLAY_BOX_PADDING_X
        text_y = y + OVERLAY_BOX_PADDING_Y
        if SHOW_OVERLAY_TITLE:
            draw.text((text_x, text_y), OVERLAY_TITLE, font=self.title_font, fill=OVERLAY_TITLE_COLOR)
            text_y += title_height + OVERLAY_LINE_SPACING

        for index, line in enumerate(lines):
            draw.text((text_x, text_y), line, font=self.body_font, fill=OVERLAY_TEXT_COLOR)
            text_y += line_heights[index] + OVERLAY_LINE_SPACING

        return image

    def get_overlay_lines(self, shared_state: dict | None) -> list[str]:
        lines = []
        if shared_state:
            lines = list(shared_state.get("live_inference_lines", []))[:MAX_TEXT_LINES]
            last_image_error = shared_state.get("last_image_error")
            if last_image_error:
                error_text = str(last_image_error).replace("\n", " ").strip()
                if len(error_text) > 140:
                    error_text = f"{error_text[:137]}..."
                lines.extend(
                    [f"image generation error: {error_text}"][:MAX_ERROR_LINES]
                )

        if not lines:
            return ["waiting for interpreted sensor state"]
        return lines

    def resolve_overlay_position(self, image_width: int, image_height: int, box_width: int, box_height: int) -> tuple[int, int]:
        if OVERLAY_POSITION == "top_left":
            return OVERLAY_MARGIN_X, OVERLAY_MARGIN_Y
        if OVERLAY_POSITION == "top_right":
            return image_width - box_width - OVERLAY_MARGIN_X, OVERLAY_MARGIN_Y
        if OVERLAY_POSITION == "bottom_right":
            return image_width - box_width - OVERLAY_MARGIN_X, image_height - box_height - OVERLAY_MARGIN_Y
        return OVERLAY_MARGIN_X, image_height - box_height - OVERLAY_MARGIN_Y

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
