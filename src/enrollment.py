from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
import datetime
import math
import os
import shutil

import cv2

from camera import open_capture, prompt_camera_source
from config import DEFAULT_CONFIG, AppConfig
from data_policy import ensure_private_directory, atomic_output, _safe_path, safe_file_path


ENROLLMENTS_DIR = DEFAULT_CONFIG.enrollments_dir
PAGE_SIZE = 5
INVALID_LABEL_CHARS = set('<>:"/\\|?*')
MAX_LABEL_LENGTH = 80
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}
RED_TEXT = "\033[91m"
RESET_TEXT = "\033[0m"


class MenuState(Enum):
    MENU = auto()
    HELP = auto()
    ENROLL_GET_NAME = auto()
    ENROLL_DUPLICATE = auto()
    ENROLL_GET_COUNT = auto()
    ENROLL_CAPTURE = auto()
    ENROLL_COMPLETE = auto()
    ENROLL_ABORT = auto()
    DELETE_CHOOSE = auto()
    DELETE_CONFIRM = auto()
    DELETE_COMPLETE = auto()
    DELETE_ABORT = auto()
    DETECT = auto()


ENROLLMENT_STATES = {
    MenuState.ENROLL_GET_NAME,
    MenuState.ENROLL_DUPLICATE,
    MenuState.ENROLL_GET_COUNT,
    MenuState.ENROLL_CAPTURE,
    MenuState.ENROLL_COMPLETE,
    MenuState.ENROLL_ABORT,
}

DELETE_STATES = {
    MenuState.DELETE_CHOOSE,
    MenuState.DELETE_CONFIRM,
    MenuState.DELETE_COMPLETE,
    MenuState.DELETE_ABORT,
}


@dataclass
class EnrollmentSession:
    label: str = ""
    folder: Path | None = None
    target_image_count: int = 0
    saved_count: int = 0

    def reset(self) -> None:
        self.label = ""
        self.folder = None
        self.target_image_count = 0
        self.saved_count = 0


@dataclass
class DeleteSession:
    names: list[str]
    page_number: int = 0
    selected_name: str = ""


def enrollment_folders(enrollments_dir: Path = ENROLLMENTS_DIR) -> list[str]:
    ensure_private_directory(enrollments_dir)
    return sorted(_safe_path(path).name for path in enrollments_dir.iterdir() if path.is_dir())


def sanitize_enrollment_label(value: str) -> str:
    label = (
        "".join(
            char for char in value.strip() if char not in INVALID_LABEL_CHARS and ord(char) >= 32
        )
        .strip(". ")[:MAX_LABEL_LENGTH]
        .strip(". ")
    )
    if not label or label.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES:
        return ""
    return label


def clear_terminal() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def prompt_user(error_message: str = "") -> str:
    if error_message:
        print(f"{RED_TEXT}{error_message}{RESET_TEXT}")
    return input(">> ").strip()


def display_state(state: MenuState, delete_session: DeleteSession) -> None:
    clear_terminal()
    if state == MenuState.MENU:
        print("Integrated Live Demo")
        print("Type 'detect' to start live monitoring.")
        print("Commands: detect | enroll | delete | export | doctor | help | q")
    elif state == MenuState.HELP:
        print("----- List of Commands -----")
        print("[q] - Quit the program")
        print("[help] - Access the help screen")
        print("[menu] - Go back to the main menu")
        print("[enroll] - Enroll a person into the database")
        print("[delete] - Delete a currently enrolled person")
        print("[detect] - Start live monitoring")
        print("[export] - Process supported files from the input folder")
        print("[doctor] - Check dependencies and model readiness")
    elif state == MenuState.ENROLL_GET_NAME:
        print("----- Enter a name for the enrolled person -----")
        print("Enter 'exit' to abort enrolling")
    elif state == MenuState.ENROLL_DUPLICATE:
        print("That person already exists. Overwrite or add more photos?")
        print("[a] - Overwrite")
        print("[b] - Add more photos")
        print("Enter 'exit' to abort enrolling")
    elif state == MenuState.ENROLL_GET_COUNT:
        print("----- Enter the number of pictures you want to take -----")
    elif state == MenuState.ENROLL_CAPTURE:
        print("Opening camera...")
    elif state == MenuState.ENROLL_COMPLETE:
        print("----- Enrollment is complete. Enter 'exit' to return to the main menu -----")
    elif state == MenuState.ENROLL_ABORT:
        print("----- Enrollment aborted -----")
        print("Do you want to resume or exit?")
        print("[a] - Resume")
        print("[b] - Exit")
    elif state == MenuState.DELETE_CHOOSE:
        print("----- Choose an enrollment to delete -----")
        print("Enter 'exit' to return to main menu")
        display_delete_page(delete_session)
    elif state == MenuState.DELETE_CONFIRM:
        print(f"Delete '{delete_session.selected_name}'?")
        print("[y/n] - yes/no")
    elif state == MenuState.DELETE_COMPLETE:
        print("----- Successfully deleted -----")
        print("Enter 'exit' to return to main menu")
    elif state == MenuState.DELETE_ABORT:
        print("----- Delete aborted -----")
        print("Enter 'exit' to return to main menu")


def display_delete_page(delete_session: DeleteSession) -> None:
    if not delete_session.names:
        print("No enrolled person. Enroll someone first.")
        return

    offset = delete_session.page_number * PAGE_SIZE
    for index, name in enumerate(delete_session.names[offset : offset + PAGE_SIZE], start=offset):
        print(f"[{index}] - {name}")

    if len(delete_session.names) > PAGE_SIZE:
        print("----- [n/p] - next/prev -----")


def capture_enrollment_images(
    session: EnrollmentSession,
    config: AppConfig | None = None,
) -> MenuState:
    if session.folder is None:
        raise RuntimeError("Enrollment folder has not been selected.")

    runtime_config = config or AppConfig.from_env()
    if not runtime_config.identity_enabled:
        print("Enrollment is disabled while identity recognition is off.")
        return MenuState.ENROLL_ABORT
    quality_identifier = _create_enrollment_identifier(runtime_config)
    if quality_identifier is None and not runtime_config.allow_unchecked_enrollment:
        print("Enrollment aborted: face-quality validation is unavailable.")
        return MenuState.ENROLL_ABORT
    ensure_private_directory(session.folder)
    camera_source = prompt_camera_source()
    capture = open_capture(camera_source)
    if not capture.isOpened():
        print(f"Could not open camera source {camera_source}.")
        input("Press Enter to continue...")
        return MenuState.MENU

    count = session.saved_count
    quality_message = "Use varied angles and neutral lighting"
    quality_color = (0, 180, 0)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Failed to grab frame. Returning to menu.")
                input("Press Enter to continue...")
                return MenuState.MENU

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s") and count < session.target_image_count:
                accepted, quality_message = validate_enrollment_frame(
                    quality_identifier,
                    frame,
                    allow_unchecked=runtime_config.allow_unchecked_enrollment,
                )
                quality_color = (0, 180, 0) if accepted else (0, 0, 255)
                if accepted:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    image_path = session.folder / f"{session.label}_{count}_{timestamp}.jpg"
                    with atomic_output(image_path) as temporary:
                        saved = cv2.imwrite(str(temporary), frame)
                        if not saved:
                            raise OSError("Failed to write enrollment image")
                    if saved:
                        count += 1
                        if count >= session.target_image_count:
                            session.saved_count = 0
                            return MenuState.ENROLL_COMPLETE
                    else:
                        print("Failed to write image to disk.")
                else:
                    print(quality_message)

            cv2.putText(
                frame,
                f"Images: {count}/{session.target_image_count}. Press 's' to save, 'q' to stop",
                (10, 24),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 0, 0),
                1,
            )
            cv2.putText(
                frame,
                quality_message,
                (10, 48),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                quality_color,
                1,
            )
            cv2.imshow("Enrollment Capture", frame)

            if key == ord("q") and count < session.target_image_count:
                session.saved_count = count
                return MenuState.ENROLL_ABORT
    finally:
        capture.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def validate_enrollment_frame(
    identifier, frame, *, allow_unchecked: bool = False
) -> tuple[bool, str]:
    """Require one clear, sufficiently large face before saving enrollment data."""
    if identifier is None:
        return (
            (True, "Unchecked enrollment explicitly enabled")
            if allow_unchecked
            else (False, "Face-quality validation unavailable; enrollment rejected")
        )
    try:
        faces = identifier.detect_faces(frame)
        if not faces:
            return False, "No face found; face the camera and try again"
        if len(faces) > 1:
            return False, "More than one face found; only one person may enroll at a time"
        if not identifier.is_face_usable(faces[0]):
            return False, "Face is too small or uncertain; move closer and improve lighting"
        return True, "Face accepted; change angle slightly for the next photo"
    except Exception:
        return False, "Face-quality validation failed; enrollment rejected"


def _create_enrollment_identifier(config: AppConfig):
    try:
        from identity import OpenCVFaceIdentifier
        from model_manager import ensure_models, identity_model_specs

        ensure_models(identity_model_specs(config), config)
        return OpenCVFaceIdentifier(
            detector_model_path=config.face_detector_model_path,
            recognizer_model_path=config.face_recognizer_model_path,
            cosine_threshold=config.identity_cosine_threshold,
            min_score_margin=config.identity_min_score_margin,
            min_face_size=config.identity_min_face_size,
            min_face_confidence=config.identity_min_face_confidence,
        )
    except Exception as exc:
        print(f"Enrollment quality checks unavailable: {exc}")
        return None


def handle_enrollment_state(
    state: MenuState,
    command: str,
    session: EnrollmentSession,
    enrollments_dir: Path = ENROLLMENTS_DIR,
) -> tuple[MenuState, str]:
    normalized_command = command.strip().casefold()
    if normalized_command == "exit":
        return MenuState.MENU, ""

    if state == MenuState.ENROLL_GET_NAME:
        if normalized_command == "q":
            return state, "Enter a name for the person you want to enroll."

        label = sanitize_enrollment_label(command)
        if not label:
            return state, "Enter a valid name using letters or numbers."

        session.label = label
        session.folder = enrollments_dir / label
        _safe_path(session.folder)
        if not session.folder.exists():
            ensure_private_directory(session.folder)
            return MenuState.ENROLL_GET_COUNT, ""
        return MenuState.ENROLL_DUPLICATE, ""

    if state == MenuState.ENROLL_DUPLICATE:
        if session.folder is None:
            return MenuState.MENU, "Enrollment folder is missing. Returning to menu."

        if normalized_command == "a":
            try:
                _remove_enrollment_folder(session.folder, enrollments_dir)
            except (OSError, ValueError) as exc:
                return state, f"Could not safely overwrite enrollment: {exc}"
            ensure_private_directory(session.folder)
        elif normalized_command != "b":
            return state, "Invalid command. Choose 'a' or 'b'."
        return MenuState.ENROLL_GET_COUNT, ""

    if state == MenuState.ENROLL_GET_COUNT:
        try:
            target_count = int(command)
        except ValueError:
            return state, "Invalid input. Please enter an integer number."

        if target_count <= 0:
            return state, "Enter a value greater than 0."

        session.target_image_count = target_count
        return MenuState.ENROLL_CAPTURE, ""

    if state == MenuState.ENROLL_ABORT:
        return (MenuState.ENROLL_CAPTURE if normalized_command == "a" else MenuState.MENU), ""

    return MenuState.MENU, ""


def handle_delete_state(
    state: MenuState,
    command: str,
    delete_session: DeleteSession,
    enrollments_dir: Path = ENROLLMENTS_DIR,
) -> tuple[MenuState, str]:
    normalized_command = command.strip().casefold()
    delete_session.names = enrollment_folders(enrollments_dir)
    if normalized_command == "q":
        return state, "Enter 'exit' instead."
    if normalized_command == "exit":
        return MenuState.MENU, ""

    if state == MenuState.DELETE_CHOOSE:
        max_page = max(0, math.ceil(len(delete_session.names) / PAGE_SIZE) - 1)
        if normalized_command == "n":
            delete_session.page_number = min(max_page, delete_session.page_number + 1)
            return state, ""
        if normalized_command == "p":
            delete_session.page_number = max(0, delete_session.page_number - 1)
            return state, ""

        try:
            index = int(command)
        except ValueError:
            return state, "Input is not an integer value."

        if 0 <= index < len(delete_session.names):
            delete_session.selected_name = delete_session.names[index]
            return MenuState.DELETE_CONFIRM, ""
        return state, "Input is out of range. Choose from the provided list."

    if state == MenuState.DELETE_CONFIRM:
        if normalized_command == "y":
            selected_path = enrollments_dir / delete_session.selected_name
            if not selected_path.exists():
                return MenuState.DELETE_ABORT, "Enrollment folder no longer exists."
            try:
                _remove_enrollment_folder(selected_path, enrollments_dir)
            except (OSError, ValueError) as exc:
                return MenuState.DELETE_ABORT, f"Could not safely delete enrollment: {exc}"
            return MenuState.DELETE_COMPLETE, ""
        if normalized_command == "n":
            return MenuState.DELETE_ABORT, ""
        return state, "Invalid command. Enter either 'n' or 'y'."

    return MenuState.MENU, ""


def main(config: AppConfig | None = None) -> None:
    config = config or AppConfig.from_env()
    state = MenuState.MENU
    error = ""
    enrollment_session = EnrollmentSession()
    delete_session = DeleteSession(names=[])

    while True:
        if state == MenuState.ENROLL_CAPTURE:
            state = capture_enrollment_images(enrollment_session, config)
            continue

        if state == MenuState.DETECT:
            from detection import run_detection

            run_detection(source=prompt_camera_source(), config=config)
            state = MenuState.MENU
            continue

        display_state(state, delete_session)
        command = prompt_user(error)
        normalized_command = command.strip().casefold()
        error = ""

        if state in ENROLLMENT_STATES:
            state, error = handle_enrollment_state(
                state,
                command,
                enrollment_session,
                config.enrollments_dir,
            )
            continue

        if state in DELETE_STATES:
            state, error = handle_delete_state(
                state,
                command,
                delete_session,
                config.enrollments_dir,
            )
            continue

        if normalized_command == "q" and state in {MenuState.MENU, MenuState.HELP}:
            break
        if normalized_command == "help":
            state = MenuState.HELP
        elif normalized_command == "menu":
            state = MenuState.MENU
        elif normalized_command in {"enroll", "enrol"}:
            if not config.identity_enabled:
                error = "Enrollment is disabled while identity recognition is off."
                continue
            enrollment_session.reset()
            state = MenuState.ENROLL_GET_NAME
        elif normalized_command == "delete":
            delete_session = DeleteSession(names=enrollment_folders(config.enrollments_dir))
            state = MenuState.DELETE_CHOOSE
        elif normalized_command == "detect":
            state = MenuState.DETECT
        elif normalized_command == "doctor":
            from diagnostics import print_diagnostics, run_diagnostics

            clear_terminal()
            print_diagnostics(run_diagnostics(config))
            input("\nPress Enter to return to the menu...")
            state = MenuState.MENU
        elif normalized_command == "export":
            from media_export import export_media

            clear_terminal()
            export_media(config.input_dir, config.output_dir, config=config)
            input("\nPress Enter to return to the menu...")
            state = MenuState.MENU
        else:
            error = "Invalid command. Go to 'help' to see the list of commands."


def _remove_enrollment_folder(folder: Path, enrollments_dir: Path) -> None:
    root = _safe_path(enrollments_dir)
    resolved = _safe_path(folder)
    if resolved.parent != root:
        raise ValueError("path is outside the enrollment directory")
    for directory, directories, files in os.walk(resolved, followlinks=False):
        for name in directories:
            _safe_path(Path(directory) / name)
        for name in files:
            safe_file_path(Path(directory) / name)
    shutil.rmtree(_safe_path(folder))


if __name__ == "__main__":
    main()
