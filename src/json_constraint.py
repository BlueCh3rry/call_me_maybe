from enum import Enum
import json


class State(Enum):
    ROOT_OPEN = 1
    ROOT_KEY = 2
    ROOT_COLON = 3
    FUNCTION_NAME = 4
    PARAMETERS_KEY = 5
    PARAMETERS_OPEN = 6
    PARAMETER_KEY = 7
    PARAMETER_COLON = 8
    PARAMETER_VALUE = 9
    PARAMETERS_CLOSE = 10
    ROOT_CLOSE = 11
    COMPLETE = 12

@dataclass
class DecoderState:
    state: State
    function_name: str | None = None
    parameter_name: str | None = None
    generated: str = ""


def load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: file not found: {path}")
        return None
    except json.JSONDecodeError as error:
        print(f"Error: invalid JSON in {path}")
        print(error)
        return None


def allowed_characters(state: State) -> set[str]:
    if state == State.ROOT_OPEN:
        return {"{"}
    if state == State.ROOT_KEY:
        return {'"'}
    if state == State.ROOT_COLON:
        return {":"}
    if state == State.FUNCTION_NAME:
        return {'"'}
    if state == State.PARAMETERS_KEY:
        return {'"'}
    if state == State.PARAMETERS_OPEN:
        return {"{"}
    if state == State.PARAMETER_KEY:
        return {'"'}
    if state == State.PARAMETER_COLON:
        return {":"}
    if state == State.PARAMETER_VALUE:
        return set()
    if state == State.PARAMETERS_CLOSE:
        return {"}"}
    if state == State.ROOT_CLOSE:
        return {"}"}
    if state == State.COMPLETE:
        return set()

    return set()