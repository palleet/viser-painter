from __future__ import annotations

import dataclasses
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
)

from ._messages import GuiAddMessage, GuiRemoveMessage, GuiSetMessage

if TYPE_CHECKING:
    from ._message_api import MessageApi
    from ._server import ClientId


T = TypeVar("T")


@dataclasses.dataclass
class _GuiHandleState(Generic[T]):
    """Internal API for GUI elements."""

    name: str
    typ: Type[T]
    api: MessageApi
    value: T
    last_updated: float

    folder_label: str
    """Name of the folder this GUI input was placed into."""

    update_cb: List[Callable[[GuiHandle[T]], None]]
    """Registered functions to call when this input is updated."""

    leva_conf: Dict[str, Any]
    """Input config for Leva."""

    is_button: bool
    """Indicates a button element, which requires special handling."""

    sync_cb: Optional[Callable[[ClientId, T], None]] = None
    """Callback for synchronizing inputs across clients."""

    cleanup_cb: Optional[Callable[[], Any]] = None
    """Function to call when GUI element is removed."""


@dataclasses.dataclass(frozen=True)
class GuiHandle(Generic[T]):
    """Handle for a particular GUI input in our visualizer.

    Lets us get values, set values, and detect updates."""

    # Let's shove private implementation details in here...
    _impl: _GuiHandleState[T]

    def on_update(
        self, func: Callable[[GuiHandle[T]], None]
    ) -> Callable[[GuiHandle[T]], None]:
        """Attach a function to call when a GUI input is updated. Happens in a thread.

        Callbacks are passed the originating GUI handle, which can be useful in loops.
        """
        self._impl.update_cb.append(func)
        return func

    def value(self) -> T:
        return self._impl.value

    def last_updated(self) -> float:
        return self._impl.last_updated

    def set_value(self, value: T) -> None:
        if not self._impl.is_button:
            self._impl.api._queue(GuiSetMessage(self._impl.name, value))
        self._impl.value = value
        self._impl.last_updated = time.time()

    def set_disabled(self, disabled: bool) -> None:
        if self._impl.is_button:
            self._impl.api._queue(
                GuiAddMessage(
                    self._impl.name,
                    self._impl.folder_label,
                    {**self._impl.leva_conf, "settings": {"disabled": disabled}},
                ),
            )
        else:
            self._impl.api._queue(
                GuiAddMessage(
                    self._impl.name,
                    self._impl.folder_label,
                    {**self._impl.leva_conf, "disabled": disabled},
                ),
            )

    def remove(self) -> None:
        """Remove this GUI element from the visualizer."""
        self._impl.api._queue(GuiRemoveMessage(self._impl.name))
        assert self._impl.cleanup_cb is not None
        self._impl.cleanup_cb()
