"""Implements with_streamlit_context.

Created by Wessel Valkenburg, 2024-03-27.
Modified by Juan Manuel Servera, 2025-06-01
"""

from typing import Any, Callable, TypeVar, cast

from streamlit.errors import NoSessionContext
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

T = TypeVar("T", bound=Callable[..., Any])


def with_streamlit_context(fn: T) -> T:
    """
    Decorator to fix Streamlit NoSessionContext errors in threaded environments.

    This decorator ensures that the Streamlit script run context is properly
    propagated to functions that might be called in different threads or contexts,
    preventing NoSessionContext errors.

    Args:
        fn: The function to wrap with Streamlit context

    Returns:
        T: The wrapped function with Streamlit context management

    Raises:
        NoSessionContext: If called outside of a Streamlit context
    """
    ctx = get_script_run_ctx()

    if ctx is None:
        raise NoSessionContext(
            "with_streamlit_context must be called inside a context; "
            "construct your function on the fly, not earlier."
        )

    def _cb(*args: Any, **kwargs: Any) -> Any:
        """
        Wrapper function that sets Streamlit context before calling the original function.

        Args:
            *args: Positional arguments to pass to the wrapped function
            **kwargs: Keyword arguments to pass to the wrapped function

        Returns:
            Any: The return value from the wrapped function
        """

        add_script_run_ctx(ctx=ctx)

        # Call the callback.
        ret = fn(*args, **kwargs)
        return ret

    return cast(T, _cb)
