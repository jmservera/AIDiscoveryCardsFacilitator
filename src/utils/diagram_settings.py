"""
UI component for Mermaid diagram settings

This file provides a Streamlit component for the sidebar that allows users
to adjust Mermaid diagram settings.
"""

import streamlit as st

from utils.openai_utils import get_diagram_scale_factor, set_diagram_scale_factor


def render_diagram_settings_sidebar() -> None:
    """Render the diagram settings UI in the sidebar.

    This function should be called from the main app when rendering the sidebar.
    """
    with st.sidebar.expander("Diagram Settings", expanded=False):
        scale_factor = st.slider(
            "Diagram Size",
            min_value=0.5,
            max_value=2.0,
            value=get_diagram_scale_factor(),
            step=0.1,
            help="Adjust the size of Mermaid diagrams (1.0 is the default size)",
        )
        set_diagram_scale_factor(scale_factor)

        st.caption("Changes apply to newly rendered diagrams.")
