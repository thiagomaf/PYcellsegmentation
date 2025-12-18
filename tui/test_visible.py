"""Simple test to verify text visibility in Textual."""
from textual.app import App, ComposeResult
from textual.widgets import Button, Static
from textual.containers import Container, Vertical


class TestApp(App):
    """Simple test app to verify text visibility."""
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    .test-container {
        width: 50;
        height: 20;
        background: $panel;
        border: wide $primary;
        padding: 2;
    }
    
    .test-button {
        color: white;
        background: $boost;
        text-style: bold;
        margin: 1;
    }
    
    .test-static {
        color: green;
        text-style: bold;
        margin: 1;
    }
    """
    
    def compose(self) -> ComposeResult:
        with Container(classes="test-container"):
            yield Static("TEST: If you see this, Static works", classes="test-static")
            yield Button("TEST: If you see this text, Button works", classes="test-button")
            yield Static("ASCII Test:", classes="test-static")
            yield Static("╔═══╗\n║ X ║\n╚═══╝", classes="test-static")


if __name__ == "__main__":
    app = TestApp()
    app.run()








