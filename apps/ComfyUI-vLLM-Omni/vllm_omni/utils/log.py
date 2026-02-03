import pprint
from typing import Any


class OmitBase64PrettyPrinter(pprint.PrettyPrinter):
    """
    A PrettyPrinter that redacts specific field names with '...'
    wherever they appear in nested structures.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _format(
        self, obj: Any, stream, indent: int, allowance: int, context, level: int
    ) -> None:
        # Check if this is a dict with redacted keys
        if isinstance(obj, dict):
            # Create a copy with redacted values
            display_obj = {}
            for key, value in obj.items():
                if key == "data" or key == "url":
                    if value.startswith("data:"):
                        base64_header = value.split(",", 1)[0]
                        display_obj[key] = f"{base64_header},***"
                    elif value.startswith("http://") or value.startswith("https://"):
                        display_obj[key] = value
                    elif len(value) > 10:
                        display_obj[key] = f"{value[:10]}***"
                    else:
                        display_obj[key] = value
                else:
                    display_obj[key] = value
            obj = display_obj

        # Handle list/tuple/set containing dicts that might have redacted keys
        # (pprint will recursively call _format on nested items, so this
        # handles arbitrary nesting automatically)

        super()._format(obj, stream, indent, allowance, context, level)


pretty_printer = OmitBase64PrettyPrinter()


# ========== USAGE EXAMPLES ==========

if __name__ == "__main__":
    data = {
        "messages": [
            {
                "role": "system",
                "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What sound is it, and what is the drawing about?",
                    },
                    {
                        "type": "text",
                        "text": "What sound is it, and what is the drawing about?",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAwAAAA"
                        },
                    },
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": "data:audio/mpeg;base64,SUQzBAAAAAAAIlRTU0UAAAAOAAADT="
                        },
                    },
                ],
            },
        ],
        "extra_body": {"mm_processor_kwargs": {"use_audio_in_video": False}},
        "modalities": ["text"],
    }

    # Create printer that redacts 'password' and 'token' fields

    print("\nRedactingPrettyPrinter (hides secrets):")
    pretty_printer.pprint(data)
