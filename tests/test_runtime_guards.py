from __future__ import annotations

import unittest
from types import SimpleNamespace

from runtime_guards import RuntimeStageError, first_dict_item, first_tool_call, message_content


class RuntimeGuardsTests(unittest.TestCase):
    def test_message_content_raises_on_empty_choices(self) -> None:
        response = SimpleNamespace(choices=[])
        with self.assertRaises(RuntimeStageError) as ctx:
            message_content(
                response,
                context="planner",
                required=True,
                retry_trigger="auto_no_agents_retry",
            )
        self.assertEqual(ctx.exception.retry_trigger, "auto_no_agents_retry")
        self.assertIn("planner", str(ctx.exception))

    def test_first_tool_call_raises_on_empty_tool_calls(self) -> None:
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[]))]
        )
        with self.assertRaises(RuntimeStageError) as ctx:
            first_tool_call(response, context="observer", retry_trigger="auto_no_agents_retry")
        self.assertEqual(ctx.exception.retry_trigger, "auto_no_agents_retry")
        self.assertIn("observer", str(ctx.exception))

    def test_first_dict_item_raises_on_empty_list(self) -> None:
        with self.assertRaises(RuntimeStageError) as ctx:
            first_dict_item([], context="reaction_results", retry_trigger="auto_recovery_retry")
        self.assertEqual(ctx.exception.retry_trigger, "auto_recovery_retry")
        self.assertIn("reaction_results", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
