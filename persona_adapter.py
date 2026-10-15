#!/usr/bin/env python3
"""
Adapter to make the Persona CLI app compatible with PersonaMem-v2 benchmark.
This module provides a QueryLLM-compatible interface for the Persona chatbot.
"""

import hashlib
import json
import os
import subprocess
import tempfile
from typing import Any, Dict, List


class PersonaAdapter:
    """Adapter to query the Persona CLI app in a benchmark-compatible way."""

    def __init__(self, args: Dict[str, Any]):
        """
        Initialize the Persona adapter.

        Args:
            args: Configuration dict (compatible with PersonaMem-v2 format)
        """
        self.args = args
        # Absolute path to the Persona project
        self.project_root = "/Users/danborthwick/Developer/AI/Persona"
        self.verbose = args.get('verbose', False)

        # Check that the Persona app is built
        self._verify_build()

    def _verify_build(self):
        """Verify that the Persona app is built and ready."""
        dist_dir = os.path.join(self.project_root, "dist")
        if not os.path.exists(dist_dir):
            raise RuntimeError(
                f"Persona app not built. Run 'yarn build' in {self.project_root}"
            )

    def query_llm(self, messages: List[Dict[str, str]], use_history: bool = True) -> str:
        """
        Query the Persona chatbot with a conversation history.

        This method simulates a conversation by:
        1. Creating a temporary user ID
        2. Replaying the conversation history
        3. Getting the final response

        Args:
            messages: List of message dicts with 'role' and 'content'
            use_history: Whether to use conversation history (always True for Persona)

        Returns:
            The assistant's response to the last user message
        """
        if not messages:
            return ""

        # Extract the conversation turns (ignore system messages for now)
        conversation_messages = []
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')

            # Skip system messages - they'll be handled via Jev personality analysis
            if role == 'system':
                continue

            if role in ('user', 'assistant'):
                conversation_messages.append({'role': role, 'content': content})

        if not conversation_messages:
            return ""

        # Get the final user message
        final_message = conversation_messages[-1]['content']

        # Build conversation history (all messages except the last one)
        history = conversation_messages[:-1] if len(conversation_messages) > 1 else []

        # Create a unique user ID for this evaluation
        user_id = hashlib.md5(str(messages).encode()).hexdigest()[:12]

        try:
            # Run the Persona CLI in non-interactive mode
            response = self._run_persona_chat(user_id, history, final_message)
            return response
        except Exception as e:
            print(f"Error querying Persona: {e}")
            raise

    def _run_persona_chat(self, user_id: str, history: List[Dict[str, str]], final_message: str) -> str:
        """
        Run the Persona CLI app with a given conversation.

        Args:
            user_id: Unique identifier for this conversation
            history: Previous messages in the conversation (dicts with 'role' and 'content')
            final_message: The final message to respond to

        Returns:
            The assistant's response
        """
        # Create stdin input lines for the CLI
        lines = []

        # Switch to the user
        lines.append(f"/user {user_id}")

        # Add history messages (each as separate lines, not as /history command)
        for msg in history:
            lines.append(json.dumps(msg))

        # Send the final message
        lines.append(final_message)

        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            input_file = f.name
            f.write('\n'.join(lines))

        try:
            # Run the Persona CLI
            env = os.environ.copy()
            # Disable debug mode for cleaner output during benchmark
            env['DEBUG'] = 'false'
            env['LOG_LEVEL'] = 'error'

            result = subprocess.run(
                ['node', 'dist/src/index.js', '--passthrough'],
                cwd=self.project_root,
                stdin=open(input_file, 'r'),
                capture_output=True,
                text=True,
                env=env,
                timeout=600  # Timeout per query
            )

            if result.returncode != 0:
                raise RuntimeError(f"Persona CLI failed: {result.stderr}")

            # Parse the output to extract the final assistant response
            response = self._extract_final_response(result.stdout)
            return response

        finally:
            # Clean up temp file
            os.unlink(input_file)

    def _extract_final_response(self, cli_output: str) -> str:
        """
        Extract the final assistant response from CLI output.

        The CLI outputs lines like:
        You: <message>
        Assistant: <response>

        We need to extract the last Assistant response.

        Args:
            cli_output: The raw output from the CLI

        Returns:
            The last assistant response
        """
        lines = cli_output.split('\n')

        # Find all "Assistant:" lines
        assistant_responses = []
        for line in lines:
            if line.strip().startswith('Assistant:'):
                # Extract everything after "Assistant: "
                response = line.strip()[len('Assistant:'):].strip()
                if response:
                    assistant_responses.append(response)

        if not assistant_responses:
            # Fallback: return empty or raise error
            if self.verbose:
                print(f"Warning: No assistant response found in output:\n{cli_output}")
            return ""

        # Return the last response
        return assistant_responses[-1]

    def cleanup_caches(self):
        """No-op for Persona adapter (included for compatibility)."""
        pass
