#!/usr/bin/env python3
"""
Persona client that mimics the Anthropic client API.
Provides a messages.create() interface compatible with query_llm.py usage patterns.
"""

import hashlib
import json
import os
import subprocess
import tempfile
from typing import Any, Dict, List


class PersonaResponse:
    """Response object that mimics Anthropic's response structure."""

    def __init__(self, text: str):
        self.content = [type('Content', (), {'text': text})()]
        self.usage = type('Usage', (), {
            'input_tokens': 0,
            'cache_read_input_tokens': 0,
            'cache_creation_input_tokens': 0
        })()


class PersonaMessages:
    """Messages API that mimics Anthropic's messages interface."""

    def __init__(self, project_root: str, verbose: bool = False):
        self.project_root = project_root
        self.verbose = verbose
        self._verify_build()

    def _verify_build(self):
        """Verify that the Persona app is built and ready."""
        dist_dir = os.path.join(self.project_root, "dist")
        if not os.path.exists(dist_dir):
            raise RuntimeError(
                f"Persona app not built. Run 'yarn build' in {self.project_root}"
            )

    def create(self, model: str, max_tokens: int, messages: List[Dict[str, Any]]) -> PersonaResponse:
        """
        Create a message completion (mimics anthropic.messages.create).

        Args:
            model: Model name (ignored, Persona uses its own model)
            max_tokens: Max tokens (ignored, Persona uses its own limits)
            messages: List of message dicts with 'role' and 'content'

        Returns:
            PersonaResponse object with text content
        """
        # Filter out system messages and extract conversation
        conversation_messages = []
        for msg in messages:
            role = msg.get('role', '')
            content_raw = msg.get('content', '')

            # Extract text from content (handle both string and list formats)
            if isinstance(content_raw, str):
                content = content_raw
            elif isinstance(content_raw, list):
                # Extract text from structured content (handle cache_control format)
                text_parts = []
                for item in content_raw:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                content = ' '.join(text_parts)
            else:
                content = ''

            # Skip system messages (handled via Jev personality analysis)
            if role == 'system':
                continue

            if role in ('user', 'assistant') and content:
                conversation_messages.append({'role': role, 'content': content})

        if not conversation_messages:
            return PersonaResponse("")

        # Get final message and history
        final_message = conversation_messages[-1]['content']
        history = conversation_messages[:-1] if len(conversation_messages) > 1 else []

        # Create unique user ID for this conversation
        user_id = hashlib.md5(str(messages).encode()).hexdigest()[:12]

        # Run Persona chat
        response_text = self._run_persona_chat(user_id, history, final_message)
        return PersonaResponse(response_text)

    def _run_persona_chat(self, user_id: str, history: List[Dict[str, str]], final_message: str) -> str:
        """
        Run the Persona CLI app with a given conversation.

        Args:
            user_id: Unique identifier for this conversation
            history: Previous messages in the conversation
            final_message: The final message to respond to

        Returns:
            The assistant's response
        """
        # Create stdin input lines
        lines = [f"/user {user_id}"]

        # Add history messages
        for msg in history:
            lines.append(json.dumps(msg))

        # Add final message
        lines.append(final_message)

        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            input_file = f.name
            f.write('\n'.join(lines))

        try:
            # Run Persona CLI
            env = os.environ.copy()
            env['DEBUG'] = 'false'
            env['LOG_LEVEL'] = 'error'

            result = subprocess.run(
                ['node', 'dist/src/index.js', '--passthrough'],
                cwd=self.project_root,
                stdin=open(input_file, 'r'),
                capture_output=True,
                text=True,
                env=env,
                timeout=600
            )

            if result.returncode != 0:
                raise RuntimeError(f"Persona CLI failed: {result.stderr}")

            # Extract final response from output
            return self._extract_final_response(result.stdout)

        finally:
            os.unlink(input_file)

    def _extract_final_response(self, cli_output: str) -> str:
        """
        Extract the final assistant response from CLI output.

        Args:
            cli_output: Raw output from the CLI

        Returns:
            The last assistant response
        """
        lines = cli_output.split('\n')
        assistant_responses = []

        for line in lines:
            if line.strip().startswith('Assistant:'):
                response = line.strip()[len('Assistant:'):].strip()
                if response:
                    assistant_responses.append(response)

        if not assistant_responses:
            if self.verbose:
                print(f"Warning: No assistant response found in output:\n{cli_output}")
            return ""

        return assistant_responses[-1]


class PersonaClient:
    """Client that mimics the Anthropic client API."""

    def __init__(self, project_root: str = "/Users/danborthwick/Developer/AI/Persona", verbose: bool = False):
        """
        Initialize the Persona client.

        Args:
            project_root: Absolute path to the Persona project
            verbose: Enable verbose logging
        """
        self.messages = PersonaMessages(project_root, verbose)

    def cleanup_caches(self):
        """No-op for compatibility with QueryLLM interface."""
        pass
