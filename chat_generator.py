#!/usr/bin/env python3
import asyncio
from openai import AsyncOpenAI
from multiprocessing import Process, Queue, Event
from typing import List, Dict


class ChatGenerator:
    @staticmethod
    def stream_in_process(messages: List[Dict], model: str, api_key: str, api_base: str,
                          queue: Queue, stop_event: Event):
        """Function to run in separate process for streaming"""
        try:
            # Initialize client in the child process
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=api_base
            )

            async def stream_response():
                try:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        stream=True
                    )

                    async for chunk in response:
                        # Check stop_event more frequently
                        if stop_event.is_set():
                            return

                        if hasattr(chunk.choices[0].delta, 'reasoning_content'):
                            if chunk.choices[0].delta.reasoning_content is not None:
                                queue.put(('reasoning', chunk.choices[0].delta.reasoning_content))
                        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
                            queue.put(('content', chunk.choices[0].delta.content))

                except Exception as e:
                    queue.put(('error', str(e)))
                finally:
                    queue.put(('done', None))

            # Run the async function with proper signal handling
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            def signal_handler(signum, frame):
                loop.stop()
                raise KeyboardInterrupt

            import signal
            signal.signal(signal.SIGINT, signal_handler)

            try:
                loop.run_until_complete(stream_response())
            except KeyboardInterrupt:
                # Clean shutdown of the event loop
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                queue.put(('error', 'Generation interrupted'))
            finally:
                loop.close()

        except Exception as e:
            queue.put(('error', str(e)))

    @staticmethod
    def generate_completion(client, model: str, messages: List[Dict], stream: bool = False):
        """Generate non-streaming completion"""
        return client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream
        )