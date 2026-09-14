import json
import unittest

import httpx

from app.services.llm.gemini import GeminiClient


class GeminiTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_maps_history_and_filters_thoughts(self):
        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(body['contents'][1]['role'], 'model')
            self.assertEqual(body['systemInstruction']['parts'][0]['text'], 'French')
            self.assertEqual(body['generationConfig']['thinkingConfig'], {'thinkingBudget': 0})
            events = [{'candidates': [{'content': {'parts': parts}}]} for parts in
                      [[{'text': 'hidden', 'thought': True}, {'text': 'Bonjour'}], [{'text': ' !'}]]]
            return httpx.Response(200, text=''.join('data: '+json.dumps(e)+'\n\n' for e in events))
        async with httpx.AsyncClient(base_url='https://example.test/', transport=httpx.MockTransport(handler)) as http:
            client = GeminiClient('test', 'gemini-2.5-flash-lite', http)
            async with client.messages.stream(model='ignored', system='French', max_tokens=80,
                messages=[{'role':'user','content':'Hi'}, {'role':'assistant','content':'Hello'}]) as stream:
                self.assertEqual([text async for text in stream.text_stream], ['Bonjour', ' !'])

    async def test_gemini3_uses_minimal_thinking_level(self):
        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(body['generationConfig']['thinkingConfig'], {'thinkingLevel': 'minimal'})
            return httpx.Response(200, text='data: {"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}\n\n')
        async with httpx.AsyncClient(base_url='https://example.test/', transport=httpx.MockTransport(handler)) as http:
            client = GeminiClient('test', 'gemini-3.1-flash-lite', http)
            await client.create(model='ignored', system='', max_tokens=8, messages=[{'role':'user','content':'Hi'}])

    async def test_empty_blocked_response_is_error(self):
        async with httpx.AsyncClient(base_url='https://example.test/', transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text='data: {"promptFeedback":{"blockReason":"SAFETY"}}\n\n')
        )) as http:
            client = GeminiClient('test', 'gemini-2.5-flash-lite', http)
            with self.assertRaisesRegex(RuntimeError, 'no speech text'):
                await client.create(model='ignored', system='', max_tokens=80, messages=[])
