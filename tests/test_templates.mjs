import test from 'node:test';
import assert from 'node:assert/strict';
import { STYLE_TEMPLATES, generationSettings } from '../qwen_studio/static/templates.mjs';

const subject = {
  prompt: 'A lighthouse above the clouds', negative_prompt: '',
  width: 1536, height: 896, steps: 30, seed: '9223372036854775807',
};

test('each style composes with the subject without altering generation settings', () => {
  for (const style of STYLE_TEMPLATES) {
    const result = generationSettings({ ...subject, style_prompt: style.prompt, style_negative: style.negative });
    assert.ok(result.prompt.startsWith(`${subject.prompt}\n\n`));
    assert.ok(result.prompt.endsWith(style.prompt));
    assert.equal(result.seed, subject.seed);
    assert.equal(result.width, subject.width);
    assert.equal(result.height, subject.height);
    assert.equal(result.steps, subject.steps);
    assert.equal(result.negative_prompt, style.negative);
  }
});
test('switching styles replaces the style layer without accumulating instructions', () => {
  const anime = STYLE_TEMPLATES.find(style => style.id === 'anime');
  const ink = STYLE_TEMPLATES.find(style => style.id === 'inkpen');
  const draft = { ...subject, style_prompt: anime.prompt };
  generationSettings(draft);
  draft.style_prompt = ink.prompt;
  const result = generationSettings(draft);
  assert.ok(!result.prompt.includes(anime.prompt));
  assert.ok(result.prompt.endsWith(ink.prompt));
  assert.equal(draft.prompt, subject.prompt);
});
test('no style and blank negatives round-trip unchanged', () => {
  assert.deepEqual(generationSettings(subject), subject);
});
test('custom style edits and existing negatives reach the request', () => {
  const result = generationSettings({ ...subject, negative_prompt: 'watermark', style_prompt: 'Blue ink only', style_negative: 'color fills' });
  assert.equal(result.prompt, 'A lighthouse above the clouds\n\nBlue ink only');
  assert.equal(result.negative_prompt, 'watermark, color fills');
  assert.ok(!('style_id' in result));
});
test('reused composed prompt with no template is not applied twice', () => {
  const generated = generationSettings({ ...subject, style_prompt: STYLE_TEMPLATES[0].prompt });
  assert.deepEqual(generationSettings(generated), generated);
});
