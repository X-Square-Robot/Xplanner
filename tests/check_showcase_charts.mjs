import assert from 'node:assert/strict';
import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const url = process.argv[2] || 'http://127.0.0.1:18879/';
const output = await mkdtemp(join(tmpdir(), 'xplanner-charts-'));
const browser = await chromium.launch({ args: ['--no-sandbox'] });
try {
  for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }, { width: 320, height: 740 }]) {
    const page = await browser.newPage({ viewport });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('response', response => {
      if (response.status() >= 400) errors.push(`${response.status()} ${response.url()}`);
    });
    await page.goto(url);
    await page.getByRole('tab', { name: 'Spell RoboDojo', exact: true }).waitFor();
    assert.deepEqual(await page.getByRole('tab').allTextContents(), ['Spell RoboDojo', 'Build Tower', 'Fill Egg Holder']);
    assert.equal(await page.getByRole('tab', { name: 'Spell RoboDojo', exact: true }).getAttribute('aria-selected'), 'true');
    assert.equal(await page.locator('#case-real-robot').count(), 0);
    assert((await page.locator('.citation').innerText()).includes('@article{xplanner2026event'));
    assert.equal(await page.locator('.benchmark-links a').getAttribute('href'), 'https://huggingface.co/datasets/x-square-robot/xplanner-benchmark');
    assert.equal(await page.locator('#benchmark .section-title').textContent(), 'XPlanner-Benchmark');
    assert.equal(await page.locator('#abstract img').count(), 0);
    assert.equal(await page.locator('#method img').getAttribute('src'), 'assets/X-Planner.jpg');
    assert(await page.locator('#method img').evaluate(image => image.complete && image.naturalWidth === 4000 && image.naturalHeight === 2250));
    assert.equal(await page.locator('.method-details h3').count(), 4);
    const benchmark = await page.evaluate(() => fetch('data/benchmark-cases.json').then(response => response.json()));
    assert(benchmark.cases.length >= 6);
    assert.equal(await page.locator('#benchmark-cases video').count(), benchmark.cases.length);
    for (const [index, item] of benchmark.cases.entries()) {
      assert.equal(item.model_predictions, false);
      const video = page.locator('#benchmark-cases video').nth(index);
      assert.equal(await video.locator('source').getAttribute('src'), item.video);
      await video.evaluate(async node => { await node.play(); });
      await page.waitForFunction(index => document.querySelectorAll('#benchmark-cases video')[index].currentTime > .1, index);
      assert(await video.evaluate(node => node.videoWidth > 0 && node.videoHeight > 0));
      await video.evaluate(node => { node.pause(); node.currentTime = 4; });
      await page.waitForFunction(index => !document.querySelectorAll('#benchmark-cases video')[index].seeking, index);
      const pixels = await video.evaluate(node => {
        const canvas = document.createElement('canvas');
        canvas.width = 32; canvas.height = 24;
        const context = canvas.getContext('2d');
        context.drawImage(node, 0, 0, 32, 24);
        return [...context.getImageData(0, 0, 32, 24).data].filter((_, i) => i % 4 !== 3);
      });
      assert(Math.max(...pixels) - Math.min(...pixels) > 100, `Blank benchmark footage: ${item.slug}`);
    }
    const payload = await page.evaluate(() => fetch('data/demos.json').then(response => response.json()));
    for (const demo of payload.demos) {
      await page.getByRole('tab', { name: demo.name, exact: true }).click();
      const clip = page.locator(`#case-${demo.slug}`);
      assert.equal(await clip.locator('.episode-explorer').getAttribute('open'), null);
      await clip.getByRole('button', { name: 'Initial plan', exact: true }).click();
      await page.waitForFunction(slug => {
        const video = document.querySelector(`#case-${slug} video`);
        return video.readyState >= 2 && !video.seeking;
      }, demo.slug);
      await clip.screenshot({ path: join(output, `${viewport.width}-${demo.slug}-initial-plan.png`) });
      await clip.getByRole('button', { name: 'Inference replay', exact: true }).click();
      await clip.locator('.episode-explorer > summary').click();
      assert.equal(await clip.locator('.episode-layout').count(), 1);
      assert.equal(await clip.locator('.episode-heading h3').textContent(), demo.name);
      assert.equal(await clip.locator('.episode-environment').textContent(), demo.environment);
      assert.equal(await clip.getByRole('link', { name: 'Full episode', exact: true }).getAttribute('href'), demo.annotated_video || demo.video);
      assert.equal(await clip.locator('.chart-panel svg').count(), 2);
      assert.equal(await clip.locator('.prediction-step').count(), demo.states.length);
      assert.equal(await clip.locator('.plan-step').count(), demo.plan.length);
      const duration = demo.duration_seconds - demo.recording_offset_seconds;
      assert((await clip.locator('.progress-panel .chart-label').allTextContents()).includes(`${Number(duration.toFixed(1))} s`));
      const bars = await clip.locator('.memory-bar').evaluateAll(nodes => nodes.map(node => Number(node.getAttribute('height'))));
      const memory = demo.states.map(state => state.memory_input.long_memory.length);
      const maximum = Math.max(...memory, 1);
      bars.forEach((height, index) => assert(Math.abs(height - 112 * memory[index] / maximum) < 0.001));
      const planLabels = await clip.locator('.plan-step').evaluateAll(nodes => nodes.map(node => node.getAttribute('aria-label')));
      demo.plan.forEach((step, index) => assert.equal(planLabels[index], `Initial plan step ${index + 1}: ${step}`));
      assert.deepEqual(await clip.locator('.plan-list li').allTextContents(), demo.plan);
      for (const state of [demo.states[0], demo.states[8], demo.states.at(-1), demo.states[2]]) {
        await clip.locator('video').evaluate(async (video, time) => {
          if (video.readyState < 1) await new Promise(resolve => video.addEventListener('loadedmetadata', resolve, { once: true }));
          video.pause();
          video.currentTime = time;
        }, state.time_seconds + demo.recording_offset_seconds + 0.5 / demo.fps);
        await page.waitForFunction(({ slug, frame }) => document.querySelector(`#case-${slug} .planning-context`).dataset.anchorFrame === String(frame), { slug: demo.slug, frame: state.frame });
        const current = state.output.predictions?.find(item => item.role === 'current');
        const next = state.output.predictions?.find(item => item.role === 'next');
        assert.equal(await clip.locator('.action-caption').first().textContent(), current?.action?.caption || 'No current action in model output');
        assert.equal(await clip.locator('.action-caption').last().textContent(), next?.action?.caption || 'No next action in model output');
        assert.deepEqual(await clip.locator('.long-memory-list li').allTextContents(), state.memory_input.long_memory.map(item => item.action));
        assert.equal(await clip.locator('.short-memory-caption').textContent(), state.memory_input.short_memory?.prediction1?.action?.caption || 'No previous prediction');
        const active = clip.locator('.prediction-step.is-active');
        assert.equal(await active.count(), 1);
        assert.equal(await active.getAttribute('aria-label'), current?.action?.caption || 'No current action in model output');
        const dot = clip.locator('.progress-panel .chart-dot');
        if (typeof state.output.task_progress_percent === 'number') {
          assert(Math.abs(Number(await dot.getAttribute('cy')) - (134 - 112 * state.output.task_progress_percent / 100)) < 0.001);
        } else assert.equal(await dot.evaluate(node => node.style.display), 'none');
      }
      const moment = demo.moments[0];
      await clip.getByRole('button', { name: moment.label, exact: true }).click();
      assert(Math.abs(await clip.locator('video').evaluate(video => video.currentTime) - (moment.time_seconds + demo.recording_offset_seconds + 0.5 / demo.fps)) < 0.01);
      await clip.locator('.progress-panel svg').click({ position: { x: 90, y: 50 } });
      assert(await clip.locator('video').evaluate(video => video.paused));
      await clip.locator('video').evaluate(video => video.play());
      await page.waitForFunction(slug => document.querySelector(`#case-${slug} video`).currentTime > 5, demo.slug);
      await clip.locator('video').evaluate(video => video.pause());
      assert(await clip.locator('video').evaluate(video => video.videoWidth === 1920 && video.videoHeight === 1080));
      const pixels = await clip.locator('video').evaluate(video => {
        const canvas = document.createElement('canvas');
        canvas.width = 32; canvas.height = 24;
        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, 32, 24);
        return [...context.getImageData(0, 0, 32, 24).data].filter((_, i) => i % 4 !== 3);
      });
      assert(Math.max(...pixels) - Math.min(...pixels) > 100, 'Camera footage is blank');
      await clip.locator('.progress-panel svg').focus();
      await page.keyboard.press('Home');
      await page.waitForFunction(({ slug, offset }) => Math.abs(document.querySelector(`#case-${slug} video`).currentTime - offset) < .1,
        { slug: demo.slug, offset: demo.recording_offset_seconds });
      await page.keyboard.press('ArrowRight');
      assert(await clip.locator('video').evaluate(video => video.currentTime >= 1));
      await clip.locator('.plan-step').nth(4).focus();
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, 'Tooltip overflow');
      await page.evaluate(() => document.activeElement.blur());
      await clip.scrollIntoViewIfNeeded();
      await page.screenshot({ path: join(output, `${viewport.width}-${demo.slug}.png`) });
      await clip.locator('.episode-explorer > summary').click();
    }
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, 'Horizontal overflow');
    const visibleText = await page.locator('body').innerText();
    assert(!/Observation \d|Decision:|Predicted progress:/.test(visibleText));
    await page.getByRole('tab', { name: payload.demos[0].name, exact: true }).click();
    await page.evaluate(() => document.querySelectorAll('.reveal').forEach(node => node.classList.add('is-in')));
    await page.waitForTimeout(1000);
    await page.screenshot({ path: join(output, `${viewport.width}-page.png`), fullPage: true });
    assert.deepEqual(errors, []);
    await page.close();
  }
  console.log(`Charts, video pixels, seeking and responsive layout passed. Screenshots: ${output}`);
} finally {
  await browser.close();
}
