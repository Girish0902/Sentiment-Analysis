import { pipeline, env } from '@huggingface/transformers';

export const IMAGE_PROMPTS: { label: string; text: string }[] = [
  { label: 'positive', text: 'a joyful and cheerful scene' },
  { label: 'positive', text: 'a bright and happy moment' },
  { label: 'positive', text: 'a warm and uplifting atmosphere' },
  { label: 'neutral', text: 'an ordinary everyday scene' },
  { label: 'neutral', text: 'a plain and uneventful moment' },
  { label: 'neutral', text: 'a calm and factual setting' },
  { label: 'negative', text: 'a sad and gloomy scene' },
  { label: 'negative', text: 'a harsh and unpleasant moment' },
  { label: 'negative', text: 'a dark and troubling atmosphere' },
];

env.allowLocalModels = false;

type ProgressMessage = { type: 'progress'; stage: string; progress: number | null };
type ResultMessage = { type: 'result'; payload: unknown };
type ErrorMessage = { type: 'error'; message: string };

const post = (message: ProgressMessage | ResultMessage | ErrorMessage) =>
  self.postMessage(message);

type Extractor = (input: unknown, options?: Record<string, unknown>) => Promise<unknown>;

let extractor: Extractor | null = null;

async function getExtractor(): Promise<Extractor> {
  if (!extractor) {
    extractor = (await pipeline('zero-shot-image-classification', 'Xenova/clip-vit-base-patch32', {
      dtype: 'q8',
      progress_callback: (update: { status: string; progress?: number; file?: string }) => {
        post({
          type: 'progress',
          stage: update.status === 'progress' ? `Downloading ${(update.file ?? 'model').split('/').pop()}` : 'Loading model',
          progress: typeof update.progress === 'number' ? update.progress / 100 : null,
        });
      },
    })) as unknown as Extractor;
  }
  return extractor;
}

self.onmessage = async (event: MessageEvent<{ image: Blob }>) => {
  try {
    post({ type: 'progress', stage: 'Analyzing image', progress: null });
    const pipe = await getExtractor();
    const outputs = (await pipe(event.data.image, {
      candidate_labels: IMAGE_PROMPTS.map((prompt) => prompt.text),
    })) as { label: string; score: number }[];

    const scores: Record<string, number> = { positive: 0, neutral: 0, negative: 0 };
    for (const prompt of IMAGE_PROMPTS) {
      const match = outputs.find((output) => output.label === prompt.text);
      scores[prompt.label] += match ? match.score : 0;
    }
    const ordered = ['positive', 'neutral', 'negative'] as const;
    let label: (typeof ordered)[number] = ordered[0];
    for (const candidate of ordered) {
      if (scores[candidate] > scores[label]) label = candidate;
    }
    const sorted = [...ordered].sort((a, b) => scores[b] - scores[a]);
    const unclear = scores[sorted[0]] < 0.5 || scores[sorted[0]] - scores[sorted[1]] < 0.1;

    post({
      type: 'result',
      payload: {
        label,
        unclear,
        probabilities: scores,
        matches: outputs
          .slice()
          .sort((a, b) => b.score - a.score)
          .slice(0, 3)
          .map((output) => ({ label: output.label, score: output.score })),
        model: 'Xenova/clip-vit-base-patch32',
        engine: 'browser',
      },
    });
  } catch (error) {
    post({
      type: 'error',
      message: error instanceof Error ? error.message : 'Image model failed to run.',
    });
  }
};
