/**
 * Browser text-sentiment engine and CSV helpers.
 *
 * Mirrors backend/engine.py: same preprocessing, stemmer, TF-IDF vectorization
 * and softmax so browser and Python modes return matching results.
 */

export type SentimentLabel = 'positive' | 'neutral' | 'negative';

export const CLASSES: SentimentLabel[] = ['positive', 'neutral', 'negative'];
export const MAX_ITEMS = 200;
export const MAX_CHARS = 5000;
export const MAX_CSV_BYTES = 1_048_576;

export interface FeatureWeight {
  term: string;
  weight: number;
  count: number;
}

export interface AnalysisResult {
  text: string;
  label: SentimentLabel;
  probabilities: Record<SentimentLabel, number>;
  insufficient: boolean;
  warnings: string[];
  cleaned: string;
  removedStopWords: string[];
  tokens: string[];
  features: FeatureWeight[];
  coverage: { known: number; total: number };
  actual?: SentimentLabel;
  correct?: boolean;
}

export interface ClassMetrics {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface EvaluationMetrics {
  accuracy: number;
  macroPrecision: number;
  macroRecall: number;
  macroF1: number;
  perClass: Record<SentimentLabel, ClassMetrics>;
  confusionMatrix: { labels: SentimentLabel[]; rows: number[][] };
}

export interface ModelArtifact {
  version: string;
  classes: SentimentLabel[];
  terms: string[];
  idf: number[];
  coefficients: number[][];
  intercepts: number[];
  prior: Record<string, number>;
  training: Record<string, unknown>;
  report: Record<string, unknown>;
}

export interface TextItem {
  text: string;
  label?: SentimentLabel;
}

const URL_RE = /(?:https?:\/\/|www\.)\S+/g;
const WORD_RE = /[a-z]+/g;

const CONTRACTIONS: Record<string, string> = {
  "i'm": 'i am',
  "i've": 'i have',
  "i'd": 'i would',
  "i'll": 'i will',
  "you're": 'you are',
  "you've": 'you have',
  "you'd": 'you would',
  "you'll": 'you will',
  "he's": 'he is',
  "he'd": 'he would',
  "he'll": 'he will',
  "she's": 'she is',
  "she'd": 'she would',
  "she'll": 'she will',
  "it's": 'it is',
  "it'd": 'it would',
  "it'll": 'it will',
  "that's": 'that is',
  "there's": 'there is',
  "here's": 'here is',
  "what's": 'what is',
  "who's": 'who is',
  "let's": 'let us',
  "we're": 'we are',
  "we've": 'we have',
  "we'd": 'we would',
  "we'll": 'we will',
  "they're": 'they are',
  "they've": 'they have',
  "they'd": 'they would',
  "they'll": 'they will',
  "can't": 'can not',
  cannot: 'can not',
  "won't": 'will not',
  "wouldn't": 'would not',
  "shouldn't": 'should not',
  "couldn't": 'could not',
  "mustn't": 'must not',
  "shan't": 'shall not',
  "don't": 'do not',
  "doesn't": 'does not',
  "didn't": 'did not',
  "isn't": 'is not',
  "aren't": 'are not',
  "wasn't": 'was not',
  "weren't": 'were not',
  "haven't": 'have not',
  "hasn't": 'has not',
  "hadn't": 'had not',
};

const CONTRACTION_ENTRIES = Object.entries(CONTRACTIONS).sort((a, b) => b[0].length - a[0].length);

const STOP_WORDS = new Set(
  `a about above after again against all am an and any are as at be because
   been before being below between both by can could did do does doing down
   during each few for from further had has have having he her here hers
   herself him himself his how i if in into is it its itself just me more
   most my myself of off on once only or other our ours ourselves out over
   own same she should so some such than that the their theirs them
   themselves then there these they this those through to too under until
   up very was we were what when where which while who whom why will with
   you your yours yourself yourselves`
    .split(/\s+/)
    .filter(Boolean)
);

const VOWELS = new Set(['a', 'e', 'i', 'o', 'u']);

function undouble(base: string): string {
  if (base.length >= 3 && base[base.length - 1] === base[base.length - 2] && !VOWELS.has(base[base.length - 1])) {
    return base.slice(0, -1);
  }
  return base;
}

export function stem(word: string): string {
  if (word.length <= 3) return word;
  let w = word;
  if (w.endsWith('ies') && w.length > 4) {
    w = w.slice(0, -3) + 'y';
  } else if (['sses', 'ches', 'shes', 'xes', 'zes'].some((s) => w.endsWith(s)) && w.length > 5) {
    w = w.slice(0, -2);
  } else if (w.endsWith('s') && !w.endsWith('ss')) {
    w = w.slice(0, -1);
  }
  if (w.endsWith('ingly') && w.length > 6 && w.length - 5 >= 3) {
    w = w.slice(0, -5);
  } else if (w.endsWith('edly') && w.length > 5 && w.length - 4 >= 3) {
    w = w.slice(0, -4);
  } else if (w.endsWith('ing') && w.length > 4 && w.length - 3 >= 3) {
    w = undouble(w.slice(0, -3));
  } else if (w.endsWith('ed') && w.length > 4) {
    let base = w.slice(0, -2);
    if (base.length >= 3) {
      base =
        base[base.length - 1] === 't' && 'kpsfxh'.includes(base[base.length - 2])
          ? base.slice(0, -1)
          : undouble(base);
      w = base;
    }
  }
  if (w.endsWith('ly') && w.length > 4 && w.length - 2 >= 3) {
    w = w.slice(0, -2);
  } else if (w.endsWith('ness') && w.length > 5) {
    w = w.slice(0, -4);
  } else if (w.endsWith('ment') && w.length > 5) {
    w = w.slice(0, -4);
  } else if (w.endsWith('ation') && w.length > 6 && w.length - 5 >= 3) {
    w = w.slice(0, -5);
  }
  return w;
}

export function preprocess(text: string): {
  cleaned: string;
  removedStopWords: string[];
  tokens: string[];
} {
  const lowered = text.toLowerCase().replace(/[\u2018\u2019]/g, "'");
  const withoutUrls = lowered.replace(URL_RE, ' ');
  let expanded = withoutUrls;
  for (const [key, value] of CONTRACTION_ENTRIES) {
    const pattern = new RegExp(`\\b${key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'g');
    expanded = expanded.replace(pattern, value);
  }
  const words = expanded.match(WORD_RE) ?? [];
  const removed: string[] = [];
  const kept: string[] = [];
  for (const word of words) {
    if (STOP_WORDS.has(word)) removed.push(word);
    else kept.push(word);
  }
  return {
    cleaned: kept.join(' '),
    removedStopWords: removed,
    tokens: kept.map(stem),
  };
}

export function extractCounts(tokens: string[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (let i = 0; i < tokens.length; i += 1) {
    counts.set(tokens[i], (counts.get(tokens[i]) ?? 0) + 1);
    if (i + 1 < tokens.length) {
      const bigram = `${tokens[i]} ${tokens[i + 1]}`;
      counts.set(bigram, (counts.get(bigram) ?? 0) + 1);
    }
  }
  return counts;
}

const indexCache = new WeakMap<ModelArtifact, Map<string, number>>();

function termIndex(model: ModelArtifact): Map<string, number> {
  let index = indexCache.get(model);
  if (!index) {
    index = new Map();
    model.terms.forEach((term, position) => index!.set(term, position));
    indexCache.set(model, index);
  }
  return index;
}

export function vectorize(counts: Map<string, number>, model: ModelArtifact): number[] {
  const index = termIndex(model);
  const vector = new Array<number>(model.terms.length).fill(0);
  for (const [term, count] of counts) {
    const position = index.get(term);
    if (position === undefined) continue;
    vector[position] = (1 + Math.log(count)) * model.idf[position];
  }
  let norm = 0;
  for (const value of vector) norm += value * value;
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (let i = 0; i < vector.length; i += 1) vector[i] /= norm;
  }
  return vector;
}

function softmax(scores: number[]): number[] {
  const peak = Math.max(...scores);
  const exps = scores.map((score) => Math.exp(score - peak));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
}

export class InputError extends Error {}

function validateText(text: unknown): string {
  if (typeof text !== 'string' || !text.trim()) {
    throw new InputError('Text is empty. Enter a non-blank English text.');
  }
  if (text.length > MAX_CHARS) {
    throw new InputError(`Text is longer than the ${MAX_CHARS} character limit.`);
  }
  return text;
}

export function analyzeWithModel(text: string, model: ModelArtifact): AnalysisResult {
  const cleanText = validateText(text);
  const processed = preprocess(cleanText);
  const counts = extractCounts(processed.tokens);
  const index = termIndex(model);
  const knownTerms: string[] = [];
  for (const term of counts.keys()) {
    if (index.has(term)) knownTerms.push(term);
  }
  const vector = vectorize(counts, model);

  const warnings: string[] = [];
  const insufficient = knownTerms.length === 0;
  const probabilities = {} as Record<SentimentLabel, number>;

  if (insufficient) {
    for (const label of CLASSES) probabilities[label] = model.prior[label];
    warnings.push(
      'No known model features were found in this text, so the scores reflect the training class prior.'
    );
  } else {
    const scores = model.coefficients.map(
      (row, r) =>
        model.intercepts[r] + row.reduce((sum, weight, i) => sum + weight * vector[i], 0)
    );
    const values = softmax(scores);
    CLASSES.forEach((label, i) => {
      probabilities[label] = values[i];
    });
    if (knownTerms.length < counts.size) {
      const hidden = counts.size - knownTerms.length;
      warnings.push(
        `${hidden} of ${counts.size} text features are not in the model vocabulary and were ignored.`
      );
    }
  }

  let label: SentimentLabel = CLASSES[0];
  for (const candidate of CLASSES) {
    if (probabilities[candidate] > probabilities[label]) label = candidate;
  }

  const features: FeatureWeight[] = knownTerms
    .map((term) => ({
      term,
      weight: vector[index.get(term)!],
      count: counts.get(term)!,
    }))
    .sort((a, b) => b.weight - a.weight || (a.term < b.term ? -1 : a.term > b.term ? 1 : 0));

  return {
    text: cleanText,
    label,
    probabilities,
    insufficient,
    warnings,
    cleaned: processed.cleaned,
    removedStopWords: processed.removedStopWords,
    tokens: processed.tokens,
    features,
    coverage: { known: knownTerms.length, total: counts.size },
  };
}

function prf(tp: number, fp: number, fn: number) {
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
  return { precision, recall, f1 };
}

export function computeMetrics(results: AnalysisResult[]): EvaluationMetrics {
  const total = results.length;
  const correct = results.filter((item) => item.label === item.actual).length;
  const index = new Map(CLASSES.map((label, i) => [label, i]));
  const matrix = CLASSES.map(() => new Array<number>(CLASSES.length).fill(0));
  for (const item of results) {
    matrix[index.get(item.actual!)!][index.get(item.label)!] += 1;
  }
  const perClass = {} as Record<SentimentLabel, ClassMetrics>;
  CLASSES.forEach((label, i) => {
    const tp = matrix[i][i];
    const fp = CLASSES.reduce((sum, _, row) => sum + (row !== i ? matrix[row][i] : 0), 0);
    const fn = CLASSES.reduce((sum, _, col) => sum + (col !== i ? matrix[i][col] : 0), 0);
    const { precision, recall, f1 } = prf(tp, fp, fn);
    perClass[label] = { precision, recall, f1, support: matrix[i].reduce((a, b) => a + b, 0) };
  });
  const mean = (pick: (m: ClassMetrics) => number) =>
    CLASSES.reduce((sum, label) => sum + pick(perClass[label]), 0) / CLASSES.length;
  return {
    accuracy: total ? correct / total : 0,
    macroPrecision: mean((m) => m.precision),
    macroRecall: mean((m) => m.recall),
    macroF1: mean((m) => m.f1),
    perClass,
    confusionMatrix: { labels: [...CLASSES], rows: matrix },
  };
}

export function analyzeItems(items: TextItem[], model: ModelArtifact): AnalysisResult[] {
  if (!Array.isArray(items) || items.length === 0) {
    throw new InputError('Provide at least one text item.');
  }
  if (items.length > MAX_ITEMS) {
    throw new InputError(`A maximum of ${MAX_ITEMS} texts can be processed at once.`);
  }
  return items.map((item) => analyzeWithModel(validateText(item.text), model));
}

export function evaluateItems(
  items: TextItem[],
  model: ModelArtifact
): { results: AnalysisResult[]; metrics: EvaluationMetrics } {
  if (!Array.isArray(items) || items.length === 0) {
    throw new InputError('Provide at least one text item.');
  }
  if (items.length > MAX_ITEMS) {
    throw new InputError(`A maximum of ${MAX_ITEMS} texts can be processed at once.`);
  }
  const results = items.map((item, position) => {
    if (item.label && !CLASSES.includes(item.label)) {
      throw new InputError(
        `Item ${position + 1} has an invalid label. Use positive, neutral or negative.`
      );
    }
    if (!item.label) {
      throw new InputError(
        `Item ${position + 1} is missing a label. Every row needs positive, neutral or negative.`
      );
    }
    const result = analyzeWithModel(validateText(item.text), model);
    result.actual = item.label;
    result.correct = result.label === item.label;
    return result;
  });
  return { results, metrics: computeMetrics(results) };
}

export function parseLabel(value: string): SentimentLabel | null {
  const normalized = value.trim().toLowerCase();
  return (CLASSES as string[]).includes(normalized) ? (normalized as SentimentLabel) : null;
}

export interface ParsedCsv {
  headers: string[];
  rows: string[][];
}

export function parseCsv(content: string): ParsedCsv {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = '';
  let inQuotes = false;
  let i = 0;
  while (i < content.length) {
    const char = content[i];
    if (inQuotes) {
      if (char === '"') {
        if (content[i + 1] === '"') {
          field += '"';
          i += 2;
          continue;
        }
        inQuotes = false;
        i += 1;
        continue;
      }
      field += char;
      i += 1;
      continue;
    }
    if (char === '"' && field === '') {
      inQuotes = true;
      i += 1;
      continue;
    }
    if (char === ',') {
      row.push(field);
      field = '';
      i += 1;
      continue;
    }
    if (char === '\r') {
      i += 1;
      continue;
    }
    if (char === '\n') {
      row.push(field);
      rows.push(row);
      row = [];
      field = '';
      i += 1;
      continue;
    }
    field += char;
    i += 1;
  }
  if (field !== '' || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  if (rows.length === 0) throw new InputError('The CSV file is empty.');
  const headers = rows[0];
  return { headers, rows: rows.slice(1) };
}

export function itemsFromCsv(
  content: string,
  requireLabels: boolean
): TextItem[] {
  const { headers, rows } = parseCsv(content);
  const textIndex = headers.findIndex((header) => header.trim().toLowerCase() === 'text');
  if (textIndex === -1) {
    throw new InputError('The CSV file needs a "text" column.');
  }
  const labelIndex = headers.findIndex((header) => header.trim().toLowerCase() === 'label');
  if (rows.length > MAX_ITEMS) {
    throw new InputError(`A maximum of ${MAX_ITEMS} texts can be processed at once.`);
  }
  return rows.map((row, position) => {
    const text = (row[textIndex] ?? '').trim();
    const item: TextItem = { text };
    if (labelIndex !== -1 && row[labelIndex] !== undefined && row[labelIndex].trim() !== '') {
      const label = parseLabel(row[labelIndex]);
      if (!label) {
        throw new InputError(
          `Row ${position + 1} has an invalid label. Use positive, neutral or negative.`
        );
      }
      item.label = label;
    } else if (requireLabels) {
      throw new InputError(
        `Row ${position + 1} is missing a label. Every row needs positive, neutral or negative.`
      );
    }
    validateText(text);
    return item;
  });
}

function csvField(value: string): string {
  if (/[",\r\n]/.test(value)) return `"${value.replace(/"/g, '""')}"`;
  return value;
}

export function resultsToCsv(results: AnalysisResult[]): string {
  const lines = ['text,sentiment,score,actual'];
  for (const result of results) {
    const score = Math.max(...CLASSES.map((label) => result.probabilities[label]));
    lines.push(
      [
        csvField(result.text),
        result.label,
        score.toFixed(4),
        result.actual ?? '',
      ].join(',')
    );
  }
  return lines.join('\n');
}
