import { StyleTextToSpeech2Model, AutoTokenizer, Tensor, RawAudio } from "@huggingface/transformers";
import { phonemize } from "./phonemize.js";
import { TextSplitterStream } from "./splitter.js";
import { getVoiceData, VOICES } from "./voices.js";
import { stack, tidy, tensor1d } from '@tensorflow/tfjs';

const STYLE_DIM = 256;
const SAMPLE_RATE = 24000;

/**
 * @typedef {Object} GenerateOptions
 * @property {keyof any} [voice="af_heart"] The voice
 * @property {number} [speed=1] The speaking speed
 */

/**
 * @typedef {Object} StreamProperties
 * @property {RegExp} [split_pattern] The pattern to split the input text. If unset, the default sentence splitter will be used.
 * @typedef {GenerateOptions & StreamProperties} StreamGenerateOptions
 */

export class KokoroTTS {
  /**
   * Create a new KokoroTTS instance.
   * @param {import('@huggingface/transformers').StyleTextToSpeech2Model} model The model
   * @param {import('@huggingface/transformers').PreTrainedTokenizer} tokenizer The tokenizer
   */
  constructor(model, tokenizer) {
    this.model = model;
    this.tokenizer = tokenizer;
  }

  /**
   * Load a KokoroTTS model from the Hugging Face Hub.
   * @param {string} model_id The model id
   * @param {Object} options Additional options
   * @param {"fp32"|"fp16"|"q8"|"q4"|"q4f16"} [options.dtype="fp32"] The data type to use.
   * @param {"wasm"|"webgpu"|"cpu"|null} [options.device=null] The device to run the model on.
   * @param {import("@huggingface/transformers").ProgressCallback} [options.progress_callback=null] A callback function that is called with progress information.
   * @returns {Promise<KokoroTTS>} The loaded model
   */
  static async from_pretrained(model_id, { dtype = "fp32", device = null, progress_callback = null } = {}) {
    const model = StyleTextToSpeech2Model.from_pretrained(model_id, { progress_callback, dtype, device });
    const tokenizer = AutoTokenizer.from_pretrained(model_id, { progress_callback });

    const info = await Promise.all([model, tokenizer]);
    return new KokoroTTS(...info);
  }

  get voices() {
    return VOICES;
  }

  list_voices() {
    console.table(VOICES);
  }

  _validate_voice(voice) {
    // if (!VOICES.hasOwnProperty(voice)) {
    //   console.error(`Voice "${voice}" not found. Available voices:`);
    //   console.table(VOICES);
    //   throw new Error(`Voice "${voice}" not found. Should be one of: ${Object.keys(VOICES).join(", ")}.`);
    // }
    const language = /** @type {"a"|"b"|"j"|"z"|"e"|"h"|"i"|"p"} */ (voice.at(0));
    return language;
  }

  /**
   * Generate audio from text.
   *
   * @param {string} text The input text
   * @param {GenerateOptions} options Additional options
   * @returns {Promise<RawAudio>} The generated audio
   */
  async generate(text, { voice = "af_heart", speed = 1 } = {}) {
    const language = this._validate_voice(voice);

    const phonemes = await phonemize(text, language);
    const { input_ids } = this.tokenizer(phonemes, {
      truncation: true,
    });

    return this.generate_from_ids(input_ids, { voice, speed });
  }

  /**
   * Generate audio from input ids.
   * @param {Tensor} input_ids The input ids
   * @param {GenerateOptions} options Additional options
   * @returns {Promise<RawAudio>} The generated audio
   */
  async generate_from_ids(input_ids, { voice = "af_heart", speed = 1 } = {}) {
    // Select voice style based on number of input tokens
    const num_tokens = Math.min(Math.max(input_ids.dims.at(-1) - 2, 0), 509);

    // Load voice style
    const data = await getBlendedVoiceData(voice, num_tokens);
    console.log("DATA")
    console.log(data)

    // Prepare model inputs
    const inputs = {
      input_ids,
      style: new Tensor("float32", data, [1, STYLE_DIM]),
      speed: new Tensor("float32", [speed], [1]),
    };

    // Generate audio
    const { waveform } = await this.model(inputs);
    return new RawAudio(waveform.data, SAMPLE_RATE);
  }

  /**
   * Generate audio from text in a streaming fashion.
   * @param {string|TextSplitterStream} text The input text
   * @param {StreamGenerateOptions} options Additional options
   * @returns {AsyncGenerator<{text: string, phonemes: string, audio: RawAudio}, void, void>}
   */
  async *stream(text, { voice = "af_heart", speed = 1, split_pattern = null } = {}) {
    const language = this._validate_voice(voice);

    /** @type {TextSplitterStream} */
    let splitter;
    if (text instanceof TextSplitterStream) {
      splitter = text;
    } else if (typeof text === "string") {
      splitter = new TextSplitterStream();
      const chunks = split_pattern
        ? text
            .split(split_pattern)
            .map((chunk) => chunk.trim())
            .filter((chunk) => chunk.length > 0)
        : [text];
      splitter.push(...chunks);
    } else {
      throw new Error("Invalid input type. Expected string or TextSplitterStream.");
    }
    for await (const sentence of splitter) {
      const phonemes = await phonemize(sentence, language);
      const { input_ids } = this.tokenizer(phonemes, {
        truncation: true,
      });

      // TODO: There may be some cases where - even with splitting - the text is too long.
      // In that case, we should split the text into smaller chunks and process them separately.
      // For now, we just truncate these exceptionally long chunks
      const audio = await this.generate_from_ids(input_ids, { voice, speed });
      yield { text: sentence, phonemes, audio };
    }
  }
}


/**
 * Parse a voice formula string and return the combined voice tensor.
 * Format: "0.3*af_heart + 0.7*af_bella"
 * @param {string} formula - Voice formula string
 * @param {number} numTokens - Number of tokens to determine slice offset
 * @returns {Promise<Float32Array>} - Blended voice tensor
 */
async function parseVoiceFormula(formula, numTokens) {
  if (!formula || !formula.trim()) {
    throw new Error("Empty voice formula");
  }
  
  // Split the formula into terms
  const terms = formula.trim().split('+');
  
  // Initialize weighted sum
  const offset = numTokens * STYLE_DIM;
  
  const voices = terms.map(async term => {
    // Parse each term (format: "0.333 * voice_name")
    const [weightStr, voiceName] = term.trim().split('*').map(part => part.trim());
    const weight = parseFloat(weightStr);
    
    if (isNaN(weight)) {
      throw new Error(`Invalid weight in term: ${term}`);
    }
    
    // Get the voice tensor
    const voiceData = await getVoiceData(voiceName);
    return [voiceData.slice(offset, offset + STYLE_DIM), weight];
  })
  Promise.all(voices).then(async (voiceTuple) => {
    const tensors = voiceTuple.map(tuple => tuple[0]);
    const weights = voiceTuple.map(tuple => tuple[1]);
    const blended = await blendTensorsWeighted(tensors, weights).data()
    return new Float32Array(blended)
  })
  return null
}

function blendTensorsWeighted(tensors, weights) {
  // Input validation
  if (!tensors || tensors.length === 0) {
    throw new Error('No tensors provided for blending');
  }
  
  if (!weights || weights.length !== tensors.length) {
    throw new Error(`Weights array length doesn't match tensors length`);
  }
  
  return tidy(() => {
    // Convert weights to tensor
    const weightsTensor = tensor1d(weights);
    
    // Normalize weights to sum to 1
    const weightSum = weightsTensor.sum();
    const normalizedWeights = weightsTensor.div(weightSum);
    
    // Stack tensors - shape will be [numTensors, ...tensorDimensions]
    const stacked = stack(tensors);
    
    // Get original tensor shape to determine reshape pattern
    const stackedShape = stacked.shape;
    
    // Create reshape dimensions for broadcasting
    // We'll add a 1 for each dimension of the original tensor
    const broadcastShape = [stackedShape[0]];
    for (let i = 1; i < stackedShape.length; i++) {
      broadcastShape.push(1);
    }
    
    // Reshape weights for broadcasting
    const reshapedWeights = normalizedWeights.reshape(broadcastShape);
    
    // Apply weights
    const weighted = stacked.mul(reshapedWeights);
    
    // Sum along the first dimension
    return weighted.sum(0);
  });
}

/**
 * Extended version of getVoiceData that supports both single voice names 
 * and voice formulas
 * @param {any} voiceInput - Either a single voice name or a blend formula
 * @param {number} numTokens - Number of tokens for offset calculation
 * @returns {Promise<Float32Array>} - Voice tensor data
 */
async function getBlendedVoiceData(voiceInput, numTokens) {
  // Check if the input contains blending operators
  if (voiceInput.includes('*') && voiceInput.includes('+')) {
    return await parseVoiceFormula(voiceInput, numTokens);
  } else {
    // Original single voice behavior
    const data = await getVoiceData(voiceInput);
    const offset = numTokens * STYLE_DIM;
    return data.slice(offset, offset + STYLE_DIM);
  }
}

export { TextSplitterStream };
