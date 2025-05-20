/**
 * Phonemize text using the eSpeak-NG phonemizer
 * @param {string} text The text to phonemize
 * @param {"a"|"b"|"j"|"z"|"e"|"h"|"i"|"p"} language The language to use
 * @param {boolean} norm Whether to normalize the text
 * @returns {Promise<string>} The phonemized text
 */
export function phonemize(text: string, language?: "a" | "b" | "j" | "z" | "e" | "h" | "i" | "p", norm?: boolean): Promise<string>;
//# sourceMappingURL=phonemize.d.ts.map