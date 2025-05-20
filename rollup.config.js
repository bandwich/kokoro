import terser from "@rollup/plugin-terser";
import commonjs from '@rollup/plugin-commonjs';
import { nodeResolve } from "@rollup/plugin-node-resolve";

const plugins = (browser) => [commonjs(), nodeResolve({ browser }), terser({ format: { comments: false } })];

const OUTPUT_CONFIGS = [
  // Node versions
  {
    file: "./build/dist/kokoro.cjs",
    format: "cjs",
  },
  {
    file: "./build/dist/kokoro.js",
    format: "esm",
  },

  // Web version
  {
    file: "./build/dist/kokoro.web.js",
    format: "esm",
  },
];

const WEB_SPECIFIC_CONFIG = {
  onwarn: (warning, warn) => {
    if (!warning.message.includes("@huggingface/transformers")) warn(warning);
  },
};

const NODE_SPECIFIC_CONFIG = {
  external: ["@huggingface/transformers", "phonemizer"],
};

export default OUTPUT_CONFIGS.map((output) => {
  const web = output.file.endsWith(".web.js");
  return {
    input: "./src/kokoro.js",
    output,
    plugins: plugins(web),
    ...(web ? WEB_SPECIFIC_CONFIG : NODE_SPECIFIC_CONFIG),
  };
});
