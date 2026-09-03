/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Getting Started',
      link: {type: 'generated-index', description: 'Install DeepSpeed, run your first job, and understand ZeRO.'},
      items: [
        'getting-started/installation',
        'getting-started/quick-start',
        'getting-started/deepspeed-zero-stages',
      ],
    },
    {
      type: 'category',
      label: 'Tutorials',
      link: {type: 'generated-index', description: 'Worked examples from a two-parameter linear model up to 560B-parameter multimodal training.'},
      items: [
        {
          type: 'category',
          label: '01 · Basics',
          items: [
            'tutorials/basic/neural-network',
            'tutorials/basic/convnet',
            'tutorials/basic/cifar10',
            'tutorials/basic/rnn',
          ],
        },
        {
          type: 'category',
          label: '02 · Intermediate',
          items: [
            'tutorials/intermediate/bayesian-nn',
            'tutorials/intermediate/stock-prediction',
            'tutorials/intermediate/mean-reversion-forecasting',
            // Ranking: 03 varies the OBJECTIVE, 04 varies the ARCHITECTURE.
            'tutorials/intermediate/learning-to-rank',
            'tutorials/intermediate/groupwise-ranking',
          ],
        },
        {
          type: 'category',
          label: '03 · HuggingFace',
          items: [
            'tutorials/huggingface/overview',
            'tutorials/huggingface/llm-finetuning',
            'tutorials/huggingface/glm53-moe-finetuning',
            'tutorials/huggingface/qwen38-hybrid-attention',
            'tutorials/huggingface/trl-function-calling',
            'tutorials/huggingface/ocr-vision-language',
            // The alignment thread, in the order the literature arrived:
            // RLHF (2017-22) -> DPO family (2023-24) -> GRPO (Feb 2024)
            // -> online methods (2024) -> post-GRPO fixes (2025).
            'tutorials/huggingface/rlhf-reward-modeling',
            'tutorials/huggingface/preference-optimization',
            'tutorials/huggingface/grpo-training',
            'tutorials/huggingface/grpo-worked-example',
            'tutorials/huggingface/online-preference-methods',
            'tutorials/huggingface/beyond-grpo',
            'tutorials/huggingface/gpt-oss-finetuning',
            'tutorials/huggingface/multi-agent',
          ],
        },
        {
          type: 'category',
            label: '04 · Video-Text',
            items: [
              // Ordered to match 04_video_text/01..05 in the repository.
              'tutorials/multimodal/video-text-training',   // 01_hf_baseline
              'tutorials/multimodal/qwen-video-baseline',   // 02_qwen25vl
              'tutorials/multimodal/token-compression',     // 03_token_compression
              'tutorials/multimodal/streaming-video',       // 04_streaming_memory
              'tutorials/multimodal/video-evaluation',      // 05_video_eval
            ],
          },
          {
            type: 'category',
            label: '05 · Video-Speech',
            items: [
              // Ordered to match 05_video_speech/01..04 in the repository.
              'tutorials/multimodal/video-speech-training', // 01_longcat_omni
              'tutorials/multimodal/omni-thinker-talker',   // 02_thinker_talker
              'tutorials/multimodal/duplex-streaming',      // 03_duplex_streaming
              'tutorials/multimodal/omni-evaluation',       // 04_omni_eval
            ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Guides',
      link: {type: 'generated-index', description: 'Deploying to SLURM clusters and single-user pods, and choosing hardware.'},
      items: [
        'guides/slurm-deployment',
        'guides/coreweave-setup',
        'guides/runpod-setup',
        'guides/hardware-requirements',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      link: {type: 'generated-index', description: 'DeepSpeed configuration keys and troubleshooting.'},
      items: [
        'reference/deepspeed-config',
        'reference/troubleshooting',
      ],
    },
    'contributing',
  ],
};

export default sidebars;
