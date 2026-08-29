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
          label: 'Basic Examples',
          items: [
            'tutorials/basic/neural-network',
            'tutorials/basic/convnet',
            'tutorials/basic/cifar10',
            'tutorials/basic/rnn',
          ],
        },
        {
          type: 'category',
          label: 'Intermediate Examples',
          items: [
            'tutorials/intermediate/bayesian-nn',
            'tutorials/intermediate/stock-prediction',
          ],
        },
        {
          type: 'category',
          label: 'HuggingFace Integration',
          items: [
            'tutorials/huggingface/overview',
            'tutorials/huggingface/llm-finetuning',
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
          label: 'Advanced Multimodal',
          items: [
            'tutorials/multimodal/video-text-training',
            'tutorials/multimodal/video-speech-training',
            'tutorials/multimodal/qwen-video-baseline',
            'tutorials/multimodal/token-compression',
            'tutorials/multimodal/streaming-video',
            'tutorials/multimodal/video-evaluation',
            'tutorials/multimodal/omni-thinker-talker',
            'tutorials/multimodal/duplex-streaming',
            'tutorials/multimodal/omni-evaluation',
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
