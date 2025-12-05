/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'getting-started/installation',
        'getting-started/quick-start',
        'getting-started/deepspeed-zero-stages',
      ],
    },
    {
      type: 'category',
      label: 'Tutorials',
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
            'tutorials/huggingface/trl-function-calling',
            'tutorials/huggingface/ocr-vision-language',
            'tutorials/huggingface/grpo-training',
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
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Guides',
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
      items: [
        'reference/deepspeed-config',
        'reference/troubleshooting',
      ],
    },
  ],
};

export default sidebars;
