import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Get Started
          </Link>
          <Link
            className="button button--outline button--lg"
            style={{marginLeft: '1rem', color: 'white', borderColor: 'white'}}
            href="https://github.com/yiqiao-yin/deepspeed-course">
            View on GitHub
          </Link>
        </div>
      </div>
    </header>
  );
}

const FeatureList = [
  {
    title: 'DeepSpeed Distributed Training',
    description: (
      <>
        Learn Microsoft DeepSpeed's powerful optimization techniques including ZeRO stages 1-3,
        mixed precision training (FP16/BF16), and CPU offloading. Train models that wouldn't
        fit on a single GPU by intelligently partitioning optimizer states, gradients, and
        parameters across multiple devices.
      </>
    ),
  },
  {
    title: 'Multi-GPU & Multi-Node Scaling',
    description: (
      <>
        Master distributed training across multiple GPUs and nodes. From single RTX 3070 setups
        to 8x H200 clusters, learn how to efficiently scale your training with proper batch
        configuration, gradient accumulation, and NCCL communication optimization. Includes
        SLURM job scheduling for HPC clusters like CoreWeave and RunPod.
      </>
    ),
  },
  {
    title: 'Comprehensive Training Portfolio',
    description: (
      <>
        Progress from basic neural networks to cutting-edge multimodal AI. This course covers
        CNNs for image classification, LSTMs for time series, Bayesian neural networks,
        HuggingFace LLM fine-tuning with TRL and GRPO, vision-language models like Qwen2-VL,
        and video-speech training with 560B parameter models.
      </>
    ),
  },
];

function Feature({title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="padding-horiz--md padding-vert--lg">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Master distributed deep learning with DeepSpeed - from basic neural networks to advanced multimodal AI training">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
