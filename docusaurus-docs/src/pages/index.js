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
    title: 'The Memory Arithmetic, Derived',
    description: (
      <>
        Mixed-precision Adam costs <strong>16 bytes per parameter</strong>, so a
        7B model needs 112&nbsp;GB of model states before a single activation is
        allocated. This course derives that number, shows why ZeRO stages 1 and 2
        eliminate the redundancy at <em>zero</em> extra communication, and why
        stage 3 costs exactly 1.5&times;. Every recommendation elsewhere follows
        from it.
      </>
    ),
  },
  {
    title: 'Worked Examples, Honestly Documented',
    description: (
      <>
        Fourteen runnable examples from a two-parameter linear model to a 560B
        omni-modal system. Including the failures: a CIFAR-10 run that produced
        NaN at exactly chance accuracy and how it was diagnosed to root cause, a
        look-ahead-bias bug in the stock example, and which examples are
        infrastructure tests rather than trainable models.
      </>
    ),
  },
  {
    title: 'Verifiable Without a Cluster',
    description: (
      <>
        Most of these examples cannot run on a laptop &mdash; so the repository
        ships logic tests that validate configs, data handling and reward
        functions with <strong>no GPU and no model download</strong>. They run in
        CI on every push, and have already caught a config bug that manual review
        missed. <code>./tests/run_all.sh</code>
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
