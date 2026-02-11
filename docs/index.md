---
hide:
  - navigation
  - toc
title: Timefence — Temporal correctness for ML training data
---

<style>
.md-content__inner > h1 { display: none; }
.hero h1 { display: none !important; }
.md-sidebar--secondary { display: none; }
.md-sidebar--primary { display: none; }
.md-main__inner.md-grid { max-width: 100% !important; }
.md-content { max-width: 100%; margin: 0; padding: 0; }
.md-content__inner { padding: 0; margin: 0; max-width: 100%; }
.md-content__button { display: none; }
.hero .headerlink,
.home-section .headerlink,
.bottom-cta .headerlink { display: none !important; }
</style>

<div class="hero" markdown>

<img src="assets/logo.png" alt="Timefence" width="120" class="logo">

## Your features know the future.<br>Your model shouldn't.

<p class="tagline">
When you join features to labels, future data leaks in — no error, no warning.<br>
Timefence finds it, fixes it, and proves every row is clean.
</p>

<div class="install-block" onclick="navigator.clipboard.writeText('pip install timefence').then(function(){var el=this;el.classList.add('copied');setTimeout(function(){el.classList.remove('copied')},2000)}.bind(this))" title="Click to copy">pip install timefence<span class="copy-icon">
<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
</span><span class="copied-label">Copied!</span></div>

<div class="badges">
<a href="https://github.com/gauthierpiarrette/timefence/actions/workflows/ci.yml"><img src="https://github.com/gauthierpiarrette/timefence/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<a href="https://codecov.io/gh/gauthierpiarrette/timefence"><img src="https://codecov.io/gh/gauthierpiarrette/timefence/graph/badge.svg" alt="codecov"></a>
<a href="https://pypi.org/project/timefence/"><img src="https://img.shields.io/pypi/v/timefence" alt="PyPI"></a>
<a href="https://pypi.org/project/timefence/"><img src="https://img.shields.io/pypi/pyversions/timefence" alt="Python"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
</div>

[Get Started](getting-started/quickstart.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/gauthierpiarrette/timefence){ .md-button }

</div>

<div class="grid cards" markdown>

-   :material-shield-check: **Guaranteed Correctness**

    ---

    Enforce `feature_time < label_time` for every row. Embargo, staleness, and lookback — all configurable.

    [:octicons-arrow-right-24: Learn more](concepts/temporal-correctness.md)

-   :material-flash: **Lightning Fast**

    ---

    Built on DuckDB. Process millions of rows locally in seconds. No Spark cluster or cloud infrastructure needed.

    [:octicons-arrow-right-24: See benchmarks](getting-started/installation.md#performance)

-   :material-file-document-check: **Audit Any Dataset**

    ---

    Don't rebuild — just audit. Point at any existing training set and get a full leakage report instantly.

    [:octicons-arrow-right-24: Audit guide](guides/audit.md)

-   :material-pipe: **CI/CD Ready**

    ---

    `--strict` exits code 1 on leakage. Add temporal correctness to your pipeline in one line.

    [:octicons-arrow-right-24: CI guide](guides/ci.md)

</div>

<div class="home-section" markdown>

<h2>See it in action</h2>
<p class="section-sub">Point Timefence at any training set. If future data leaked in, it finds it.</p>

<div class="terminal-wrapper">
<div class="terminal-window">
<div class="terminal-header">
<div class="terminal-controls">
<div class="control close"></div>
<div class="control minimize"></div>
<div class="control maximize"></div>
</div>
<div class="terminal-title">timefence</div>
</div>
<div class="terminal-content">
<pre><code><span class="prompt">$</span> <span class="cmd">timefence audit train_fraud_v3.parquet</span>

<span class="dim">TEMPORAL AUDIT REPORT</span>
<span class="dim">Scanned 1,247,392 rows across 6 features</span>

<span class="warning">WARNING  LEAKAGE DETECTED in 2 of 6 features</span>

  <span class="error">LEAK</span>  <span class="cmd">merchant_risk_score</span>
        <span class="dim">41,580 rows (3.3%) — future data</span>
        <span class="dim">Severity:</span> <span class="warning">MEDIUM</span>

  <span class="error">LEAK</span>  <span class="cmd">rolling_txn_count_7d</span>
        <span class="dim">98,423 rows (7.9%) — future data</span>
        <span class="dim">Severity:</span> <span class="error">HIGH</span>

  <span class="success">OK</span>    <span class="cmd">account_age_days</span> <span class="dim">— clean</span>
  <span class="success">OK</span>    <span class="cmd">avg_txn_amount_30d</span> <span class="dim">— clean</span>
  <span class="success">OK</span>    <span class="cmd">device_fingerprint_count</span> <span class="dim">— clean</span>
  <span class="success">OK</span>    <span class="cmd">customer_tenure_months</span> <span class="dim">— clean</span>

<span class="prompt">$</span> <span class="cmd">timefence build -o train_fraud_v3_clean.parquet</span>
<span class="prompt">$</span> <span class="cmd">timefence audit train_fraud_v3_clean.parquet</span>
<span class="success">ALL CLEAN — no temporal leakage detected</span></code></pre>
</div>
</div>
</div>

<div style="text-align: center; margin-top: 2rem;" markdown>

[Try the quickstart :octicons-arrow-right-24:](getting-started/quickstart.md){ .md-button }

</div>

</div>

<div class="steps-section">
<div class="steps-header">
<h2>How it works</h2>
<p>Define your data, and Timefence handles point-in-time correctness for every row.</p>
</div>
<div class="steps-grid">
<div class="step">
<div class="step-number">1</div>
<h3>Define Sources</h3>
<p>Tell Timefence where your raw data lives and which columns represent time and entities.</p>
<pre><code>src = <span class="fn">timefence.Source</span>(
  path=<span class="st">"txns.parquet"</span>,
  keys=[<span class="st">"user_id"</span>],
  timestamp=<span class="st">"ts"</span>,
)</code></pre>
</div>
<div class="step">
<div class="step-number">2</div>
<h3>Build Clean Data</h3>
<p>For every label, Timefence finds the most recent feature value <em>strictly before</em> the label timestamp — respecting embargo, lookback, and staleness.</p>
<pre><code>df = <span class="fn">timefence.build</span>(
  features=features,
  labels=labels,
  output=<span class="st">"train.parquet"</span>,
)</code></pre>
</div>
<div class="step">
<div class="step-number">3</div>
<h3>Verify &amp; Ship</h3>
<p>Audit the output to confirm zero leakage. A build manifest records exactly what happened.</p>
<pre><code>report = <span class="fn">timefence.audit</span>(
  <span class="st">"train.parquet"</span>,
  features=features,
  labels=labels,
)
report.<span class="fn">assert_clean</span>()</code></pre>
</div>
</div>
</div>

<div class="stats-bar">
  <div class="stat">
    <div class="stat-value">1M+</div>
    <div class="stat-label">rows in seconds</div>
  </div>
  <div class="stat">
    <div class="stat-value">Zero</div>
    <div class="stat-label">infrastructure needed</div>
  </div>
  <div class="stat">
    <div class="stat-value">1 line</div>
    <div class="stat-label">to add to CI/CD</div>
  </div>
  <div class="stat">
    <div class="stat-value">100%</div>
    <div class="stat-label">row-level guarantees</div>
  </div>
</div>

<div class="bottom-cta" markdown>

<h2>Ready to find out if your training data is clean?</h2>

[Get Started in 60 Seconds](getting-started/quickstart.md){ .md-button .md-button--primary }
[Read the Docs](getting-started/installation.md){ .md-button }

</div>
