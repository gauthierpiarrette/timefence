# FeatureSet

A named group of features for reuse across builds and audits.

::: timefence.FeatureSet
    options:
      show_root_heading: true
      members: []

## Why use FeatureSet

When you have a stable set of features that get reused across multiple models or experiments, group them into a `FeatureSet` instead of managing individual lists. This gives you:

- A **single named reference** for a group of features
- **Version tracking** — name the set `"churn_features_v2"` and swap it in one place
- **Composability** — combine FeatureSets with additional features in `build()` and `audit()`

## Example

```python
import timefence

# Define individual features
rolling_spend = timefence.Feature(source=transactions, columns=["amount"], embargo="1d", name="spend")
user_country = timefence.Feature(source=users, columns=["country"])
login_count = timefence.Feature(source=logins, columns=["login_count"])

# Group into a named set
base_features = timefence.FeatureSet(
    name="churn_features_v1",
    features=[rolling_spend, user_country, login_count],
)

# Use directly in build() — Timefence flattens it automatically
result = timefence.build(
    labels=labels,
    features=[base_features],
    output="train.parquet",
)
```

## Combining FeatureSets with individual features

You can mix FeatureSets and individual Features in any call:

```python
# Add an experimental feature alongside the base set
result = timefence.build(
    labels=labels,
    features=[base_features, new_experimental_feature],
    output="train.parquet",
)

# Audit with the same features
report = timefence.audit(
    data="train.parquet",
    features=[base_features],
    keys=["user_id"],
    label_time="label_time",
)
```

## Iteration and length

FeatureSet supports standard Python iteration:

```python
print(len(base_features))  # 3

for feature in base_features:
    print(feature.name)  # "spend", "country", "login_count"
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Human-readable name for this group. |
| `features` | `Sequence[Feature]` | List of Feature objects. |

!!! note
    FeatureSets are flat — they contain only `Feature` objects, not other FeatureSets. There is no nesting.
